"""
Autonomous Practice Loop for SAIL

The robot practices tasks autonomously:
1. Execute episode with current policy
2. VLM judges success
3. Store trajectory to dataset
4. Periodically retrain policy
5. Track improvement over time

Based on SOAR paper's autonomous practice approach.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR

from sail.rewards.vlm_detector import VLMSuccessDetector
from sail.rewards.vip import VIPReward


@dataclass
class PracticeConfig:
    """Configuration for autonomous practice."""
    
    # Policy
    policy_path: str  # Path to baseline ACT checkpoint
    policy_type: str = "act"
    
    # Robot
    robot_type: str = "so101_follower"
    robot_port: str = None  # Will auto-detect if None
    
    # Task
    task_description: str = "pick up the cube and place it in the target zone"
    episode_length: int = 900  # Max frames per episode (30 sec at 30fps)
    fps: int = 30
    
    # Practice settings
    num_practice_episodes: int = 50  # How many practice attempts
    retrain_frequency: int = 10  # Retrain every N successful episodes
    
    # Dataset
    dataset_path: str = None  # Path to local dataset
    dataset_repo_id: str = None  # Or HuggingFace repo ID
    
    # Evaluation
    use_vlm: bool = True  # Use VLM for success detection
    use_vip: bool = True  # Use VIP for progress tracking
    success_threshold: float = 0.8  # VIP progress threshold
    
    # Output
    output_dir: str = "outputs/sail_practice"
    save_videos: bool = True
    save_logs: bool = True
    
    # Safety
    max_retries: int = 3  # Max retries if episode fails
    emergency_stop: bool = True


@dataclass
class EpisodeResult:
    """Result of a single practice episode."""
    episode_num: int
    success: bool
    vlm_success: bool
    vlm_confidence: float
    vip_progress: float
    duration: float  # seconds
    num_frames: int
    trajectory_saved: bool
    error: Optional[str] = None


class PracticeSession:
    """Manages an autonomous practice session."""
    
    def __init__(self, config: PracticeConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_path / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results: List[EpisodeResult] = []
        self.num_successes = 0
        self.num_failures = 0
        
        # Initialize components (lazy loading)
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.robot = None
        self.vlm = None
        self.vip = None
        self.dataset = None
        self.dataset_metadata = None
        
        print(f"Practice session initialized: {self.session_id}")
    
    def load_policy(self):
        """Load the ACT policy and pre/post processors."""
        if self.policy is not None:
            return
        
        print(f"Loading policy from: {self.config.policy_path}")
        
        # Load policy using from_pretrained (handles checkpoint loading)
        # If policy_path is a directory, it should contain model files
        # If it's a .pt file, we need to handle it differently
        policy_path = Path(self.config.policy_path)
        
        if policy_path.is_file() and policy_path.suffix == ".pt":
            # Legacy checkpoint format - load manually
            print("  Loading from .pt checkpoint...")
            checkpoint = torch.load(policy_path, map_location=self.device)
            
            # Try to get config from checkpoint
            if "config" in checkpoint:
                from lerobot.policies.act.configuration_act import ACTConfig
                if isinstance(checkpoint["config"], dict):
                    policy_config = ACTConfig.from_dict(checkpoint["config"])
                else:
                    policy_config = checkpoint["config"]
            else:
                # Fallback: try to infer from dataset
                self.load_dataset()
                if self.dataset_metadata:
                    # Use dataset stats to create config
                    from lerobot.policies.act.configuration_act import ACTConfig
                    policy_config = ACTConfig()
                else:
                    raise ValueError("Cannot determine policy config. Need dataset metadata.")
            
            self.policy = ACTPolicy(policy_config)
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.policy.to(self.device)
            self.policy.eval()
        else:
            # Directory or HuggingFace repo - use from_pretrained
            self.policy = ACTPolicy.from_pretrained(str(policy_path))
            self.policy.to(self.device)
            self.policy.eval()
        
        # Load dataset metadata for pre/post processors
        self.load_dataset()
        
        # Create pre/post processors - MUST load from pretrained to get correct normalization!
        print("  Creating pre/post processors...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.config.policy_path),  # Critical: load from checkpoint!
            dataset_stats=self.dataset_metadata.stats if self.dataset_metadata else None
        )
        
        print(f"[OK] Policy loaded")
    
    def load_robot(self):
        """Initialize robot connection."""
        if self.robot is not None:
            return
        
        print(f"Connecting to robot: {self.config.robot_type}")
        
        # Load dataset metadata to get camera config
        self.load_dataset()
        
        # Get camera configuration from dataset metadata
        camera_config = {}
        if self.dataset_metadata and hasattr(self.dataset_metadata, 'info'):
            # Extract camera info from dataset
            info = self.dataset_metadata.info
            if 'cameras' in info:
                for cam_name, cam_info in info['cameras'].items():
                    camera_config[cam_name] = OpenCVCameraConfig(
                        index_or_path=0,  # Will need to be set correctly
                        width=cam_info.get('width', 640),
                        height=cam_info.get('height', 480),
                        fps=cam_info.get('fps', 30)
                    )
        
        # Default camera config if not found
        if not camera_config:
            camera_config = {
                "gripper": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
                "station": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
            }
        
        # Create robot config
        robot_cfg = SO101FollowerConfig(
            port=self.config.robot_port,
            id="follower_so101",  # Default ID, can be configured
            cameras=camera_config
        )
        
        # Create and connect robot
        self.robot = SO101Follower(robot_cfg)
        # Connect with calibrate=False to skip calibration if already calibrated
        self.robot.connect(calibrate=False)
        
        # Check cameras
        if hasattr(self.robot, 'cameras') and self.robot.cameras:
            print(f"  Cameras configured: {list(self.robot.cameras.keys())}")
            # Test camera read
            try:
                test_obs = self.robot.get_observation()
                camera_keys = [k for k, v in test_obs.items() 
                             if isinstance(v, np.ndarray) and len(v.shape) >= 2 and not k.endswith('.pos')]
                if camera_keys:
                    print(f"  [OK] Cameras working: {camera_keys}")
                else:
                    print(f"  [WARNING] Cameras configured but not returning images")
            except Exception as e:
                print(f"  [WARNING] Camera test failed: {e}")
        else:
            print(f"  [WARNING] No cameras configured!")
        
        print(f"[OK] Robot connected")
    
    def load_evaluators(self):
        """Load VLM and VIP evaluators."""
        if self.config.use_vlm and self.vlm is None:
            print("Loading VLM success detector...")
            self.vlm = VLMSuccessDetector(device=self.device, use_4bit=True)
            print("[OK] VLM loaded")
        
        if self.config.use_vip and self.vip is None:
            print("Loading VIP reward model...")
            self.vip = VIPReward(device=self.device)
            print("[OK] VIP loaded")
    
    def load_dataset(self):
        """Load dataset and metadata for adding new trajectories."""
        if self.dataset is not None and self.dataset_metadata is not None:
            return
        
        dataset_path = self.config.dataset_path or self.config.dataset_repo_id
        if not dataset_path:
            raise ValueError("Must provide dataset_path or dataset_repo_id")
        
        print(f"Loading dataset: {dataset_path}")
        self.dataset = LeRobotDataset(dataset_path)
        self.dataset_metadata = LeRobotDatasetMetadata(dataset_path)
        print(f"[OK] Dataset loaded: {self.dataset.num_episodes} episodes")
    
    def execute_episode(self, episode_num: int) -> EpisodeResult:
        """Execute a single practice episode."""
        
        start_time = time.time()
        
        try:
            # Ensure robot is ready (skip if method doesn't exist)
            if hasattr(self.robot, 'teleop_safety_stop'):
                self.robot.teleop_safety_stop()  # Reset safety
            
            # Reset policy state (clear action queue)
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
                print(f"  Policy reset (cleared action queue)")
            
            # Collect trajectory
            observations = []
            actions = []
            
            # IMPORTANT: Reset to home/sitting position
            # The policy was trained with robot starting from this position
            print(f"  Resetting to home position...")
            home_position = {
                'shoulder_pan.pos': 0.0,
                'shoulder_lift.pos': -100.0,  # Sitting position
                'elbow_flex.pos': 100.0,       # Folded
                'wrist_flex.pos': 67.0,
                'wrist_roll.pos': 10.0,
                'gripper.pos': 9.0,  # Open
            }
            self.robot.send_action(home_position)
            time.sleep(2.0)  # Wait for robot to reach home position
            
            # Get initial observation to check starting state
            initial_obs_check = self.robot.get_observation()
            print(f"  Starting state:")
            for key in ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos']:
                if key in initial_obs_check:
                    print(f"    {key}: {initial_obs_check[key]:.2f}")
            
            # Check if images are in observation (critical for policy)
            # Images come as camera keys (e.g., "gripper", "station") or numpy arrays
            image_keys = []
            for key, value in initial_obs_check.items():
                # Check if it's an image (numpy array with 2+ dimensions, not a joint position)
                if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                    if not key.endswith('.pos'):  # Not a joint position
                        image_keys.append(key)
            
            if image_keys:
                print(f"  [OK] Images found: {len(image_keys)} cameras ({', '.join(image_keys)})")
            else:
                print(f"  [WARNING] No images in observation!")
                print(f"    Available keys: {list(initial_obs_check.keys())[:10]}")
                print(f"    Policy needs images to work! Check camera configuration.")
            
            # Run episode
            print(f"  Executing episode {episode_num} ({self.config.episode_length} steps)...")
            
            # Track action magnitudes for debugging
            action_magnitudes = []
            
            for step in range(self.config.episode_length):
                # Get observation from robot
                obs = self.robot.get_observation()
                observations.append(obs)
                
                # Build dataset frame from observation (converts to expected format)
                obs_frame = build_dataset_frame(
                    self.dataset_metadata.features,
                    obs,
                    prefix=OBS_STR
                )
                
                # Use the standard LeRobot predict_action function (same as lerobot-record)
                # This handles observation preparation, preprocessing, policy inference, and postprocessing
                action = predict_action(
                    observation=obs_frame,
                    policy=self.policy,
                    device=get_safe_torch_device(self.policy.config.device),
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                    use_amp=self.policy.config.use_amp,
                    task=self.config.task_description,
                    robot_type=self.config.robot_type,
                )
                
                # Format action for robot
                robot_action = make_robot_action(action, self.dataset_metadata.features)
                actions.append(robot_action)  # Store as dict
                
                # Track action magnitude (for debugging)
                if step < 5 or step % 50 == 0:  # Log first 5 and every 50th
                    action_values = [v if isinstance(v, (int, float)) else v.item() if isinstance(v, torch.Tensor) else 0.0 
                                   for v in robot_action.values()]
                    max_action = max(abs(v) for v in action_values if isinstance(v, (int, float)))
                    action_magnitudes.append(max_action)
                    if step < 5:
                        print(f"    Step {step+1}: max action magnitude = {max_action:.2f}")
                
                # Execute action
                self.robot.send_action(robot_action)
                
                # Sleep to maintain FPS
                time.sleep(1.0 / self.config.fps)
                
                # Progress indicator
                if (step + 1) % 50 == 0:
                    avg_magnitude = sum(action_magnitudes[-10:]) / min(10, len(action_magnitudes)) if action_magnitudes else 0
                    print(f"    Step {step+1}/{self.config.episode_length}... (avg action: {avg_magnitude:.2f})")
                
                # Check for emergency stop (if robot supports it)
                if self.config.emergency_stop and hasattr(self.robot, 'is_stopped') and self.robot.is_stopped():
                    raise RuntimeError("Emergency stop triggered")
            
            # Check if actions were too small
            if action_magnitudes:
                avg_magnitude = sum(action_magnitudes) / len(action_magnitudes)
                max_magnitude = max(action_magnitudes)
                print(f"  Episode {episode_num} completed ({len(observations)} steps)")
                print(f"    Action stats: avg={avg_magnitude:.2f}, max={max_magnitude:.2f}")
                if avg_magnitude < 5.0:
                    print(f"    [WARNING] Actions are very small (avg < 5 degrees)")
                    print(f"    Policy might be stuck or actions not scaled correctly")
            
            # Episode complete - evaluate
            duration = time.time() - start_time
            print(f"  Evaluating episode {episode_num}...")
            
            # Get final observation for evaluation
            final_obs = observations[-1]
            initial_obs = observations[0]
            
            # VLM evaluation
            vlm_success = False
            vlm_confidence = 0.0
            if self.vlm:
                try:
                    # Extract image from observation
                    final_image = self._extract_image(final_obs, camera="station")
                    vlm_success, vlm_confidence = self.vlm.detect_success(
                        final_image, 
                        self.config.task_description
                    )
                    print(f"    VLM: {vlm_success} (confidence: {vlm_confidence:.2f})")
                except Exception as e:
                    print(f"    VLM evaluation failed: {e}")
            
            # VIP evaluation
            vip_progress = 0.0
            if self.vip:
                try:
                    # For autonomous practice, we don't have a goal image
                    # Instead, measure how much the scene changed (robot movement proxy)
                    initial_image = self._extract_image(initial_obs, camera="station")
                    final_image = self._extract_image(final_obs, camera="station")
                    
                    # Simple approach: measure visual similarity/difference
                    # If images are very different, robot moved a lot
                    pixel_diff = np.abs(final_image.astype(float) - initial_image.astype(float)).mean()
                    
                    # Normalize pixel difference to [0, 1] range
                    # Typical successful episodes have 10-50 mean pixel difference
                    # Scale so that 30+ difference = high progress
                    vip_progress = min(pixel_diff / 30.0, 1.0)
                    
                    print(f"    Visual change: {pixel_diff:.1f} pixels -> VIP progress: {vip_progress:.2f}")
                    
                    # Alternative: Use VIP embeddings to measure distance
                    # But we'd need a proper goal image from demonstrations
                    
                except Exception as e:
                    print(f"    VIP evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Determine success
            success = vlm_success if self.vlm else (vip_progress > self.config.success_threshold)
            
            # Save trajectory if successful
            trajectory_saved = False
            if success:
                self._save_trajectory(episode_num, observations, actions)
                trajectory_saved = True
                self.num_successes += 1
            else:
                self.num_failures += 1
            
            result = EpisodeResult(
                episode_num=episode_num,
                success=success,
                vlm_success=vlm_success,
                vlm_confidence=vlm_confidence,
                vip_progress=vip_progress,
                duration=duration,
                num_frames=len(observations),
                trajectory_saved=trajectory_saved,
            )
            
        except Exception as e:
            # Episode failed
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  ✗ Episode {episode_num} failed: {error_msg}")
            
            result = EpisodeResult(
                episode_num=episode_num,
                success=False,
                vlm_success=False,
                vlm_confidence=0.0,
                vip_progress=0.0,
                duration=duration,
                num_frames=len(observations) if 'observations' in locals() else 0,
                trajectory_saved=False,
                error=error_msg,
            )
            self.num_failures += 1
        
        self.results.append(result)
        return result
    
    def _extract_image(self, obs: Dict, camera: str = "station") -> np.ndarray:
        """Extract image from observation for VLM/VIP."""
        # Try different possible keys
        possible_keys = [
            f"observation.images.{camera}",
            f"images.{camera}",
            camera,
        ]
        
        img = None
        for key in possible_keys:
            if key in obs:
                img = obs[key]
                break
        
        if img is None:
            # Try to find any image key
            for key, value in obs.items():
                if "image" in key.lower() or camera in key.lower():
                    img = value
                    break
        
        if img is None:
            raise KeyError(f"Camera {camera} not found in observation. Available keys: {list(obs.keys())}")
        
        # Ensure correct format (HWC, uint8)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Handle different formats
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Handle CHW -> HWC conversion
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        return img
    
    def _save_trajectory(self, episode_num: int, observations: List, actions: List):
        """Save successful trajectory to dataset."""
        # TODO: Implement trajectory saving to LeRobot dataset format
        # This requires adding new episode to the dataset
        
        trajectory_path = self.session_dir / f"episode_{episode_num:04d}.pt"
        torch.save({
            "observations": observations,
            "actions": actions,
            "episode_num": episode_num,
        }, trajectory_path)
        
        print(f"  Saved trajectory: {trajectory_path.name}")
    
    def retrain_policy(self):
        """Retrain policy on expanded dataset."""
        print("\n" + "=" * 60)
        print("Retraining policy on expanded dataset...")
        print("=" * 60)
        
        # TODO: Implement retraining
        # This would call the training script with updated dataset
        
        # For now, just log that retraining would happen
        print("[WARNING] Retraining not implemented yet")
        print("    In production, this would:")
        print("    1. Merge new trajectories into dataset")
        print("    2. Run ACT training script")
        print("    3. Load new checkpoint")
    
    def run(self):
        """Run the full practice session."""
        print("\n" + "=" * 60)
        print(f"Starting Autonomous Practice Session")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Task: {self.config.task_description}")
        print(f"Episodes: {self.config.num_practice_episodes}")
        print(f"Retrain frequency: {self.config.retrain_frequency}")
        print("=" * 60)
        
        # Load components
        self.load_policy()
        self.load_robot()
        self.load_evaluators()
        self.load_dataset()
        
        # Practice loop
        print(f"\nStarting practice loop...")
        
        for ep_num in tqdm(range(1, self.config.num_practice_episodes + 1), desc="Practice"):
            result = self.execute_episode(ep_num)
            
            # Log result
            status = "[SUCCESS]" if result.success else "[FAIL]"
            tqdm.write(f"Episode {ep_num}: {status} (VLM: {result.vlm_success}, VIP: {result.vip_progress:.2f})")
            
            # Check if retrain needed
            if result.trajectory_saved and self.num_successes % self.config.retrain_frequency == 0:
                self.retrain_policy()
                # Reload policy after retraining
                # self.policy = None
                # self.load_policy()
        
        # Session complete
        self._save_session_summary()
        self._print_summary()
    
    def _save_session_summary(self):
        """Save session summary to JSON."""
        summary_path = self.session_dir / "summary.json"
        
        summary = {
            "session_id": self.session_id,
            "config": {
                "task": self.config.task_description,
                "num_episodes": self.config.num_practice_episodes,
                "retrain_frequency": self.config.retrain_frequency,
            },
            "results": {
                "total_episodes": len(self.results),
                "successes": self.num_successes,
                "failures": self.num_failures,
                "success_rate": self.num_successes / len(self.results) if self.results else 0.0,
            },
            "episodes": [
                {
                    "episode": int(r.episode_num),
                    "success": bool(r.success),
                    "vlm_success": bool(r.vlm_success),
                    "vlm_confidence": float(r.vlm_confidence),
                    "vip_progress": float(r.vip_progress),
                    "duration": float(r.duration),
                    "saved": bool(r.trajectory_saved),
                    "error": str(r.error) if r.error else None,
                }
                for r in self.results
            ]
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSession summary saved: {summary_path}")
    
    def _print_summary(self):
        """Print session summary."""
        print("\n" + "=" * 60)
        print("Practice Session Complete!")
        print("=" * 60)
        print(f"Total episodes: {len(self.results)}")
        print(f"Successes: {self.num_successes}")
        print(f"Failures: {self.num_failures}")
        print(f"Success rate: {self.num_successes / len(self.results) * 100:.1f}%")
        print(f"Output: {self.session_dir}")
        print("=" * 60)


def run_autonomous_practice(config: PracticeConfig):
    """Run autonomous practice session."""
    session = PracticeSession(config)
    session.run()
    return session

