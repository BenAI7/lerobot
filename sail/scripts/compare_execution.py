"""
Compare action execution between LeRobot baseline and SAIL.

This script logs:
1. Predicted actions from policy
2. Actual robot positions after each action
3. Action magnitudes and ranges

This helps identify differences in action scaling, normalization, or execution.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import time

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR
from lerobot.policies.utils import make_robot_action


class ExecutionLogger:
    """Logs detailed execution data for comparison."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.actions = []
        self.positions = []
        self.step_times = []
        
    def log_step(self, step: int, predicted_action: Dict, robot_position: Dict):
        """Log a single execution step."""
        self.actions.append({
            'step': step,
            'action': {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                      for k, v in predicted_action.items()}
        })
        self.positions.append({
            'step': step,
            'position': {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                        for k, v in robot_position.items() if k.endswith('.pos')}
        })
        
    def compute_stats(self):
        """Compute statistics about the actions."""
        if not self.actions:
            return {}
            
        # Extract action values
        all_action_values = []
        for action_log in self.actions:
            action = action_log['action']
            values = [abs(v) for v in action.values() if isinstance(v, (int, float))]
            all_action_values.extend(values)
        
        # Extract position changes
        position_changes = []
        if len(self.positions) > 1:
            for i in range(1, len(self.positions)):
                prev = self.positions[i-1]['position']
                curr = self.positions[i]['position']
                for key in prev.keys():
                    if key in curr:
                        change = abs(curr[key] - prev[key])
                        position_changes.append(change)
        
        return {
            'method': self.method_name,
            'num_steps': len(self.actions),
            'action_stats': {
                'mean': float(np.mean(all_action_values)) if all_action_values else 0,
                'std': float(np.std(all_action_values)) if all_action_values else 0,
                'min': float(np.min(all_action_values)) if all_action_values else 0,
                'max': float(np.max(all_action_values)) if all_action_values else 0,
            },
            'position_change_stats': {
                'mean': float(np.mean(position_changes)) if position_changes else 0,
                'std': float(np.std(position_changes)) if position_changes else 0,
                'min': float(np.min(position_changes)) if position_changes else 0,
                'max': float(np.max(position_changes)) if position_changes else 0,
                'total_movement': float(np.sum(position_changes)) if position_changes else 0,
            }
        }
    
    def save_to_file(self, output_path: Path):
        """Save logs to JSON file."""
        stats = self.compute_stats()
        data = {
            'stats': stats,
            'actions': self.actions[:20],  # First 20 actions for inspection
            'positions': self.positions[:20],
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"{self.method_name} Execution Report")
        print(f"{'='*60}")
        print(f"Total steps: {stats['num_steps']}")
        print(f"\nAction Statistics:")
        print(f"  Mean magnitude: {stats['action_stats']['mean']:.2f}")
        print(f"  Std: {stats['action_stats']['std']:.2f}")
        print(f"  Range: [{stats['action_stats']['min']:.2f}, {stats['action_stats']['max']:.2f}]")
        print(f"\nPosition Change Statistics:")
        print(f"  Mean change: {stats['position_change_stats']['mean']:.2f} degrees")
        print(f"  Total movement: {stats['position_change_stats']['total_movement']:.2f} degrees")
        print(f"  Range: [{stats['position_change_stats']['min']:.2f}, {stats['position_change_stats']['max']:.2f}]")
        print(f"\nFull log saved: {output_path}")
        print(f"{'='*60}\n")


def run_sail_method(policy_path: str, dataset_path: str, robot_port: str, num_steps: int = 300):
    """Run SAIL's method and log execution."""
    
    print("\n" + "="*60)
    print("RUNNING SAIL METHOD")
    print("="*60)
    
    logger = ExecutionLogger("SAIL")
    
    # Load policy
    print("Loading policy...")
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    
    # Load dataset for metadata
    dataset = LeRobotDataset(dataset_path)
    
    # Create preprocessor/postprocessor - MUST load from pretrained!
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,  # Critical: load normalization from checkpoint!
        dataset_stats=dataset.meta.stats if hasattr(dataset.meta, 'stats') else None
    )
    
    # Connect robot
    print("Connecting to robot...")
    robot_config = SO101FollowerConfig(
        port=robot_port,
        cameras={
            "gripper": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "station": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        }
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    
    # Reset to home
    print("Resetting to home position...")
    home_position = {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': -100.0,
        'elbow_flex.pos': 100.0,
        'wrist_flex.pos': 67.0,
        'wrist_roll.pos': 10.0,
        'gripper.pos': 9.0,
    }
    robot.send_action(home_position)
    time.sleep(2.0)
    
    # Reset policy
    if hasattr(policy, 'reset'):
        policy.reset()
    
    print(f"Executing {num_steps} steps with SAIL method...")
    
    for step in range(num_steps):
        # Get observation
        obs = robot.get_observation()
        
        # Build dataset frame (SAIL's way)
        obs_frame = build_dataset_frame(
            dataset.meta.features,
            obs,
            prefix=OBS_STR
        )
        
        # Predict action using SAIL's method
        action = predict_action(
            observation=obs_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task="pick up the cube and place it in the target zone",
            robot_type="so101_follower",
        )
        
        # Format action for robot
        robot_action = make_robot_action(action, dataset.meta.features)
        
        # Get current position before sending action
        current_pos = robot.get_observation()
        
        # Log
        logger.log_step(step, robot_action, current_pos)
        
        # Execute action
        robot.send_action(robot_action)
        
        # Progress indicator
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}")
        
        # Maintain FPS
        time.sleep(1.0 / 30)
    
    print("SAIL execution complete!")
    robot.disconnect()
    
    # Save logs
    output_dir = Path("outputs/execution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.save_to_file(output_dir / "sail_execution.json")
    
    return logger


def run_baseline_method(policy_path: str, dataset_path: str, robot_port: str, num_steps: int = 300):
    """Run LeRobot baseline method and log execution."""
    
    print("\n" + "="*60)
    print("RUNNING LEROBOT BASELINE METHOD")
    print("="*60)
    
    logger = ExecutionLogger("LeRobot Baseline")
    
    # Load policy
    print("Loading policy...")
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    
    # Load dataset for metadata
    dataset = LeRobotDataset(dataset_path)
    
    # Create preprocessor/postprocessor - MUST load from pretrained!
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,  # Critical: load normalization from checkpoint!
        dataset_stats=dataset.meta.stats if hasattr(dataset.meta, 'stats') else None
    )
    
    # Connect robot
    print("Connecting to robot...")
    robot_config = SO101FollowerConfig(
        port=robot_port,
        cameras={
            "gripper": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "station": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        }
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    
    # Reset to home (same as SAIL)
    print("Resetting to home position...")
    home_position = {
        'shoulder_pan.pos': 0.0,
        'shoulder_lift.pos': -100.0,
        'elbow_flex.pos': 100.0,
        'wrist_flex.pos': 67.0,
        'wrist_roll.pos': 10.0,
        'gripper.pos': 9.0,
    }
    robot.send_action(home_position)
    time.sleep(2.0)
    
    # Reset policy
    if hasattr(policy, 'reset'):
        policy.reset()
    
    print(f"Executing {num_steps} steps with baseline method...")
    
    for step in range(num_steps):
        # Get observation
        obs = robot.get_observation()
        
        # Build dataset frame (same as lerobot_record.py)
        obs_frame = build_dataset_frame(
            dataset.meta.features,
            obs,
            prefix=OBS_STR
        )
        
        # Predict action (same as lerobot_record.py)
        action = predict_action(
            observation=obs_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task="pick up cube",  # Baseline uses this
            robot_type="so101_follower",
        )
        
        # Format action for robot
        robot_action = make_robot_action(action, dataset.meta.features)
        
        # Get current position before sending action
        current_pos = robot.get_observation()
        
        # Log
        logger.log_step(step, robot_action, current_pos)
        
        # Execute action
        robot.send_action(robot_action)
        
        # Progress indicator
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}")
        
        # Maintain FPS
        time.sleep(1.0 / 30)
    
    print("Baseline execution complete!")
    robot.disconnect()
    
    # Save logs
    output_dir = Path("outputs/execution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.save_to_file(output_dir / "baseline_execution.json")
    
    return logger


def compare_results(sail_logger: ExecutionLogger, baseline_logger: ExecutionLogger):
    """Compare and report differences."""
    
    sail_stats = sail_logger.compute_stats()
    baseline_stats = baseline_logger.compute_stats()
    
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    
    print("\nAction Magnitude Comparison:")
    print(f"  SAIL Mean:     {sail_stats['action_stats']['mean']:.2f}")
    print(f"  Baseline Mean: {baseline_stats['action_stats']['mean']:.2f}")
    print(f"  Difference:    {abs(sail_stats['action_stats']['mean'] - baseline_stats['action_stats']['mean']):.2f}")
    
    print("\nTotal Robot Movement Comparison:")
    print(f"  SAIL Total:     {sail_stats['position_change_stats']['total_movement']:.2f} degrees")
    print(f"  Baseline Total: {baseline_stats['position_change_stats']['total_movement']:.2f} degrees")
    print(f"  Ratio:          {sail_stats['position_change_stats']['total_movement'] / baseline_stats['position_change_stats']['total_movement'] if baseline_stats['position_change_stats']['total_movement'] > 0 else 0:.2%}")
    
    print("\n" + "="*60)
    print("Files saved to: outputs/execution_comparison/")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare SAIL vs Baseline execution")
    parser.add_argument("--policy", required=True, help="Path to policy checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--robot-port", default="COM6", help="Robot port")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps to run")
    parser.add_argument("--method", choices=["sail", "baseline", "both"], default="both", 
                       help="Which method to run")
    
    args = parser.parse_args()
    
    sail_logger = None
    baseline_logger = None
    
    if args.method in ["sail", "both"]:
        sail_logger = run_sail_method(args.policy, args.dataset, args.robot_port, args.steps)
    
    if args.method in ["baseline", "both"]:
        baseline_logger = run_baseline_method(args.policy, args.dataset, args.robot_port, args.steps)
    
    if sail_logger and baseline_logger:
        compare_results(sail_logger, baseline_logger)

