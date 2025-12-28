"""
Debug script for autonomous practice.

Checks:
1. Policy predictions
2. Action scaling/normalization
3. Observation processing
4. VIP evaluation
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sail.autonomous.practice_loop import PracticeConfig, PracticeSession


def debug_policy_predictions(session: PracticeSession, num_steps: int = 10):
    """Check what the policy is predicting."""
    print("\n" + "=" * 60)
    print("Debug: Policy Predictions")
    print("=" * 60)
    
    from lerobot.policies.utils import build_inference_frame
    
    # Get observations
    print("Getting observations from robot...")
    observations = []
    for i in range(num_steps):
        obs = session.robot.get_observation()
        observations.append(obs)
        print(f"  Step {i+1}: Got observation with keys: {list(obs.keys())[:5]}...")
    
    print(f"\n✓ Collected {len(observations)} observations")
    
    # Check observation values
    print("\nSample observation values:")
    obs = observations[0]
    for key, value in list(obs.items())[:5]:
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, min={value.min():.3f}, max={value.max():.3f}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Process observations and get predictions
    print("\n\nProcessing observations through policy...")
    actions_raw = []
    actions_processed = []
    actions_robot = []
    
    for i, obs in enumerate(observations):
        # Prepare observation
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=session.dataset_metadata.features,
            device=session.device
        )
        obs_processed = session.preprocessor(obs_frame)
        
        # Get policy prediction
        with torch.no_grad():
            action = session.policy.select_action(obs_processed)
            actions_raw.append(action.clone())
            
            action_post = session.postprocessor(action)
            actions_processed.append(action_post.clone())
            
            # Format for robot
            from lerobot.policies.utils import make_robot_action
            robot_action = make_robot_action(action_post, session.dataset_metadata.features)
            actions_robot.append(robot_action)
    
    print(f"✓ Processed {len(actions_raw)} predictions")
    
    # Analyze predictions
    print("\n\nAction Analysis:")
    print("-" * 60)
    
    # Check raw actions
    actions_raw_tensor = torch.stack(actions_raw).cpu().numpy()
    print(f"\nRaw actions (before postprocessing):")
    print(f"  Shape: {actions_raw_tensor.shape}")
    print(f"  Min: {actions_raw_tensor.min():.4f}")
    print(f"  Max: {actions_raw_tensor.max():.4f}")
    print(f"  Mean: {actions_raw_tensor.mean():.4f}")
    print(f"  Std: {actions_raw_tensor.std():.4f}")
    print(f"\n  First action: {actions_raw_tensor[0, 0, :6]}")
    print(f"  Last action:  {actions_raw_tensor[-1, 0, :6]}")
    
    # Check processed actions
    actions_proc_tensor = torch.stack(actions_processed).cpu().numpy()
    print(f"\nProcessed actions (after postprocessing):")
    print(f"  Shape: {actions_proc_tensor.shape}")
    print(f"  Min: {actions_proc_tensor.min():.4f}")
    print(f"  Max: {actions_proc_tensor.max():.4f}")
    print(f"  Mean: {actions_proc_tensor.mean():.4f}")
    print(f"  Std: {actions_proc_tensor.std():.4f}")
    print(f"\n  First action: {actions_proc_tensor[0, 0, :6]}")
    print(f"  Last action:  {actions_proc_tensor[-1, 0, :6]}")
    
    # Check robot actions
    print(f"\nRobot actions (formatted for robot):")
    print(f"  First action keys: {list(actions_robot[0].keys())}")
    print(f"  First action values:")
    for key, value in actions_robot[0].items():
        if isinstance(value, (int, float)):
            print(f"    {key}: {value:.4f}")
        elif isinstance(value, torch.Tensor):
            print(f"    {key}: {value.item():.4f}")
    
    # Check action diversity
    print(f"\n\nAction Diversity Check:")
    print(f"  Action range (std across time):")
    for dim in range(min(6, actions_proc_tensor.shape[-1])):
        std_across_time = actions_proc_tensor[:, 0, dim].std()
        print(f"    Dim {dim}: std={std_across_time:.4f}")
    
    if actions_proc_tensor[:, 0, :].std() < 0.01:
        print("\n  ⚠️  WARNING: Actions have very low variance!")
        print("  Policy might be stuck or actions aren't being scaled correctly")
    
    return observations, actions_robot


def debug_vip_evaluation(session: PracticeSession, observations: list):
    """Check VIP evaluation."""
    print("\n" + "=" * 60)
    print("Debug: VIP Evaluation")
    print("=" * 60)
    
    if not session.vip or len(observations) < 2:
        print("Skipping (no VIP or not enough observations)")
        return
    
    initial_obs = observations[0]
    final_obs = observations[-1]
    
    # Extract images
    print("Extracting images...")
    try:
        initial_image = session._extract_image(initial_obs, camera="station")
        final_image = session._extract_image(final_obs, camera="station")
        print(f"  Initial image shape: {initial_image.shape}")
        print(f"  Final image shape: {final_image.shape}")
        
        # Check if images are different
        if np.array_equal(initial_image, final_image):
            print("  ⚠️  WARNING: Initial and final images are IDENTICAL!")
        else:
            diff = np.abs(initial_image.astype(float) - final_image.astype(float)).mean()
            print(f"  Image difference (mean pixel): {diff:.2f}")
    
    except Exception as e:
        print(f"  ✗ Failed to extract images: {e}")
        return
    
    # Prepare for VIP
    def prepare_image_for_vip(img):
        if len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).to(session.device)
        return tensor
    
    initial_tensor = prepare_image_for_vip(initial_image)
    final_tensor = prepare_image_for_vip(final_image)
    
    print("\nComputing VIP progress...")
    print(f"  Initial tensor shape: {initial_tensor.shape}")
    print(f"  Final tensor shape: {final_tensor.shape}")
    
    # Compute VIP progress
    vip_progress = session.vip.compute_progress(
        final_tensor,
        initial_tensor,
        final_tensor  # Using final as goal
    ).item()
    
    print(f"\n  VIP Progress: {vip_progress:.4f}")
    
    if vip_progress > 0.95:
        print("  ⚠️  WARNING: Very high VIP progress!")
        print("  This might indicate the robot didn't move much")
        print("  (VIP compares current to goal, and we're using final as goal)")


def main():
    parser = argparse.ArgumentParser(description="Debug autonomous practice")
    
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to baseline ACT checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default=None,
        help="Robot COM port (e.g., COM3)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps to debug (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Create session
    config = PracticeConfig(
        policy_path=args.policy,
        robot_type="so101_follower",
        robot_port=args.robot_port,
        task_description="test",
        dataset_path=args.dataset,
        num_practice_episodes=1,
        retrain_frequency=10,
        episode_length=args.steps,
        fps=30,
        use_vlm=False,
        use_vip=True,
    )
    
    print("Initializing session...")
    session = PracticeSession(config)
    session.load_policy()
    session.load_robot()
    session.load_evaluators()
    session.load_dataset()
    
    # Debug policy predictions
    observations, actions = debug_policy_predictions(session, args.steps)
    
    # Debug VIP evaluation
    debug_vip_evaluation(session, observations)
    
    # Cleanup
    if session.robot and session.robot.is_connected:
        session.robot.disconnect()
    
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

