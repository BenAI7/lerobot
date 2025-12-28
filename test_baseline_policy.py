"""
Test baseline policy directly on robot (no SAIL, just policy execution).

This will show us if the policy works correctly on its own.
"""

import torch
import time
import numpy as np
from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def test_policy_on_robot(
    policy_path: str,
    dataset_path: str,
    robot_port: str,
    num_episodes: int = 3,
    episode_length: int = 300,
):
    """Run policy on robot and observe behavior."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "=" * 60)
    print("Testing Baseline Policy on Robot")
    print("=" * 60)
    
    # Load policy
    print(f"\n1. Loading policy from: {policy_path}")
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()
    print("[OK] Policy loaded")
    
    # Load dataset metadata
    print(f"\n2. Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_path)
    print(f"[OK] Dataset metadata loaded")
    
    # Create preprocessors
    print(f"\n3. Creating pre/post processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats
    )
    print("[OK] Processors created")
    
    # Connect to robot
    print(f"\n4. Connecting to robot on {robot_port}...")
    camera_config = {
        "gripper": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "station": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    }
    robot_cfg = SO101FollowerConfig(
        port=robot_port,
        id="follower_so101",
        cameras=camera_config
    )
    robot = SO101Follower(robot_cfg)
    robot.connect(calibrate=False)
    print("[OK] Robot connected")
    
    # Run episodes
    print(f"\n5. Running {num_episodes} episodes...")
    print("=" * 60)
    
    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        
        # Reset policy
        if hasattr(policy, 'reset'):
            policy.reset()
            print("Policy reset")
        
        # Reset to home position
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
        
        # Get starting position
        obs = robot.get_observation()
        print(f"Starting position:")
        print(f"  shoulder_pan: {obs['shoulder_pan.pos']:.2f}°")
        print(f"  shoulder_lift: {obs['shoulder_lift.pos']:.2f}°")
        print(f"  elbow_flex: {obs['elbow_flex.pos']:.2f}°")
        print(f"  wrist_flex: {obs['wrist_flex.pos']:.2f}°")
        
        # Execute episode
        print(f"\nExecuting {episode_length} steps...")
        
        # Track positions for analysis
        positions = []
        actions_sent = []
        
        for step in range(episode_length):
            # Get observation
            obs = robot.get_observation()
            
            # Track current position
            if step % 50 == 0:
                pos = [
                    obs['shoulder_pan.pos'],
                    obs['shoulder_lift.pos'],
                    obs['elbow_flex.pos'],
                    obs['wrist_flex.pos']
                ]
                positions.append(pos)
            
            # Prepare observation for policy
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_metadata.features,
                device=device
            )
            obs_processed = preprocessor(obs_frame)
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(obs_processed)
                action = postprocessor(action)
            
            # Format and send action
            robot_action = make_robot_action(action, dataset_metadata.features)
            robot.send_action(robot_action)
            
            # Track action
            if step < 5 or step % 50 == 0:
                actions_sent.append({
                    'step': step,
                    'shoulder_pan': robot_action['shoulder_pan.pos'],
                    'shoulder_lift': robot_action['shoulder_lift.pos'],
                    'elbow_flex': robot_action['elbow_flex.pos'],
                    'wrist_flex': robot_action['wrist_flex.pos'],
                })
                
                # Print progress
                if step < 5:
                    print(f"  Step {step+1}: Commanded shoulder_pan={robot_action['shoulder_pan.pos']:.1f}°, shoulder_lift={robot_action['shoulder_lift.pos']:.1f}°")
            
            # Maintain FPS
            time.sleep(1.0 / 30.0)
        
        # Analyze movement
        print(f"\n[OK] Episode {ep+1} completed")
        print(f"\nPosition changes during episode:")
        if len(positions) > 1:
            for i, pos in enumerate(positions):
                print(f"  Step {i*50}: shoulder_pan={pos[0]:.1f}°, shoulder_lift={pos[1]:.1f}°, elbow={pos[2]:.1f}°")
            
            # Calculate total movement
            total_movement = sum(
                abs(positions[-1][j] - positions[0][j]) 
                for j in range(4)
            )
            print(f"\n  Total joint movement: {total_movement:.1f}° (sum of absolute changes)")
            
            if total_movement < 10:
                print(f"  [WARNING] Very little movement! Robot might be stuck.")
            elif total_movement > 100:
                print(f"  [OK] Good movement range - robot is executing task")
            else:
                print(f"  ~ Moderate movement")
        
        print(f"\nActions commanded:")
        for act in actions_sent[-5:]:  # Last 5 actions
            print(f"  Step {act['step']}: pan={act['shoulder_pan']:.1f}°, lift={act['shoulder_lift']:.1f}°")
        
        if ep < num_episodes - 1:
            print(f"\nPress Enter to continue to next episode...")
            input()
    
    # Cleanup
    robot.disconnect()
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Run with your settings
    test_policy_on_robot(
        policy_path=r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\train_session_01\checkpoints\040000\pretrained_model",
        dataset_path=r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2",
        robot_port="COM6",
        num_episodes=1,
        episode_length=300,
    )

