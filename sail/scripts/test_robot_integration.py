"""
Test robot integration for autonomous practice.

This script tests:
1. Robot connection
2. Policy loading
3. Single episode execution
4. Observation/action processing

Run this before running the full practice loop!
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sail.autonomous.practice_loop import PracticeConfig, PracticeSession


def test_robot_connection(config: PracticeConfig):
    """Test robot connection only."""
    print("\n" + "=" * 60)
    print("Test 1: Robot Connection")
    print("=" * 60)
    
    session = PracticeSession(config)
    
    try:
        session.load_robot()
        print("✓ Robot connected successfully!")
        
        # Test getting observation
        print("\nTesting observation...")
        obs = session.robot.get_observation()
        print(f"✓ Got observation with keys: {list(obs.keys())}")
        
        # Test sending a safe action (all zeros = stay still)
        print("\nTesting action sending...")
        # Get action format from dataset
        session.load_dataset()
        action_format = {}
        for key in session.dataset_metadata.features:
            if key.startswith("action."):
                action_format[key] = 0.0  # Safe default
        
        if action_format:
            session.robot.send_action(action_format)
            print("✓ Action sent successfully (zero action = stay still)")
        else:
            print("⚠️  Could not determine action format")
        
        return True
        
    except Exception as e:
        print(f"✗ Robot connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if session.robot and session.robot.is_connected:
            session.robot.disconnect()
            print("\nRobot disconnected.")


def test_policy_loading(config: PracticeConfig):
    """Test policy loading."""
    print("\n" + "=" * 60)
    print("Test 2: Policy Loading")
    print("=" * 60)
    
    session = PracticeSession(config)
    
    try:
        session.load_policy()
        print("✓ Policy loaded successfully!")
        print(f"  Policy type: {type(session.policy).__name__}")
        print(f"  Device: {next(session.policy.parameters()).device}")
        print(f"  Preprocessor: {type(session.preprocessor).__name__}")
        print(f"  Postprocessor: {type(session.postprocessor).__name__}")
        return True
        
    except Exception as e:
        print(f"✗ Policy loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_episode(config: PracticeConfig, num_steps: int = 10):
    """Test executing a single short episode."""
    print("\n" + "=" * 60)
    print(f"Test 3: Single Episode Execution ({num_steps} steps)")
    print("=" * 60)
    
    session = PracticeSession(config)
    
    try:
        # Load all components
        session.load_policy()
        session.load_robot()
        session.load_dataset()
        
        print("\nExecuting episode...")
        print("(Robot will move - watch carefully!)")
        
        observations = []
        actions = []
        
        for step in range(num_steps):
            print(f"  Step {step+1}/{num_steps}...", end=" ", flush=True)
            
            # Get observation
            obs = session.robot.get_observation()
            observations.append(obs)
            
            # Prepare observation
            from lerobot.policies.utils import build_inference_frame
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=session.dataset_metadata.features,
                device=session.device
            )
            obs_processed = session.preprocessor(obs_frame)
            
            # Predict action
            import torch
            with torch.no_grad():
                action = session.policy.select_action(obs_processed)
                action = session.postprocessor(action)
            
            # Format for robot
            from lerobot.policies.utils import make_robot_action
            robot_action = make_robot_action(action, session.dataset_metadata.features)
            actions.append(robot_action)
            
            # Execute action
            session.robot.send_action(robot_action)
            
            time.sleep(1.0 / config.fps)
            print("✓")
        
        print(f"\n✓ Episode completed successfully!")
        print(f"  Collected {len(observations)} observations")
        print(f"  Executed {len(actions)} actions")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Episode execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if session.robot and session.robot.is_connected:
            session.robot.disconnect()
            print("\nRobot disconnected.")


def main():
    parser = argparse.ArgumentParser(description="Test robot integration")
    
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
        "--test",
        type=str,
        choices=["all", "robot", "policy", "episode"],
        default="all",
        help="Which test to run (default: all)"
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=10,
        help="Number of steps for episode test (default: 10)"
    )
    
    args = parser.parse_args()
    
    config = PracticeConfig(
        policy_path=args.policy,
        robot_type="so101_follower",
        robot_port=args.robot_port,
        task_description="test",
        dataset_path=args.dataset,
        num_practice_episodes=1,
        retrain_frequency=10,
        episode_length=args.episode_steps,
        fps=30,
        use_vlm=False,
        use_vip=False,
    )
    
    results = {}
    
    if args.test in ["all", "robot"]:
        results["robot"] = test_robot_connection(config)
    
    if args.test in ["all", "policy"]:
        results["policy"] = test_policy_loading(config)
    
    if args.test in ["all", "episode"]:
        if results.get("robot", True) and results.get("policy", True):
            results["episode"] = test_single_episode(config, args.episode_steps)
        else:
            print("\n⚠️  Skipping episode test (robot or policy test failed)")
            results["episode"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper()}: {status}")
    print("=" * 60)
    
    if all(results.values()):
        print("\n🎉 All tests passed! Ready for full practice loop.")
    else:
        print("\n⚠️  Some tests failed. Fix issues before running practice loop.")


if __name__ == "__main__":
    main()

