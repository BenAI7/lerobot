"""
Test if robot actually moves when we send actions.

This script:
1. Gets current joint positions
2. Sends a small movement command
3. Checks if robot moved
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def test_robot_movement(port: str):
    """Test if robot responds to action commands."""
    
    print("\n" + "=" * 60)
    print("Test: Robot Movement")
    print("=" * 60)
    
    # Create robot config
    camera_config = {
        "gripper": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "station": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    }
    
    robot_cfg = SO101FollowerConfig(
        port=port,
        id="follower_so101",
        cameras=camera_config
    )
    
    # Connect robot
    print("Connecting to robot...")
    robot = SO101Follower(robot_cfg)
    robot.connect(calibrate=False)  # Skip calibration
    print("✓ Robot connected")
    
    try:
        # Get initial position
        print("\n1. Getting initial position...")
        initial_obs = robot.get_observation()
        initial_pos = {
            key: value for key, value in initial_obs.items()
            if key.endswith('.pos') and not key.startswith('observation')
        }
        print("Initial joint positions:")
        for joint, pos in initial_pos.items():
            print(f"  {joint}: {pos:.2f}")
        
        # Create a small movement action (move shoulder_pan by 5 degrees)
        print("\n2. Sending movement command (+5 degrees on shoulder_pan)...")
        target_action = {}
        for joint, pos in initial_pos.items():
            if joint == 'shoulder_pan.pos':
                target_action[joint] = pos + 5.0  # Move 5 degrees
            else:
                target_action[joint] = pos  # Keep same position
        
        print("Target positions:")
        for joint, pos in target_action.items():
            print(f"  {joint}: {pos:.2f}")
        
        # Send action
        robot.send_action(target_action)
        print("✓ Action sent")
        
        # Wait for movement
        print("\n3. Waiting 2 seconds for movement...")
        time.sleep(2.0)
        
        # Get final position
        print("\n4. Getting final position...")
        final_obs = robot.get_observation()
        final_pos = {
            key: value for key, value in final_obs.items()
            if key.endswith('.pos') and not key.startswith('observation')
        }
        print("Final joint positions:")
        for joint, pos in final_pos.items():
            print(f"  {joint}: {pos:.2f}")
        
        # Check movement
        print("\n5. Checking movement...")
        print("Position changes:")
        moved = False
        for joint in initial_pos:
            diff = final_pos[joint] - initial_pos[joint]
            print(f"  {joint}: {diff:+.2f} degrees")
            if abs(diff) > 1.0:  # Moved more than 1 degree
                moved = True
        
        if moved:
            print("\n✓ SUCCESS: Robot moved in response to actions!")
        else:
            print("\n✗ FAIL: Robot did NOT move (all changes < 1 degree)")
            print("\nPossible issues:")
            print("  - Actions might be in wrong format")
            print("  - Robot might be in wrong mode")
            print("  - Motor torque might be disabled")
        
        return moved
        
    finally:
        # Disconnect robot
        if robot.is_connected:
            robot.disconnect()
            print("\nRobot disconnected.")


def main():
    parser = argparse.ArgumentParser(description="Test robot movement")
    parser.add_argument(
        "--robot-port",
        type=str,
        required=True,
        help="Robot COM port (e.g., COM6)"
    )
    
    args = parser.parse_args()
    
    success = test_robot_movement(args.robot_port)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Robot responds to actions correctly!")
    else:
        print("✗ Robot movement test failed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

