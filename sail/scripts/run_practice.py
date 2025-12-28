"""
Run autonomous practice session with robot.

This script executes the full autonomous practice loop:
1. Load baseline ACT policy
2. Connect to robot
3. Execute practice episodes
4. Evaluate with VLM/VIP
5. Save successful trajectories
6. Periodically retrain policy
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sail.autonomous.practice_loop import PracticeConfig, run_autonomous_practice


def main():
    parser = argparse.ArgumentParser(description="Run autonomous practice session")
    
    # Policy
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to baseline ACT checkpoint (directory or .pt file)"
    )
    
    # Robot
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so101_follower",
        help="Robot type (default: so101_follower)"
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default=None,
        help="Robot COM port (e.g., COM3). Auto-detect if not specified"
    )
    
    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the cube and place it in the target zone",
        help="Task description for VLM evaluation"
    )
    
    # Practice settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of practice episodes (default: 50)"
    )
    parser.add_argument(
        "--retrain-freq",
        type=int,
        default=10,
        help="Retrain every N successful episodes (default: 10)"
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=900,
        help="Max frames per episode (default: 900 = 30s at 30fps)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency (default: 30)"
    )
    
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (for adding new trajectories)"
    )
    
    # Evaluation
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Use VLM for success detection (slower but more accurate)"
    )
    parser.add_argument(
        "--use-vip",
        action="store_true",
        default=True,
        help="Use VIP for progress tracking (default: True)"
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.8,
        help="VIP progress threshold for success (default: 0.8)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sail_practice",
        help="Output directory (default: outputs/sail_practice)"
    )
    parser.add_argument(
        "--no-save-videos",
        action="store_true",
        help="Don't save episode videos"
    )
    
    # Safety
    parser.add_argument(
        "--no-emergency-stop",
        action="store_true",
        help="Disable emergency stop checking"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PracticeConfig(
        policy_path=args.policy,
        policy_type="act",
        robot_type=args.robot_type,
        robot_port=args.robot_port,
        task_description=args.task,
        num_practice_episodes=args.episodes,
        retrain_frequency=args.retrain_freq,
        episode_length=args.episode_length,
        fps=args.fps,
        dataset_path=args.dataset,
        use_vlm=args.use_vlm,
        use_vip=args.use_vip,
        success_threshold=args.success_threshold,
        output_dir=args.output,
        save_videos=not args.no_save_videos,
        save_logs=True,
        emergency_stop=not args.no_emergency_stop,
    )
    
    # Run practice session
    print("\n" + "=" * 60)
    print("Starting Autonomous Practice Session")
    print("=" * 60)
    print(f"Policy: {config.policy_path}")
    print(f"Robot: {config.robot_type} ({config.robot_port or 'auto-detect'})")
    print(f"Task: {config.task_description}")
    print(f"Episodes: {config.num_practice_episodes}")
    print(f"Retrain frequency: {config.retrain_frequency}")
    print("=" * 60 + "\n")
    
    session = run_autonomous_practice(config)
    
    print("\n" + "=" * 60)
    print("Practice Session Complete!")
    print("=" * 60)
    print(f"Results saved to: {session.session_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

