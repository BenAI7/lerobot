"""
Test the unified Reward Manager (VIP + VLM combined).
Run: python sail/test_reward_manager.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sail.rewards.reward_manager import create_reward_manager


def load_dataset():
    """Load your pick-and-place dataset."""
    dataset_path = Path("C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2")
    
    if dataset_path.exists():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(str(dataset_path))
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("yeyian/self_improve_pickplace_v2")
    
    return dataset


def test_reward_manager():
    """Test unified reward manager on your dataset."""
    
    print("=" * 60)
    print("Testing Unified Reward Manager (VIP + VLM)")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = load_dataset()
        print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Create reward manager
    print("\nCreating Reward Manager...")
    print("(This loads both VIP and VLM models)")
    
    manager = create_reward_manager(
        device="cuda",
        use_vlm=True,  # Set to False for faster testing without VLM
        vip_model="resnet50",
        vlm_model="Qwen/Qwen2-VL-2B-Instruct"
    )
    
    # Test on one episode
    print("\nTesting on Episode 0...")
    
    task_description = "pick up the cube and place it in the target zone"
    
    # Find frames for episode 0
    # Iterate through dataset to find frames with episode_index == 0
    episode_frames = []
    for idx in range(min(1000, len(dataset))):  # Check first 1000 frames
        frame = dataset[idx]
        if frame["episode_index"].item() == 0:
            episode_frames.append(idx)
        elif len(episode_frames) > 0:
            # We've passed episode 0
            break
    
    if not episode_frames:
        print("✗ Could not find episode 0 frames")
        return False
    
    initial_img = dataset[episode_frames[0]]["observation.images.station"]
    final_img = dataset[episode_frames[-1]]["observation.images.station"]
    
    # Set up for episode
    manager.set_initial(initial_img)
    manager.set_goal(final_img)  # Using final frame as goal
    
    # Compute step reward at different points
    print("\n--- Step Rewards (VIP only) ---")
    
    num_samples = 5
    sample_indices = np.linspace(0, len(episode_frames) - 1, num_samples, dtype=int)
    sample_indices = [episode_frames[i] for i in sample_indices]
    
    for i, idx in enumerate(sample_indices):
        frame = dataset[idx]["observation.images.station"]
        signals = manager.compute_step_reward(frame)
        
        progress_bar = "█" * int(signals.vip_progress * 20) + "░" * (20 - int(signals.vip_progress * 20))
        print(f"Frame {i+1}/{num_samples}: [{progress_bar}] {signals.vip_progress:.3f}")
    
    # Compute episode reward (with VLM)
    print("\n--- Episode Reward (VIP + VLM) ---")
    
    final_signals = manager.compute_episode_reward(final_img, task_description)
    
    print(f"VIP Progress: {final_signals.vip_progress:.3f}")
    print(f"VIP Reward: {final_signals.vip_reward:.3f}")
    print(f"VLM Success: {'YES ✓' if final_signals.vlm_success else 'NO ✗'}")
    print(f"VLM Confidence: {final_signals.vlm_confidence:.3f}")
    print(f"Combined Reward: {final_signals.combined_reward:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Reward Manager Test Complete!")
    print("=" * 60)
    
    print("\nThe Reward Manager is working if:")
    print("1. VIP progress increases from 0 → 1 through episode")
    print("2. VLM gives a yes/no answer with confidence")
    print("3. Combined reward reflects both signals")
    
    print("\nNext step: python sail/scripts/validate_rewards.py")
    
    return True


if __name__ == "__main__":
    test_reward_manager()

