"""
Test VIP reward on your existing dataset.
Run: python sail/test_vip.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sail.rewards.vip import create_vip_reward


def load_dataset():
    """Load your pick-and-place dataset."""
    # Try to load from local path first
    dataset_path = Path("C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2")
    
    if dataset_path.exists():
        print(f"Loading dataset from local path...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(str(dataset_path))
    else:
        # Try HuggingFace Hub
        print(f"Loading dataset from HuggingFace Hub...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("yeyian/self_improve_pickplace_v2")
    
    return dataset


def test_vip_on_dataset():
    """Test VIP reward using your actual pick-and-place dataset."""
    
    print("=" * 60)
    print("Testing VIP Reward Model")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = load_dataset()
        print(f"✓ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("\nTrying with synthetic data instead...")
        dataset = None
    
    # Create VIP model
    print("\nCreating VIP model...")
    vip = create_vip_reward(model_name="resnet50", device="cuda")
    
    if dataset is not None:
        # Test on real data
        print("\nTesting VIP progress computation on episodes...")
        
        # Get episode indices
        test_episodes = [0, min(10, dataset.num_episodes-1), min(25, dataset.num_episodes-1)]
        test_episodes = list(set(test_episodes))  # Remove duplicates
        
        for ep_idx in test_episodes:
            print(f"\n--- Episode {ep_idx} ---")
            
            try:
                # Find frames for this episode
                episode_frames = []
                for idx in range(len(dataset)):
                    frame = dataset[idx]
                    if frame["episode_index"].item() == ep_idx:
                        episode_frames.append(idx)
                    elif len(episode_frames) > 0:
                        break  # We've passed this episode
                
                if not episode_frames:
                    print(f"No frames found for episode {ep_idx}")
                    continue
                
                # Get first, middle, and last frames
                first_idx = episode_frames[0]
                middle_idx = episode_frames[len(episode_frames) // 2]
                last_idx = episode_frames[-1]
                
                # Get frames (use station camera for workspace view)
                initial_img = dataset[first_idx]["observation.images.station"]
                middle_img = dataset[middle_idx]["observation.images.station"]
                final_img = dataset[last_idx]["observation.images.station"]
                
                # Compute progress at different timepoints
                progress_at_start = vip.compute_progress(
                    initial_img.unsqueeze(0),
                    initial_img.unsqueeze(0),
                    final_img.unsqueeze(0)
                )
                
                progress_at_middle = vip.compute_progress(
                    middle_img.unsqueeze(0),
                    initial_img.unsqueeze(0),
                    final_img.unsqueeze(0)
                )
                
                progress_at_end = vip.compute_progress(
                    final_img.unsqueeze(0),
                    initial_img.unsqueeze(0),
                    final_img.unsqueeze(0)
                )
                
                print(f"Progress at start:  {progress_at_start.item():.3f} (expect ~0.0)")
                print(f"Progress at middle: {progress_at_middle.item():.3f} (expect ~0.3-0.7)")
                print(f"Progress at end:    {progress_at_end.item():.3f} (expect ~1.0)")
                
                # Check if progress makes sense
                if progress_at_end.item() > progress_at_start.item():
                    print("✓ Progress increases correctly!")
                else:
                    print("⚠️  Progress doesn't increase - check camera/data")
                    
            except Exception as e:
                print(f"Error processing episode {ep_idx}: {e}")
                
    else:
        # Test with synthetic data
        print("\nTesting VIP with synthetic data...")
        
        # Create dummy images
        initial = torch.rand(1, 3, 480, 640).cuda()
        current = torch.rand(1, 3, 480, 640).cuda()
        goal = torch.rand(1, 3, 480, 640).cuda()
        
        # Test encoding
        emb = vip.encode(initial)
        print(f"✓ Embedding shape: {emb.shape}")
        
        # Test reward
        reward = vip.compute_reward(current, goal)
        print(f"✓ Reward computed: {reward.item():.4f}")
        
        # Test progress
        progress = vip.compute_progress(current, initial, goal)
        print(f"✓ Progress computed: {progress.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ VIP Test Complete!")
    print("=" * 60)
    
    if dataset is not None:
        print("\nIf progress values make sense (0 → middle → 1),")
        print("VIP is working correctly for your task!")
        print("\nNext step: python sail/test_vlm.py")
    
    return True


if __name__ == "__main__":
    test_vip_on_dataset()

