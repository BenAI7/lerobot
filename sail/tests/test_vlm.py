"""
Test VLM success detection on your dataset.
Run: python sail/test_vlm.py

Note: This will download ~4GB model on first run.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sail.rewards.vlm_detector import create_vlm_detector


def load_dataset():
    """Load your pick-and-place dataset."""
    dataset_path = Path("C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2")
    
    if dataset_path.exists():
        print(f"Loading dataset from local path...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(str(dataset_path))
    else:
        print(f"Loading dataset from HuggingFace Hub...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("yeyian/self_improve_pickplace_v2")
    
    return dataset


def test_vlm_on_dataset():
    """Test VLM success detection on your pick-and-place episodes."""
    
    print("=" * 60)
    print("Testing VLM Success Detector")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = load_dataset()
        print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("\nRunning with synthetic test instead...")
        dataset = None
    
    # Create VLM detector
    print("\nCreating VLM detector...")
    print("⏳ This downloads ~4GB model on first run. Please wait...")
    
    try:
        vlm = create_vlm_detector(device="cuda", use_4bit=False)
    except Exception as e:
        print(f"\n✗ Failed to load VLM: {e}")
        print("\nPossible fixes:")
        print("1. Install transformers: pip install transformers>=4.36.0")
        print("2. Check CUDA memory (need ~4-6GB free)")
        print("3. Try smaller model by editing vlm_detector.py")
        return False
    
    # Test task description
    task_description = "pick up the cube and place it in the target zone"
    
    print(f"\nTask: '{task_description}'")
    print("\nLabeling final frames of episodes...")
    
    results = []
    
    if dataset is not None:
        # Test on real data
        test_episodes = [0, 5, 10, 15, 20]
        test_episodes = [ep for ep in test_episodes if ep < dataset.num_episodes]
        
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
                        break
                
                if not episode_frames:
                    print(f"No frames found for episode {ep_idx}")
                    continue
                
                final_img = dataset[episode_frames[-1]]["observation.images.station"]
                
                # Detect success
                success, confidence = vlm.detect_success(final_img, task_description)
                
                print(f"VLM says: {'SUCCESS ✓' if success else 'FAILURE ✗'} (confidence: {confidence:.2f})")
                
                results.append({
                    "episode": ep_idx,
                    "vlm_success": success,
                    "vlm_confidence": confidence
                })
                
            except Exception as e:
                print(f"Error on episode {ep_idx}: {e}")
    else:
        # Test with synthetic image
        print("\nTesting with synthetic image...")
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        success, confidence = vlm.detect_success(dummy_img, task_description)
        print(f"VLM response: {'SUCCESS' if success else 'FAILURE'} (confidence: {confidence:.2f})")
        print("(This is random data, so result doesn't mean anything)")
    
    print("\n" + "=" * 60)
    print("✅ VLM Test Complete!")
    print("=" * 60)
    
    if results:
        print("\nSummary:")
        successes = sum(1 for r in results if r["vlm_success"])
        print(f"VLM labeled {successes}/{len(results)} episodes as SUCCESS")
        
        print("\nNow manually verify these results by watching the episodes.")
        print("Does VLM agree with your judgment?")
        print("Target: >75% agreement")
    
    print("\nNext step: python sail/test_reward_manager.py")
    
    return True


if __name__ == "__main__":
    test_vlm_on_dataset()

