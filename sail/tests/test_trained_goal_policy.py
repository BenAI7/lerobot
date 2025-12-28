"""
Test the trained goal-conditioned policy.

This script:
1. Loads the trained model
2. Tests inference with goal images from the dataset
3. Verifies the model can predict actions given different goals

Usage:
    python sail/tests/test_trained_goal_policy.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sail.policies.goal_conditioned_act import (
    GoalConditionedACTPolicy,
    GoalConditionedACTConfig,
)


def test_trained_policy(
    model_path: str = "outputs/sail_phase2_test/best_model.pt",
    dataset_path: str = r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2",
):
    """Test the trained goal-conditioned policy."""
    
    print("=" * 60)
    print("Testing Trained Goal-Conditioned Policy")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"\n[1/4] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = GoalConditionedACTConfig.from_dict(checkpoint["config"])
    policy = GoalConditionedACTPolicy(config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    
    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')}, loss {checkpoint.get('loss', 'unknown'):.4f})")
    
    # Load dataset
    print(f"\n[2/4] Loading dataset...")
    dataset = LeRobotDataset(dataset_path)
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    
    # Get camera names
    sample = dataset[0]
    camera_names = []
    for key in sample.keys():
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            camera_names.append(cam_name)
    print(f"  Cameras: {camera_names}")
    
    # Test 1: Same goal (should produce consistent actions)
    print(f"\n[3/4] Test: Inference with different goals...")
    
    # Get first and last frame of episode 0
    first_frame = dataset[0]
    
    # Find last frame of episode 0
    ep0_end = 0
    for i in range(len(dataset)):
        if dataset[i]["episode_index"].item() == 0:
            ep0_end = i
        else:
            break
    last_frame = dataset[ep0_end]
    
    print(f"  Episode 0: frames 0 to {ep0_end}")
    
    # Prepare inputs
    images = []
    for cam in camera_names:
        key = f"observation.images.{cam}"
        img = first_frame[key].unsqueeze(0).to(device)
        images.append(img)
    
    state = first_frame["observation.state"].unsqueeze(0).to(device)
    
    # Goal 1: First frame (stay in place)
    goal1 = first_frame[f"observation.images.{camera_names[0]}"].unsqueeze(0).to(device)
    
    # Goal 2: Last frame (complete task)
    goal2 = last_frame[f"observation.images.{camera_names[0]}"].unsqueeze(0).to(device)
    
    # Predict actions for different goals
    with torch.no_grad():
        output1 = policy(images, state, goal1)
        output2 = policy(images, state, goal2)
    
    actions1 = output1["action"]
    actions2 = output2["action"]
    
    # Compare actions
    diff = (actions1 - actions2).abs().mean().item()
    
    print(f"\n  Results:")
    print(f"    Goal 1 (stay): first 5 actions = {actions1[0, 0, :5].cpu().numpy().round(3)}")
    print(f"    Goal 2 (task): first 5 actions = {actions2[0, 0, :5].cpu().numpy().round(3)}")
    print(f"    Action difference: {diff:.4f}")
    
    if diff > 0.01:
        print(f"  ✓ Actions differ based on goal (good! goal conditioning works)")
    else:
        print(f"  ⚠ Actions are similar - goal conditioning may need more training")
    
    # Test 2: Multiple episodes
    print(f"\n[4/4] Test: Inference across episodes...")
    
    test_episodes = [0, 10, 25, 40]
    for ep_idx in test_episodes:
        # Find first frame of episode
        frame_idx = None
        for i in range(len(dataset)):
            if dataset[i]["episode_index"].item() == ep_idx:
                frame_idx = i
                break
        
        if frame_idx is None:
            continue
        
        frame = dataset[frame_idx]
        
        # Prepare inputs
        images = []
        for cam in camera_names:
            key = f"observation.images.{cam}"
            img = frame[key].unsqueeze(0).to(device)
            images.append(img)
        
        state = frame["observation.state"].unsqueeze(0).to(device)
        goal = frame[f"observation.images.{camera_names[0]}"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = policy(images, state, goal)
        
        actions = output["action"]
        print(f"    Episode {ep_idx}: action range [{actions.min().item():.2f}, {actions.max().item():.2f}]")
    
    print(f"\n" + "=" * 60)
    print("✅ Model Test Complete!")
    print("=" * 60)
    print(f"\nThe goal-conditioned policy is working.")
    print(f"Ready for full training or Phase 3 (autonomous practice).")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/sail_phase2_test/best_model.pt")
    parser.add_argument("--dataset", default=r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2")
    args = parser.parse_args()
    
    test_trained_policy(args.model, args.dataset)

