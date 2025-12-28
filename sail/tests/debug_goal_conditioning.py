"""
Debug script to check if goal conditioning is actually working.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sail.policies.goal_conditioned_act import (
    GoalConditionedACTPolicy,
    GoalConditionedACTConfig,
)


def debug_goal_conditioning(
    model_path: str = "outputs/sail_phase2_overnight/checkpoint_005000.pt",
    dataset_path: str = r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2",
):
    print("=" * 60)
    print("Debugging Goal Conditioning")
    print("=" * 60)
    
    device = "cuda"
    
    # Load model
    print("\n[1] Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    config = GoalConditionedACTConfig.from_dict(checkpoint["config"])
    policy = GoalConditionedACTPolicy(config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    print(f"✓ Model loaded. Goal fusion: {config.goal_fusion}")
    
    # Load dataset
    print("\n[2] Loading dataset...")
    dataset = LeRobotDataset(dataset_path)
    
    # Get sample frames
    frame1 = dataset[0]  # Episode 0, start
    frame2 = dataset[400]  # Episode 0, middle-ish
    
    # Find a frame from a different episode
    for i in range(len(dataset)):
        if dataset[i]["episode_index"].item() != 0:
            frame3 = dataset[i]
            break
    
    camera = "gripper"  # Use gripper camera
    
    # Prepare observation (same for all tests)
    obs_img = frame1[f"observation.images.{camera}"].unsqueeze(0).to(device)
    station_img = frame1["observation.images.station"].unsqueeze(0).to(device)
    images = [obs_img, station_img]
    state = frame1["observation.state"].unsqueeze(0).to(device)
    
    print("\n[3] Testing goal encoder...")
    
    # Get different goals
    goal1 = frame1[f"observation.images.{camera}"].unsqueeze(0).to(device)  # Same as obs
    goal2 = frame2[f"observation.images.{camera}"].unsqueeze(0).to(device)  # Same episode, later
    goal3 = frame3[f"observation.images.{camera}"].unsqueeze(0).to(device)  # Different episode
    
    # Encode goals
    with torch.no_grad():
        emb1 = policy.encode_goal(goal1)
        emb2 = policy.encode_goal(goal2)
        emb3 = policy.encode_goal(goal3)
    
    diff_12 = (emb1 - emb2).abs().mean().item()
    diff_13 = (emb1 - emb3).abs().mean().item()
    diff_23 = (emb2 - emb3).abs().mean().item()
    
    print(f"  Goal embedding differences:")
    print(f"    Same episode (start vs mid): {diff_12:.6f}")
    print(f"    Different episodes (ep0 vs ep1): {diff_13:.6f}")
    print(f"    Different episodes (mid vs ep1): {diff_23:.6f}")
    
    if diff_12 < 0.001 and diff_13 < 0.001:
        print("  ⚠️ PROBLEM: Goal encoder outputs nearly identical embeddings!")
        print("     The goal encoder might not be trained or connected properly.")
    else:
        print("  ✓ Goal encoder produces different embeddings for different goals.")
    
    print("\n[4] Testing full forward pass...")
    
    # Forward pass with different goals
    with torch.no_grad():
        out1 = policy(images, state, goal1)
        out2 = policy(images, state, goal2)
        out3 = policy(images, state, goal3)
    
    act_diff_12 = (out1["action"] - out2["action"]).abs().mean().item()
    act_diff_13 = (out1["action"] - out3["action"]).abs().mean().item()
    
    print(f"  Action differences:")
    print(f"    Same episode goals: {act_diff_12:.6f}")
    print(f"    Different episode goals: {act_diff_13:.6f}")
    
    if act_diff_12 < 0.001 and act_diff_13 < 0.001:
        print("  ⚠️ PROBLEM: Actions are identical regardless of goal!")
        print("     The goal is not influencing the output.")
    else:
        print("  ✓ Actions differ based on goal!")
    
    print("\n[5] Checking goal fusion mechanism...")
    
    # Check if cross-attention is being used
    if hasattr(policy, 'goal_cross_attn'):
        print("  ✓ Cross-attention module exists")
    else:
        print("  ⚠️ Cross-attention module NOT found!")
    
    # Check FiLM
    if hasattr(policy, 'goal_film'):
        print("  ✓ FiLM module exists")
    else:
        print("  ℹ️ FiLM module not used (expected for cross_attention)")
    
    print("\n[6] Checking encoder features before/after goal fusion...")
    
    # We need to trace through the forward pass
    with torch.no_grad():
        # Encode images
        image_features = policy.encode_images(images)
        state_features = policy.encode_state(state)
        
        # Get latent (zeros for inference)
        z = torch.zeros(1, config.latent_dim, device=device)
        latent_features = policy.encoder_latent_input_proj(z).unsqueeze(1)
        
        # Add positional embeddings
        pos_embed = policy.encoder_1d_feature_pos_embed.weight.unsqueeze(0)
        latent_features = latent_features + pos_embed[:, 0:1]
        state_features = state_features + pos_embed[:, 1:2]
        
        # Before fusion
        encoder_input_before = torch.cat([latent_features, state_features, image_features], dim=1)
        
        # With goal 1
        goal_emb1 = policy.encode_goal(goal1)
        encoder_input_with_goal1 = policy.fuse_goal(encoder_input_before.clone(), goal_emb1)
        
        # With goal 3 (different episode)
        goal_emb3 = policy.encode_goal(goal3)
        encoder_input_with_goal3 = policy.fuse_goal(encoder_input_before.clone(), goal_emb3)
        
        feature_diff = (encoder_input_with_goal1 - encoder_input_with_goal3).abs().mean().item()
        
    print(f"  Feature difference after goal fusion: {feature_diff:.6f}")
    
    if feature_diff < 0.001:
        print("  ⚠️ PROBLEM: Goal fusion is not changing the features!")
    else:
        print("  ✓ Goal fusion is modifying features!")
    
    print("\n" + "=" * 60)
    print("Debug Summary")
    print("=" * 60)
    
    issues = []
    if diff_12 < 0.001 and diff_13 < 0.001:
        issues.append("Goal encoder not differentiating goals")
    if act_diff_12 < 0.001 and act_diff_13 < 0.001:
        issues.append("Actions not influenced by goal")
    if feature_diff < 0.001:
        issues.append("Goal fusion not working")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected - goal conditioning should be working!")
    
    return len(issues) == 0


if __name__ == "__main__":
    debug_goal_conditioning()

