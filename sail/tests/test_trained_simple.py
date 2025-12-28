"""Simple test of trained model - loads directly without config mismatch."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def test_simple(
    model_path="outputs/sail_phase2_proof/final_model.pt",
    dataset_path=r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2",
):
    print("=" * 60)
    print("Simple Goal Test")
    print("=" * 60)
    
    device = "cuda"
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load the FULL model state (not recreating from config)
    from sail.policies.goal_conditioned_act import create_goal_conditioned_act
    
    # Create model with same config from checkpoint
    config_dict = checkpoint["config"]
    print(f"Model config: {config_dict.get('goal_fusion', 'unknown')} fusion")
    
    policy = create_goal_conditioned_act(
        chunk_size=config_dict.get("chunk_size", 100),
        dim_model=config_dict.get("dim_model", 512),
        use_vae=config_dict.get("use_vae", True),
        goal_fusion=config_dict.get("goal_fusion", "cross_attention"),
        camera_names=config_dict.get("camera_names", ["gripper", "station"]),
        state_dim=config_dict.get("state_dim", 6),
        action_dim=config_dict.get("action_dim", 6),
    ).to(device)
    
    # Load weights
    try:
        policy.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading: {e}")
        print("\nTrying strict=False...")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("✓ Model loaded (some weights skipped)")
    
    policy.eval()
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = LeRobotDataset(dataset_path)
    print(f"✓ Loaded {dataset.num_episodes} episodes")
    
    # Test with real data
    frame1 = dataset[0]  # Episode 0 start
    frame2 = dataset[800]  # Episode 0 end
    
    # Find different episode
    frame3_idx = None
    for i in range(len(dataset)):
        if dataset[i]["episode_index"].item() == 1:
            frame3_idx = i
            break
    
    if frame3_idx:
        frame3 = dataset[frame3_idx]
    else:
        frame3 = dataset[1000]
    
    # Prepare inputs
    images = [
        frame1["observation.images.gripper"].unsqueeze(0).to(device),
        frame1["observation.images.station"].unsqueeze(0).to(device),
    ]
    state = frame1["observation.state"].unsqueeze(0).to(device)
    
    # Three different goals
    goal1 = frame1["observation.images.station"].unsqueeze(0).to(device)  # Same as obs
    goal2 = frame2["observation.images.station"].unsqueeze(0).to(device)  # Same episode end
    goal3 = frame3["observation.images.station"].unsqueeze(0).to(device)  # Different episode
    
    print("\nTesting inference...")
    with torch.no_grad():
        out1 = policy(images, state, goal1)
        out2 = policy(images, state, goal2)
        out3 = policy(images, state, goal3)
    
    diff_12 = (out1["action"] - out2["action"]).abs().mean().item()
    diff_13 = (out1["action"] - out3["action"]).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Action diff (same episode start vs end): {diff_12:.6f}")
    print(f"  Action diff (different episodes):        {diff_13:.6f}")
    
    if diff_12 > 0.01 or diff_13 > 0.01:
        print(f"\n✅ SUCCESS! Goal conditioning is working!")
        print(f"   The model responds to different goals.")
    else:
        print(f"\n⚠️  Goal influence is still weak.")
        print(f"   May need more training steps.")
    
    print("\n" + "=" * 60)
    return diff_12, diff_13

if __name__ == "__main__":
    test_simple()

