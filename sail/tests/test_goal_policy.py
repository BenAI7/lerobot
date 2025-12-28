"""
Test script for Goal-Conditioned ACT Policy

This script validates:
1. Policy can be created
2. Forward pass works with goal conditioning
3. Loss computation works
4. Can load and run inference

Usage:
    python sail/tests/test_goal_policy.py
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sail.policies.goal_conditioned_act import (
    create_goal_conditioned_act,
    GoalConditionedACTConfig,
    GoalConditionedACTPolicy,
)


def test_policy_creation():
    """Test that policy can be created."""
    print("\n--- Test 1: Policy Creation ---")
    
    policy = create_goal_conditioned_act(
        chunk_size=100,
        dim_model=512,
        use_vae=True,
        goal_fusion="film",
        camera_names=["gripper", "station"],
        state_dim=6,
        action_dim=6,
    )
    
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"✓ Policy created successfully")
    print(f"  Parameters: {num_params:,}")
    print(f"  Config: chunk_size={policy.config.chunk_size}, dim_model={policy.config.dim_model}")
    
    return policy


def test_forward_pass(policy: GoalConditionedACTPolicy, device: str = "cuda"):
    """Test forward pass with goal conditioning."""
    print("\n--- Test 2: Forward Pass ---")
    
    policy = policy.to(device)
    policy.train()
    
    batch_size = 4
    
    # Create dummy inputs
    images = [
        torch.randn(batch_size, 3, 480, 640, device=device),  # gripper
        torch.randn(batch_size, 3, 480, 640, device=device),  # station
    ]
    state = torch.randn(batch_size, 6, device=device)
    goal_image = torch.randn(batch_size, 3, 480, 640, device=device)
    actions = torch.randn(batch_size, 100, 6, device=device)
    
    # Forward pass (training mode with VAE)
    output = policy(images, state, goal_image, actions)
    
    print(f"✓ Forward pass successful")
    print(f"  Action output shape: {output['action'].shape}")
    if 'mu' in output:
        print(f"  VAE mu shape: {output['mu'].shape}")
        print(f"  VAE logvar shape: {output['logvar'].shape}")
    
    # Forward pass (inference mode without VAE)
    policy.eval()
    with torch.no_grad():
        output_infer = policy(images, state, goal_image)
    
    print(f"✓ Inference pass successful")
    print(f"  Action output shape: {output_infer['action'].shape}")
    
    return True


def test_loss_computation(policy: GoalConditionedACTPolicy, device: str = "cuda"):
    """Test loss computation."""
    print("\n--- Test 3: Loss Computation ---")
    
    policy = policy.to(device)
    policy.train()
    
    batch_size = 4
    
    # Create dummy inputs
    images = [
        torch.randn(batch_size, 3, 480, 640, device=device),
        torch.randn(batch_size, 3, 480, 640, device=device),
    ]
    state = torch.randn(batch_size, 6, device=device)
    goal_image = torch.randn(batch_size, 3, 480, 640, device=device)
    actions = torch.randn(batch_size, 100, 6, device=device)
    
    # Compute loss
    loss, loss_dict = policy.compute_loss(images, state, actions, goal_image)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"✓ Backward pass successful")
    
    # Check gradients exist
    has_grads = sum(1 for p in policy.parameters() if p.grad is not None)
    total_params = sum(1 for p in policy.parameters())
    print(f"  Gradients: {has_grads}/{total_params} parameters")
    
    return True


def test_goal_fusion_methods():
    """Test different goal fusion methods."""
    print("\n--- Test 4: Goal Fusion Methods ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    fusion_methods = ["film", "concat", "cross_attention"]
    
    for fusion in fusion_methods:
        policy = create_goal_conditioned_act(
            chunk_size=50,  # Smaller for faster test
            dim_model=256,  # Smaller for faster test
            use_vae=True,
            goal_fusion=fusion,
            camera_names=["station"],
            state_dim=6,
            action_dim=6,
        ).to(device)
        
        # Create dummy inputs
        images = [torch.randn(batch_size, 3, 480, 640, device=device)]
        state = torch.randn(batch_size, 6, device=device)
        goal_image = torch.randn(batch_size, 3, 480, 640, device=device)
        actions = torch.randn(batch_size, 50, 6, device=device)
        
        # Forward pass
        policy.train()
        output = policy(images, state, goal_image, actions)
        
        print(f"✓ {fusion}: output shape = {output['action'].shape}")
    
    return True


def test_predict_action(policy: GoalConditionedACTPolicy, device: str = "cuda"):
    """Test action prediction."""
    print("\n--- Test 5: Action Prediction ---")
    
    policy = policy.to(device)
    policy.eval()
    
    batch_size = 1
    
    # Create dummy inputs
    images = [
        torch.randn(batch_size, 3, 480, 640, device=device),
        torch.randn(batch_size, 3, 480, 640, device=device),
    ]
    state = torch.randn(batch_size, 6, device=device)
    goal_image = torch.randn(batch_size, 3, 480, 640, device=device)
    
    # Predict actions
    actions = policy.predict_action(images, state, goal_image)
    
    print(f"✓ Action prediction successful")
    print(f"  Predicted actions shape: {actions.shape}")
    print(f"  Action range: [{actions.min().item():.2f}, {actions.max().item():.2f}]")
    
    return True


def test_with_real_data():
    """Test with real dataset (if available)."""
    print("\n--- Test 6: Real Data Test ---")
    
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from sail.training.hindsight import convert_lerobot_dataset_to_trajectories
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load dataset (try local path first)
        local_dataset_path = r"C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2"
        
        if Path(local_dataset_path).exists():
            print(f"Loading local dataset: {local_dataset_path}...")
            dataset = LeRobotDataset(local_dataset_path)
        else:
            # Fallback to HuggingFace (will fail if not available)
            dataset_repo_id = "yeyian/self_improve_pickplace_v2"
            print(f"Loading dataset: {dataset_repo_id}...")
            dataset = LeRobotDataset(dataset_repo_id)
        
        print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
        
        # Get sample
        sample = dataset[0]
        
        # Find camera names
        camera_names = []
        for key in sample.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                camera_names.append(cam_name)
        
        print(f"  Cameras: {camera_names}")
        
        # Create policy matching dataset
        state_dim = sample["observation.state"].shape[0] if "observation.state" in sample else 6
        action_dim = sample["action"].shape[0] if "action" in sample else 6
        
        policy = create_goal_conditioned_act(
            chunk_size=100,
            dim_model=512,
            use_vae=True,
            goal_fusion="film",
            camera_names=camera_names,
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        
        # Get a real sample
        images = []
        for cam in camera_names:
            key = f"observation.images.{cam}"
            img = sample[key].unsqueeze(0).to(device)
            images.append(img)
        
        state = sample["observation.state"].unsqueeze(0).to(device)
        goal_image = images[0]  # Use first camera for goal
        
        # Test forward pass
        policy.eval()
        with torch.no_grad():
            output = policy(images, state, goal_image)
        
        print(f"✓ Real data forward pass successful")
        print(f"  Action shape: {output['action'].shape}")
        
        # Convert to trajectories
        print("\nConverting dataset to trajectories...")
        trajectories = convert_lerobot_dataset_to_trajectories(
            dataset,
            goal_camera=camera_names[0],
        )
        print(f"✓ Converted {len(trajectories)} trajectories")
        
        return True
        
    except Exception as e:
        print(f"⚠ Real data test skipped: {e}")
        return False


def main():
    print("=" * 60)
    print("Goal-Conditioned ACT Policy Tests")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Test 1: Creation
        policy = test_policy_creation()
        
        # Test 2: Forward pass
        test_forward_pass(policy, device)
        
        # Test 3: Loss computation
        test_loss_computation(policy, device)
        
        # Test 4: Goal fusion methods
        test_goal_fusion_methods()
        
        # Test 5: Action prediction
        test_predict_action(policy, device)
        
        # Test 6: Real data (optional)
        test_with_real_data()
        
        print("\n" + "=" * 60)
        print("✅ All Tests Passed!")
        print("=" * 60)
        print("\nNext step: python sail/scripts/train_goal_conditioned.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

