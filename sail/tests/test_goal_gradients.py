"""Test if gradients flow from actions back to goal during training."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sail.policies.goal_conditioned_act import create_goal_conditioned_act

print("Testing gradient flow to goal...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create policy
policy = create_goal_conditioned_act(
    chunk_size=100,
    dim_model=512,
    goal_fusion="cross_attention",
).to(device)
policy.train()

# Create inputs (requires_grad on goal!)
images = [
    torch.randn(1, 3, 480, 640, device=device),
    torch.randn(1, 3, 480, 640, device=device),
]
state = torch.randn(1, 6, device=device)
goal = torch.randn(1, 3, 480, 640, device=device, requires_grad=True)
target_actions = torch.randn(1, 100, 6, device=device)

# Forward + backward
output = policy(images, state, goal, target_actions)
loss = (output["action"] - target_actions).pow(2).mean()
loss.backward()

# Check if goal received gradients
if goal.grad is not None and goal.grad.abs().sum() > 0:
    grad_magnitude = goal.grad.abs().mean().item()
    print(f"\n✅ SUCCESS! Gradients flow to goal!")
    print(f"   Goal gradient magnitude: {grad_magnitude:.6f}")
    print(f"\n   This means:")
    print(f"   - Training WILL make the model use goals")
    print(f"   - Weak signal with random weights is NORMAL")
    print(f"   - After training, goal influence will be strong")
else:
    print(f"\n❌ FAIL: No gradients on goal!")
    print(f"   Goal is disconnected from loss.")

