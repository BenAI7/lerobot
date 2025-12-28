"""Quick test: Does the architecture produce different actions for different goals?"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sail.policies.goal_conditioned_act import create_goal_conditioned_act

print("Testing if architecture responds to different goals...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create fresh policy (random weights)
policy = create_goal_conditioned_act(
    chunk_size=100,
    dim_model=512,
    goal_fusion="cross_attention",
).to(device)
policy.eval()

# Create dummy inputs
batch = 1
images = [
    torch.randn(batch, 3, 480, 640, device=device),
    torch.randn(batch, 3, 480, 640, device=device),
]
state = torch.randn(batch, 6, device=device)

# Two different goals
goal1 = torch.randn(batch, 3, 480, 640, device=device)
goal2 = torch.randn(batch, 3, 480, 640, device=device)  # Different random goal

with torch.no_grad():
    out1 = policy(images, state, goal1)
    out2 = policy(images, state, goal2)

diff = (out1["action"] - out2["action"]).abs().mean().item()
print(f"\nAction difference with different goals: {diff:.6f}")

if diff > 0.01:
    print("✅ SUCCESS! Architecture responds to different goals!")
    print("   Ready for training.")
else:
    print("❌ FAIL: Architecture still not using goals properly.")

