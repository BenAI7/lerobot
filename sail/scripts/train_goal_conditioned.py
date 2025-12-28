"""
Training script for goal-conditioned ACT policy.

This script trains a goal-conditioned ACT policy using:
1. Your baseline dataset (from Phase 0)
2. Hindsight Experience Replay for data augmentation
3. FiLM conditioning for goal images

Usage:
    python sail/scripts/train_goal_conditioned.py --dataset <local_path>
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sail.policies.goal_conditioned_act import (
    create_goal_conditioned_act,
    GoalConditionedACTConfig,
)


def build_episode_index_and_cache(dataset):
    """Build episode index AND cache all actions (FAST!)."""
    print("  Building episode index and caching actions...")
    
    episode_starts = {}
    episode_ends = {}
    action_cache = {}  # idx -> action tensor
    state_cache = {}   # idx -> state tensor
    
    # Single pass through dataset - cache actions and states (they're small)
    for i in tqdm(range(len(dataset)), desc="  Indexing & caching", leave=False):
        item = dataset[i]
        ep_idx = item["episode_index"].item()
        
        if ep_idx not in episode_starts:
            episode_starts[ep_idx] = i
        episode_ends[ep_idx] = i
        
        # Cache action and state (small tensors, fast to store)
        action_cache[i] = item["action"].clone()
        if "observation.state" in item:
            state_cache[i] = item["observation.state"].clone()
    
    episode_list = sorted(episode_starts.keys())
    print(f"  ✓ Indexed {len(episode_list)} episodes")
    print(f"  ✓ Cached {len(action_cache)} actions")
    
    return episode_starts, episode_ends, episode_list, action_cache, state_cache


def sample_batch_fast(
    dataset,
    episode_starts,
    episode_ends,
    episode_list,
    batch_size,
    chunk_size,
    camera_names,
    goal_camera,
    hindsight_ratio,
    device,
    action_cache,
    state_cache,
):
    """Sample a batch with hindsight relabeling (OPTIMIZED - uses cached actions)."""
    
    observations = {cam: [] for cam in camera_names}
    states = []
    actions = []
    goals = []
    
    for _ in range(batch_size):
        # Random episode
        ep_idx = np.random.choice(episode_list)
        ep_start = episode_starts[ep_idx]
        ep_end = episode_ends[ep_idx]
        ep_len = ep_end - ep_start + 1
        
        # Random timestep (with room for action chunk)
        max_t = max(0, ep_len - chunk_size)
        t_offset = np.random.randint(0, max(1, max_t))
        t = ep_start + t_offset
        
        # Get current frame (only images - actions/states from cache)
        item = dataset[t]
        
        # Collect observations (images only - this is the slow part)
        for cam in camera_names:
            key = f"observation.images.{cam}"
            observations[cam].append(item[key])
        
        # State from cache (FAST!)
        if t in state_cache:
            states.append(state_cache[t])
        else:
            states.append(torch.zeros(6))
        
        # Action chunk from cache (FAST! No video decoding)
        action_chunk = []
        for i in range(chunk_size):
            idx = min(t + i, ep_end)
            action_chunk.append(action_cache[idx])
        actions.append(torch.stack(action_chunk))
        
        # Goal with hindsight
        use_hindsight = np.random.random() < hindsight_ratio
        if use_hindsight:
            # Random future frame as goal
            goal_t = np.random.randint(t, ep_end + 1)
        else:
            # Final frame as intended goal
            goal_t = ep_end
        
        goal_item = dataset[goal_t]
        goal_key = f"observation.images.{goal_camera}"
        goals.append(goal_item[goal_key])
    
    # Stack and move to device
    batch = {
        "observations": {
            cam: torch.stack(observations[cam]).to(device)
            for cam in camera_names
        },
        "states": torch.stack(states).to(device),
        "actions": torch.stack(actions).to(device),
        "goals": torch.stack(goals).to(device),
    }
    
    return batch


def train_goal_conditioned(
    dataset_path: str,
    output_dir: str,
    num_steps: int = 25000,
    batch_size: int = 8,
    chunk_size: int = 100,
    learning_rate: float = 1e-4,
    hindsight_ratio: float = 0.5,
    goal_fusion: str = "film",
    goal_camera: str = "station",
    use_amp: bool = True,
    save_freq: int = 5000,
    log_freq: int = 100,
    device: str = "cuda",
):
    """
    Train goal-conditioned ACT policy.
    
    Args:
        dataset_path: Local path to dataset directory OR HuggingFace dataset ID
        output_dir: Where to save checkpoints
        num_steps: Training steps
        batch_size: Batch size
        chunk_size: Action chunk size
        learning_rate: Learning rate
        hindsight_ratio: Fraction of batch to use hindsight goals
        goal_fusion: Goal conditioning method ("film", "concat", "cross_attention")
        goal_camera: Camera for goal images
        use_amp: Use automatic mixed precision
        save_freq: Save checkpoint every N steps
        log_freq: Log every N steps
        device: Training device
    """
    print("=" * 60)
    print("SAIL Phase 2: Goal-Conditioned ACT Training")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n[1/5] Loading dataset...")
    
    if Path(dataset_path).exists():
        print(f"  Loading from local path: {dataset_path}")
        dataset = LeRobotDataset(dataset_path)
    else:
        print(f"  Loading from HuggingFace: {dataset_path}")
        dataset = LeRobotDataset(dataset_path)
    
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes, {len(dataset)} frames")
    
    # Determine dimensions from dataset
    sample = dataset[0]
    
    # Find state dimension
    if "observation.state" in sample:
        state_dim = sample["observation.state"].shape[0]
    else:
        state_dim = 6  # Default for SO-101
    
    # Find action dimension
    if "action" in sample:
        action_dim = sample["action"].shape[0]
    else:
        action_dim = 6  # Default for SO-101
    
    # Find camera names
    camera_names = []
    for key in sample.keys():
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            camera_names.append(cam_name)
    
    if not camera_names:
        raise ValueError("No camera images found in dataset!")
    
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Cameras: {camera_names}")
    
    # Use specified goal camera or default to first available
    if goal_camera not in camera_names:
        goal_camera = camera_names[0]
        print(f"  Using {goal_camera} for goal images")
    
    # Create policy
    print(f"\n[2/5] Creating goal-conditioned ACT policy...")
    policy = create_goal_conditioned_act(
        chunk_size=chunk_size,
        dim_model=512,
        use_vae=True,
        goal_fusion=goal_fusion,
        camera_names=camera_names,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"✓ Policy created: {num_params:,} parameters")
    print(f"  Goal fusion: {goal_fusion}")
    print(f"  Using VAE: {policy.config.use_vae}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
    scaler = GradScaler('cuda') if use_amp else None
    
    # Build episode index and cache actions (single pass)
    print(f"\n[3/5] Building episode index and caching actions...")
    episode_starts, episode_ends, episode_list, action_cache, state_cache = build_episode_index_and_cache(dataset)
    print(f"✓ Ready for training with hindsight ratio: {hindsight_ratio}")
    
    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "dataset_path": dataset_path,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "learning_rate": learning_rate,
            "hindsight_ratio": hindsight_ratio,
            "goal_fusion": goal_fusion,
            "goal_camera": goal_camera,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "camera_names": camera_names,
            "policy_config": policy.config.to_dict(),
        }, f, indent=2)
    
    # Training loop
    print(f"\n[4/5] Starting training for {num_steps} steps...")
    print("-" * 60)
    
    policy.train()
    losses = []
    best_loss = float("inf")
    
    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Sample batch with hindsight (uses cached actions - FAST!)
        batch = sample_batch_fast(
            dataset=dataset,
            episode_starts=episode_starts,
            episode_ends=episode_ends,
            episode_list=episode_list,
            batch_size=batch_size,
            chunk_size=chunk_size,
            camera_names=camera_names,
            goal_camera=goal_camera,
            hindsight_ratio=hindsight_ratio,
            device=device,
            action_cache=action_cache,
            state_cache=state_cache,
        )
        
        # Prepare inputs
        images = [batch["observations"][cam] for cam in camera_names]
        states = batch["states"]
        actions = batch["actions"]
        goals = batch["goals"]
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        if use_amp:
            with autocast('cuda'):
                loss, loss_dict = policy.compute_loss(
                    images=images,
                    state=states,
                    actions=actions,
                    goal_image=goals,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = policy.compute_loss(
                images=images,
                state=states,
                actions=actions,
                goal_image=goals,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
        
        scheduler.step()
        losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Logging
        if step % log_freq == 0 and step > 0:
            avg_loss = sum(losses[-log_freq:]) / len(losses[-log_freq:])
            lr = scheduler.get_last_lr()[0]
            
            log_msg = f"Step {step}: loss={avg_loss:.4f}"
            if "l1_loss" in loss_dict:
                log_msg += f", l1={loss_dict['l1_loss']:.4f}"
            if "kl_loss" in loss_dict:
                log_msg += f", kl={loss_dict['kl_loss']:.4f}"
            log_msg += f", lr={lr:.2e}"
            
            tqdm.write(log_msg)
        
        # Save checkpoint
        if step % save_freq == 0 and step > 0:
            checkpoint_path = output_path / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": policy.config.to_dict(),
                "loss": sum(losses[-log_freq:]) / len(losses[-log_freq:]),
            }, checkpoint_path)
            tqdm.write(f"✓ Saved checkpoint: {checkpoint_path.name}")
            
            # Track best
            current_loss = sum(losses[-log_freq:]) / len(losses[-log_freq:])
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = output_path / "best_model.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": policy.state_dict(),
                    "config": policy.config.to_dict(),
                    "loss": best_loss,
                }, best_path)
                tqdm.write(f"✓ New best model saved (loss={best_loss:.4f})")
    
    # Save final model
    print("\n[5/5] Saving final model...")
    final_path = output_path / "final_model.pt"
    torch.save({
        "step": num_steps,
        "model_state_dict": policy.state_dict(),
        "config": policy.config.to_dict(),
        "loss": sum(losses[-100:]) / len(losses[-100:]) if losses else 0,
    }, final_path)
    
    # Save training metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "final_loss": sum(losses[-100:]) / len(losses[-100:]) if losses else 0,
            "best_loss": best_loss,
            "num_steps": num_steps,
            "dataset_path": dataset_path,
            "hindsight_ratio": hindsight_ratio,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"  Final loss: {sum(losses[-100:]) / len(losses[-100:]) if losses else 0:.4f}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Output: {output_path}")
    
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Train goal-conditioned ACT policy for SAIL Phase 2"
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        required=True,
        help="Local path to dataset directory OR HuggingFace dataset ID"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="outputs/sail_goal_conditioned",
        help="Output directory"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=25000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Action chunk size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--hindsight_ratio",
        type=float,
        default=0.5,
        help="Fraction of batch to use hindsight goals"
    )
    parser.add_argument(
        "--goal_fusion",
        type=str,
        default="film",
        choices=["film", "concat", "cross_attention"],
        help="Goal conditioning method"
    )
    parser.add_argument(
        "--goal_camera",
        type=str,
        default="station",
        help="Camera for goal images"
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )
    args = parser.parse_args()
    
    train_goal_conditioned(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_steps=args.steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        learning_rate=args.lr,
        hindsight_ratio=args.hindsight_ratio,
        goal_fusion=args.goal_fusion,
        goal_camera=args.goal_camera,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
