"""
Hindsight Experience Replay for SAIL

Core idea: Failed attempts at task A are successful demonstrations
for reaching the state that was actually achieved.

This is what makes VLM labeling errors tolerable - we learn from
every trajectory regardless of success/failure.

Based on:
- HER: "Hindsight Experience Replay" (Andrychowicz et al., 2017)
- SOAR paper's hindsight relabeling approach
"""

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Trajectory:
    """Single trajectory with observations, actions, and metadata."""
    
    # Core data (stored as tensors)
    observations: Dict[str, Tensor]  # camera_name -> (T, C, H, W)
    states: Tensor  # (T, state_dim)
    actions: Tensor  # (T, action_dim)
    
    # Goal information
    intended_goal: Tensor  # (C, H, W) what we tried to achieve
    achieved_goal: Tensor  # (C, H, W) what we actually achieved (final frame)
    
    # Metadata
    success: bool  # Did we achieve intended goal?
    task_description: str = ""
    episode_idx: int = -1
    
    @property
    def length(self) -> int:
        """Get trajectory length."""
        return self.states.shape[0]
    
    def get_frame(self, t: int, camera: str = "station") -> Tensor:
        """Get observation at timestep t."""
        return self.observations[camera][t]
    
    def get_action_chunk(self, t: int, chunk_size: int) -> Tensor:
        """Get action chunk starting at timestep t."""
        end_t = min(t + chunk_size, self.length)
        chunk = self.actions[t:end_t]
        
        # Pad if necessary
        if chunk.shape[0] < chunk_size:
            padding = torch.zeros(
                chunk_size - chunk.shape[0], 
                chunk.shape[1],
                device=chunk.device,
                dtype=chunk.dtype
            )
            chunk = torch.cat([chunk, padding], dim=0)
        
        return chunk
    
    def to_device(self, device: str) -> "Trajectory":
        """Move trajectory to device."""
        return Trajectory(
            observations={k: v.to(device) for k, v in self.observations.items()},
            states=self.states.to(device),
            actions=self.actions.to(device),
            intended_goal=self.intended_goal.to(device),
            achieved_goal=self.achieved_goal.to(device),
            success=self.success,
            task_description=self.task_description,
            episode_idx=self.episode_idx,
        )


class HindsightBuffer:
    """
    Replay buffer with hindsight experience replay.
    
    For each trajectory, we can relabel the goal to be:
    1. The intended goal (if successful)
    2. The achieved state (hindsight - always "successful")
    3. Any intermediate state (future hindsight)
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        hindsight_ratio: float = 0.5,
        future_k: int = 4,  # Number of future goals to sample per trajectory
        goal_camera: str = "station",
    ):
        """
        Args:
            capacity: Maximum number of trajectories
            hindsight_ratio: Fraction of samples to relabel with hindsight goals
            future_k: Number of future goal samples per trajectory
            goal_camera: Which camera to use for goal images
        """
        self.capacity = capacity
        self.hindsight_ratio = hindsight_ratio
        self.future_k = future_k
        self.goal_camera = goal_camera
        
        self.trajectories: List[Trajectory] = []
        self.successful_indices: List[int] = []
        self.failed_indices: List[int] = []
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def add(self, trajectory: Trajectory) -> None:
        """Add trajectory to buffer."""
        if len(self.trajectories) >= self.capacity:
            # Remove oldest
            removed_idx = 0
            if removed_idx in self.successful_indices:
                self.successful_indices.remove(removed_idx)
            if removed_idx in self.failed_indices:
                self.failed_indices.remove(removed_idx)
            self.trajectories.pop(0)
            
            # Decrement all indices
            self.successful_indices = [i - 1 for i in self.successful_indices]
            self.failed_indices = [i - 1 for i in self.failed_indices]
        
        idx = len(self.trajectories)
        self.trajectories.append(trajectory)
        
        if trajectory.success:
            self.successful_indices.append(idx)
        else:
            self.failed_indices.append(idx)
    
    def sample_batch(
        self,
        batch_size: int,
        chunk_size: int,
        device: str = "cuda",
    ) -> Dict[str, Tensor]:
        """
        Sample batch with hindsight relabeling.
        
        Returns:
            Dict with:
                - observations: Dict[camera] -> (B, C, H, W)
                - states: (B, state_dim)
                - actions: (B, chunk_size, action_dim)
                - goals: (B, C, H, W)
                - is_hindsight: (B,) bool tensor
        """
        observations = {cam: [] for cam in self.trajectories[0].observations.keys()}
        states = []
        actions = []
        goals = []
        is_hindsight = []
        
        for _ in range(batch_size):
            # Select trajectory
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            
            # Select timestep (ensure enough future steps for chunk)
            max_t = max(1, traj.length - chunk_size)
            t = np.random.randint(0, max_t)
            
            # Decide on goal (hindsight or intended)
            use_hindsight = np.random.random() < self.hindsight_ratio
            
            if use_hindsight or not traj.success:
                # Use hindsight goal: sample from future states
                future_t = np.random.randint(
                    min(t + chunk_size, traj.length - 1),
                    traj.length
                )
                goal = traj.get_frame(future_t, self.goal_camera)
                is_hindsight.append(True)
            else:
                # Use intended goal
                goal = traj.intended_goal
                is_hindsight.append(False)
            
            # Extract data
            for cam in observations.keys():
                observations[cam].append(traj.get_frame(t, cam))
            states.append(traj.states[t])
            actions.append(traj.get_action_chunk(t, chunk_size))
            goals.append(goal)
        
        # Stack and move to device
        return {
            "observations": {
                cam: torch.stack(obs).to(device) 
                for cam, obs in observations.items()
            },
            "states": torch.stack(states).to(device),
            "actions": torch.stack(actions).to(device),
            "goals": torch.stack(goals).to(device),
            "is_hindsight": torch.tensor(is_hindsight, device=device),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        total = len(self.trajectories)
        successful = len(self.successful_indices)
        
        return {
            "total_trajectories": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "capacity": self.capacity,
            "hindsight_ratio": self.hindsight_ratio,
        }


class HindsightRelabeler:
    """
    Relabel trajectories with achieved goals.
    
    Converts failed task attempts into successful demonstrations
    for reaching the achieved state.
    """
    
    def __init__(
        self,
        future_strategy: str = "final",
        goal_camera: str = "station",
    ):
        """
        Args:
            future_strategy: How to select hindsight goal
                - "final": Always use final state
                - "future": Random future state
                - "episode": Random state from same episode
            goal_camera: Which camera to use for goal images
        """
        self.future_strategy = future_strategy
        self.goal_camera = goal_camera
    
    def relabel(
        self,
        trajectory: Trajectory,
        timestep: int,
    ) -> Tuple[Tensor, str]:
        """
        Get hindsight goal for given timestep.
        
        Args:
            trajectory: Trajectory to relabel
            timestep: Current timestep
            
        Returns:
            goal_image: Relabeled goal
            task_description: Updated task description
        """
        if self.future_strategy == "final":
            goal = trajectory.achieved_goal
            task_desc = "reach the final achieved state"
        
        elif self.future_strategy == "future":
            # Random future timestep
            future_t = np.random.randint(timestep + 1, trajectory.length)
            goal = trajectory.get_frame(future_t, self.goal_camera)
            task_desc = f"reach state at timestep {future_t}"
        
        else:  # episode - any timestep
            random_t = np.random.randint(0, trajectory.length)
            goal = trajectory.get_frame(random_t, self.goal_camera)
            task_desc = f"reach state at timestep {random_t}"
        
        return goal, task_desc
    
    def create_relabeled_batch(
        self,
        trajectories: List[Trajectory],
        batch_size: int,
        chunk_size: int,
        hindsight_ratio: float = 0.5,
        device: str = "cuda",
    ) -> Dict[str, Tensor]:
        """
        Create training batch with hindsight relabeling.
        
        Args:
            trajectories: List of trajectories
            batch_size: Batch size
            chunk_size: Action chunk size
            hindsight_ratio: Fraction to relabel
            device: Target device
            
        Returns:
            Batch dictionary
        """
        camera_names = list(trajectories[0].observations.keys())
        
        batch = {
            "observations": {cam: [] for cam in camera_names},
            "states": [],
            "actions": [],
            "goals": [],
            "is_hindsight": [],
        }
        
        for _ in range(batch_size):
            # Sample trajectory
            traj = np.random.choice(trajectories)
            
            # Sample timestep
            max_t = max(1, traj.length - chunk_size)
            t = np.random.randint(0, max_t)
            
            # Decide relabeling
            use_hindsight = np.random.random() < hindsight_ratio
            
            if use_hindsight or not traj.success:
                goal, _ = self.relabel(traj, t)
                is_hindsight = True
            else:
                goal = traj.intended_goal
                is_hindsight = False
            
            # Add to batch
            for cam in camera_names:
                batch["observations"][cam].append(traj.get_frame(t, cam))
            batch["states"].append(traj.states[t])
            batch["actions"].append(traj.get_action_chunk(t, chunk_size))
            batch["goals"].append(goal)
            batch["is_hindsight"].append(is_hindsight)
        
        # Stack tensors
        return {
            "observations": {
                cam: torch.stack(obs).to(device)
                for cam, obs in batch["observations"].items()
            },
            "states": torch.stack(batch["states"]).to(device),
            "actions": torch.stack(batch["actions"]).to(device),
            "goals": torch.stack(batch["goals"]).to(device),
            "is_hindsight": torch.tensor(batch["is_hindsight"], device=device),
        }


class GoalConditionedDataset(Dataset):
    """
    PyTorch Dataset for goal-conditioned training with hindsight.
    
    Wraps a LeRobotDataset and provides goal-conditioned samples
    with automatic hindsight relabeling.
    """
    
    def __init__(
        self,
        lerobot_dataset,
        chunk_size: int = 100,
        hindsight_ratio: float = 0.5,
        goal_camera: str = "station",
        transform=None,
    ):
        """
        Args:
            lerobot_dataset: LeRobotDataset instance
            chunk_size: Action chunk size
            hindsight_ratio: Fraction of samples to relabel
            goal_camera: Camera name for goal images
            transform: Optional transform for images
        """
        self.dataset = lerobot_dataset
        self.chunk_size = chunk_size
        self.hindsight_ratio = hindsight_ratio
        self.goal_camera = goal_camera
        self.transform = transform
        
        # Build episode index
        self._build_episode_index()
    
    def _build_episode_index(self):
        """Build index mapping samples to episodes."""
        self.episode_starts = {}
        self.episode_ends = {}
        
        # Get unique episode indices
        episode_indices = set()
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            ep_idx = item["episode_index"].item()
            episode_indices.add(ep_idx)
        
        # Find start/end for each episode
        for ep_idx in sorted(episode_indices):
            start = None
            end = None
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if item["episode_index"].item() == ep_idx:
                    if start is None:
                        start = i
                    end = i
            self.episode_starts[ep_idx] = start
            self.episode_ends[ep_idx] = end
        
        self.episode_list = sorted(episode_indices)
        self.num_episodes = len(self.episode_list)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _get_goal_image(self, episode_idx: int, current_idx: int) -> Tensor:
        """Get goal image with hindsight relabeling."""
        use_hindsight = np.random.random() < self.hindsight_ratio
        
        ep_start = self.episode_starts[episode_idx]
        ep_end = self.episode_ends[episode_idx]
        
        if use_hindsight:
            # Sample future frame from same episode
            if current_idx < ep_end:
                goal_idx = np.random.randint(current_idx + 1, ep_end + 1)
            else:
                goal_idx = ep_end
        else:
            # Use final frame as intended goal
            goal_idx = ep_end
        
        goal_item = self.dataset[goal_idx]
        goal_key = f"observation.images.{self.goal_camera}"
        
        if goal_key in goal_item:
            goal_image = goal_item[goal_key]
        else:
            # Fallback to first available camera
            for key in goal_item.keys():
                if key.startswith("observation.images."):
                    goal_image = goal_item[key]
                    break
        
        return goal_image
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a training sample with goal conditioning."""
        item = self.dataset[idx]
        episode_idx = item["episode_index"].item()
        
        # Get goal image
        goal_image = self._get_goal_image(episode_idx, idx)
        
        # Get action chunk
        ep_end = self.episode_ends[episode_idx]
        action_chunk = []
        action_mask = []
        
        for i in range(self.chunk_size):
            if idx + i <= ep_end:
                future_item = self.dataset[idx + i]
                action_chunk.append(future_item["action"])
                action_mask.append(1.0)
            else:
                # Pad with last action
                action_chunk.append(action_chunk[-1] if action_chunk else item["action"])
                action_mask.append(0.0)
        
        # Prepare output
        output = {
            "goal_image": goal_image,
            "action": torch.stack(action_chunk),
            "action_mask": torch.tensor(action_mask),
        }
        
        # Copy observation keys
        for key, value in item.items():
            if key.startswith("observation."):
                output[key] = value
        
        return output


def convert_lerobot_dataset_to_trajectories(
    dataset,
    goal_camera: str = "station",
    all_successful: bool = True,
) -> List[Trajectory]:
    """
    Convert LeRobotDataset to list of Trajectory objects.
    
    Args:
        dataset: LeRobotDataset instance
        goal_camera: Camera to use for goal images
        all_successful: Assume all demo trajectories are successful
        
    Returns:
        List of Trajectory objects
    """
    trajectories = []
    
    # Get unique episode indices
    episode_indices = set()
    for i in range(len(dataset)):
        item = dataset[i]
        ep_idx = item["episode_index"].item()
        episode_indices.add(ep_idx)
    
    for ep_idx in sorted(episode_indices):
        # Collect all frames for this episode
        frames = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item["episode_index"].item() == ep_idx:
                frames.append((i, item))
        
        if not frames:
            continue
        
        # Sort by frame index
        frames.sort(key=lambda x: x[0])
        
        # Extract data
        observations = {}
        states = []
        actions = []
        
        # Find camera keys
        camera_keys = [k for k in frames[0][1].keys() if k.startswith("observation.images.")]
        camera_names = [k.replace("observation.images.", "") for k in camera_keys]
        
        for cam in camera_names:
            observations[cam] = []
        
        for _, item in frames:
            # Get observations
            for cam in camera_names:
                key = f"observation.images.{cam}"
                if key in item:
                    observations[cam].append(item[key])
            
            # Get state
            if "observation.state" in item:
                states.append(item["observation.state"])
            
            # Get action
            if "action" in item:
                actions.append(item["action"])
        
        # Stack tensors
        for cam in camera_names:
            if observations[cam]:
                observations[cam] = torch.stack(observations[cam])
        
        if states:
            states = torch.stack(states)
        else:
            states = torch.zeros(len(frames), 6)  # Default state dim
        
        if actions:
            actions = torch.stack(actions)
        else:
            actions = torch.zeros(len(frames), 6)  # Default action dim
        
        # Get goal images
        goal_key = goal_camera if goal_camera in camera_names else camera_names[0]
        intended_goal = observations[goal_key][-1]  # Final frame
        achieved_goal = observations[goal_key][-1]
        
        traj = Trajectory(
            observations=observations,
            states=states,
            actions=actions,
            intended_goal=intended_goal,
            achieved_goal=achieved_goal,
            success=all_successful,
            task_description="pick and place",
            episode_idx=ep_idx,
        )
        
        trajectories.append(traj)
    
    return trajectories

