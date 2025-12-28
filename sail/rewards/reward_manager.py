"""
Unified Reward Manager for SAIL
Combines VIP (progress) + VLM (success) signals
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from sail.rewards.vip import VIPReward, create_vip_reward
from sail.rewards.vlm_detector import VLMSuccessDetector, create_vlm_detector


@dataclass
class RewardSignals:
    """Container for all reward signals."""
    vip_progress: float  # [0, 1] progress toward goal
    vip_reward: float  # Raw VIP reward (negative distance)
    vlm_success: bool  # Binary success from VLM
    vlm_confidence: float  # VLM confidence [0, 1]
    combined_reward: float  # Weighted combination


class RewardManager:
    """
    Manages reward computation for SAIL.
    
    Combines:
    - VIP: Dense progress signal (every timestep)
    - VLM: Sparse success signal (end of episode)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        vip_weight: float = 1.0,
        success_bonus: float = 10.0,
        use_vlm: bool = True,
        vip_model: str = "resnet50",
        vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    ):
        self.device = device
        self.vip_weight = vip_weight
        self.success_bonus = success_bonus
        self.use_vlm = use_vlm
        
        # Initialize reward models
        print("=" * 50)
        print("Initializing SAIL Reward Manager...")
        print("=" * 50)
        
        print("\n[1/2] Loading VIP reward model...")
        self.vip = create_vip_reward(model_name=vip_model, device=device)
        
        if use_vlm:
            print("\n[2/2] Loading VLM success detector...")
            self.vlm = create_vlm_detector(model_name=vlm_model, device=device, use_4bit=False)
        else:
            self.vlm = None
            print("\n[2/2] VLM disabled (use_vlm=False)")
        
        print("\n" + "=" * 50)
        print("[OK] Reward Manager ready!")
        print("=" * 50)
        
        # Cache for goal embedding
        self.goal_embedding: Optional[torch.Tensor] = None
        self.initial_embedding: Optional[torch.Tensor] = None
        
    def set_goal(self, goal_image: np.ndarray | torch.Tensor) -> None:
        """
        Set the goal image for reward computation.
        Call at start of episode.
        """
        goal_tensor = self._to_tensor(goal_image)
        self.goal_embedding = self.vip.encode(goal_tensor)
        
    def set_initial(self, initial_image: np.ndarray | torch.Tensor) -> None:
        """
        Set the initial image for progress computation.
        Call at start of episode.
        """
        initial_tensor = self._to_tensor(initial_image)
        self.initial_embedding = self.vip.encode(initial_tensor)
    
    def _to_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert image to tensor."""
        if isinstance(image, torch.Tensor):
            tensor = image.clone()
        else:
            tensor = torch.from_numpy(image.copy())
        
        # Convert to float if needed
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        elif tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        # Handle HWC -> CHW if needed
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        
        # Add batch dim if needed
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def compute_step_reward(
        self,
        current_image: np.ndarray | torch.Tensor
    ) -> RewardSignals:
        """
        Compute reward for a single timestep.
        
        Returns RewardSignals with VIP progress/reward
        """
        if self.goal_embedding is None:
            raise ValueError("Must call set_goal() before computing rewards")
        
        current_tensor = self._to_tensor(current_image)
        current_embedding = self.vip.encode(current_tensor)
        
        # VIP reward (negative distance)
        distance = torch.norm(current_embedding - self.goal_embedding).item()
        vip_reward = -distance
        
        # VIP progress (requires initial embedding)
        if self.initial_embedding is not None:
            initial_dist = torch.norm(self.goal_embedding - self.initial_embedding).item()
            current_dist = distance
            progress = 1.0 - (current_dist / (initial_dist + 1e-8))
            progress = max(0.0, min(1.0, progress))
        else:
            progress = 0.0
        
        return RewardSignals(
            vip_progress=progress,
            vip_reward=vip_reward,
            vlm_success=False,  # Not computed for intermediate steps
            vlm_confidence=0.0,
            combined_reward=vip_reward * self.vip_weight
        )
    
    def compute_episode_reward(
        self,
        final_image: np.ndarray | torch.Tensor,
        task_description: str
    ) -> RewardSignals:
        """
        Compute final episode reward including VLM success.
        
        Args:
            final_image: Final observation
            task_description: Task for VLM to evaluate
            
        Returns:
            RewardSignals with all components
        """
        # Get VIP signals
        signals = self.compute_step_reward(final_image)
        
        # Get VLM success
        if self.use_vlm and self.vlm is not None:
            success, confidence = self.vlm.detect_success(
                final_image, task_description
            )
            signals = RewardSignals(
                vip_progress=signals.vip_progress,
                vip_reward=signals.vip_reward,
                vlm_success=success,
                vlm_confidence=confidence,
                combined_reward=signals.combined_reward + (self.success_bonus if success else 0)
            )
        
        return signals
    
    def propose_practice_tasks(
        self,
        current_image: np.ndarray | torch.Tensor,
        available_tasks: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Propose feasible tasks for autonomous practice.
        """
        if self.vlm is None:
            return [(t, 1.0) for t in available_tasks]
        
        return self.vlm.propose_tasks(current_image, available_tasks)


def create_reward_manager(
    device: str = "cuda",
    use_vlm: bool = True,
    vip_model: str = "resnet50",
    vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"
) -> RewardManager:
    """Factory function to create reward manager."""
    return RewardManager(
        device=device,
        vip_weight=1.0,
        success_bonus=10.0,
        use_vlm=use_vlm,
        vip_model=vip_model,
        vlm_model=vlm_model
    )

