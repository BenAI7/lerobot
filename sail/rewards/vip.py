"""
VIP (Value-Implicit Pre-training) Reward Model
Zero-shot reward from goal image similarity

Uses pre-trained ResNet features to compute visual similarity.
Paper: https://arxiv.org/abs/2210.00030
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from typing import Optional
import numpy as np


class VIPReward(nn.Module):
    """
    Computes reward as negative L2 distance between
    current observation embedding and goal embedding.
    
    Uses pretrained vision encoder (ResNet/ViT) trained on ImageNet.
    Surprisingly effective for robot manipulation tasks!
    """
    
    def __init__(
        self, 
        model_name: str = "resnet50",
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        
        # Load pretrained encoder
        print(f"Loading VIP encoder: {model_name}...")
        self.encoder = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        self.encoder.eval()
        self.encoder.to(device)
        
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.embedding_dim = self.encoder.num_features
        print(f"[OK] VIP encoder loaded. Embedding dim: {self.embedding_dim}")
        
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embedding space.
        
        Args:
            images: (B, C, H, W) tensor, values in [0, 1] or [0, 255]
            
        Returns:
            embeddings: (B, embedding_dim) L2-normalized tensor
        """
        # Ensure float tensor
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.0:
            images = images / 255.0
        
        # Move to device
        images = images.to(self.device)
        
        # Apply transforms
        images = self.transform(images)
        
        # Encode
        embeddings = self.encoder(images)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    @torch.no_grad()
    def compute_reward(
        self, 
        current_obs: torch.Tensor, 
        goal_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward as negative distance to goal.
        
        Args:
            current_obs: (B, C, H, W) current observation
            goal_obs: (B, C, H, W) or (C, H, W) goal observation
            
        Returns:
            rewards: (B,) reward values (higher = closer to goal)
        """
        current_emb = self.encode(current_obs)
        
        if goal_obs.dim() == 3:
            goal_obs = goal_obs.unsqueeze(0)
        goal_emb = self.encode(goal_obs)
        
        # Expand goal_emb if batch sizes don't match
        if goal_emb.shape[0] == 1 and current_emb.shape[0] > 1:
            goal_emb = goal_emb.expand(current_emb.shape[0], -1)
        
        # Negative L2 distance as reward
        distances = torch.norm(current_emb - goal_emb, dim=-1)
        rewards = -distances
        
        return rewards
    
    @torch.no_grad()
    def compute_progress(
        self,
        current_obs: torch.Tensor,
        initial_obs: torch.Tensor,
        goal_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute progress as ratio of distance traveled.
        
        Returns value in [0, 1] where:
        - 0 = at initial state
        - 1 = at goal state
        """
        current_emb = self.encode(current_obs)
        initial_emb = self.encode(initial_obs)
        goal_emb = self.encode(goal_obs)
        
        # Expand dimensions if needed
        if initial_emb.shape[0] == 1 and current_emb.shape[0] > 1:
            initial_emb = initial_emb.expand(current_emb.shape[0], -1)
        if goal_emb.shape[0] == 1 and current_emb.shape[0] > 1:
            goal_emb = goal_emb.expand(current_emb.shape[0], -1)
        
        initial_to_goal = torch.norm(goal_emb - initial_emb, dim=-1)
        current_to_goal = torch.norm(goal_emb - current_emb, dim=-1)
        
        # Progress = how much of the distance has been covered
        progress = 1.0 - (current_to_goal / (initial_to_goal + 1e-8))
        progress = torch.clamp(progress, 0.0, 1.0)
        
        return progress


def create_vip_reward(
    model_name: str = "resnet50",
    device: str = "cuda"
) -> VIPReward:
    """Factory function to create VIP reward model."""
    return VIPReward(model_name=model_name, device=device)

