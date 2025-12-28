"""
Goal-Conditioned ACT Policy for SAIL

Extends standard ACT to accept goal images as conditioning.
Enables hindsight experience replay during self-improvement.

Key features:
- Goal image encoding with shared or separate encoder
- FiLM (Feature-wise Linear Modulation) for goal conditioning
- Compatible with LeRobot's ACT architecture
- Supports loading from baseline ACT checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import math

import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


@dataclass
class GoalConditionedACTConfig:
    """Configuration for goal-conditioned ACT policy."""
    
    # Core ACT parameters (matching baseline)
    chunk_size: int = 100
    n_action_steps: int = 100
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    dropout: float = 0.1
    
    # VAE parameters
    use_vae: bool = True
    latent_dim: int = 32
    kl_weight: float = 10.0
    n_vae_encoder_layers: int = 4
    
    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    
    # Input/output dimensions
    state_dim: int = 6  # SO-101 joint positions
    action_dim: int = 6  # SO-101 joint commands
    image_channels: int = 3
    image_height: int = 480
    image_width: int = 640
    
    # Camera configuration (matching baseline)
    camera_names: List[str] = field(default_factory=lambda: ["gripper", "station"])
    
    # Goal conditioning parameters
    use_goal_image: bool = True
    goal_encoder_type: str = "shared"  # "shared" or "separate"
    goal_fusion: str = "film"  # "film", "concat", or "cross_attention"
    goal_camera: str = "station"  # Which camera to use for goal images
    
    # Hindsight relabeling
    hindsight_ratio: float = 0.5  # Fraction of batch to relabel with hindsight goals
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "dim_model": self.dim_model,
            "n_heads": self.n_heads,
            "dim_feedforward": self.dim_feedforward,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "dropout": self.dropout,
            "use_vae": self.use_vae,
            "latent_dim": self.latent_dim,
            "kl_weight": self.kl_weight,
            "n_vae_encoder_layers": self.n_vae_encoder_layers,
            "vision_backbone": self.vision_backbone,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "camera_names": self.camera_names,
            "use_goal_image": self.use_goal_image,
            "goal_encoder_type": self.goal_encoder_type,
            "goal_fusion": self.goal_fusion,
            "goal_camera": self.goal_camera,
            "hindsight_ratio": self.hindsight_ratio,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "GoalConditionedACTConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def create_sinusoidal_pos_embedding(num_positions: int, dim: int) -> Tensor:
    """Create sinusoidal positional embeddings."""
    position = torch.arange(num_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal position embedding for image features."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to feature map.
        
        Args:
            x: (B, C, H, W) feature map
            
        Returns:
            (B, C, H, W) feature map with positional encoding added
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create position encodings
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)
        
        # Normalize to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        # Create sinusoidal embeddings
        dim_t = torch.arange(self.dim, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.dim)
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_x, pos_y], dim=-1)  # (H, W, 2*dim)
        pos = pos[:, :, :C].permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        return x + pos


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation for goal conditioning.
    
    Applies: output = gamma * features + beta
    where gamma and beta are learned from the conditioning signal.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        
        # Initialize to identity transformation
        nn.init.ones_(self.gamma.weight.data.diag())
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
    
    def forward(self, features: Tensor, conditioning: Tensor) -> Tensor:
        """Apply FiLM conditioning.
        
        Args:
            features: (B, T, D) or (B, D) features to modulate
            conditioning: (B, D) conditioning signal
            
        Returns:
            Modulated features with same shape as input
        """
        gamma = self.gamma(conditioning)
        beta = self.beta(conditioning)
        
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)
        
        return gamma * features + beta


class GoalEncoder(nn.Module):
    """Encodes goal image to conditioning vector.
    
    Can share weights with the observation encoder or use separate weights.
    """
    
    def __init__(
        self, 
        config: GoalConditionedACTConfig,
        shared_backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config
        
        if config.goal_encoder_type == "shared" and shared_backbone is not None:
            # Share backbone with observation encoder
            self.backbone = shared_backbone
            self.shared = True
        else:
            # Separate backbone for goals
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model, 
                return_layers={"layer4": "feature_map"}
            )
            self.shared = False
        
        # Get backbone output dimension
        if config.vision_backbone == "resnet18":
            backbone_dim = 512
        elif config.vision_backbone == "resnet34":
            backbone_dim = 512
        elif config.vision_backbone == "resnet50":
            backbone_dim = 2048
        else:
            backbone_dim = 512
        
        # Projection to model dimension
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_dim, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )
    
    def forward(self, goal_image: Tensor) -> Tensor:
        """Encode goal image.
        
        Args:
            goal_image: (B, C, H, W) goal observation
            
        Returns:
            goal_embedding: (B, dim_model) conditioning vector
        """
        # Extract features
        features = self.backbone(goal_image)["feature_map"]
        
        # Project to embedding
        embedding = self.projection(features)
        
        return embedding


class GoalConditionedACTPolicy(nn.Module):
    """
    ACT policy conditioned on goal images.
    
    Key changes from standard ACT:
    1. Accepts goal_image in forward pass
    2. Fuses goal embedding with encoder features via FiLM
    3. Supports hindsight experience replay during training
    4. Can load weights from baseline ACT checkpoint
    """
    
    def __init__(self, config: GoalConditionedACTConfig):
        super().__init__()
        self.config = config
        
        # Vision backbone (ResNet)
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(
            backbone_model, 
            return_layers={"layer4": "feature_map"}
        )
        
        # Get backbone dimensions
        if config.vision_backbone in ["resnet18", "resnet34"]:
            self.backbone_dim = 512
        else:
            self.backbone_dim = 2048
        
        # Image feature projection
        self.encoder_img_feat_input_proj = nn.Conv2d(
            self.backbone_dim, config.dim_model, kernel_size=1
        )
        
        # State projection
        self.encoder_robot_state_input_proj = nn.Linear(config.state_dim, config.dim_model)
        
        # Latent projection (for VAE)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        
        # Positional embeddings
        n_1d_tokens = 2  # latent + state
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2d(config.dim_model // 2)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        
        # Decoder positional embeddings (action queries)
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        
        # Action head - input is decoder output + goal embedding (concatenated)
        action_head_input_dim = config.dim_model * 2 if config.use_goal_image else config.dim_model
        self.action_head = nn.Linear(action_head_input_dim, config.action_dim)
        
        # VAE components
        if config.use_vae:
            # VAE encoder
            vae_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.vae_encoder = nn.TransformerEncoder(
                vae_encoder_layer, 
                num_layers=config.n_vae_encoder_layers
            )
            
            # VAE input projections
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            self.vae_encoder_robot_state_input_proj = nn.Linear(config.state_dim, config.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(config.action_dim, config.dim_model)
            
            # VAE output projection
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            
            # VAE positional embeddings
            num_vae_tokens = 2 + config.chunk_size  # cls + state + actions
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_vae_tokens, config.dim_model).unsqueeze(0)
            )
        
        # Goal conditioning
        if config.use_goal_image:
            self.goal_encoder = GoalEncoder(config, self.backbone if config.goal_encoder_type == "shared" else None)
            
            if config.goal_fusion == "film":
                self.goal_film = FiLMConditioning(config.dim_model)
            elif config.goal_fusion == "cross_attention":
                self.goal_cross_attn = nn.MultiheadAttention(
                    config.dim_model, config.n_heads, 
                    dropout=config.dropout, batch_first=True
                )
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier-uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode_images(self, images: List[Tensor]) -> Tensor:
        """Encode multiple camera images.
        
        Args:
            images: List of (B, C, H, W) tensors, one per camera
            
        Returns:
            (B, N, D) encoded image features where N is total spatial positions
        """
        all_features = []
        
        for img in images:
            # Extract backbone features
            features = self.backbone(img)["feature_map"]  # (B, C, H', W')
            
            # Project to model dimension
            features = self.encoder_img_feat_input_proj(features)  # (B, D, H', W')
            
            # Add positional encoding
            features = self.encoder_cam_feat_pos_embed(features)
            
            # Flatten spatial dimensions
            B, D, H, W = features.shape
            features = features.flatten(2).permute(0, 2, 1)  # (B, H'*W', D)
            
            all_features.append(features)
        
        # Concatenate all camera features
        return torch.cat(all_features, dim=1)  # (B, N_total, D)
    
    def encode_state(self, state: Tensor) -> Tensor:
        """Encode robot state.
        
        Args:
            state: (B, state_dim) robot joint positions
            
        Returns:
            (B, 1, D) encoded state
        """
        state_emb = self.encoder_robot_state_input_proj(state)
        return state_emb.unsqueeze(1)
    
    def encode_goal(self, goal_image: Tensor) -> Tensor:
        """Encode goal image.
        
        Args:
            goal_image: (B, C, H, W) goal observation
            
        Returns:
            (B, D) goal embedding
        """
        return self.goal_encoder(goal_image)
    
    def fuse_goal(self, features: Tensor, goal_embedding: Tensor) -> Tensor:
        """Fuse goal embedding with encoder features.
        
        Args:
            features: (B, N, D) encoder input features
            goal_embedding: (B, D) goal conditioning
            
        Returns:
            (B, N, D) conditioned features
        """
        if self.config.goal_fusion == "concat":
            # Add goal embedding to all positions
            goal_expanded = goal_embedding.unsqueeze(1).expand(-1, features.shape[1], -1)
            return features + goal_expanded
        
        elif self.config.goal_fusion == "film":
            # Feature-wise modulation
            return self.goal_film(features, goal_embedding)
        
        elif self.config.goal_fusion == "cross_attention":
            # Cross-attention
            goal_seq = goal_embedding.unsqueeze(1)  # (B, 1, D)
            fused, _ = self.goal_cross_attn(features, goal_seq, goal_seq)
            return features + fused
        
        return features
    
    def vae_encode(self, state: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode actions to latent space (for training).
        
        Args:
            state: (B, state_dim) robot state
            actions: (B, chunk_size, action_dim) action sequence
            
        Returns:
            mu: (B, latent_dim) latent mean
            logvar: (B, latent_dim) latent log variance
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Prepare VAE encoder input: [cls, state, action_1, ..., action_T]
        cls_token = self.vae_encoder_cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        state_token = self.vae_encoder_robot_state_input_proj(state).unsqueeze(1)
        action_tokens = self.vae_encoder_action_input_proj(actions)
        
        tokens = torch.cat([cls_token, state_token, action_tokens], dim=1)
        
        # Add positional encoding
        tokens = tokens + self.vae_encoder_pos_enc[:, :tokens.shape[1]]
        
        # Encode
        encoded = self.vae_encoder(tokens)
        
        # Extract latent from CLS token
        cls_output = encoded[:, 0]
        latent_params = self.vae_encoder_latent_output_proj(cls_output)
        
        mu, logvar = latent_params.chunk(2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        images: List[Tensor],
        state: Tensor,
        goal_image: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass.
        
        Args:
            images: List of (B, C, H, W) camera images
            state: (B, state_dim) robot state
            goal_image: (B, C, H, W) goal observation (optional)
            actions: (B, chunk_size, action_dim) ground truth actions (for VAE training)
            
        Returns:
            Dict with:
                - "action": (B, chunk_size, action_dim) predicted actions
                - "mu": (B, latent_dim) VAE mean (if training with VAE)
                - "logvar": (B, latent_dim) VAE log variance (if training with VAE)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Encode images
        image_features = self.encode_images(images)  # (B, N_img, D)
        
        # Encode state
        state_features = self.encode_state(state)  # (B, 1, D)
        
        # Get latent
        if self.config.use_vae and actions is not None and self.training:
            # Training with VAE: encode actions
            mu, logvar = self.vae_encode(state, actions)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference: sample from prior
            z = torch.zeros(batch_size, self.config.latent_dim, device=device)
            mu = None
            logvar = None
        
        latent_features = self.encoder_latent_input_proj(z).unsqueeze(1)  # (B, 1, D)
        
        # Add positional embeddings to 1D tokens
        pos_embed = self.encoder_1d_feature_pos_embed.weight.unsqueeze(0)
        latent_features = latent_features + pos_embed[:, 0:1]
        state_features = state_features + pos_embed[:, 1:2]
        
        # Concatenate all encoder inputs
        encoder_input = torch.cat([latent_features, state_features, image_features], dim=1)
        
        # Apply goal conditioning
        goal_embedding = None
        if self.config.use_goal_image and goal_image is not None:
            goal_embedding = self.encode_goal(goal_image)
            encoder_input = self.fuse_goal(encoder_input, goal_embedding)
        
        # Encode with transformer
        memory = self.encoder(encoder_input)
        
        # Decoder queries (action positions)
        query_pos = self.decoder_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query = torch.zeros_like(query_pos)
        
        # Decode
        decoder_output = self.decoder(query + query_pos, memory)
        
        # BULLETPROOF: Concatenate goal to EVERY timestep before action prediction
        # This physically forces the action head to see the goal
        if goal_embedding is not None:
            goal_expanded = goal_embedding.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
            # Concatenate decoder output + goal
            decoder_with_goal = torch.cat([decoder_output, goal_expanded], dim=-1)
        else:
            decoder_with_goal = decoder_output
        
        # Predict actions (input is now decoder_output + goal)
        actions_pred = self.action_head(decoder_with_goal)
        
        result = {"action": actions_pred}
        if mu is not None:
            result["mu"] = mu
            result["logvar"] = logvar
        
        return result
    
    def predict_action(
        self,
        images: List[Tensor],
        state: Tensor,
        goal_image: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict action chunk for inference.
        
        Args:
            images: List of (B, C, H, W) camera images
            state: (B, state_dim) robot state
            goal_image: (B, C, H, W) goal observation
            
        Returns:
            (B, chunk_size, action_dim) predicted actions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images, state, goal_image)
        return output["action"]
    
    def compute_loss(
        self,
        images: List[Tensor],
        state: Tensor,
        actions: Tensor,
        goal_image: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute training loss.
        
        Args:
            images: List of (B, C, H, W) camera images
            state: (B, state_dim) robot state
            actions: (B, chunk_size, action_dim) ground truth actions
            goal_image: (B, C, H, W) goal observation
            action_mask: (B, chunk_size) mask for valid actions
            
        Returns:
            loss: scalar loss
            loss_dict: dictionary of loss components
        """
        output = self.forward(images, state, goal_image, actions)
        
        # L1 loss for actions
        if action_mask is not None:
            l1_loss = (F.l1_loss(output["action"], actions, reduction="none") * 
                       action_mask.unsqueeze(-1)).mean()
        else:
            l1_loss = F.l1_loss(output["action"], actions)
        
        loss_dict = {"l1_loss": l1_loss.item()}
        
        # KL divergence for VAE
        if self.config.use_vae and "mu" in output:
            mu = output["mu"]
            logvar = output["logvar"]
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            loss_dict["kl_loss"] = kl_loss.item()
            loss = l1_loss + self.config.kl_weight * kl_loss
        else:
            loss = l1_loss
        
        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict


def create_goal_conditioned_act(
    chunk_size: int = 100,
    dim_model: int = 512,
    use_vae: bool = True,
    goal_fusion: str = "film",
    camera_names: List[str] = None,
    state_dim: int = 6,
    action_dim: int = 6,
) -> GoalConditionedACTPolicy:
    """Factory function to create goal-conditioned ACT.
    
    Args:
        chunk_size: Action chunk size (default 100)
        dim_model: Transformer hidden dimension
        use_vae: Whether to use VAE
        goal_fusion: Goal conditioning method ("film", "concat", "cross_attention")
        camera_names: List of camera names
        state_dim: Robot state dimension
        action_dim: Action dimension
        
    Returns:
        GoalConditionedACTPolicy instance
    """
    if camera_names is None:
        camera_names = ["gripper", "station"]
    
    config = GoalConditionedACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        dim_model=dim_model,
        use_vae=use_vae,
        goal_fusion=goal_fusion,
        camera_names=camera_names,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    
    return GoalConditionedACTPolicy(config)


def load_goal_conditioned_act(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[GoalConditionedACTPolicy, GoalConditionedACTConfig]:
    """Load goal-conditioned ACT from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        (policy, config) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = GoalConditionedACTConfig.from_dict(checkpoint["config"])
    policy = GoalConditionedACTPolicy(config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    
    return policy, config

