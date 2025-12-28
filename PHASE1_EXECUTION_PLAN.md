# SAIL Phase 1: Reward Infrastructure - Execution Plan

**Status:** ✅ Phase 0 Complete - Starting Phase 1  
**Objective:** Implement VIP + VLM rewards for autonomous success detection  
**Time Estimate:** 1-2 weeks  
**Hardware:** SO-101, RTX 4070 (12GB), Windows 11

---

## ✅ What You Already Have (Phase 0 Complete)

| Component | Status | Details |
|-----------|--------|---------|
| **Baseline Policy** | ✅ Complete | ACT at 40K steps, works "really good" |
| **Dataset** | ✅ Complete | 50 episodes, 40,914 frames |
| **Cameras** | ✅ Complete | Gripper cam + Station cam (dual view) |
| **Robot** | ✅ Complete | SO-101 calibrated and working |

**Your Baseline Checkpoint:**
```
C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\train_session_01\checkpoints\040000
```

---

## 🎯 Phase 1 Goals

By the end of Phase 1, you will have:

1. **VIP Reward Model** - Gives dense progress signals from images (zero-shot)
2. **VLM Success Detector** - Labels task success automatically (~75%+ accuracy)
3. **Unified Reward Manager** - Combines VIP + VLM into single interface
4. **Validation Results** - Proof that rewards correlate with human labels

**Why This Matters:** These rewards enable Phase 2-3 autonomous practice where the robot:
- Practices tasks without human supervision
- Labels its own success/failure
- Learns from every trajectory (even failures!)

---

## Step 1.1: Install Dependencies

### Required Packages

```powershell
# Activate your conda environment first
conda activate lerobot  # or whatever your env is called

# Install reward model dependencies
pip install transformers>=4.36.0
pip install timm>=0.9.12
pip install Pillow>=10.0.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0  # For 4-bit quantization
```

### Verify Installation

Create test script: `sail/test_dependencies.py`

```python
"""Test that all Phase 1 dependencies are installed."""

def test_imports():
    print("Testing Phase 1 dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
    
    try:
        import timm
        print(f"✓ TIMM {timm.__version__}")
    except ImportError as e:
        print(f"✗ TIMM: {e}")
    
    try:
        from PIL import Image
        print(f"✓ Pillow (PIL)")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
    
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes (for 4-bit quantization)")
    except ImportError as e:
        print(f"✗ BitsAndBytes: {e}")
        print("  Note: BitsAndBytes may not work on Windows. VLM will use FP16 instead.")
    
    print("\n✅ Dependency check complete!")

if __name__ == "__main__":
    test_imports()
```

**Run it:**
```powershell
python sail/test_dependencies.py
```

---

## Step 1.2: Implement VIP Reward Model

VIP provides **zero-shot progress estimation** from images without any training data.

### Create VIP Module

Create file: `sail/rewards/vip.py`

```python
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
        print(f"✓ VIP encoder loaded. Embedding dim: {self.embedding_dim}")
        
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embedding space.
        
        Args:
            images: (B, C, H, W) tensor, values in [0, 1]
            
        Returns:
            embeddings: (B, embedding_dim) L2-normalized tensor
        """
        # Ensure values are in [0, 1]
        if images.max() > 1.0:
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
        if initial_emb.shape[0] == 1:
            initial_emb = initial_emb.expand(current_emb.shape[0], -1)
        if goal_emb.shape[0] == 1:
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
```

### Test VIP Reward

Create file: `sail/test_vip.py`

```python
"""Test VIP reward on your existing dataset."""

import torch
import numpy as np
from pathlib import Path
import sys

# Add sail to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sail.rewards.vip import create_vip_reward
from lerobot.datasets.factory import make_dataset


def test_vip_on_dataset():
    """Test VIP reward using your actual pick-and-place dataset."""
    
    print("=" * 60)
    print("Testing VIP Reward Model")
    print("=" * 60)
    
    # Load your dataset
    dataset_path = "C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2"
    
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = make_dataset(dataset_path)
    
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
    # Create VIP model
    print("\nCreating VIP model...")
    vip = create_vip_reward(model_name="resnet50", device="cuda")
    
    # Test on a few episodes
    print("\nTesting VIP progress computation on episodes...")
    
    for ep_idx in [0, 10, 25]:  # Test a few episodes
        print(f"\n--- Episode {ep_idx} ---")
        
        # Get episode frames from station camera (workspace view)
        episode_indices = dataset.episode_data_index["from"][ep_idx], dataset.episode_data_index["to"][ep_idx]
        
        # Get first, middle, and last frames
        first_idx = episode_indices[0]
        middle_idx = (episode_indices[0] + episode_indices[1]) // 2
        last_idx = episode_indices[1] - 1
        
        initial_img = dataset[first_idx]["observation.images.station"]
        middle_img = dataset[middle_idx]["observation.images.station"]
        final_img = dataset[last_idx]["observation.images.station"]
        
        # Compute progress at different timepoints
        progress_at_start = vip.compute_progress(
            initial_img.unsqueeze(0),
            initial_img.unsqueeze(0),
            final_img.unsqueeze(0)
        )
        
        progress_at_middle = vip.compute_progress(
            middle_img.unsqueeze(0),
            initial_img.unsqueeze(0),
            final_img.unsqueeze(0)
        )
        
        progress_at_end = vip.compute_progress(
            final_img.unsqueeze(0),
            initial_img.unsqueeze(0),
            final_img.unsqueeze(0)
        )
        
        print(f"Progress at start: {progress_at_start.item():.3f} (expect ~0.0)")
        print(f"Progress at middle: {progress_at_middle.item():.3f} (expect ~0.3-0.7)")
        print(f"Progress at end: {progress_at_end.item():.3f} (expect ~1.0)")
        
        # Compute reward
        reward_at_end = vip.compute_reward(
            final_img.unsqueeze(0),
            final_img.unsqueeze(0)
        )
        print(f"Reward when at goal: {reward_at_end.item():.3f} (expect close to 0)")
    
    print("\n" + "=" * 60)
    print("✅ VIP Test Complete!")
    print("=" * 60)
    print("\nIf progress values make sense (0 → middle → 1),")
    print("VIP is working correctly for your task!")


if __name__ == "__main__":
    test_vip_on_dataset()
```

**Run it:**
```powershell
python sail/test_vip.py
```

**Expected Output:**
- Progress should be ~0.0 at start
- Progress should increase in middle frames
- Progress should be ~0.9-1.0 at end

---

## Step 1.3: Implement VLM Success Detector

VLM will label whether the task succeeded by looking at the final image.

### Create VLM Module

Create file: `sail/rewards/vlm_detector.py`

```python
"""
VLM-based Success Detector
Uses vision-language model to label task success/failure

Based on SOAR paper's approach. We use Qwen2-VL for accessibility on RTX 4070.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List


class VLMSuccessDetector:
    """
    Detect task success using VLM visual question answering.
    
    Uses binary question format:
    Q: "Has the robot successfully [task]? Answer yes or no."
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        use_4bit: bool = False  # 4-bit may not work on Windows
    ):
        self.device = device
        
        print(f"Loading VLM: {model_name}...")
        
        # Load model with appropriate quantization
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("✓ Loaded in 4-bit mode")
            except Exception as e:
                print(f"⚠️  4-bit loading failed: {e}")
                print("  Falling back to FP16...")
                use_4bit = False
        
        if not use_4bit:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✓ Loaded in FP16 mode")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"✓ VLM loaded successfully")
        
    def _prepare_image(self, image: np.ndarray | torch.Tensor) -> Image.Image:
        """Convert numpy/torch array to PIL Image."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.dtype != np.uint8:
            # Assume [0, 1] range
            image = (image * 255).astype(np.uint8)
        
        # Handle CHW -> HWC if needed
        if image.shape[0] == 3 and len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        return Image.fromarray(image)
    
    @torch.no_grad()
    def detect_success(
        self,
        image: np.ndarray | torch.Tensor,
        task_description: str
    ) -> Tuple[bool, float]:
        """
        Determine if task was successful.
        
        Args:
            image: Final observation image
            task_description: Natural language task description
            
        Returns:
            success: Boolean indicating success
            confidence: Confidence score [0, 1]
        """
        pil_image = self._prepare_image(image)
        
        # Construct prompt
        prompt = f"""Look at this image of a robot workspace after completing a task.

Task: {task_description}

Question: Did the robot successfully complete this task?
Answer with ONLY 'yes' or 'no'."""
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Decode response
        response = self.processor.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).lower().strip()
        
        # Parse response
        success = "yes" in response
        
        # Estimate confidence from logits
        if outputs.scores:
            first_token_logits = outputs.scores[0][0]
            probs = torch.softmax(first_token_logits, dim=-1)
            confidence = probs.max().item()
        else:
            confidence = 0.5
        
        return success, confidence
    
    @torch.no_grad()
    def propose_tasks(
        self,
        image: np.ndarray | torch.Tensor,
        available_tasks: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Propose feasible tasks given current workspace state.
        
        Used for autonomous practice: VLM suggests what robot can practice.
        
        Args:
            image: Current workspace observation
            available_tasks: List of possible task descriptions
            
        Returns:
            List of (task, feasibility_score) tuples, sorted by score
        """
        pil_image = self._prepare_image(image)
        
        task_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(available_tasks)])
        
        prompt = f"""Look at this robot workspace image.

Available tasks:
{task_list}

Which tasks could the robot attempt right now based on what objects are visible and their positions?
List the task numbers that are feasible, separated by commas."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
        
        response = self.processor.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response to get task indices
        feasible_tasks = []
        for i, task in enumerate(available_tasks):
            if str(i+1) in response:
                feasible_tasks.append((task, 1.0))
            else:
                feasible_tasks.append((task, 0.0))
        
        # Sort by feasibility
        feasible_tasks.sort(key=lambda x: x[1], reverse=True)
        
        return feasible_tasks


def create_vlm_detector(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    use_4bit: bool = False
) -> VLMSuccessDetector:
    """Factory function to create VLM detector."""
    return VLMSuccessDetector(
        model_name=model_name,
        device=device,
        use_4bit=use_4bit
    )
```

### Test VLM Detector

Create file: `sail/test_vlm.py`

```python
"""Test VLM success detection on your dataset."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sail.rewards.vlm_detector import create_vlm_detector
from lerobot.datasets.factory import make_dataset


def test_vlm_on_dataset():
    """Test VLM success detection on your pick-and-place episodes."""
    
    print("=" * 60)
    print("Testing VLM Success Detector")
    print("=" * 60)
    
    # Load dataset
    dataset_path = "C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2"
    
    print(f"\nLoading dataset...")
    dataset = make_dataset(dataset_path)
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    
    # Create VLM detector
    print("\nCreating VLM detector...")
    print("(This will download ~4GB model on first run)")
    vlm = create_vlm_detector(device="cuda", use_4bit=False)
    
    # Test on a few episodes
    task_description = "pick up the cube and place it in the target zone"
    
    print(f"\nTesting VLM on task: '{task_description}'")
    print("\nLabeling final frames of episodes...")
    
    results = []
    test_episodes = [0, 5, 10, 15, 20]  # Test 5 episodes
    
    for ep_idx in test_episodes:
        print(f"\n--- Episode {ep_idx} ---")
        
        # Get final frame from station camera
        episode_indices = dataset.episode_data_index["from"][ep_idx], dataset.episode_data_index["to"][ep_idx]
        final_idx = episode_indices[1] - 1
        
        final_img = dataset[final_idx]["observation.images.station"]
        
        # Detect success
        success, confidence = vlm.detect_success(final_img, task_description)
        
        print(f"VLM says: {'SUCCESS' if success else 'FAILURE'} (confidence: {confidence:.2f})")
        
        results.append({
            "episode": ep_idx,
            "vlm_success": success,
            "vlm_confidence": confidence,
            "human_label": None  # You'll fill this in manually
        })
    
    print("\n" + "=" * 60)
    print("✅ VLM Test Complete!")
    print("=" * 60)
    
    print("\nNow manually verify these results:")
    for r in results:
        print(f"Episode {r['episode']}: VLM says {'SUCCESS' if r['vlm_success'] else 'FAILURE'}")
    
    print("\nNext step: Compare VLM labels to your judgment")
    print("Target: >75% agreement with human labels")
    
    return results


if __name__ == "__main__":
    test_vlm_on_dataset()
```

**Run it:**
```powershell
python sail/test_vlm.py
```

**This will take a few minutes on first run (downloads 4GB model).**

---

## Step 1.4: Create Unified Reward Manager

Combines VIP (progress) + VLM (success) into single interface.

Create file: `sail/rewards/reward_manager.py`

```python
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
        print("Initializing Reward Manager...")
        print("  Loading VIP...")
        self.vip = create_vip_reward(model_name=vip_model, device=device)
        
        if use_vlm:
            print("  Loading VLM...")
            self.vlm = create_vlm_detector(model_name=vlm_model, device=device, use_4bit=False)
        else:
            self.vlm = None
        
        print("✓ Reward Manager ready!")
        
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
            tensor = image
        else:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            tensor = torch.from_numpy(image)
        
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
            signals.vlm_success = success
            signals.vlm_confidence = confidence
            
            # Add success bonus
            if success:
                signals.combined_reward += self.success_bonus
        
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
```

---

## Step 1.5: Validate Rewards Against Human Labels

Now test that VIP + VLM rewards correlate with human judgment.

Create file: `sail/scripts/validate_rewards.py`

```python
"""
Validate reward signals against human labels.
Goal: >75% agreement between VLM success and human judgment.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.factory import make_dataset
from sail.rewards.reward_manager import create_reward_manager


def validate_rewards(
    dataset_path: str,
    task_description: str,
    num_episodes: int = 20,
    output_path: str = "sail/experiments/reward_validation.json"
):
    """
    Validate reward signals on your dataset.
    
    This will:
    1. Load your pick-and-place dataset
    2. Run VIP + VLM on episodes
    3. Save results for you to manually label
    4. Compute agreement after you add human labels
    """
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = make_dataset(dataset_path)
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    
    # Create reward manager
    print("\nInitializing reward manager (VIP + VLM)...")
    manager = create_reward_manager(device="cuda", use_vlm=True)
    
    # Test on episodes
    results = []
    test_episodes = np.linspace(0, min(num_episodes, dataset.num_episodes) - 1, num_episodes, dtype=int)
    
    print(f"\nValidating on {len(test_episodes)} episodes...")
    
    for ep_idx in test_episodes:
        print(f"  Episode {ep_idx}...", end=" ")
        
        # Get episode frames
        from_idx, to_idx = dataset.episode_data_index["from"][ep_idx], dataset.episode_data_index["to"][ep_idx]
        
        initial_frame = dataset[from_idx]["observation.images.station"]
        final_frame = dataset[to_idx - 1]["observation.images.station"]
        
        # Set up reward manager
        manager.set_initial(initial_frame)
        manager.set_goal(final_frame)
        
        # Get VIP progress
        vip_signals = manager.compute_step_reward(final_frame)
        
        # Get VLM success
        vlm_success, vlm_confidence = manager.vlm.detect_success(
            final_frame, task_description
        )
        
        results.append({
            "episode": int(ep_idx),
            "vlm_success": bool(vlm_success),
            "vlm_confidence": float(vlm_confidence),
            "vip_progress": float(vip_signals.vip_progress),
            "human_label": None  # To be filled manually
        })
        
        print(f"VLM={'YES' if vlm_success else 'NO'} (conf={vlm_confidence:.2f}), VIP prog={vip_signals.vip_progress:.2f}")
    
    # Save for human labeling
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("NEXT STEP: Manual Labeling")
    print("=" * 60)
    print(f"1. Open: {output_path}")
    print("2. For each episode, set 'human_label' to true or false")
    print("3. Run: python sail/scripts/validate_rewards.py compute")
    print("=" * 60)


def compute_agreement(validation_path: str = "sail/experiments/reward_validation.json"):
    """Compute agreement after human labels are added."""
    print("Computing VLM agreement with human labels...")
    
    with open(validation_path) as f:
        results = json.load(f)
    
    # Filter to episodes with human labels
    labeled = [r for r in results if r["human_label"] is not None]
    
    if not labeled:
        print("✗ No human labels found!")
        print(f"Please edit {validation_path} and add human_label (true/false) for each episode.")
        return
    
    print(f"\nFound {len(labeled)} labeled episodes")
    
    # Compute agreement
    agreements = sum(1 for r in labeled if r["vlm_success"] == r["human_label"])
    accuracy = agreements / len(labeled)
    
    # Compute recall (VLM says yes when human says yes)
    human_positives = [r for r in labeled if r["human_label"]]
    if human_positives:
        recall = sum(1 for r in human_positives if r["vlm_success"]) / len(human_positives)
    else:
        recall = 0.0
    
    # Compute precision (when VLM says yes, human agrees)
    vlm_positives = [r for r in labeled if r["vlm_success"]]
    if vlm_positives:
        precision = sum(1 for r in vlm_positives if r["human_label"]) / len(vlm_positives)
    else:
        precision = 0.0
    
    print("\n" + "=" * 60)
    print("VLM Validation Results")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.1%} ({agreements}/{len(labeled)})")
    print(f"Precision: {precision:.1%} (when VLM says yes, human agrees)")
    print(f"Recall:    {recall:.1%} (VLM catches all true successes)")
    print("=" * 60)
    print(f"\nTarget: >75% accuracy, >90% recall")
    
    if accuracy >= 0.75:
        print("✅ PASSED: VLM is ready for autonomous practice!")
    else:
        print("⚠️  Below target. Consider:")
        print("   - Refining task description prompt")
        print("   - Using larger VLM model")
        print("   - Adjusting camera angle for clearer view")
    
    # VIP correlation
    print("\n" + "=" * 60)
    print("VIP Progress Correlation")
    print("=" * 60)
    
    successful = [r for r in labeled if r["human_label"]]
    failed = [r for r in labeled if not r["human_label"]]
    
    if successful and failed:
        avg_progress_success = np.mean([r["vip_progress"] for r in successful])
        avg_progress_fail = np.mean([r["vip_progress"] for r in failed])
        
        print(f"Avg VIP progress on successes: {avg_progress_success:.3f}")
        print(f"Avg VIP progress on failures:  {avg_progress_fail:.3f}")
        print(f"Separation: {avg_progress_success - avg_progress_fail:.3f}")
        
        if avg_progress_success > avg_progress_fail + 0.1:
            print("✅ VIP progress correlates with success!")
        else:
            print("⚠️  VIP may not be discriminative enough")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute_agreement()
    else:
        validate_rewards(
            dataset_path="C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2",
            task_description="pick up the cube and place it in the target zone",
            num_episodes=20
        )
```

**Run it:**
```powershell
# Generate VIP + VLM predictions
python sail/scripts/validate_rewards.py

# Then manually label the JSON file
# Then compute agreement:
python sail/scripts/validate_rewards.py compute
```

---

## ✅ Phase 1 Success Criteria

Before proceeding to Phase 2, you should have:

| Metric | Target | Status |
|--------|--------|--------|
| **VIP Progress** | Increases from 0 → 1 during episode | ⬜ |
| **VLM Accuracy** | >75% agreement with human labels | ⬜ |
| **VLM Recall** | >90% (catches all real successes) | ⬜ |
| **Integration** | Reward manager works on real images | ⬜ |
| **Documentation** | Validation results saved | ⬜ |

---

## 🎯 Deliverables

By end of Phase 1, you'll have these files:

```
sail/
├── rewards/
│   ├── __init__.py
│   ├── vip.py                    ← VIP reward model
│   ├── vlm_detector.py           ← VLM success detector
│   └── reward_manager.py         ← Unified manager
├── scripts/
│   └── validate_rewards.py       ← Validation script
├── experiments/
│   └── reward_validation.json    ← Results with human labels
├── test_dependencies.py
├── test_vip.py
└── test_vlm.py
```

---

## 🚀 Next Steps to Phase 2

Once Phase 1 is complete, we'll implement:

1. **Goal-Conditioned ACT** - Extend your existing policy to accept goal images
2. **Hindsight Experience Replay** - Learn from "failed" attempts
3. **Enable self-improvement** - Policy learns from VIP/VLM feedback

---

## 💡 Tips & Troubleshooting

### If VLM is too slow:
- Use smaller model: `Qwen/Qwen2-VL-1.5B` instead of 2B
- Run VLM on CPU, keep VIP on GPU
- Cache VLM results for validation

### If VIP doesn't correlate:
- Try different backbone: `resnet101` or `vit_base_patch16_224`
- Use gripper cam instead of station cam (or vice versa)
- Ensure goal images are truly representative

### If VLM accuracy is low:
- Refine task description to be more specific
- Provide examples in the prompt (few-shot)
- Use larger model if VRAM allows

### Windows-specific:
- BitsAndBytes 4-bit may not work → use FP16
- COM ports instead of /dev/ttyUSB*
- Use forward slashes in paths or escape backslashes

---

**Status:** Ready to start Phase 1 implementation  
**First Command:** `python sail/test_dependencies.py`  
**Estimated Time:** 1-2 weeks

---

*Let's begin! Run the dependency test and share the results.*

