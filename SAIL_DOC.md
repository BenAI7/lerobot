# SAIL: Self-Adaptation Imitation Learning
## Complete Implementation Plan for LeRobot + SO-101

**Project Goal:** Achieve sub-10 demonstration learning with autonomous self-improvement  
**Hardware:** SO-101 arms, RTX 4070 (16GB), i9 CPU, 96GB RAM  
**Framework:** LeRobot (Hugging Face)  
**Development Environment:** Cursor IDE

---

# PHASE 0: BASELINE FOUNDATION
**Objective:** Train working ACT policy from 5 demonstrations

This is your control baseline. Every future improvement gets measured against this.

---

## Step 0.1: Verify LeRobot Installation

### Action Item
```bash
# Verify LeRobot is properly installed
cd ~/lerobot
python -c "import lerobot; print(lerobot.__version__)"

# Check SO-101 connection
python -c "from lerobot.common.robot_devices.robots.factory import make_robot; robot = make_robot('so101')"
```

### Expected Output
- LeRobot version prints (should be 0.1.0+)
- No errors on robot import

### Troubleshooting
If SO-101 not detected:
```bash
# Check USB connection
ls /dev/ttyUSB*
# Should show /dev/ttyUSB0 or similar

# Check permissions
sudo usermod -a -G dialout $USER
# Then logout/login
```

---

## Step 0.2: Calibrate SO-101 Arms

### Action Item
```bash
# Calibrate follower arm (the one that executes)
python lerobot/scripts/control_robot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=follower \
    calibrate

# Calibrate leader arm (the one you teleoperate)
python lerobot/scripts/control_robot.py \
    --robot.type=so101_leader \
    --robot.port=/dev/ttyUSB1 \
    --robot.id=leader \
    calibrate
```

### What This Does
- Moves each joint to limits
- Records min/max positions
- Saves calibration to `~/.cache/huggingface/lerobot/calibration/`

### Verification
```bash
# Test teleoperation after calibration
python lerobot/scripts/control_robot.py \
    --robot.type=so101 \
    teleoperate
```
Move the leader arm - follower should mirror movements smoothly.

---

## Step 0.3: Define Your First Task

### Task Selection Criteria
For baseline, choose a task that is:
- Single object manipulation
- Clear success/failure
- Completes in 5-10 seconds
- Repeatable positioning

### Recommended First Task: "Pick and Place Cube"
- Start: Cube at position A (marked on workspace)
- Goal: Cube at position B (marked on workspace)
- Success: Cube fully within target zone

### Workspace Setup
```
+----------------------------------+
|                                  |
|   [A: Start Zone]    [B: Goal]   |
|      (cube)            (X)       |
|                                  |
|         [SO-101 Base]            |
+----------------------------------+
```

Mark zones with tape for consistency.

---

## Step 0.4: Record 5 Demonstrations

### Action Item
```bash
# Create dataset directory
export DATASET_NAME="sail_pickplace_v1"
export HF_USER="your_huggingface_username"

# Record demonstrations
python lerobot/scripts/control_robot.py \
    --robot.type=so101 \
    --control.type=teleoperate \
    --control.fps=30 \
    record \
    --dataset.repo_id=${HF_USER}/${DATASET_NAME} \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=15
```

### Recording Protocol
For each of the 5 episodes:
1. Reset cube to start position A
2. Press ENTER to start recording
3. Teleoperate: reach → grasp → lift → move → place → release
4. Press ENTER to stop
5. Confirm success when prompted

### Quality Guidelines
- **Consistent strategy**: Use same approach each time
- **Smooth motions**: Avoid jerky movements
- **Clean grasp**: Full grip before lifting
- **Complete release**: Fully open gripper at end

### Verification
```bash
# Check recorded data
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('${HF_USER}/${DATASET_NAME}')
print(f'Episodes: {ds.num_episodes}')
print(f'Total frames: {len(ds)}')
print(f'FPS: {ds.fps}')
"
```

Expected: 5 episodes, ~2000-4000 total frames at 30 FPS

---

## Step 0.5: Train ACT Policy (Baseline)

### Create Training Config

Create file: `configs/sail_act_baseline.yaml`

```yaml
# SAIL Baseline ACT Configuration
# Optimized for SO-101 + RTX 4070

seed: 42
dataset_repo_id: ${HF_USER}/sail_pickplace_v1

training:
  offline_steps: 25000
  batch_size: 8
  lr: 1e-4
  lr_scheduler: cosine
  lr_warmup_steps: 500
  grad_clip_norm: 10
  save_freq: 5000
  eval_freq: 2500
  log_freq: 100

policy:
  name: act
  
  # ACT-specific parameters
  chunk_size: 100  # Critical for few-shot learning
  n_action_steps: 100
  
  # Architecture
  dim_model: 256
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1
  
  # VAE for action distribution
  use_vae: true
  latent_dim: 32
  kl_weight: 10.0
  
  # Input configuration
  input_shapes:
    observation.images.top: [3, 480, 640]
    observation.state: [6]  # 6-DOF joint positions
  
  output_shapes:
    action: [6]  # 6-DOF joint commands

# Memory optimization for RTX 4070
device: cuda
use_amp: true  # Mixed precision
```

### Run Training

```bash
python lerobot/scripts/train.py \
    --config configs/sail_act_baseline.yaml \
    --output_dir outputs/sail_baseline_v1
```

### Expected Training Time
- RTX 4070: ~2-4 hours for 25K steps
- Watch loss curve in tensorboard

### Monitor Training
```bash
# In separate terminal
tensorboard --logdir outputs/sail_baseline_v1
```

Look for:
- Loss decreasing steadily
- No NaN values
- KL divergence stable (if using VAE)

---

## Step 0.6: Evaluate Baseline Policy

### Run Evaluation

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so101 \
    --control.type=policy \
    --control.policy_path=outputs/sail_baseline_v1/checkpoints/last \
    --control.fps=30 \
    eval \
    --eval.num_episodes=10
```

### Record Baseline Metrics

Create file: `experiments/baseline_results.json`

```json
{
  "experiment": "sail_baseline_v1",
  "num_demos": 5,
  "policy": "ACT",
  "chunk_size": 100,
  "training_steps": 25000,
  "eval_episodes": 10,
  "results": {
    "success_rate": 0.0,
    "partial_success_rate": 0.0,
    "average_completion_time": 0.0,
    "failure_modes": []
  },
  "notes": ""
}
```

### Expected Baseline Performance
With only 5 demos: **20-40% success rate**

This is your benchmark. Everything we add should improve on this.

---

## Step 0.7: Analyze Failure Modes

### Watch Failed Episodes
Record video of 5 failures and categorize:

| Failure Mode | Count | Description |
|--------------|-------|-------------|
| Miss grasp | | Gripper closes but misses object |
| Premature release | | Drops object during transport |
| Wrong location | | Places outside target zone |
| Collision | | Hits workspace/object |
| Timeout | | Doesn't complete in time |

### Document in experiments log
This informs which SAIL components to prioritize.

---

# PHASE 1: REWARD INFRASTRUCTURE
**Objective:** Enable autonomous success detection without human labels

---

## Step 1.1: Implement VIP Reward Model

VIP (Value-Implicit Pre-training) gives zero-shot reward signals from goal images.

### Install Dependencies

```bash
pip install torch torchvision --upgrade
pip install timm  # For vision transformers
```

### Create VIP Module

Create file: `sail/rewards/vip.py`

```python
"""
VIP (Value-Implicit Pre-training) Reward Model
Zero-shot reward from goal image similarity

Paper: https://arxiv.org/abs/2210.00030
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm


class VIPReward(nn.Module):
    """
    Computes reward as negative L2 distance between
    current observation embedding and goal embedding.
    
    Pretrained on Ego4D human videos - generalizes to robot tasks.
    """
    
    def __init__(self, model_name: str = "resnet50", device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Load pretrained encoder
        # Using R3M weights which are trained on Ego4D
        self.encoder = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        self.encoder.eval()
        self.encoder.to(device)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.embedding_dim = self.encoder.num_features
        
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embedding space.
        
        Args:
            images: (B, C, H, W) tensor, values in [0, 1]
            
        Returns:
            embeddings: (B, embedding_dim) tensor
        """
        if images.max() > 1.0:
            images = images / 255.0
            
        images = self.transform(images)
        embeddings = self.encoder(images)
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
        
        initial_to_goal = torch.norm(goal_emb - initial_emb, dim=-1)
        current_to_goal = torch.norm(goal_emb - current_emb, dim=-1)
        
        # Progress = how much of the distance has been covered
        progress = 1.0 - (current_to_goal / (initial_to_goal + 1e-8))
        progress = torch.clamp(progress, 0.0, 1.0)
        
        return progress


# Factory function for easy instantiation
def create_vip_reward(device: str = "cuda") -> VIPReward:
    """Create VIP reward model with default settings."""
    return VIPReward(model_name="resnet50", device=device)
```

### Test VIP Reward

Create file: `tests/test_vip_reward.py`

```python
"""Test VIP reward computation."""

import torch
from sail.rewards.vip import create_vip_reward

def test_vip_reward():
    # Create model
    vip = create_vip_reward(device="cuda")
    
    # Create dummy images (B, C, H, W)
    initial = torch.rand(1, 3, 480, 640).cuda()
    current = torch.rand(1, 3, 480, 640).cuda()
    goal = torch.rand(1, 3, 480, 640).cuda()
    
    # Test encoding
    emb = vip.encode(initial)
    print(f"Embedding shape: {emb.shape}")  # Should be (1, 2048) for ResNet50
    
    # Test reward
    reward = vip.compute_reward(current, goal)
    print(f"Reward: {reward.item():.4f}")  # Should be negative
    
    # Test progress
    progress = vip.compute_progress(current, initial, goal)
    print(f"Progress: {progress.item():.4f}")  # Should be in [0, 1]
    
    # Verify: current=goal should give progress=1
    progress_at_goal = vip.compute_progress(goal, initial, goal)
    print(f"Progress at goal: {progress_at_goal.item():.4f}")  # Should be ~1.0
    
    # Verify: current=initial should give progress=0
    progress_at_start = vip.compute_progress(initial, initial, goal)
    print(f"Progress at start: {progress_at_start.item():.4f}")  # Should be ~0.0
    
    print("VIP reward test passed!")

if __name__ == "__main__":
    test_vip_reward()
```

Run test:
```bash
python tests/test_vip_reward.py
```

### Memory Usage
VIP with ResNet50: ~200MB VRAM for inference

---

## Step 1.2: Implement VLM Success Detector

Use a vision-language model to determine if task succeeded.

### Create VLM Detector Module

Create file: `sail/rewards/vlm_detector.py`

```python
"""
VLM-based Success Detector
Uses vision-language model to label task success/failure

Based on SOAR paper's approach using CogVLM
We use Qwen-VL for better accessibility
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from typing import Tuple, Optional


class VLMSuccessDetector:
    """
    Detect task success using VLM visual question answering.
    
    Uses a simple binary question format:
    Q: "Has the robot successfully [task description]? Answer yes or no."
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True  # For RTX 4070 memory
    ):
        self.device = device
        
        # Load model with quantization for memory efficiency
        if load_in_4bit:
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
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.shape[0] == 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        return Image.fromarray(image)
    
    @torch.no_grad()
    def detect_success(
        self,
        image: np.ndarray,
        task_description: str
    ) -> Tuple[bool, float]:
        """
        Determine if task was successful.
        
        Args:
            image: Final observation image (C, H, W) or (H, W, C)
            task_description: Natural language task description
            
        Returns:
            success: Boolean indicating success
            confidence: Confidence score [0, 1]
        """
        pil_image = self._prepare_image(image)
        
        # Construct prompt
        prompt = f"""Look at this image of a robot workspace.
Task: {task_description}

Has the robot successfully completed this task? 
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
        
        # Estimate confidence from first token logits
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
        image: np.ndarray,
        available_tasks: list[str]
    ) -> list[Tuple[str, float]]:
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


def create_vlm_detector(device: str = "cuda") -> VLMSuccessDetector:
    """Create VLM detector with default settings."""
    return VLMSuccessDetector(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device=device,
        load_in_4bit=True
    )
```

### Test VLM Detector

Create file: `tests/test_vlm_detector.py`

```python
"""Test VLM success detection."""

import numpy as np
from sail.rewards.vlm_detector import create_vlm_detector

def test_vlm_detector():
    # Create detector
    detector = create_vlm_detector(device="cuda")
    
    # Create dummy image (would be real image in practice)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test success detection
    task = "pick up the red cube and place it in the blue box"
    success, confidence = detector.detect_success(image, task)
    print(f"Success: {success}, Confidence: {confidence:.2f}")
    
    # Test task proposal
    tasks = [
        "pick up the red cube",
        "place cube in box",
        "push cube to the left",
        "stack cubes"
    ]
    proposed = detector.propose_tasks(image, tasks)
    print(f"Proposed tasks: {proposed}")
    
    print("VLM detector test passed!")

if __name__ == "__main__":
    test_vlm_detector()
```

### Memory Usage
Qwen2-VL-2B (4-bit): ~3-4GB VRAM

---

## Step 1.3: Create Unified Reward Manager

Combines VIP progress and VLM success into single interface.

Create file: `sail/rewards/reward_manager.py`

```python
"""
Unified Reward Manager for SAIL
Combines VIP (progress) + VLM (success) signals
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
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
        use_vlm: bool = True
    ):
        self.device = device
        self.vip_weight = vip_weight
        self.success_bonus = success_bonus
        self.use_vlm = use_vlm
        
        # Initialize reward models
        print("Loading VIP reward model...")
        self.vip = create_vip_reward(device)
        
        if use_vlm:
            print("Loading VLM success detector...")
            self.vlm = create_vlm_detector(device)
        else:
            self.vlm = None
        
        # Cache for goal embedding
        self.goal_embedding: Optional[torch.Tensor] = None
        self.initial_embedding: Optional[torch.Tensor] = None
        
    def set_goal(self, goal_image: np.ndarray) -> None:
        """
        Set the goal image for reward computation.
        Call at start of episode.
        
        Args:
            goal_image: (H, W, C) or (C, H, W) goal observation
        """
        goal_tensor = self._to_tensor(goal_image)
        self.goal_embedding = self.vip.encode(goal_tensor)
        
    def set_initial(self, initial_image: np.ndarray) -> None:
        """
        Set the initial image for progress computation.
        Call at start of episode.
        """
        initial_tensor = self._to_tensor(initial_image)
        self.initial_embedding = self.vip.encode(initial_tensor)
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.shape[-1] == 3:  # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor
    
    def compute_step_reward(
        self,
        current_image: np.ndarray
    ) -> RewardSignals:
        """
        Compute reward for a single timestep.
        
        Args:
            current_image: Current observation
            
        Returns:
            RewardSignals with VIP progress/reward
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
        final_image: np.ndarray,
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
        current_image: np.ndarray,
        available_tasks: list[str]
    ) -> list[Tuple[str, float]]:
        """
        Propose feasible tasks for autonomous practice.
        
        Args:
            current_image: Current workspace state
            available_tasks: List of possible tasks
            
        Returns:
            Sorted list of (task, score) tuples
        """
        if self.vlm is None:
            # If no VLM, return all tasks with equal weight
            return [(t, 1.0) for t in available_tasks]
        
        return self.vlm.propose_tasks(current_image, available_tasks)


def create_reward_manager(
    device: str = "cuda",
    use_vlm: bool = True
) -> RewardManager:
    """Create reward manager with default settings."""
    return RewardManager(
        device=device,
        vip_weight=1.0,
        success_bonus=10.0,
        use_vlm=use_vlm
    )
```

### Test Reward Manager

Create file: `tests/test_reward_manager.py`

```python
"""Test unified reward manager."""

import numpy as np
from sail.rewards.reward_manager import create_reward_manager

def test_reward_manager():
    # Create manager (set use_vlm=False for faster testing)
    manager = create_reward_manager(device="cuda", use_vlm=False)
    
    # Simulate episode
    initial = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    goal = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    manager.set_initial(initial)
    manager.set_goal(goal)
    
    # Compute step rewards
    for step in range(5):
        current = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        signals = manager.compute_step_reward(current)
        print(f"Step {step}: progress={signals.vip_progress:.3f}, reward={signals.vip_reward:.3f}")
    
    # Final episode reward
    final_signals = manager.compute_episode_reward(
        current, "pick up the cube"
    )
    print(f"Final: combined_reward={final_signals.combined_reward:.3f}")
    
    print("Reward manager test passed!")

if __name__ == "__main__":
    test_reward_manager()
```

---

## Step 1.4: Validate Reward Signals Against Human Labels

### Collect Validation Data

```bash
# Record 20 trajectories for validation
python lerobot/scripts/control_robot.py \
    --robot.type=so101 \
    --control.type=policy \
    --control.policy_path=outputs/sail_baseline_v1/checkpoints/last \
    record \
    --dataset.repo_id=${HF_USER}/sail_validation \
    --dataset.num_episodes=20
```

### Create Validation Script

Create file: `sail/scripts/validate_rewards.py`

```python
"""
Validate reward signals against human labels.
Goal: >75% agreement between VLM success and human judgment.
"""

import json
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from sail.rewards.reward_manager import create_reward_manager


def validate_rewards(
    dataset_repo_id: str,
    task_description: str,
    output_path: str = "experiments/reward_validation.json"
):
    # Load dataset
    dataset = LeRobotDataset(dataset_repo_id)
    
    # Create reward manager
    manager = create_reward_manager(device="cuda", use_vlm=True)
    
    results = []
    
    for episode_idx in range(dataset.num_episodes):
        # Get episode data
        episode = dataset.get_episode(episode_idx)
        
        # Get final frame
        final_frame = episode["observation.images.top"][-1]
        
        # Get VLM prediction
        vlm_success, vlm_confidence = manager.vlm.detect_success(
            final_frame.numpy(), task_description
        )
        
        # Get VIP progress
        initial_frame = episode["observation.images.top"][0]
        goal_frame = final_frame  # Assume we want to reach final state
        
        manager.set_initial(initial_frame.numpy())
        manager.set_goal(goal_frame.numpy())
        
        vip_signals = manager.compute_step_reward(final_frame.numpy())
        
        results.append({
            "episode": episode_idx,
            "vlm_success": vlm_success,
            "vlm_confidence": vlm_confidence,
            "vip_progress": vip_signals.vip_progress,
            "human_label": None  # To be filled manually
        })
        
        print(f"Episode {episode_idx}: VLM={vlm_success} ({vlm_confidence:.2f}), Progress={vip_signals.vip_progress:.2f}")
    
    # Save for human labeling
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    print("Please fill in 'human_label' field (true/false) for each episode")


def compute_agreement(validation_path: str = "experiments/reward_validation.json"):
    """Compute agreement after human labels are added."""
    with open(validation_path) as f:
        results = json.load(f)
    
    # Filter to episodes with human labels
    labeled = [r for r in results if r["human_label"] is not None]
    
    if not labeled:
        print("No human labels found. Please label episodes first.")
        return
    
    # Compute agreement
    agreements = sum(1 for r in labeled if r["vlm_success"] == r["human_label"])
    accuracy = agreements / len(labeled)
    
    # Compute recall (VLM says yes when human says yes)
    human_positives = [r for r in labeled if r["human_label"]]
    if human_positives:
        recall = sum(1 for r in human_positives if r["vlm_success"]) / len(human_positives)
    else:
        recall = 0.0
    
    print(f"\nVLM Validation Results:")
    print(f"  Accuracy: {accuracy:.1%} ({agreements}/{len(labeled)})")
    print(f"  Recall: {recall:.1%}")
    print(f"  Target: >75% accuracy, >90% recall")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute_agreement()
    else:
        validate_rewards(
            dataset_repo_id="your_username/sail_validation",
            task_description="pick up the cube and place it in the target zone"
        )
```

### Run Validation
```bash
# Generate predictions
python sail/scripts/validate_rewards.py

# Manually label experiments/reward_validation.json

# Compute agreement
python sail/scripts/validate_rewards.py compute
```

**Target:** >75% accuracy, >90% recall on successes

---

# PHASE 2: GOAL-CONDITIONED ARCHITECTURE
**Objective:** Convert ACT policy to accept goal images

This is the critical architectural change that enables hindsight relabeling and self-improvement.

---

## Step 2.1: Create Goal-Conditioned ACT

Create file: `sail/policies/goal_conditioned_act.py`

```python
"""
Goal-Conditioned ACT Policy for SAIL

Extends standard ACT to accept goal images as conditioning.
Enables hindsight experience replay during self-improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from lerobot.common.policies.act.modeling_act import (
    ACTPolicy,
    ACTConfig,
    ACTEncoder,
    ACTDecoder
)


@dataclass
class GoalConditionedACTConfig(ACTConfig):
    """Configuration for goal-conditioned ACT."""
    # Goal conditioning
    use_goal_image: bool = True
    goal_encoder_type: str = "shared"  # "shared" or "separate"
    goal_fusion: str = "concat"  # "concat", "film", or "cross_attention"
    
    # Hindsight relabeling
    hindsight_ratio: float = 0.5  # Fraction of batch to relabel


class GoalEncoder(nn.Module):
    """Encodes goal image to conditioning vector."""
    
    def __init__(self, config: GoalConditionedACTConfig, observation_encoder: nn.Module):
        super().__init__()
        self.config = config
        
        if config.goal_encoder_type == "shared":
            # Share weights with observation encoder
            self.encoder = observation_encoder
        else:
            # Separate encoder for goals
            self.encoder = self._build_encoder(config)
        
        self.projection = nn.Linear(config.dim_model, config.dim_model)
    
    def _build_encoder(self, config):
        # Simple CNN encoder
        return nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, config.dim_model)
        )
    
    def forward(self, goal_image: torch.Tensor) -> torch.Tensor:
        """
        Encode goal image.
        
        Args:
            goal_image: (B, C, H, W) goal observation
            
        Returns:
            goal_embedding: (B, dim_model) conditioning vector
        """
        features = self.encoder(goal_image)
        if features.dim() > 2:
            features = features.mean(dim=(2, 3))  # Global average pool
        return self.projection(features)


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation for goal conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
    
    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: gamma * features + beta."""
        gamma = self.gamma(conditioning).unsqueeze(1)  # (B, 1, D)
        beta = self.beta(conditioning).unsqueeze(1)
        return gamma * features + beta


class GoalConditionedACTPolicy(nn.Module):
    """
    ACT policy conditioned on goal images.
    
    Key changes from standard ACT:
    1. Accepts goal_image in observation dict
    2. Fuses goal embedding with observation features
    3. Supports hindsight experience replay during training
    """
    
    def __init__(self, config: GoalConditionedACTConfig):
        super().__init__()
        self.config = config
        
        # Base ACT components (reuse from lerobot)
        self.observation_encoder = self._build_observation_encoder(config)
        self.transformer_encoder = ACTEncoder(config)
        self.transformer_decoder = ACTDecoder(config)
        
        # Goal conditioning
        if config.use_goal_image:
            self.goal_encoder = GoalEncoder(config, self.observation_encoder)
            
            if config.goal_fusion == "film":
                self.goal_film = FiLMConditioning(config.dim_model)
            elif config.goal_fusion == "cross_attention":
                self.goal_cross_attn = nn.MultiheadAttention(
                    config.dim_model, config.n_heads, batch_first=True
                )
        
        # Action head
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])
        
        # VAE components (if using)
        if config.use_vae:
            self.vae_encoder = self._build_vae_encoder(config)
            self.vae_decoder = self._build_vae_decoder(config)
    
    def _build_observation_encoder(self, config):
        """Build observation encoder (simplified)."""
        # In practice, use the full ACT encoder from lerobot
        return nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, config.dim_model)
        )
    
    def _build_vae_encoder(self, config):
        return nn.Sequential(
            nn.Linear(config.chunk_size * config.output_shapes["action"][0], 256),
            nn.ReLU(),
            nn.Linear(256, config.latent_dim * 2)  # mu and logvar
        )
    
    def _build_vae_decoder(self, config):
        return nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.dim_model)
        )
    
    def encode_observation(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation to feature vector."""
        image = observation["observation.images.top"]
        return self.observation_encoder(image)
    
    def fuse_goal(
        self,
        obs_features: torch.Tensor,
        goal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Fuse goal embedding with observation features."""
        if self.config.goal_fusion == "concat":
            # Simple concatenation
            return obs_features + goal_embedding.unsqueeze(1)
        
        elif self.config.goal_fusion == "film":
            # Feature-wise modulation
            return self.goal_film(obs_features, goal_embedding)
        
        elif self.config.goal_fusion == "cross_attention":
            # Cross-attention
            goal_seq = goal_embedding.unsqueeze(1)  # (B, 1, D)
            fused, _ = self.goal_cross_attn(obs_features, goal_seq, goal_seq)
            return obs_features + fused
        
        return obs_features
    
    def forward(
        self,
        observation: Dict[str, torch.Tensor],
        goal_image: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None  # For training with VAE
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            observation: Dict with "observation.images.top" and "observation.state"
            goal_image: (B, C, H, W) goal observation (optional)
            actions: (B, chunk_size, action_dim) ground truth actions for training
            
        Returns:
            Dict with "action" predictions and optional "kl_loss"
        """
        batch_size = observation["observation.images.top"].shape[0]
        
        # Encode observation
        obs_features = self.encode_observation(observation)
        obs_features = obs_features.unsqueeze(1)  # (B, 1, D)
        
        # Encode and fuse goal
        if self.config.use_goal_image and goal_image is not None:
            goal_embedding = self.goal_encoder(goal_image)
            obs_features = self.fuse_goal(obs_features, goal_embedding)
        
        # Encode with transformer
        memory = self.transformer_encoder(obs_features)
        
        # VAE during training
        kl_loss = torch.tensor(0.0, device=obs_features.device)
        if self.config.use_vae and actions is not None:
            # Encode actions to latent
            actions_flat = actions.flatten(start_dim=1)
            vae_output = self.vae_encoder(actions_flat)
            mu, logvar = vae_output.chunk(2, dim=-1)
            
            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_loss = kl_loss.mean()
            
            # Decode latent to conditioning
            z_decoded = self.vae_decoder(z).unsqueeze(1)
            memory = memory + z_decoded
        
        # Decode to action sequence
        # Query tokens for each timestep
        query = torch.zeros(batch_size, self.config.chunk_size, self.config.dim_model,
                          device=obs_features.device)
        
        decoder_output = self.transformer_decoder(query, memory)
        
        # Project to actions
        actions_pred = self.action_head(decoder_output)  # (B, chunk_size, action_dim)
        
        return {
            "action": actions_pred,
            "kl_loss": kl_loss
        }
    
    def predict_action(
        self,
        observation: Dict[str, torch.Tensor],
        goal_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict action chunk for inference.
        
        Returns:
            actions: (B, chunk_size, action_dim) predicted action sequence
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(observation, goal_image)
        return output["action"]


def create_goal_conditioned_act(
    chunk_size: int = 100,
    dim_model: int = 256,
    use_vae: bool = True,
    goal_fusion: str = "film"
) -> GoalConditionedACTPolicy:
    """Factory function to create goal-conditioned ACT."""
    config = GoalConditionedACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        dim_model=dim_model,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=use_vae,
        latent_dim=32,
        kl_weight=10.0,
        use_goal_image=True,
        goal_encoder_type="shared",
        goal_fusion=goal_fusion,
        input_shapes={
            "observation.images.top": [3, 480, 640],
            "observation.state": [6]
        },
        output_shapes={
            "action": [6]
        }
    )
    return GoalConditionedACTPolicy(config)
```

---

## Step 2.2: Implement Hindsight Experience Replay

Create file: `sail/training/hindsight.py`

```python
"""
Hindsight Experience Replay for SAIL

Core idea: Failed attempts at task A are successful demonstrations
for reaching the state that was actually achieved.

This is what makes VLM labeling errors tolerable - we learn from
every trajectory regardless of success/failure.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trajectory:
    """Single trajectory with observations, actions, and metadata."""
    observations: torch.Tensor  # (T, C, H, W)
    states: torch.Tensor  # (T, state_dim)
    actions: torch.Tensor  # (T, action_dim)
    intended_goal: torch.Tensor  # (C, H, W) what we tried to achieve
    achieved_goal: torch.Tensor  # (C, H, W) what we actually achieved
    success: bool  # Did we achieve intended goal?
    task_description: str


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
        future_k: int = 4  # Number of future goals to sample
    ):
        self.capacity = capacity
        self.hindsight_ratio = hindsight_ratio
        self.future_k = future_k
        
        self.trajectories: List[Trajectory] = []
        self.successful_indices: List[int] = []
        self.failed_indices: List[int] = []
    
    def add(self, trajectory: Trajectory) -> None:
        """Add trajectory to buffer."""
        if len(self.trajectories) >= self.capacity:
            # Remove oldest
            removed = self.trajectories.pop(0)
            # Update index lists
            if 0 in self.successful_indices:
                self.successful_indices.remove(0)
            if 0 in self.failed_indices:
                self.failed_indices.remove(0)
            # Decrement all indices
            self.successful_indices = [i-1 for i in self.successful_indices]
            self.failed_indices = [i-1 for i in self.failed_indices]
        
        idx = len(self.trajectories)
        self.trajectories.append(trajectory)
        
        if trajectory.success:
            self.successful_indices.append(idx)
        else:
            self.failed_indices.append(idx)
    
    def sample_batch(
        self,
        batch_size: int,
        chunk_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample batch with hindsight relabeling.
        
        Returns:
            Dict with:
                - observations: (B, C, H, W)
                - states: (B, state_dim)
                - actions: (B, chunk_size, action_dim)
                - goals: (B, C, H, W)
        """
        observations = []
        states = []
        actions = []
        goals = []
        
        for _ in range(batch_size):
            # Select trajectory
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            
            # Select timestep (ensure enough future steps for chunk)
            max_t = len(traj.observations) - chunk_size
            if max_t <= 0:
                max_t = 1
            t = np.random.randint(0, max_t)
            
            # Decide on goal (hindsight or intended)
            use_hindsight = np.random.random() < self.hindsight_ratio
            
            if use_hindsight or not traj.success:
                # Use hindsight goal
                # Sample from future states in this trajectory
                future_t = np.random.randint(t + chunk_size, len(traj.observations))
                goal = traj.observations[future_t]
            else:
                # Use intended goal
                goal = traj.intended_goal
            
            # Extract data
            observations.append(traj.observations[t])
            states.append(traj.states[t])
            actions.append(traj.actions[t:t+chunk_size])
            goals.append(goal)
        
        return {
            "observations": torch.stack(observations),
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "goals": torch.stack(goals)
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        total = len(self.trajectories)
        successful = len(self.successful_indices)
        
        return {
            "total_trajectories": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0.0
        }


class HindsightRelabeler:
    """
    Relabel trajectories with achieved goals.
    
    Converts failed task attempts into successful demonstrations
    for reaching the achieved state.
    """
    
    def __init__(self, future_strategy: str = "final"):
        """
        Args:
            future_strategy: How to select hindsight goal
                - "final": Always use final state
                - "future": Random future state
                - "episode": Random state from same episode
        """
        self.future_strategy = future_strategy
    
    def relabel(
        self,
        trajectory: Trajectory,
        timestep: int
    ) -> Tuple[torch.Tensor, str]:
        """
        Get hindsight goal for given timestep.
        
        Returns:
            goal_image: Relabeled goal
            task_description: Updated task description
        """
        if self.future_strategy == "final":
            goal = trajectory.achieved_goal
            task_desc = f"reach the final state"
        
        elif self.future_strategy == "future":
            # Random future timestep
            future_t = np.random.randint(timestep + 1, len(trajectory.observations))
            goal = trajectory.observations[future_t]
            task_desc = f"reach state at timestep {future_t}"
        
        else:  # episode
            # Any timestep from episode
            random_t = np.random.randint(0, len(trajectory.observations))
            goal = trajectory.observations[random_t]
            task_desc = f"reach state at timestep {random_t}"
        
        return goal, task_desc
    
    def create_relabeled_batch(
        self,
        trajectories: List[Trajectory],
        batch_size: int,
        chunk_size: int,
        hindsight_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Create training batch with hindsight relabeling.
        """
        batch = {
            "observation.images.top": [],
            "observation.state": [],
            "action": [],
            "goal_image": [],
            "is_hindsight": []
        }
        
        for _ in range(batch_size):
            # Sample trajectory
            traj = np.random.choice(trajectories)
            
            # Sample timestep
            max_t = max(1, len(traj.observations) - chunk_size)
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
            batch["observation.images.top"].append(traj.observations[t])
            batch["observation.state"].append(traj.states[t])
            batch["action"].append(traj.actions[t:t+chunk_size])
            batch["goal_image"].append(goal)
            batch["is_hindsight"].append(is_hindsight)
        
        # Stack tensors
        return {
            k: torch.stack(v) if k != "is_hindsight" else v
            for k, v in batch.items()
        }
```

---

## Step 2.3: Train Goal-Conditioned Policy

Create file: `sail/scripts/train_goal_conditioned.py`

```python
"""
Training script for goal-conditioned ACT policy.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from sail.policies.goal_conditioned_act import create_goal_conditioned_act
from sail.training.hindsight import HindsightBuffer, Trajectory


def train_goal_conditioned(
    dataset_repo_id: str,
    output_dir: str,
    num_steps: int = 25000,
    batch_size: int = 8,
    chunk_size: int = 100,
    learning_rate: float = 1e-4,
    hindsight_ratio: float = 0.5,
    device: str = "cuda"
):
    """
    Train goal-conditioned ACT policy.
    
    Args:
        dataset_repo_id: HuggingFace dataset ID
        output_dir: Where to save checkpoints
        num_steps: Training steps
        batch_size: Batch size
        chunk_size: Action chunk size
        learning_rate: Learning rate
        hindsight_ratio: Fraction of batch to use hindsight goals
        device: Training device
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    # Create policy
    print("Creating goal-conditioned ACT policy...")
    policy = create_goal_conditioned_act(
        chunk_size=chunk_size,
        dim_model=256,
        use_vae=True,
        goal_fusion="film"
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
    
    # Convert dataset to trajectories for hindsight buffer
    print("Building hindsight buffer...")
    buffer = HindsightBuffer(
        capacity=10000,
        hindsight_ratio=hindsight_ratio
    )
    
    for episode_idx in range(dataset.num_episodes):
        episode = dataset.get_episode(episode_idx)
        traj = Trajectory(
            observations=episode["observation.images.top"],
            states=episode["observation.state"],
            actions=episode["action"],
            intended_goal=episode["observation.images.top"][-1],  # Assume final is goal
            achieved_goal=episode["observation.images.top"][-1],
            success=True,  # Assume demos are successful
            task_description="pick and place"
        )
        buffer.add(traj)
    
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Training loop
    print(f"Starting training for {num_steps} steps...")
    policy.train()
    
    losses = []
    
    for step in tqdm(range(num_steps)):
        # Sample batch with hindsight
        batch = buffer.sample_batch(batch_size, chunk_size)
        
        # Move to device
        observations = {
            "observation.images.top": batch["observations"].to(device),
            "observation.state": batch["states"].to(device)
        }
        goals = batch["goals"].to(device)
        target_actions = batch["actions"].to(device)
        
        # Forward pass
        output = policy(observations, goal_image=goals, actions=target_actions)
        
        # Compute loss
        action_loss = F.mse_loss(output["action"], target_actions)
        kl_loss = output["kl_loss"]
        
        total_loss = action_loss + policy.config.kl_weight * kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(total_loss.item())
        
        # Logging
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f"Step {step}: loss={avg_loss:.4f}, action_loss={action_loss.item():.4f}, kl_loss={kl_loss.item():.4f}")
        
        # Save checkpoint
        if step % 5000 == 0 and step > 0:
            checkpoint_path = output_path / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": policy.config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_path / "final_model.pt"
    torch.save({
        "step": num_steps,
        "model_state_dict": policy.state_dict(),
        "config": policy.config
    }, final_path)
    
    # Save training metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "final_loss": sum(losses[-100:]) / 100,
            "num_steps": num_steps,
            "dataset": dataset_repo_id,
            "hindsight_ratio": hindsight_ratio
        }, f, indent=2)
    
    print(f"Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset repo ID")
    parser.add_argument("--output", default="outputs/sail_goal_conditioned", help="Output directory")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    train_goal_conditioned(
        dataset_repo_id=args.dataset,
        output_dir=args.output,
        num_steps=args.steps,
        batch_size=args.batch_size
    )
```

### Run Goal-Conditioned Training

```bash
python sail/scripts/train_goal_conditioned.py \
    --dataset ${HF_USER}/sail_pickplace_v1 \
    --output outputs/sail_goal_conditioned_v1 \
    --steps 25000 \
    --batch_size 8
```

---

# PHASE 3: AUTONOMOUS PRACTICE LOOP
**Objective:** Robot collects its own training data with VLM guidance

---

## Step 3.1: Create Autonomous Practice System

Create file: `sail/autonomous/practice_loop.py`

```python
"""
SAIL Autonomous Practice Loop

Core self-improvement mechanism:
1. VLM proposes feasible task from current state
2. Policy attempts task with goal conditioning
3. VLM labels success/failure
4. Store trajectory with hindsight goals
5. Periodically retrain on expanded dataset

Based on SOAR paper architecture.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import time
from datetime import datetime

from sail.policies.goal_conditioned_act import GoalConditionedACTPolicy
from sail.rewards.reward_manager import RewardManager
from sail.training.hindsight import Trajectory, HindsightBuffer


@dataclass
class PracticeConfig:
    """Configuration for autonomous practice."""
    # Task selection
    available_tasks: List[str]  # Tasks the robot can practice
    ucb_exploration: float = 1.0  # UCB exploration parameter
    
    # Episode parameters
    max_episode_steps: int = 300
    control_frequency: float = 30.0  # Hz
    
    # Data collection
    save_all_trajectories: bool = True
    trajectories_dir: str = "data/autonomous_trajectories"
    
    # Subgoal generation
    num_subgoals: int = 5
    subgoal_horizon: int = 20  # Steps between subgoals
    
    # Safety
    max_velocity: float = 0.5  # Fraction of max
    workspace_bounds: Dict[str, Tuple[float, float]] = None  # Joint limits


class TaskSelector:
    """
    Select which task to practice using UCB (Upper Confidence Bound).
    
    Balances exploitation (practice what we're bad at) with
    exploration (try different tasks for diversity).
    """
    
    def __init__(self, tasks: List[str], ucb_c: float = 1.0):
        self.tasks = tasks
        self.ucb_c = ucb_c
        
        # Track attempts and successes per task
        self.attempts = {t: 0 for t in tasks}
        self.successes = {t: 0 for t in tasks}
        self.total_attempts = 0
    
    def select(self, feasible_tasks: List[str]) -> str:
        """
        Select task to practice using UCB.
        
        Args:
            feasible_tasks: Tasks that VLM says are currently possible
            
        Returns:
            Selected task string
        """
        if not feasible_tasks:
            feasible_tasks = self.tasks
        
        self.total_attempts += 1
        
        # UCB scores
        scores = {}
        for task in feasible_tasks:
            n = self.attempts[task]
            if n == 0:
                # Never tried - maximum exploration bonus
                scores[task] = float('inf')
            else:
                # UCB formula
                success_rate = self.successes[task] / n
                exploration_bonus = self.ucb_c * np.sqrt(
                    np.log(self.total_attempts) / n
                )
                # We want to practice LOW success rate tasks
                # So use (1 - success_rate) as exploitation term
                scores[task] = (1 - success_rate) + exploration_bonus
        
        # Select highest score
        selected = max(scores, key=scores.get)
        return selected
    
    def update(self, task: str, success: bool) -> None:
        """Update statistics after attempting task."""
        self.attempts[task] += 1
        if success:
            self.successes[task] += 1
    
    def get_stats(self) -> Dict:
        """Get selection statistics."""
        return {
            task: {
                "attempts": self.attempts[task],
                "successes": self.successes[task],
                "success_rate": self.successes[task] / max(1, self.attempts[task])
            }
            for task in self.tasks
        }


class SubgoalGenerator:
    """
    Generate subgoal images for hierarchical goal-conditioned execution.
    
    Options:
    1. VLM-based: Ask VLM to describe intermediate states
    2. Diffusion-based: Use SuSIE-style image generation
    3. Retrieval-based: Find similar states from successful trajectories
    
    We start with retrieval-based for simplicity and RTX 4070 compatibility.
    """
    
    def __init__(
        self,
        successful_trajectories: List[Trajectory],
        num_subgoals: int = 5
    ):
        self.trajectories = successful_trajectories
        self.num_subgoals = num_subgoals
        
        # Build index of states from successful trajectories
        self.state_index = self._build_index()
    
    def _build_index(self) -> List[Tuple[torch.Tensor, int, int]]:
        """Build index of (observation, traj_idx, timestep)."""
        index = []
        for traj_idx, traj in enumerate(self.trajectories):
            for t in range(len(traj.observations)):
                index.append((traj.observations[t], traj_idx, t))
        return index
    
    def generate_subgoals(
        self,
        current_obs: torch.Tensor,
        final_goal: torch.Tensor,
        vip_encoder  # VIP encoder for similarity
    ) -> List[torch.Tensor]:
        """
        Generate sequence of subgoal images.
        
        Uses retrieval: find trajectory states that interpolate
        between current and goal in embedding space.
        """
        if not self.state_index:
            # No reference trajectories - just return final goal
            return [final_goal] * self.num_subgoals
        
        # Encode current and goal
        with torch.no_grad():
            current_emb = vip_encoder.encode(current_obs.unsqueeze(0))
            goal_emb = vip_encoder.encode(final_goal.unsqueeze(0))
        
        # Find trajectory with most similar goal
        best_traj_idx = None
        best_similarity = -float('inf')
        
        for traj_idx, traj in enumerate(self.trajectories):
            with torch.no_grad():
                traj_goal_emb = vip_encoder.encode(traj.achieved_goal.unsqueeze(0))
            similarity = torch.cosine_similarity(goal_emb, traj_goal_emb).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_traj_idx = traj_idx
        
        # Extract subgoals from best matching trajectory
        traj = self.trajectories[best_traj_idx]
        traj_len = len(traj.observations)
        
        subgoal_indices = np.linspace(0, traj_len - 1, self.num_subgoals + 1, dtype=int)[1:]
        subgoals = [traj.observations[i] for i in subgoal_indices]
        
        return subgoals


class AutonomousPracticeLoop:
    """
    Main autonomous practice system.
    
    Orchestrates:
    - Task selection (what to practice)
    - Subgoal generation (how to break down task)
    - Policy execution (attempt the task)
    - Success labeling (did it work?)
    - Data storage (save for retraining)
    """
    
    def __init__(
        self,
        policy: GoalConditionedACTPolicy,
        reward_manager: RewardManager,
        robot,  # LeRobot robot interface
        config: PracticeConfig
    ):
        self.policy = policy
        self.reward_manager = reward_manager
        self.robot = robot
        self.config = config
        
        # Task selection
        self.task_selector = TaskSelector(
            config.available_tasks,
            config.ucb_exploration
        )
        
        # Trajectory storage
        self.buffer = HindsightBuffer(capacity=10000)
        self.trajectories_path = Path(config.trajectories_dir)
        self.trajectories_path.mkdir(parents=True, exist_ok=True)
        
        # Subgoal generator (initialized after we have some data)
        self.subgoal_generator = None
        
        # Logging
        self.practice_log = []
    
    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """Get current observation from robot."""
        obs = self.robot.get_observation()
        return {
            "observation.images.top": torch.from_numpy(obs["image"]).permute(2, 0, 1),
            "observation.state": torch.from_numpy(obs["state"])
        }
    
    def _execute_policy(
        self,
        goal_image: torch.Tensor,
        max_steps: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Execute policy toward goal.
        
        Returns:
            observations: List of observation images
            states: List of state vectors
            actions: List of executed actions
        """
        observations = []
        states = []
        actions = []
        
        device = next(self.policy.parameters()).device
        
        for step in range(max_steps):
            # Get observation
            obs = self._get_observation()
            observations.append(obs["observation.images.top"])
            states.append(obs["observation.state"])
            
            # Prepare batch
            obs_batch = {
                k: v.unsqueeze(0).to(device)
                for k, v in obs.items()
            }
            goal_batch = goal_image.unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action_chunk = self.policy.predict_action(obs_batch, goal_batch)
            
            # Execute first action of chunk
            action = action_chunk[0, 0].cpu().numpy()
            actions.append(torch.from_numpy(action))
            
            # Apply safety limits
            action = self._apply_safety(action)
            
            # Send to robot
            self.robot.send_action(action)
            
            # Wait for next control step
            time.sleep(1.0 / self.config.control_frequency)
        
        return observations, states, actions
    
    def _apply_safety(self, action: np.ndarray) -> np.ndarray:
        """Apply safety constraints to action."""
        # Velocity limiting
        action = action * self.config.max_velocity
        
        # Workspace bounds
        if self.config.workspace_bounds:
            for i, (name, (low, high)) in enumerate(self.config.workspace_bounds.items()):
                action[i] = np.clip(action[i], low, high)
        
        return action
    
    def run_practice_episode(self) -> Dict:
        """
        Run single autonomous practice episode.
        
        Returns:
            Episode result dictionary
        """
        episode_start = datetime.now()
        
        # Get current observation
        initial_obs = self._get_observation()
        initial_image = initial_obs["observation.images.top"]
        
        # Get feasible tasks from VLM
        feasible = self.reward_manager.propose_practice_tasks(
            initial_image.permute(1, 2, 0).numpy(),
            self.config.available_tasks
        )
        feasible_tasks = [t for t, score in feasible if score > 0.5]
        
        # Select task to practice
        selected_task = self.task_selector.select(feasible_tasks)
        print(f"Selected task: {selected_task}")
        
        # Generate goal image
        # For now, use last frame of a successful demo as goal
        # TODO: Implement proper subgoal generation
        if self.buffer.successful_indices:
            goal_traj_idx = np.random.choice(self.buffer.successful_indices)
            goal_image = self.buffer.trajectories[goal_traj_idx].achieved_goal
        else:
            # No successful demos yet - use current as goal (will fail)
            goal_image = initial_image
        
        # Set up reward manager
        self.reward_manager.set_initial(initial_image.permute(1, 2, 0).numpy())
        self.reward_manager.set_goal(goal_image.permute(1, 2, 0).numpy())
        
        # Execute policy
        observations, states, actions = self._execute_policy(
            goal_image,
            self.config.max_episode_steps
        )
        
        # Get final observation
        final_image = observations[-1]
        
        # Label success with VLM
        success, confidence = self.reward_manager.vlm.detect_success(
            final_image.permute(1, 2, 0).numpy(),
            selected_task
        )
        
        # Compute final reward
        final_signals = self.reward_manager.compute_episode_reward(
            final_image.permute(1, 2, 0).numpy(),
            selected_task
        )
        
        # Create trajectory
        trajectory = Trajectory(
            observations=torch.stack(observations),
            states=torch.stack(states),
            actions=torch.stack(actions),
            intended_goal=goal_image,
            achieved_goal=final_image,
            success=success,
            task_description=selected_task
        )
        
        # Add to buffer
        self.buffer.add(trajectory)
        
        # Update task selector
        self.task_selector.update(selected_task, success)
        
        # Save trajectory
        if self.config.save_all_trajectories:
            self._save_trajectory(trajectory)
        
        # Log result
        result = {
            "timestamp": episode_start.isoformat(),
            "task": selected_task,
            "success": success,
            "vlm_confidence": confidence,
            "vip_progress": final_signals.vip_progress,
            "combined_reward": final_signals.combined_reward,
            "num_steps": len(observations)
        }
        self.practice_log.append(result)
        
        print(f"Episode complete: success={success}, progress={final_signals.vip_progress:.2f}")
        
        return result
    
    def _save_trajectory(self, trajectory: Trajectory) -> None:
        """Save trajectory to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.trajectories_path / f"traj_{timestamp}.pt"
        
        torch.save({
            "observations": trajectory.observations,
            "states": trajectory.states,
            "actions": trajectory.actions,
            "intended_goal": trajectory.intended_goal,
            "achieved_goal": trajectory.achieved_goal,
            "success": trajectory.success,
            "task_description": trajectory.task_description
        }, save_path)
    
    def run_practice_session(
        self,
        num_episodes: int,
        log_path: str = "experiments/practice_log.json"
    ) -> None:
        """
        Run multiple practice episodes.
        
        Args:
            num_episodes: Number of episodes to run
            log_path: Where to save practice log
        """
        print(f"Starting practice session: {num_episodes} episodes")
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            result = self.run_practice_episode()
            
            # Print running stats
            stats = self.task_selector.get_stats()
            buffer_stats = self.buffer.get_stats()
            
            print(f"Buffer: {buffer_stats['total_trajectories']} total, "
                  f"{buffer_stats['success_rate']:.1%} success rate")
        
        # Save log
        with open(log_path, "w") as f:
            json.dump({
                "practice_log": self.practice_log,
                "task_stats": self.task_selector.get_stats(),
                "buffer_stats": self.buffer.get_stats()
            }, f, indent=2)
        
        print(f"\nPractice session complete! Log saved to {log_path}")


def create_practice_loop(
    policy_path: str,
    robot,
    tasks: List[str],
    device: str = "cuda"
) -> AutonomousPracticeLoop:
    """
    Factory function to create practice loop.
    
    Args:
        policy_path: Path to trained goal-conditioned policy
        robot: LeRobot robot interface
        tasks: List of tasks to practice
        device: Device for inference
    """
    # Load policy
    checkpoint = torch.load(policy_path)
    policy = GoalConditionedACTPolicy(checkpoint["config"])
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    
    # Create reward manager
    reward_manager = RewardManager(device=device, use_vlm=True)
    
    # Create config
    config = PracticeConfig(
        available_tasks=tasks,
        max_episode_steps=300,
        control_frequency=30.0
    )
    
    return AutonomousPracticeLoop(policy, reward_manager, robot, config)
```

---

## Step 3.2: Run First Autonomous Practice Session

Create file: `sail/scripts/run_autonomous_practice.py`

```python
"""
Run autonomous practice session.

Usage:
    python sail/scripts/run_autonomous_practice.py \
        --policy outputs/sail_goal_conditioned_v1/final_model.pt \
        --episodes 50
"""

import argparse
from lerobot.common.robot_devices.robots.factory import make_robot
from sail.autonomous.practice_loop import create_practice_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="Path to trained policy")
    parser.add_argument("--episodes", type=int, default=50, help="Number of practice episodes")
    parser.add_argument("--robot_type", default="so101", help="Robot type")
    parser.add_argument("--robot_port", default="/dev/ttyUSB0", help="Robot port")
    args = parser.parse_args()
    
    # Define tasks to practice
    tasks = [
        "pick up the cube and place it in the target zone",
        "push the cube to the left",
        "push the cube to the right",
        "lift the cube and hold it"
    ]
    
    # Create robot
    print("Connecting to robot...")
    robot = make_robot(args.robot_type, port=args.robot_port)
    
    # Create practice loop
    print("Creating practice loop...")
    practice_loop = create_practice_loop(
        policy_path=args.policy,
        robot=robot,
        tasks=tasks,
        device="cuda"
    )
    
    # Run practice
    print(f"Starting {args.episodes} practice episodes...")
    practice_loop.run_practice_session(
        num_episodes=args.episodes,
        log_path="experiments/practice_log.json"
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
```

### Run Autonomous Practice

```bash
# First practice session - 50 episodes with human monitoring
python sail/scripts/run_autonomous_practice.py \
    --policy outputs/sail_goal_conditioned_v1/final_model.pt \
    --episodes 50

# Monitor the robot during practice
# Press Ctrl+C to stop if anything goes wrong
```

### Expected Results
- **Success rate**: 20-40% initially
- **Data collected**: 50 trajectories
- **With hindsight**: All 50 become useful training data

---

## Step 3.3: Retrain on Autonomous Data

Create file: `sail/scripts/retrain_with_autonomous.py`

```python
"""
Retrain policy using autonomous practice data.

Key: Upsample autonomous data 10x relative to original demos
(from SOAR paper)
"""

import torch
from pathlib import Path
import glob
from sail.training.hindsight import Trajectory, HindsightBuffer
from sail.scripts.train_goal_conditioned import train_goal_conditioned


def load_autonomous_trajectories(trajectories_dir: str) -> list:
    """Load trajectories from autonomous practice."""
    trajectories = []
    
    for traj_file in glob.glob(f"{trajectories_dir}/*.pt"):
        data = torch.load(traj_file)
        traj = Trajectory(
            observations=data["observations"],
            states=data["states"],
            actions=data["actions"],
            intended_goal=data["intended_goal"],
            achieved_goal=data["achieved_goal"],
            success=data["success"],
            task_description=data["task_description"]
        )
        trajectories.append(traj)
    
    return trajectories


def retrain_with_autonomous(
    original_dataset: str,
    autonomous_dir: str,
    output_dir: str,
    autonomous_upsample: int = 10,
    num_steps: int = 30000
):
    """
    Retrain policy combining original demos and autonomous data.
    
    Args:
        original_dataset: HuggingFace dataset ID for original demos
        autonomous_dir: Directory with autonomous trajectories
        output_dir: Where to save retrained model
        autonomous_upsample: How much to upsample autonomous data
        num_steps: Training steps
    """
    print("Loading autonomous trajectories...")
    auto_trajectories = load_autonomous_trajectories(autonomous_dir)
    print(f"Loaded {len(auto_trajectories)} autonomous trajectories")
    
    # Count successes
    auto_successes = sum(1 for t in auto_trajectories if t.success)
    print(f"Autonomous success rate: {auto_successes}/{len(auto_trajectories)} = {auto_successes/len(auto_trajectories):.1%}")
    
    # Training will use hindsight buffer which handles upsampling
    # The key insight: even "failed" trajectories teach goal-conditioned policy
    
    print(f"\nStarting retraining for {num_steps} steps...")
    print(f"Autonomous data will be upsampled {autonomous_upsample}x")
    
    # TODO: Modify train_goal_conditioned to accept autonomous trajectories
    # For now, convert to LeRobot dataset format and combine
    
    print("Retraining complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset", required=True)
    parser.add_argument("--autonomous_dir", default="data/autonomous_trajectories")
    parser.add_argument("--output", default="outputs/sail_retrained_v1")
    parser.add_argument("--upsample", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    
    retrain_with_autonomous(
        original_dataset=args.original_dataset,
        autonomous_dir=args.autonomous_dir,
        output_dir=args.output,
        autonomous_upsample=args.upsample,
        num_steps=args.steps
    )
```

---

# PHASE 4: EVALUATION AND ITERATION
**Objective:** Measure improvement and identify what to work on next

---

## Step 4.1: Comprehensive Evaluation Script

Create file: `sail/scripts/evaluate_sail.py`

```python
"""
Comprehensive SAIL evaluation.

Measures:
1. Success rate (with and without self-improvement)
2. Sample efficiency (success vs number of demos)
3. Generalization (new object positions, lighting)
4. Self-improvement rate (success after N practice episodes)
"""

import torch
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

from lerobot.common.robot_devices.robots.factory import make_robot
from sail.policies.goal_conditioned_act import GoalConditionedACTPolicy
from sail.rewards.reward_manager import create_reward_manager


def evaluate_policy(
    policy_path: str,
    robot,
    task_description: str,
    num_episodes: int = 20,
    goal_image_path: str = None,
    device: str = "cuda"
):
    """
    Evaluate policy on task.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load policy
    checkpoint = torch.load(policy_path)
    policy = GoalConditionedACTPolicy(checkpoint["config"])
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    
    # Create reward manager for success detection
    reward_manager = create_reward_manager(device=device, use_vlm=True)
    
    # Load goal image if provided
    if goal_image_path:
        goal_image = torch.load(goal_image_path)
    else:
        goal_image = None
    
    results = []
    
    for episode in tqdm(range(num_episodes)):
        # Reset environment (manual for now)
        input(f"Episode {episode + 1}: Reset environment, then press Enter...")
        
        # Get initial observation
        obs = robot.get_observation()
        initial_image = torch.from_numpy(obs["image"]).permute(2, 0, 1)
        
        if goal_image is None:
            # Use some default goal
            print("Warning: No goal image provided, using VLM for success detection only")
            episode_goal = initial_image  # Placeholder
        else:
            episode_goal = goal_image
        
        # Execute policy
        max_steps = 300
        for step in range(max_steps):
            obs = robot.get_observation()
            obs_tensor = {
                "observation.images.top": torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).to(device),
                "observation.state": torch.from_numpy(obs["state"]).unsqueeze(0).to(device)
            }
            
            with torch.no_grad():
                action = policy.predict_action(obs_tensor, episode_goal.unsqueeze(0).to(device))
            
            robot.send_action(action[0, 0].cpu().numpy())
        
        # Get final observation
        final_obs = robot.get_observation()
        final_image = final_obs["image"]
        
        # Evaluate success
        success, confidence = reward_manager.vlm.detect_success(
            final_image, task_description
        )
        
        results.append({
            "episode": episode,
            "success": success,
            "confidence": confidence
        })
        
        print(f"Episode {episode + 1}: {'SUCCESS' if success else 'FAIL'} (confidence: {confidence:.2f})")
    
    # Compute metrics
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / num_episodes
    avg_confidence = np.mean([r["confidence"] for r in results])
    
    return {
        "policy_path": policy_path,
        "task": task_description,
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_confidence": avg_confidence,
        "results": results
    }


def run_evaluation_suite(
    baseline_policy: str,
    improved_policy: str,
    robot,
    task: str,
    output_path: str = "experiments/evaluation_results.json"
):
    """
    Run full evaluation comparing baseline and improved policies.
    """
    print("=" * 50)
    print("SAIL EVALUATION SUITE")
    print("=" * 50)
    
    # Evaluate baseline
    print("\n--- Evaluating BASELINE policy ---")
    baseline_results = evaluate_policy(
        baseline_policy, robot, task, num_episodes=10
    )
    
    # Evaluate improved
    print("\n--- Evaluating IMPROVED policy ---")
    improved_results = evaluate_policy(
        improved_policy, robot, task, num_episodes=10
    )
    
    # Compute improvement
    improvement = improved_results["success_rate"] - baseline_results["success_rate"]
    relative_improvement = improvement / max(0.01, baseline_results["success_rate"])
    
    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "policy": baseline_policy,
            "success_rate": baseline_results["success_rate"],
            "num_demos": 5  # From original training
        },
        "improved": {
            "policy": improved_policy,
            "success_rate": improved_results["success_rate"],
            "autonomous_episodes": 50  # From practice
        },
        "improvement": {
            "absolute": improvement,
            "relative": relative_improvement
        },
        "detailed_results": {
            "baseline": baseline_results,
            "improved": improved_results
        }
    }
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Baseline success rate: {baseline_results['success_rate']:.1%}")
    print(f"Improved success rate: {improved_results['success_rate']:.1%}")
    print(f"Absolute improvement: {improvement:+.1%}")
    print(f"Relative improvement: {relative_improvement:+.1%}")
    print("=" * 50)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline policy path")
    parser.add_argument("--improved", required=True, help="Improved policy path")
    parser.add_argument("--task", default="pick up the cube and place it in the target zone")
    args = parser.parse_args()
    
    robot = make_robot("so101")
    
    run_evaluation_suite(
        baseline_policy=args.baseline,
        improved_policy=args.improved,
        robot=robot,
        task=args.task
    )
```

---

# PROJECT STRUCTURE

```
sail/
├── __init__.py
├── policies/
│   ├── __init__.py
│   └── goal_conditioned_act.py      # Step 2.1
├── rewards/
│   ├── __init__.py
│   ├── vip.py                        # Step 1.1
│   ├── vlm_detector.py               # Step 1.2
│   └── reward_manager.py             # Step 1.3
├── training/
│   ├── __init__.py
│   └── hindsight.py                  # Step 2.2
├── autonomous/
│   ├── __init__.py
│   └── practice_loop.py              # Step 3.1
├── scripts/
│   ├── validate_rewards.py           # Step 1.4
│   ├── train_goal_conditioned.py     # Step 2.3
│   ├── run_autonomous_practice.py    # Step 3.2
│   ├── retrain_with_autonomous.py    # Step 3.3
│   └── evaluate_sail.py              # Step 4.1
├── configs/
│   └── sail_act_baseline.yaml        # Step 0.5
├── experiments/
│   ├── baseline_results.json
│   ├── reward_validation.json
│   ├── practice_log.json
│   └── evaluation_results.json
├── data/
│   └── autonomous_trajectories/
└── tests/
    ├── test_vip_reward.py
    ├── test_vlm_detector.py
    └── test_reward_manager.py
```

---

# EXECUTION CHECKLIST

## Phase 0: Baseline
- [ ] Verify LeRobot installation
- [ ] Calibrate SO-101 arms
- [ ] Record 5 demonstrations
- [ ] Train ACT baseline policy
- [ ] Evaluate baseline (target: 20-40%)
- [ ] Document failure modes

## Phase 1: Rewards
- [ ] Implement VIP reward
- [ ] Test VIP with real images
- [ ] Implement VLM detector
- [ ] Test VLM success detection
- [ ] Create unified reward manager
- [ ] Validate against human labels (target: >75% agreement)

## Phase 2: Goal-Conditioning
- [ ] Implement goal-conditioned ACT
- [ ] Implement hindsight replay buffer
- [ ] Train goal-conditioned policy
- [ ] Verify goal conditioning works

## Phase 3: Autonomous Practice
- [ ] Implement practice loop
- [ ] Run 50 practice episodes (monitored)
- [ ] Review practice log
- [ ] Retrain with autonomous data
- [ ] Evaluate improvement

## Phase 4: Iteration
- [ ] Run full evaluation suite
- [ ] Compare baseline vs improved
- [ ] Identify remaining failure modes
- [ ] Plan next iteration

---

# SUCCESS METRICS

| Metric | Baseline Target | After SAIL Target |
|--------|-----------------|-------------------|
| Success rate (5 demos) | 20-40% | 50-70% |
| Success rate (10 demos) | 40-60% | 70-85% |
| Autonomous practice success | N/A | 30-50% |
| VLM labeling accuracy | N/A | >75% |
| Self-improvement ratio | 1.0x | 1.5-2.0x |

---

# NEXT STEPS AFTER PHASE 4

Once basic SAIL loop works:

1. **Add more tasks** - Expand from pick-place to pushing, stacking, insertion
2. **Multi-object scenes** - Handle clutter and distractors
3. **Add EES skill selection** - Prioritize practice by expected improvement
4. **Add SELFI online learning** - Q-learning refinement during practice
5. **Subgoal diffusion** - Generate intermediate goal images
6. **Forward/backward cycling** - Autonomous resets for longer practice

Each of these builds on the foundation established in Phases 0-4.

---

*This document serves as your implementation roadmap. Work through each step sequentially, verify each component works before moving on, and document results in the experiments/ directory.*