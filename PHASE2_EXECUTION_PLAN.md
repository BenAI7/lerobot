# Phase 2: Goal-Conditioned Policy

## Objective
Extend the baseline ACT policy to accept goal images as conditioning, enabling:
- **Hindsight Experience Replay (HER)**: Learn from failures by relabeling achieved states as goals
- **Multi-task learning**: Same policy can work toward different goals
- **Autonomous practice**: VLM can propose goals for self-improvement

---

## What Was Implemented

### 1. Goal-Conditioned ACT Policy (`sail/policies/goal_conditioned_act.py`)

A modified ACT architecture that:
- Accepts goal images in addition to current observations
- Uses **FiLM (Feature-wise Linear Modulation)** to condition the policy on goals
- Maintains compatibility with the original ACT architecture
- Supports VAE for action distribution (matching your baseline)

**Key Components:**
- `GoalEncoder`: Encodes goal images to conditioning vectors
- `FiLMConditioning`: Modulates features based on goal embedding
- `GoalConditionedACTPolicy`: Full policy with goal conditioning

### 2. Hindsight Experience Replay (`sail/training/hindsight.py`)

Core data augmentation for self-improvement:
- `Trajectory`: Data structure for storing episodes
- `HindsightBuffer`: Replay buffer with automatic hindsight relabeling
- `HindsightRelabeler`: Converts failed attempts into successful demonstrations
- `GoalConditionedDataset`: PyTorch Dataset wrapper with hindsight

**How HER Works:**
1. Robot attempts task A but fails
2. Instead of discarding, we ask "what DID the robot achieve?"
3. Relabel the trajectory: goal = final achieved state
4. Now it's a successful demonstration for reaching that state
5. Policy learns from both successes AND failures

### 3. Training Script (`sail/scripts/train_goal_conditioned.py`)

Complete training pipeline:
- Loads your baseline dataset
- Converts to trajectories for HER
- Trains with hindsight relabeling
- Saves checkpoints and metrics
- Uses mixed precision for efficiency on RTX 4070

---

## Architecture Details

### Goal Fusion: FiLM

The policy uses FiLM (Feature-wise Linear Modulation) to condition on goals:

```
output = γ(goal) * features + β(goal)
```

Where γ and β are learned from the goal embedding. This is more expressive than simple concatenation and more efficient than cross-attention.

### Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Step                             │
├─────────────────────────────────────────────────────────────┤
│  1. Sample batch from HindsightBuffer                        │
│     - 50% use intended goals (final state)                   │
│     - 50% use hindsight goals (random future state)          │
│                                                              │
│  2. Forward pass with goal conditioning                      │
│     - Encode images with ResNet backbone                     │
│     - Encode goal with shared backbone                       │
│     - Apply FiLM conditioning                                │
│     - Decode to action chunk with transformer                │
│                                                              │
│  3. Compute loss                                             │
│     - L1 loss on actions                                     │
│     - KL divergence for VAE (if enabled)                     │
│                                                              │
│  4. Update policy                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Created

```
sail/
├── policies/
│   ├── __init__.py                    # Package init
│   └── goal_conditioned_act.py        # Goal-conditioned policy
├── training/
│   ├── __init__.py                    # Package init
│   └── hindsight.py                   # HER implementation
├── scripts/
│   └── train_goal_conditioned.py      # Training script
└── tests/
    └── test_goal_policy.py            # Policy tests
```

---

## Configuration

### Policy Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 100 | Action chunk size (matching baseline) |
| `dim_model` | 512 | Transformer hidden dimension |
| `n_encoder_layers` | 4 | Transformer encoder layers |
| `n_decoder_layers` | 1 | Transformer decoder layers |
| `use_vae` | True | Use VAE for action distribution |
| `latent_dim` | 32 | VAE latent dimension |
| `goal_fusion` | "film" | Goal conditioning method |
| `goal_camera` | "station" | Camera for goal images |
| `hindsight_ratio` | 0.5 | Fraction of batch for HER |

### Training Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 25000 | Training steps |
| `batch_size` | 8 | Batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `use_amp` | True | Mixed precision training |

---

## Memory Usage

Estimated VRAM on RTX 4070 (12GB):

| Component | VRAM |
|-----------|------|
| Policy (FP16) | ~2 GB |
| Images (batch=8) | ~3 GB |
| Optimizer states | ~2 GB |
| Activations | ~3 GB |
| **Total** | **~10 GB** |

Mixed precision (AMP) keeps us within 12GB budget.

---

## Expected Results

After training:
- Policy can reach arbitrary goal states (not just trained task)
- HER augmentation effectively 2x the training data
- Same or better performance on original task
- Foundation for autonomous practice (Phase 3)

---

## Next Steps (Phase 3)

Once goal-conditioned policy is trained:

1. **Autonomous Practice Loop**
   - VLM proposes goals based on current state
   - Policy attempts to reach goals
   - VLM labels success/failure
   - Add trajectories to buffer with HER
   - Periodically retrain

2. **Curriculum Learning**
   - Start with easy goals (small movements)
   - Gradually increase difficulty
   - Focus on failure modes

3. **Online Fine-tuning**
   - Continuous improvement during deployment
   - Adapt to workspace changes
   - Learn from corrections

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python sail/scripts/train_goal_conditioned.py --batch_size 4

# Disable AMP (uses more memory but can help stability)
python sail/scripts/train_goal_conditioned.py --no_amp
```

### Training Instability

```bash
# Lower learning rate
python sail/scripts/train_goal_conditioned.py --lr 5e-5

# Reduce hindsight ratio
python sail/scripts/train_goal_conditioned.py --hindsight_ratio 0.3
```

### Slow Training

```bash
# Use fewer steps for testing
python sail/scripts/train_goal_conditioned.py --steps 5000
```

---

## References

- **HER**: Hindsight Experience Replay (Andrychowicz et al., 2017)
- **FiLM**: Feature-wise Linear Modulation (Perez et al., 2018)
- **ACT**: Action Chunking Transformer (Zhao et al., 2023)
- **SOAR**: Self-improvement through VLM feedback (2024)

