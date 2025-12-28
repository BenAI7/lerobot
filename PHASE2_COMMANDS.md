# Phase 2: Commands Reference

Quick reference for all Phase 2 commands.

---

## Prerequisites

Make sure you're in the correct environment:

```powershell
cd C:\Users\Yeyian\lerobot
conda activate lerobot
```

---

## Step 1: Test Goal-Conditioned Policy

First, verify the policy architecture works:

```powershell
python sail/tests/test_goal_policy.py
```

**Expected output:**
- ✓ Policy created successfully
- ✓ Forward pass successful
- ✓ Loss computation successful
- ✓ All Tests Passed!

---

## Step 2: Train Goal-Conditioned Policy

**Note:** Use your **local dataset path** (no HuggingFace needed):

### Default Training (Recommended)

```powershell
python sail/scripts/train_goal_conditioned.py --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2"
```

This will:
- Train for 25,000 steps
- Use batch size 8
- Use FiLM goal conditioning
- Use 50% hindsight relabeling
- Save to `outputs/sail_goal_conditioned/`

### Custom Training Options

```powershell
# Shorter training for testing
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --steps 5000

# Custom output directory
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --output outputs/my_experiment

# Lower batch size (if OOM)
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --batch_size 4

# Different goal fusion method
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --goal_fusion cross_attention

# All options
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --output outputs/sail_goal_conditioned ^
    --steps 25000 ^
    --batch_size 8 ^
    --chunk_size 100 ^
    --lr 1e-4 ^
    --hindsight_ratio 0.5 ^
    --goal_fusion film ^
    --goal_camera station
```

---

## Step 3: Monitor Training

Training will print progress:
```
Step 100: loss=0.0234, l1=0.0189, kl=0.0045, lr=1.00e-04
Step 200: loss=0.0198, l1=0.0156, kl=0.0042, lr=9.99e-05
...
```

Checkpoints saved every 5000 steps to output directory.

---

## Step 4: Evaluate Trained Policy

After training completes:

```powershell
# Check the output directory
dir outputs\sail_goal_conditioned

# Expected files:
# - config.json
# - checkpoint_005000.pt
# - checkpoint_010000.pt
# - ...
# - best_model.pt
# - final_model.pt
# - training_metrics.json
```

---

## Training Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | Required | Local path to dataset OR HuggingFace dataset ID |
| `--output` | `outputs/sail_goal_conditioned` | Output directory |
| `--steps` | 25000 | Training steps |
| `--batch_size` | 8 | Batch size |
| `--chunk_size` | 100 | Action chunk size |
| `--lr` | 1e-4 | Learning rate |
| `--hindsight_ratio` | 0.5 | HER relabeling fraction |
| `--goal_fusion` | film | Goal conditioning (film/concat/cross_attention) |
| `--goal_camera` | station | Camera for goals |
| `--no_amp` | False | Disable mixed precision |

---

## Troubleshooting

### CUDA Out of Memory

```powershell
# Reduce batch size
python sail/scripts/train_goal_conditioned.py --batch_size 4

# Or even smaller
python sail/scripts/train_goal_conditioned.py --batch_size 2
```

### Module Not Found

```powershell
# Make sure you're in the right directory
cd C:\Users\Yeyian\lerobot

# And environment is activated
conda activate lerobot
```

### Dataset Not Found

```powershell
# Check HuggingFace login
huggingface-cli whoami

# Login if needed
huggingface-cli login
```

---

## File Locations

| File | Path |
|------|------|
| Policy Implementation | `sail/policies/goal_conditioned_act.py` |
| HER Implementation | `sail/training/hindsight.py` |
| Training Script | `sail/scripts/train_goal_conditioned.py` |
| Policy Test | `sail/tests/test_goal_policy.py` |
| Documentation | `PHASE2_EXECUTION_PLAN.md` |

---

## Next Steps

After training completes successfully:
1. Review `training_metrics.json` for final loss
2. Ready for Phase 3: Autonomous Practice Loop

```powershell
# View training results
type outputs\sail_goal_conditioned\training_metrics.json
```

