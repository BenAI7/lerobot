# Phase 2: Local Commands (No HuggingFace)

**All data stays on your PC. No online HuggingFace needed.**

---

## Your Local Dataset Path

```
C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2
```

---

## Step 1: Test Goal-Conditioned Policy (Again with local data)

```powershell
cd C:\Users\Yeyian\lerobot
conda activate lerobot
python sail/tests/test_goal_policy.py
```

This time Test 6 should also pass since it will load from local path.

---

## Step 2: Train Goal-Conditioned Policy (Local Dataset)

### Full Training (Recommended - 25K steps, ~2-4 hours)

```powershell
python sail/scripts/train_goal_conditioned.py --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2"
```

### Quick Test (5K steps, ~30 minutes)

```powershell
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --steps 5000
```

### Custom Output Location

```powershell
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --output "C:\Users\Yeyian\outputs\sail_phase2\goal_conditioned_v1"
```

---

## Training Progress

You'll see output like:

```
============================================================
SAIL Phase 2: Goal-Conditioned ACT Training
============================================================

[1/5] Loading dataset...
  Loading from local path: C:\Users\Yeyian\outputs\...
✓ Dataset loaded: 50 episodes
  State dim: 6
  Action dim: 6
  Cameras: ['gripper', 'station']

[2/5] Creating goal-conditioned ACT policy...
✓ Policy created: 52,646,790 parameters
  Goal fusion: film
  Using VAE: True

[3/5] Building hindsight buffer...
✓ Buffer ready: 50 trajectories
  Hindsight ratio: 0.5

[4/5] Starting training for 25000 steps...
------------------------------------------------------------
Training:   0%|          | 0/25000 [00:00<?, ?it/s]
Step 100: loss=0.0234, l1=0.0189, kl=0.0045, lr=1.00e-04
Step 200: loss=0.0198, l1=0.0156, kl=0.0042, lr=9.99e-05
...
✓ Saved checkpoint: checkpoint_005000.pt
```

---

## Output Files

After training, check:

```powershell
dir outputs\sail_goal_conditioned
```

You'll see:
- `config.json` - Training configuration
- `checkpoint_005000.pt` - Checkpoint at 5K steps
- `checkpoint_010000.pt` - Checkpoint at 10K steps
- `checkpoint_015000.pt` - Checkpoint at 15K steps
- `checkpoint_020000.pt` - Checkpoint at 20K steps
- `checkpoint_025000.pt` - Checkpoint at 25K steps (if full training)
- `best_model.pt` - Best model by loss
- `final_model.pt` - Final trained model
- `training_metrics.json` - Training summary

---

## All Training Options

```powershell
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --output outputs\sail_goal_conditioned ^
    --steps 25000 ^
    --batch_size 8 ^
    --chunk_size 100 ^
    --lr 1e-4 ^
    --hindsight_ratio 0.5 ^
    --goal_fusion film ^
    --goal_camera station
```

### If You Get Out of Memory:

```powershell
# Reduce batch size to 4
python sail/scripts/train_goal_conditioned.py ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --batch_size 4
```

---

## What This Training Does

1. **Loads your 50 episodes** from local dataset
2. **Converts to trajectories** for Hindsight Experience Replay
3. **Trains with goal conditioning:**
   - 50% samples use intended goals (final frame)
   - 50% samples use hindsight goals (random future frame)
4. **Learns to reach arbitrary goals** (not just trained task)
5. **Foundation for Phase 3** autonomous practice

---

## After Training

Once complete, you'll have a goal-conditioned policy that can:
- Reach any goal state you show it
- Work with hindsight relabeling
- Ready for autonomous self-improvement (Phase 3)

---

## Time Estimates (RTX 4070)

| Steps | Time | Use Case |
|-------|------|----------|
| 5,000 | ~30 min | Quick test |
| 10,000 | ~1 hour | Short training |
| 25,000 | ~2-4 hours | Full training (recommended) |

---

## Notes

- All data stays local
- No internet required after initial package downloads
- Dataset path is long - use quotes in command line
- Training can be interrupted and resumed from checkpoints
- Best model is saved automatically based on loss

