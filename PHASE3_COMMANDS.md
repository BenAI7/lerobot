# Phase 3: Commands Reference

Quick reference for autonomous practice commands.

---

## Prerequisites

```powershell
cd C:\Users\Yeyian\lerobot
conda activate lerobot
```

---

## Step 1: Test Simulation Mode (No Robot Needed)

**Test the evaluation pipeline without hardware:**

```powershell
python sail/scripts/practice_simulation.py --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" --episodes 10
```

**What this does:**
- Loads your dataset episodes
- Runs VLM success detection on each
- Computes VIP progress scores
- Shows success rate

**Expected output:**
```
Episodes evaluated: 10
VLM successes: 8
Success rate: 80.0%
Average VIP progress: 0.95
```

---

## Step 2: Check Your Baseline Policy

**Find your best checkpoint:**

```powershell
dir C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\train_session_01\checkpoints
```

**Your best checkpoint:** `040000` (works really good)

Full path:
```
C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\train_session_01\checkpoints\040000\pretrained_model
```

---

## Step 3: Prepare for Robot Practice

### Check Robot Connection

```powershell
# Find COM port
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"

# Test robot
python -m lerobot.scripts.lerobot_teleoperate --robot so101_follower
```

### Verify Cameras

```powershell
python -m lerobot.scripts.lerobot_find_cameras
```

---

## Step 4: Run Autonomous Practice (Full Implementation Needed)

**Note:** The full autonomous practice requires completing robot integration.

**When ready, the command will be:**

```powershell
python sail/scripts/run_practice.py ^
    --policy "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\train_session_01\checkpoints\040000\pretrained_model" ^
    --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" ^
    --task "pick up the cube and place it in the target zone" ^
    --episodes 50 ^
    --retrain-freq 10 ^
    --output outputs/sail_practice
```

---

## Current Status

| Feature | Status | Command |
|---------|--------|---------|
| Simulation Mode | ✅ Ready | `practice_simulation.py` |
| VLM Evaluation | ✅ Working | Phase 1 |
| VIP Progress | ✅ Working | Phase 1 |
| Episode Execution | 📝 Needs Testing | With robot |
| Dataset Expansion | 📝 Needs Implementation | - |
| Automatic Retraining | 📝 Needs Implementation | - |

---

## What Works Now

### 1. Simulation Test

```powershell
# Test evaluation pipeline
python sail/scripts/practice_simulation.py --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" --episodes 20
```

### 2. View Results

```powershell
# Check simulation results
type outputs\sail_practice_sim\simulation_results.json
```

---

## Next Implementation Steps

### To Complete Phase 3:

1. **Robot Integration Testing**
   - Test policy execution with robot
   - Verify action sending works
   - Validate episode completion

2. **Dataset Expansion**
   - Implement adding new episodes
   - Maintain LeRobot format
   - Test data integrity

3. **Retraining Pipeline**
   - Call training script automatically
   - Reload improved policy
   - Track performance over time

---

## Monitoring Practice

### During Practice Session

```powershell
# Watch output folder
dir outputs\sail_practice\session_* /s

# View latest session
type outputs\sail_practice\session_<ID>\summary.json
```

### Expected Progress

| Episodes | Success Rate | Dataset Size |
|----------|--------------|--------------|
| Baseline | 60% | 50 episodes |
| After 10 | 65% | 56 episodes |
| After 30 | 70% | 68 episodes |
| After 50 | 75%+ | 80 episodes |

---

## Configuration Options

### PracticeConfig (When robot integration is complete)

```python
from sail.autonomous.practice_loop import PracticeConfig, run_autonomous_practice

config = PracticeConfig(
    policy_path="path/to/checkpoint",
    robot_type="so101_follower",
    robot_port="COM3",  # Or None for auto-detect
    task_description="pick up the cube and place it in the target zone",
    num_practice_episodes=50,
    retrain_frequency=10,
    episode_length=300,
    fps=30,
    dataset_path="path/to/dataset",
    success_threshold=0.8,
    output_dir="outputs/sail_practice",
    use_vlm=True,
    use_vip=True,
)

run_autonomous_practice(config)
```

---

## Troubleshooting

### Simulation Issues

```powershell
# If VLM loading fails
pip install transformers>=4.36.0

# If CUDA out of memory
# Close other applications or restart
```

### Robot Issues

```powershell
# Test robot separately
python -m lerobot.scripts.lerobot_teleoperate --robot so101_follower --port COM3

# Calibrate if needed
python -m lerobot.scripts.lerobot_calibrate --robot so101_follower
```

---

## File Locations

| File | Path |
|------|------|
| Practice Loop | `sail/autonomous/practice_loop.py` |
| Simulation Script | `sail/scripts/practice_simulation.py` |
| Phase 3 Docs | `PHASE3_EXECUTION_PLAN.md` |
| Commands Reference | `PHASE3_COMMANDS.md` (this file) |

---

## Summary

**What's Ready:**
- ✅ Simulation mode - test evaluation pipeline
- ✅ VLM + VIP integration
- ✅ Results tracking

**What's Next:**
- 📝 Complete robot integration
- 📝 Implement dataset expansion
- 📝 Add automatic retraining

**First Step:**
```powershell
python sail/scripts/practice_simulation.py --dataset "C:\Users\Yeyian\outputs\self_improve\2025-12-21_16-04-22_790120_self_improve\datasets\yeyian\self_improve_pickplace_v2" --episodes 10
```

This validates the evaluation pipeline is working correctly! 🎯

