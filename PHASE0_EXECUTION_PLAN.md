# SAIL Phase 0: Baseline Foundation - Windows Execution Plan

**Objective:** Train working ACT policy from 5-10 demonstrations  
**Time Estimate:** 2-3 days  
**Hardware:** SO-101 arms, RTX 4070, Windows 11

---

## Step 0.0: Pre-Flight Verification

Run the verification script to check all requirements:

```powershell
python sail/phase0_verify.py
```

**Expected outputs:**
- ✓ LeRobot installed
- ✓ PyTorch + CUDA (12GB VRAM detected)
- ✓ Feetech Motors SDK
- ✓ Cameras detected
- ✓ COM ports detected  
- ✓ Logged into Hugging Face

**If any checks fail, fix them before proceeding.**

---

## Step 0.1: Find Hardware Connections

### Find COM Ports for Robot Arms

```powershell
lerobot-find-port
```

**Action:** Note the COM ports (e.g., COM3, COM4)
- Identify which is leader arm
- Identify which is follower arm

### Find Camera Indices

```powershell
lerobot-find-cameras
```

**Action:** Note the camera index (e.g., 0, 1) for your workspace camera

---

## Step 0.2: Calibrate SO-101 Arms

### Calibrate Follower Arm (executes policies)

```powershell
lerobot-calibrate `
  --robot.type=so101_follower `
  --robot.port=COM3 `
  --robot.id=follower_arm
```

Replace `COM3` with your actual follower port.

**What happens:**
1. Each joint moves to its limits
2. Min/max positions recorded
3. Calibration saved to: `C:\Users\Yeyian\.cache\huggingface\lerobot\calibration\`

### Calibrate Leader Arm (for teleoperation)

```powershell
lerobot-calibrate `
  --robot.type=so101_leader `
  --robot.port=COM4 `
  --robot.id=leader_arm
```

Replace `COM4` with your actual leader port.

### Test Teleoperation

```powershell
lerobot-teleoperate `
  --robot.type=so101_follower `
  --robot.port=COM3 `
  --robot.id=follower_arm `
  --teleop.type=so101_leader `
  --teleop.port=COM4 `
  --teleop.id=leader_arm
```

**Action:** Move leader arm - follower should mirror smoothly.  
Press Ctrl+C to stop.

---

## Step 0.3: Set Up Workspace & Task

### Physical Setup

1. **Clear workspace** - Remove unnecessary objects
2. **Position camera** - Should see entire workspace
3. **Mark zones** with tape:
   - **Start Zone A**: Where cube begins (left side)
   - **Goal Zone B**: Where cube should be placed (right side)
4. **Place cube** in start zone
5. **Adjust lighting** - Even, no harsh shadows

### Task Definition

**Task:** "Pick up the cube from start zone and place it in goal zone"

**Success Criteria:**
- Cube grasped cleanly
- Lifted off table
- Placed within goal zone
- Gripper releases

**Expected Duration:** 8-12 seconds per demonstration

---

## Step 0.4: Record Demonstrations

### Set Environment Variables

```powershell
$env:HF_USER = "your_huggingface_username"
$env:DATASET_NAME = "sail_pickplace_v1"
```

Replace `your_huggingface_username` with actual username from Step 0.0.

### Record 10 Demonstrations

We'll record 10 demos (not 5) to give more training data for baseline:

```powershell
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM3 `
  --robot.id=follower_arm `
  --robot.cameras="{ top: { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }}" `
  --teleop.type=so101_leader `
  --teleop.port=COM4 `
  --teleop.id=leader_arm `
  --dataset.repo_id="$env:HF_USER/$env:DATASET_NAME" `
  --dataset.num_episodes=10 `
  --dataset.single_task="Pick up the cube from start zone and place it in goal zone" `
  --display_data=true
```

**Replace:**
- `COM3`, `COM4` with your ports
- Camera index `0` with your camera from Step 0.1

### Recording Protocol

**For each of the 10 episodes:**

1. **Reset:** Place cube in start zone A
2. **Ready:** Press Enter when ready to start recording
3. **Execute smoothly:**
   - Approach cube (2-3 sec)
   - Grasp cube (close gripper fully)
   - Lift cube (1-2 sec)
   - Move to goal zone (2-3 sec)
   - Lower cube
   - Release gripper (open fully)
   - Retract arm
4. **Stop:** Episode ends automatically after 20 seconds
5. **Confirm:** Review and confirm if good, or re-record if failed

**Quality Tips:**
- **Consistent approach angle** - Use same path each time
- **Smooth motions** - No jerky movements
- **Full gripper closure** - Ensure solid grasp
- **Complete release** - Fully open gripper at end
- **Similar timing** - Try to match duration across demos

### Verify Recorded Dataset

```powershell
python -c "from lerobot.datasets.factory import make_dataset; ds = make_dataset('$env:HF_USER/$env:DATASET_NAME'); print(f'Episodes: {ds.num_episodes}'); print(f'Total frames: {ds.num_frames}'); print(f'FPS: {ds.fps}')"
```

**Expected:**
- Episodes: 10
- Total frames: ~3000-6000 (depends on FPS and duration)
- FPS: 30

---

## Step 0.5: Train ACT Baseline Policy

### Create Training Configuration

Create file: `sail/configs/baseline_act.yaml`

```yaml
# SAIL Baseline ACT Configuration
# Optimized for SO-101 + RTX 4070

seed: 42

# Dataset
dataset:
  repo_id: ${oc.env:HF_USER}/sail_pickplace_v1
  
# Training
training:
  offline_steps: 50000  # Increased for 10 demos
  batch_size: 8
  eval_freq: 5000
  save_freq: 10000
  log_freq: 100
  save_checkpoint: true
  
# Policy
policy:
  type: act
  chunk_size: 100  # Critical for few-shot learning!
  n_action_steps: 100
  
  # Architecture
  dim_model: 256
  encoder_layers: 4
  decoder_layers: 1
  n_heads: 8
  
  # VAE for action distribution
  use_vae: true
  latent_dim: 32
  kl_weight: 10.0
  
  # Training
  lr: 1e-4
  lr_backbone: 1e-5
  weight_decay: 1e-4
  
  # Mixed precision for RTX 4070
  use_amp: true

# Environment (for evaluation)
env:
  name: so101_pickplace
  fps: 30
  episode_length: 300
  
# Logging
wandb:
  enable: true
  project: sail_baseline
  run_name: act_so101_10demos
```

### Start Training

```powershell
lerobot-train `
  --config sail/configs/baseline_act.yaml `
  --output_dir outputs/sail_baseline_act_v1 `
  --policy.device=cuda `
  --training.eval_freq=5000
```

**Training Time:** ~3-5 hours on RTX 4070 for 50K steps

**Monitor with WandB:**
- Go to https://wandb.ai/
- Watch loss curves decrease
- Check for NaN (if occurs, reduce learning rate)

**What to watch:**
- `loss/total` should decrease steadily
- `loss/l1_loss` (action prediction) should drop
- `loss/kld_loss` (VAE) should stabilize around 2-5

### Training Tips

If training crashes with OOM (out of memory):
```powershell
# Reduce batch size
lerobot-train ... --training.batch_size=4
```

If loss plateaus early:
- Check demonstrations are consistent
- Verify camera sees full workspace
- Consider recording more demos

---

## Step 0.6: Evaluate Baseline Policy

### Run Evaluation on Robot

```powershell
lerobot-eval `
  --robot.type=so101_follower `
  --robot.port=COM3 `
  --robot.id=follower_arm `
  --robot.cameras="{ top: { type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }}" `
  --policy.path=outputs/sail_baseline_act_v1/checkpoints/last/pretrained_model `
  --eval.num_episodes=10 `
  --eval.single_task="Pick up the cube from start zone and place it in goal zone"
```

**For each evaluation episode:**
1. Reset cube to start position
2. Policy executes autonomously
3. Manually label success/failure

### Record Results

Create file: `sail/experiments/baseline_results.json`

```json
{
  "experiment": "sail_baseline_v1",
  "date": "2024-XX-XX",
  "num_demos": 10,
  "policy": "ACT",
  "chunk_size": 100,
  "training_steps": 50000,
  "eval_episodes": 10,
  "results": {
    "successes": 0,
    "failures": 0,
    "success_rate": 0.0,
    "failure_modes": {
      "miss_grasp": 0,
      "drop_during_transport": 0,
      "wrong_placement": 0,
      "collision": 0,
      "timeout": 0
    }
  },
  "notes": ""
}
```

**Manually fill in after evaluation.**

**Expected Baseline Performance:**
- With 10 demos: **40-60% success rate**
- This is your benchmark for measuring SAIL improvements

---

## Step 0.7: Analyze Failures

### Review Failed Episodes

Watch the failed episodes and categorize:

| Failure Mode | Count | Notes |
|--------------|-------|-------|
| Miss grasp | | Gripper closes but misses cube |
| Premature drop | | Drops cube during transport |
| Wrong placement | | Places outside goal zone |
| Collision | | Hits workspace/objects |
| Timeout | | Doesn't complete in time |
| Other | | Describe specific issues |

### Generate Failure Report

Create file: `sail/experiments/baseline_failure_analysis.md`

Document:
1. **Most common failure mode** - This is what we'll target first
2. **Visual observations** - What does the robot do wrong?
3. **Hypothesis** - Why does it fail? (e.g., "poor depth perception")
4. **Potential fixes** - What might help? (e.g., "more demos from different angles")

**This analysis informs Phase 1 reward design!**

---

## Success Criteria for Phase 0

✅ **All checks passed:**
- [ ] Hardware verified and calibrated
- [ ] 10 clean demonstrations recorded
- [ ] Dataset uploaded to Hugging Face
- [ ] ACT policy trained for 50K steps
- [ ] Loss curves look healthy (no NaN, steady decrease)
- [ ] Baseline evaluation completed (10 episodes)
- [ ] Success rate documented: ____% (target: 40-60%)
- [ ] Failure modes analyzed and documented

✅ **Ready for Phase 1 when:**
- Baseline policy runs without crashes
- We have quantitative success rate
- We understand main failure modes

---

## Troubleshooting

### COM Port Not Found
- Check Device Manager → Ports (COM & LPT)
- Ensure USB cable is connected
- Try different USB port
- May need driver: https://github.com/TheRobotStudio/SO-ARM101

### Camera Not Detected
- Check `lerobot-find-cameras` output
- Try different USB port
- Ensure no other app is using camera

### Calibration Fails
- Ensure robot is powered on
- Check torque is enabled
- Move joints manually to check for obstructions
- Clear any saved calibration: Delete `~/.cache/huggingface/lerobot/calibration/<robot_id>/`

### Training OOM
- Reduce batch_size to 4 or 2
- Close other GPU applications
- Reduce chunk_size to 50 (but 100 is much better!)

### Low Success Rate (<20%)
- Record more demos (15-20 total)
- Improve demonstration consistency
- Check camera angle covers full workspace
- Ensure lighting is even

---

## Next Steps

**After Phase 0 completion, we proceed to Phase 1:**
- Implement VIP reward model (zero-shot progress)
- Implement VLM success detector (autonomous labeling)
- Validate reward signals against human labels

**All Phase 1 components build on this baseline.**

---

*Last Updated: Ready for Execution*

