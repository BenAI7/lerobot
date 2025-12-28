## Phase 3: Autonomous Practice Loop

**Objective:** Robot practices autonomously and improves through self-generated experience.

---

## What Was Implemented

### Core Components

1. **Autonomous Practice Loop** (`sail/autonomous/practice_loop.py`)
   - Executes episodes with baseline ACT policy
   - VLM judges success after each episode
   - Stores successful trajectories
   - Periodic retraining on expanded dataset

2. **Simulation Mode** (`sail/scripts/practice_simulation.py`)
   - Test practice loop without robot hardware
   - Evaluates existing dataset episodes
   - Validates VLM + VIP evaluation pipeline

3. **Progress Tracking**
   - Episode-by-episode results
   - Success rate monitoring
   - VLM and VIP metrics

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│              Autonomous Practice Loop                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Execute Episode                                      │
│     - Baseline ACT policy controls robot                 │
│     - Record observations + actions                      │
│                                                          │
│  2. Evaluate Success                                     │
│     - VLM: "Did the task succeed?" (yes/no)             │
│     - VIP: Progress score (0-1)                          │
│                                                          │
│  3. Store if Successful                                  │
│     - Add trajectory to dataset                          │
│     - Increment success counter                          │
│                                                          │
│  4. Periodic Retraining                                  │
│     - Every N successful episodes                        │
│     - Retrain baseline ACT on expanded dataset           │
│     - Load improved policy                               │
│                                                          │
│  5. Repeat → Robot Gets Better                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Key Differences from Original SAIL Plan

**What Changed:**
- ❌ No goal-conditioned policy (didn't work for single-task)
- ❌ No hindsight relabeling (not needed without goal conditioning)
- ✅ Baseline ACT policy (works well - 60% success)
- ✅ VLM success detection (Phase 1 validated)
- ✅ VIP progress tracking (Phase 1 validated)
- ✅ Periodic retraining instead of continuous

**Why This Works:**
- Single-task dataset → baseline ACT is sufficient
- VLM provides automated feedback → no human labeling needed
- Robot practices → dataset grows → policy improves
- Simpler than goal-conditioned approach → actually implementable

---

## Architecture

### PracticeSession Components

| Component | Purpose | Status |
|-----------|---------|--------|
| Policy | Baseline ACT from Phase 0 | ✅ Working |
| Robot | SO-101 hardware interface | ✅ LeRobot API |
| VLM | Success detection (Qwen2-VL-2B) | ✅ Phase 1 |
| VIP | Progress tracking (ResNet50) | ✅ Phase 1 |
| Dataset | Storage for new trajectories | 📝 Needs integration |
| Retraining | Periodic policy improvement | 📝 Needs implementation |

---

## Phase 3 Status

### ✅ Completed
- Core practice loop structure
- VLM/VIP integration
- Episode execution framework
- Results tracking
- Simulation mode for testing

### 📝 To Complete
1. **Robot Integration**
   - Connect to SO-101 via LeRobot API
   - Test episode execution
   - Verify action sending

2. **Dataset Expansion**
   - Add new episodes to existing dataset
   - Maintain LeRobot dataset format
   - Handle metadata correctly

3. **Retraining Pipeline**
   - Call ACT training script automatically
   - Load new checkpoint after training
   - Track improvement over time

4. **Safety Features**
   - Emergency stop handling
   - Workspace boundary checking
   - Collision detection

---

## Testing Strategy

### Step 1: Simulation (No Robot)
Test the evaluation pipeline:

```powershell
python sail/scripts/practice_simulation.py --dataset <your_dataset> --episodes 10
```

**Validates:**
- VLM can evaluate episodes
- VIP computes progress correctly
- Results tracking works

### Step 2: Single Episode (With Robot)
Execute one episode manually:

```powershell
python -c "from sail.autonomous.practice_loop import PracticeConfig, PracticeSession; ..."
```

**Validates:**
- Robot connection works
- Policy can control robot
- Episode completes successfully

### Step 3: Full Practice Loop
Run autonomous practice:

```powershell
python sail/scripts/run_practice.py --episodes 50 --retrain-freq 10
```

**Expected:**
- Initial success rate: ~60% (baseline)
- After 10 successes + retrain: ~65-70%
- After 30 successes + 3 retrains: ~75%+

---

## Expected Results

| Metric | Baseline | After 10 Episodes | After 50 Episodes |
|--------|----------|-------------------|-------------------|
| Success Rate | 60% | 65-70% | 75-80% |
| Avg VIP Progress | 0.8 | 0.85 | 0.9+ |
| Dataset Size | 50 eps | 55-60 eps | 75-80 eps |
| Training Steps | 40K | 50K | 80K |

**Key Insight:** Improvement comes from:
1. More diverse data (robot's own attempts)
2. Filtering (only successful episodes added)
3. Continuous learning (periodic retraining)

---

## Configuration

### PracticeConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `policy_path` | Required | Path to baseline ACT checkpoint |
| `robot_type` | "so101_follower" | Robot type |
| `task_description` | Required | Task for VLM evaluation |
| `num_practice_episodes` | 50 | Total practice attempts |
| `retrain_frequency` | 10 | Retrain every N successes |
| `episode_length` | 300 | Max frames (10s at 30fps) |
| `fps` | 30 | Control frequency |
| `success_threshold` | 0.8 | VIP progress threshold |
| `output_dir` | "outputs/sail_practice" | Save location |

---

## Safety Considerations

### Pre-Flight Checklist
- ✅ Robot calibrated
- ✅ Workspace clear of obstacles
- ✅ Emergency stop accessible
- ✅ Camera views unobstructed
- ✅ Baseline policy tested manually

### During Practice
- Monitor first 5 episodes closely
- Check for drift/degradation
- Verify VLM judgments are reasonable
- Stop if success rate drops below 40%

### Data Quality
- Review saved trajectories periodically
- Remove any corrupted episodes
- Ensure VLM isn't mis-labeling failures as successes

---

## Troubleshooting

### Robot Not Connecting
```powershell
# Find robot port
python -m lerobot.scripts.lerobot_find_port

# Test connection
python -m lerobot.scripts.lerobot_teleoperate --robot-type so101_follower
```

### VLM Gives Wrong Labels
- Check lighting conditions
- Verify camera angle captures target zone
- Adjust task description to be more specific
- Use VIP threshold as fallback (success_threshold=0.9)

### Policy Degradation After Retraining
- Too few episodes for retraining (increase retrain_frequency)
- Bad episodes being added (stricter VLM threshold)
- Overfitting (add regularization, reduce training steps)

---

## Next Steps

1. **Test Simulation** - Validate evaluation pipeline
2. **Connect Robot** - Test single episode execution
3. **Run Practice** - Start autonomous loop
4. **Monitor Progress** - Track improvement over time
5. **Iterate** - Adjust parameters based on results

---

## Implementation Notes

### What's Not Implemented Yet

1. **Dataset Expansion**
   - Code structure is ready
   - Needs LeRobot dataset API calls
   - Must handle episode IDs correctly

2. **Automatic Retraining**
   - Framework is in place
   - Needs to call ACT training script
   - Must reload policy after training

3. **Robot Safety**
   - Emergency stop detection works
   - Needs workspace boundaries
   - Could add collision detection

### Why These Are Separate

These require:
- Testing with real robot hardware
- Careful validation of data integrity
- Iterative debugging

**Approach:** Get simulation working → Add robot → Add retraining → Add safety

---

## References

- **SOAR Paper:** Self-supervised practice with online refinement
- **SELFI Paper:** VLM-guided self-improvement
- **LeRobot Docs:** Robot control and dataset management
- **Phase 0:** Baseline ACT policy (60% success)
- **Phase 1:** VLM + VIP evaluation (100% validation)

---

## Summary

Phase 3 provides a **working foundation** for autonomous practice:
- ✅ Core loop structure
- ✅ VLM/VIP integration
- ✅ Results tracking
- ✅ Simulation mode

**Next:** Complete robot integration → Test practice loop → Monitor improvement

