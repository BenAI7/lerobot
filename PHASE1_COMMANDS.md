# Phase 1: Commands to Execute

All code is implemented. Just run these commands in order.

---

## Step 1: Install Dependencies

```powershell
pip install transformers>=4.36.0 timm>=0.9.12 Pillow>=10.0.0 accelerate>=0.25.0
```

---

## Step 2: Verify Dependencies

```powershell
python sail/test_dependencies.py
```

**Expected:** All green checkmarks (✓)

---

## Step 3: Test VIP Reward Model

```powershell
python sail/test_vip.py
```

**Expected:** Progress increases from ~0 → ~1 during episodes

---

## Step 4: Test VLM Success Detector

```powershell
python sail/test_vlm.py
```

**Note:** Downloads ~4GB model on first run. Will take a few minutes.

**Expected:** VLM labels episodes as SUCCESS or FAILURE

---

## Step 5: Test Unified Reward Manager

```powershell
python sail/test_reward_manager.py
```

**Expected:** Both VIP progress and VLM labels working together

---

## Step 6: Run Full Validation

```powershell
python sail/scripts/validate_rewards.py
```

**What it does:**
- Tests VIP + VLM on 20 episodes
- Saves results to `sail/experiments/reward_validation.json`

---

## Step 7: Add Human Labels

**Manual step:** Edit `sail/experiments/reward_validation.json`

For each episode, change:
```json
"human_label": null
```
to:
```json
"human_label": true    // if task succeeded
```
or:
```json
"human_label": false   // if task failed
```

---

## Step 8: Compute Agreement

```powershell
python sail/scripts/validate_rewards.py compute
```

**Expected:** 
- Accuracy >75%
- Recall >90%

---

## Phase 1 Complete Checklist

- [ ] Dependencies installed (`test_dependencies.py` passes)
- [ ] VIP tested (`test_vip.py` shows progress 0→1)
- [ ] VLM tested (`test_vlm.py` gives yes/no answers)
- [ ] Reward manager tested (`test_reward_manager.py` works)
- [ ] Validation run (`validate_rewards.py` creates JSON)
- [ ] Human labels added (edited JSON file)
- [ ] Agreement computed (`validate_rewards.py compute` shows >75%)

---

## If Something Fails

**"Module not found"**
→ Run Step 1 again to install dependencies

**"CUDA out of memory"**  
→ Close other GPU apps, or edit `vlm_detector.py` to use smaller model

**"Dataset not found"**
→ Check the path in the test files matches your dataset location

---

## After Phase 1

When validation passes (>75% accuracy), you're ready for Phase 2!

Tell me the results and I'll create Phase 2 implementation.

