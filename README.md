# SAIL: Self-Adapting Imitation Learning

**Goal:** Achieve sub-10 demonstration learning with autonomous self-improvement on SO-101 robot arms.

## Quick Start

### 1. Run Pre-Flight Verification

```powershell
python sail/phase0_verify.py
```

This checks:
- ✓ LeRobot installation
- ✓ PyTorch + CUDA (RTX 4070)
- ✓ Robot motors SDK
- ✓ Cameras
- ✓ COM ports
- ✓ Hugging Face login

### 2. Follow Phase 0 Execution Plan

See `sail/PHASE0_EXECUTION_PLAN.md` for step-by-step instructions.

## Project Structure

```
sail/
├── README.md                        # This file
├── PHASE0_EXECUTION_PLAN.md         # Detailed Phase 0 steps
├── phase0_verify.py                 # Pre-flight verification script
├── __init__.py                      # Package init
├── configs/                         # Training configurations
│   └── baseline_act.yaml           # ACT baseline config
├── experiments/                     # Results and logs
│   ├── baseline_results.json       # Evaluation metrics
│   └── baseline_failure_analysis.md# Failure mode analysis
├── policies/                        # Policy implementations (Phase 2)
├── rewards/                         # Reward models (Phase 1)
├── training/                        # Training utilities (Phase 2)
└── autonomous/                      # Autonomous practice (Phase 3)
```

## Phases Overview

### Phase 0: Baseline Foundation (Current)
- ✅ Verify hardware and software
- ✅ Calibrate SO-101 arms
- ✅ Record 10 demonstrations
- ✅ Train ACT policy (50K steps)
- ✅ Evaluate baseline (target: 40-60% success)

### Phase 1: Reward Infrastructure (Next)
- Implement VIP reward model
- Implement VLM success detector
- Validate reward signals
- Target: >75% labeling accuracy

### Phase 2: Goal-Conditioned Architecture
- Extend ACT to accept goal images
- Implement hindsight experience replay
- Enable learning from failures

### Phase 3: Autonomous Practice
- VLM-guided task selection
- Autonomous data collection
- Self-improvement loop
- Target: 1.5-2× baseline improvement

### Phase 4: Evaluation & Iteration
- Compare baseline vs improved
- Analyze remaining failures
- Plan next improvements

## Hardware Requirements

- **Robot:** SO-101 follower + leader arms
- **GPU:** RTX 4070 (12GB VRAM)
- **CPU:** i9 or equivalent
- **RAM:** 96GB
- **OS:** Windows 11
- **Camera:** USB webcam (640x480, 30fps)

## Key Papers Implemented

1. **SOAR** (UC Berkeley): VLM-guided autonomous practice
2. **SELFI** (UC Berkeley/Toyota): Online model-free RL
3. **EES** (MIT): Principled skill selection
4. **ACT** (Tony Zhao): Action chunking transformers
5. **VIP** (Facebook): Value-implicit pre-training
6. **SARM** (Stage-aware reward modeling)

## Expected Timeline

- **Phase 0:** 2-3 days
- **Phase 1:** 1-2 weeks
- **Phase 2:** 2-3 weeks
- **Phase 3:** 2-3 weeks
- **Phase 4:** 1 week

**Total: ~2-3 months to full SAIL system**

## Success Metrics

| Metric | Baseline Target | SAIL Target |
|--------|----------------|-------------|
| Success rate (10 demos) | 40-60% | 70-85% |
| Autonomous practice success | N/A | 30-50% |
| VLM labeling accuracy | N/A | >75% |
| Self-improvement ratio | 1.0× | 1.5-2.0× |

## Resources

- **LeRobot Docs:** https://huggingface.co/docs/lerobot
- **SAIL Implementation Guide:** `SAIL_DOC.md`
- **Phase 0 Plan:** `sail/PHASE0_EXECUTION_PLAN.md`

---

**Status:** Phase 0 - Ready to Start  
**Next Action:** Run `python sail/phase0_verify.py`

