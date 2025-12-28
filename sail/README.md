# SAIL Implementation

Self-Adaptation Imitation Learning for SO-101 Robot

## 📁 Project Structure

```
sail/
├── rewards/              # Reward infrastructure (VIP, VLM, Manager)
│   ├── vip.py           # VIP reward model (zero-shot progress)
│   ├── vlm_detector.py  # VLM success detector (Qwen2-VL-2B)
│   └── reward_manager.py # Unified reward manager
├── scripts/             # Utility scripts
│   └── validate_rewards.py # Reward validation against human labels
├── tests/               # Test scripts for all components
│   ├── test_dependencies.py
│   ├── test_vip.py
│   ├── test_vlm.py
│   ├── test_reward_manager.py
│   └── phase0_verify.py
├── experiments/         # Experiment results and data
│   ├── phase0_baseline_summary.json
│   └── reward_validation.json
├── policies/            # Goal-conditioned policy implementations
├── training/            # Training scripts and utilities
├── autonomous/          # Autonomous practice system
└── configs/             # Configuration files
```

## 📚 Documentation

All documentation is in the root `lerobot/` folder:
- `SAIL_DOC.md` - Complete implementation plan
- `PHASE0_EXECUTION_PLAN.md` - Phase 0 baseline training
- `PHASE1_EXECUTION_PLAN.md` - Phase 1 reward infrastructure
- `PHASE1_COMMANDS.md` - Quick command reference for Phase 1
- `START_HERE.md` - Getting started guide

## 🚀 Quick Start

### Phase 1: Test Reward Infrastructure
```bash
cd C:\Users\Yeyian\lerobot
conda activate lerobot

# Test dependencies
python sail/tests/test_dependencies.py

# Test VIP reward model
python sail/tests/test_vip.py

# Test VLM success detector
python sail/tests/test_vlm.py

# Test unified reward manager
python sail/tests/test_reward_manager.py

# Validate rewards (requires manual labeling)
python sail/scripts/validate_rewards.py
```

## 📊 Current Status

- ✅ Phase 0: Baseline ACT policy trained (50 episodes, 40K frames)
- ✅ Phase 1: Reward infrastructure implemented and tested
- ⏳ Phase 2: Goal-conditioned policy (next)

## 🔧 Hardware

- **Robot**: SO-101 dual arms (Feetech motors)
- **GPU**: RTX 4070 (12GB VRAM)
- **Cameras**: 2x (gripper + station views)

## 📖 Key Papers

- **SOAR**: Self-supervision, Online data, Autonomy, Refinement
- **SELFI**: Self-improvement through VLM feedback
- **EES**: Entropy-based exploration scheduling

