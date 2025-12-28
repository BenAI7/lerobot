"""
Validate reward signals against human labels.
Goal: >75% agreement between VLM success and human judgment.

Usage:
    # Step 1: Generate predictions
    python sail/scripts/validate_rewards.py
    
    # Step 2: Manually edit sail/experiments/reward_validation.json
    #         Set "human_label" to true or false for each episode
    
    # Step 3: Compute agreement
    python sail/scripts/validate_rewards.py compute
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_dataset():
    """Load your pick-and-place dataset."""
    dataset_path = Path("C:/Users/Yeyian/outputs/self_improve/2025-12-21_16-04-22_790120_self_improve/datasets/yeyian/self_improve_pickplace_v2")
    
    if dataset_path.exists():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(str(dataset_path))
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("yeyian/self_improve_pickplace_v2")
    
    return dataset


def validate_rewards(
    task_description: str = "pick up the cube and place it in the target zone",
    num_episodes: int = 20,
    output_path: str = "sail/experiments/reward_validation.json"
):
    """
    Validate reward signals on your dataset.
    
    This will:
    1. Load your pick-and-place dataset
    2. Run VIP + VLM on episodes
    3. Save results for you to manually label
    """
    from sail.rewards.reward_manager import create_reward_manager
    
    print("=" * 60)
    print("SAIL Reward Validation")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset()
    print(f"✓ Dataset loaded: {dataset.num_episodes} episodes")
    
    # Create reward manager
    print("\nInitializing reward manager (VIP + VLM)...")
    manager = create_reward_manager(device="cuda", use_vlm=True)
    
    # Select episodes to validate
    actual_num = min(num_episodes, dataset.num_episodes)
    test_episodes = np.linspace(0, dataset.num_episodes - 1, actual_num, dtype=int)
    test_episodes = list(set(test_episodes))  # Remove duplicates
    
    print(f"\nValidating on {len(test_episodes)} episodes...")
    print(f"Task: '{task_description}'")
    
    results = []
    
    for i, ep_idx in enumerate(test_episodes):
        ep_idx = int(ep_idx)
        print(f"\n[{i+1}/{len(test_episodes)}] Episode {ep_idx}...", end=" ")
        
        try:
            # Find frames for this episode
            episode_frames = []
            for idx in range(len(dataset)):
                frame = dataset[idx]
                if frame["episode_index"].item() == ep_idx:
                    episode_frames.append(idx)
                elif len(episode_frames) > 0:
                    break
            
            if not episode_frames:
                print(f"SKIP (no frames found)")
                continue
            
            initial_frame = dataset[episode_frames[0]]["observation.images.station"]
            final_frame = dataset[episode_frames[-1]]["observation.images.station"]
            
            # Set up reward manager
            manager.set_initial(initial_frame)
            manager.set_goal(final_frame)
            
            # Get VIP progress
            vip_signals = manager.compute_step_reward(final_frame)
            
            # Get VLM success
            vlm_success, vlm_confidence = manager.vlm.detect_success(
                final_frame, task_description
            )
            
            results.append({
                "episode": ep_idx,
                "vlm_success": bool(vlm_success),
                "vlm_confidence": float(vlm_confidence),
                "vip_progress": float(vip_signals.vip_progress),
                "human_label": None  # To be filled manually
            })
            
            status = "SUCCESS" if vlm_success else "FAILURE"
            print(f"VLM={status} (conf={vlm_confidence:.2f}), VIP={vip_signals.vip_progress:.2f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Save for human labeling
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ Results saved!")
    print("=" * 60)
    
    print(f"\nFile: {output_path}")
    print("\n📝 NEXT STEPS:")
    print("1. Open the JSON file")
    print("2. For each episode, set 'human_label' to true or false")
    print("   - Watch the episode video if needed")
    print("   - true = task was completed successfully")
    print("   - false = task failed")
    print("3. Run: python sail/scripts/validate_rewards.py compute")
    
    return results


def compute_agreement(validation_path: str = "sail/experiments/reward_validation.json"):
    """Compute agreement after human labels are added."""
    
    print("=" * 60)
    print("Computing VLM Agreement with Human Labels")
    print("=" * 60)
    
    validation_path = Path(validation_path)
    
    if not validation_path.exists():
        print(f"\n✗ File not found: {validation_path}")
        print("Run validation first: python sail/scripts/validate_rewards.py")
        return
    
    with open(validation_path) as f:
        results = json.load(f)
    
    # Filter to episodes with human labels
    labeled = [r for r in results if r["human_label"] is not None]
    
    if not labeled:
        print("\n✗ No human labels found!")
        print(f"\nPlease edit {validation_path}")
        print("Set 'human_label' to true or false for each episode.")
        print("\nExample:")
        print('  "human_label": null  →  "human_label": true')
        return
    
    print(f"\nFound {len(labeled)} labeled episodes (out of {len(results)})")
    
    # Compute agreement
    agreements = sum(1 for r in labeled if r["vlm_success"] == r["human_label"])
    accuracy = agreements / len(labeled)
    
    # Compute recall (VLM says yes when human says yes)
    human_positives = [r for r in labeled if r["human_label"]]
    if human_positives:
        recall = sum(1 for r in human_positives if r["vlm_success"]) / len(human_positives)
    else:
        recall = 0.0
    
    # Compute precision (when VLM says yes, human agrees)
    vlm_positives = [r for r in labeled if r["vlm_success"]]
    if vlm_positives:
        precision = sum(1 for r in vlm_positives if r["human_label"]) / len(vlm_positives)
    else:
        precision = 0.0
    
    print("\n" + "=" * 60)
    print("VLM VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy:  {accuracy:.1%} ({agreements}/{len(labeled)})")
    print(f"Precision: {precision:.1%} (when VLM says yes, human agrees)")
    print(f"Recall:    {recall:.1%} (VLM catches all true successes)")
    
    print("\n" + "-" * 40)
    print("Target: >75% accuracy, >90% recall")
    print("-" * 40)
    
    if accuracy >= 0.75:
        print("\n✅ PASSED: VLM accuracy is sufficient!")
    else:
        print("\n⚠️  Below target accuracy. Consider:")
        print("   - Refining task description prompt")
        print("   - Using larger VLM model")
        print("   - Adjusting camera angle for clearer view")
    
    if recall >= 0.90:
        print("✅ PASSED: VLM recall is sufficient!")
    else:
        print("⚠️  Below target recall. VLM may miss some successes.")
    
    # VIP correlation analysis
    print("\n" + "=" * 60)
    print("VIP PROGRESS CORRELATION")
    print("=" * 60)
    
    successful = [r for r in labeled if r["human_label"]]
    failed = [r for r in labeled if not r["human_label"]]
    
    if successful and failed:
        avg_progress_success = np.mean([r["vip_progress"] for r in successful])
        avg_progress_fail = np.mean([r["vip_progress"] for r in failed])
        
        print(f"\nAvg VIP progress on successes: {avg_progress_success:.3f}")
        print(f"Avg VIP progress on failures:  {avg_progress_fail:.3f}")
        print(f"Separation: {avg_progress_success - avg_progress_fail:.3f}")
        
        if avg_progress_success > avg_progress_fail + 0.1:
            print("\n✅ VIP progress correlates with success!")
        else:
            print("\n⚠️  VIP may not be discriminative enough for this task")
    elif successful:
        print(f"\nAll labeled episodes were successes.")
        print(f"Avg VIP progress: {np.mean([r['vip_progress'] for r in successful]):.3f}")
    elif failed:
        print(f"\nAll labeled episodes were failures.")
        print(f"Avg VIP progress: {np.mean([r['vip_progress'] for r in failed]):.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 STATUS")
    print("=" * 60)
    
    if accuracy >= 0.75 and recall >= 0.90:
        print("\n🎉 Phase 1 COMPLETE!")
        print("Reward system is ready for Phase 2 (Goal-Conditioned Learning)")
    else:
        print("\n⚠️  Consider improving reward signals before Phase 2")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "num_labeled": len(labeled)
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute_agreement()
    else:
        validate_rewards()

