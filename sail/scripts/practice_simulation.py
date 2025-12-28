"""
Simulation mode for autonomous practice.

Tests the practice loop without requiring robot hardware.
Uses recorded dataset episodes to simulate robot execution.
"""

import torch
import numpy as np
from pathlib import Path
import json
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sail.rewards.vlm_detector import VLMSuccessDetector
from sail.rewards.vip import VIPReward


class SimulatedPracticeSession:
    """Simulates autonomous practice using dataset episodes."""
    
    def __init__(
        self,
        dataset_path: str,
        task_description: str,
        num_episodes: int = 10,
        output_dir: str = "outputs/sail_practice_sim",
        use_vlm: bool = True,
    ):
        self.dataset_path = dataset_path
        self.task_description = task_description
        self.num_episodes = num_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_vlm = use_vlm
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load dataset
        print(f"Loading dataset: {dataset_path}")
        self.dataset = LeRobotDataset(dataset_path)
        print(f"✓ Dataset loaded: {self.dataset.num_episodes} episodes")
        
        # Load evaluators
        if use_vlm:
            print("Loading VLM and VIP...")
            self.vlm = VLMSuccessDetector(device=self.device, use_4bit=True)
        else:
            print("Loading VIP only (VLM disabled for speed)...")
            self.vlm = None
        
        self.vip = VIPReward(device=self.device)
        print("✓ Evaluators loaded")
        
        # Build episode index from metadata (much faster!)
        print("Building episode index from metadata...")
        self.episode_boundaries = self._build_episode_index()
        print(f"✓ Indexed {len(self.episode_boundaries)} episodes")
        
        self.results = []
    
    def _build_episode_index(self):
        """Build index of episode start/end frames from dataset metadata."""
        episode_boundaries = {}
        
        # Use dataset's episode metadata (much faster than iterating frames!)
        if hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, 'episodes'):
            episodes = self.dataset.meta.episodes
            for i in range(len(episodes)):
                ep_data = episodes[i]
                # Get frame indices from metadata
                from_idx = ep_data.get("dataset_from_index", None)
                to_idx = ep_data.get("dataset_to_index", None)
                
                if from_idx is not None and to_idx is not None:
                    # to_idx is exclusive, so subtract 1 for last frame
                    episode_boundaries[i] = (int(from_idx), int(to_idx) - 1)
                else:
                    # Fallback: use episode_index from metadata
                    episode_boundaries[i] = (i, i)  # Placeholder, will be fixed below
        
        # If metadata doesn't have indices, fall back to scanning (but only once)
        if not episode_boundaries:
            print("  (Metadata not available, scanning dataset once...)")
            current_episode = None
            episode_start = None
            
            # Single pass through dataset
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                ep_idx = item["episode_index"].item()
                
                if ep_idx != current_episode:
                    # New episode started
                    if current_episode is not None:
                        # Save previous episode
                        episode_boundaries[current_episode] = (episode_start, i - 1)
                    # Start new episode
                    current_episode = ep_idx
                    episode_start = i
            
            # Don't forget the last episode
            if current_episode is not None:
                episode_boundaries[current_episode] = (episode_start, len(self.dataset) - 1)
        
        return episode_boundaries
    
    def evaluate_episode(self, episode_idx: int):
        """Evaluate an episode from the dataset."""
        
        import time
        start_time = time.time()
        
        print(f"[{episode_idx+1}/{self.num_episodes}] Episode {episode_idx}...", end=" ", flush=True)
        
        # Get episode boundaries from index
        if episode_idx not in self.episode_boundaries:
            print(f"⚠️ Not found")
            return None
        
        start_frame_idx, end_frame_idx = self.episode_boundaries[episode_idx]
        
        # Get initial and final frames
        initial_frame = self.dataset[start_frame_idx]
        final_frame = self.dataset[end_frame_idx]
        num_frames = end_frame_idx - start_frame_idx + 1
        
        # Extract images (use station camera)
        initial_image = initial_frame["observation.images.station"].cpu().numpy()
        final_image = final_frame["observation.images.station"].cpu().numpy()
        
        # Prepare images for evaluation
        def prepare_image(img):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if img.shape[0] == 3:  # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
            return img
        
        initial_image = prepare_image(initial_image)
        final_image = prepare_image(final_image)
        
        # VLM evaluation
        if self.vlm:
            print("VLM...", end=" ", flush=True)
            vlm_success, vlm_confidence = self.vlm.detect_success(
                final_image, 
                self.task_description
            )
        else:
            vlm_success = False
            vlm_confidence = 0.0
        
        print("VIP...", end=" ", flush=True)
        
        # VIP evaluation
        initial_tensor = torch.from_numpy(initial_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        final_tensor = torch.from_numpy(final_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        vip_progress = self.vip.compute_progress(
            final_tensor,
            initial_tensor,
            final_tensor
        ).item()
        
        elapsed = time.time() - start_time
        
        result = {
            "episode": episode_idx,
            "num_frames": num_frames,
            "vlm_success": vlm_success,
            "vlm_confidence": vlm_confidence,
            "vip_progress": vip_progress,
        }
        
        if self.use_vlm:
            status = "✓" if vlm_success else "✗"
            print(f"{status} VLM={vlm_success} (conf={vlm_confidence:.2f}), VIP={vip_progress:.2f} ({elapsed:.1f}s)")
        else:
            print(f"VIP={vip_progress:.2f} ({elapsed:.1f}s)")
        
        return result
    
    def run(self):
        """Run simulated practice session."""
        print("\n" + "=" * 60)
        print("Simulated Autonomous Practice")
        print("=" * 60)
        print(f"Task: {self.task_description}")
        print(f"Episodes to evaluate: {self.num_episodes}")
        print(f"Mode: {'VLM + VIP' if self.use_vlm else 'VIP only (fast)'}")
        print("=" * 60)
        
        # Evaluate episodes
        episode_indices = list(range(min(self.num_episodes, self.dataset.num_episodes)))
        
        print(f"\nEvaluating {len(episode_indices)} episodes...")
        
        for ep_idx in episode_indices:
            result = self.evaluate_episode(ep_idx)
            if result:
                self.results.append(result)
        
        # Summary
        self._print_summary()
        self._save_summary()
    
    def _print_summary(self):
        """Print session summary."""
        if not self.results:
            print("\nNo results to summarize")
            return
        
        num_total = len(self.results)
        avg_vip = np.mean([r["vip_progress"] for r in self.results])
        
        print("\n" + "=" * 60)
        print("Simulation Complete!")
        print("=" * 60)
        print(f"Episodes evaluated: {num_total}")
        print(f"Average VIP progress: {avg_vip:.2f}")
        
        if self.use_vlm:
            num_success = sum(1 for r in self.results if r["vlm_success"])
            success_rate = num_success / num_total * 100
            avg_vlm_conf = np.mean([r["vlm_confidence"] for r in self.results])
            print(f"VLM successes: {num_success}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average VLM confidence: {avg_vlm_conf:.2f}")
        else:
            print("(VLM was disabled for speed)")
        
        print("=" * 60)
    
    def _save_summary(self):
        """Save results to JSON."""
        summary_path = self.output_dir / "simulation_results.json"
        
        summary_stats = {
            "avg_vip_progress": float(np.mean([r["vip_progress"] for r in self.results])) if self.results else 0,
        }
        
        if self.use_vlm:
            summary_stats["success_rate"] = sum(1 for r in self.results if r["vlm_success"]) / len(self.results) if self.results else 0
            summary_stats["avg_vlm_confidence"] = float(np.mean([r["vlm_confidence"] for r in self.results])) if self.results else 0
        
        summary = {
            "task": self.task_description,
            "dataset": self.dataset_path,
            "num_episodes": len(self.results),
            "use_vlm": self.use_vlm,
            "results": self.results,
            "summary": summary_stats,
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate autonomous practice")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the cube and place it in the target zone",
        help="Task description"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sail_practice_sim",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM (faster, VIP only)"
    )
    
    args = parser.parse_args()
    
    session = SimulatedPracticeSession(
        dataset_path=args.dataset,
        task_description=args.task,
        num_episodes=args.episodes,
        output_dir=args.output,
        use_vlm=not args.skip_vlm,
    )
    
    session.run()


if __name__ == "__main__":
    main()

