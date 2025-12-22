#!/usr/bin/env python
#
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Self-improving imitation loop for real-world robots (Phase 1).

One "session" runs:
  1) Record 1 teleop demo episode (you control the robot).
  2) Train/fine-tune for a small number of steps (fast iteration).
  3) Run 1 policy-controlled episode and prompt for success/failure.
     - If marked failure, the episode can be discarded (so you only keep good self-rollouts).

You can run multiple sessions back-to-back, and confirm whether to continue after each one.
"""

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.configs.default import WandBConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.scripts.lerobot_train import train
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class SelfImproveTrainConfig:
    output_root: Path = Path("outputs/self_improve")
    policy_type: str = "act"
    device: str = "cuda"
    # Roughly ~5 minutes on many RTX 40xx setups for ACT @ batch_size=8.
    # Tune this based on your observed steps/sec.
    steps_per_session: int = 2000
    log_freq: int = 50


@dataclass
class SelfImproveConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig

    # Where to store the growing dataset (kept local).
    dataset_repo_id: str
    # Base directory for datasets. The actual dataset directory is `${dataset_root}/${dataset_repo_id}`.
    # If None, a run-local folder is used.
    dataset_root: Path | None = None
    task: str = "Pick up the object and place it in the bin"
    fps: int = 30

    # Timing
    demo_episode_time_s: int = 30
    policy_episode_time_s: int = 60
    # How many teleop demos to record at the start of each session before training.
    demos_per_session: int = 3

    # Loop control
    sessions: int = 5
    confirm_each_session: bool = True

    # Visualization
    display_data: bool = True
    play_sounds: bool = True

    # Training
    train: SelfImproveTrainConfig = field(default_factory=SelfImproveTrainConfig)


def _resolve_dataset_root(cfg: SelfImproveConfig, run_dir: Path) -> Path:
    if cfg.dataset_root is not None:
        return Path(cfg.dataset_root)
    return run_dir / "datasets"


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    default = "y" if default_yes else "n"
    while True:
        ans = input(f"{prompt} [y/n] (default={default}): ").strip().lower()
        if ans == "":
            return default_yes
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please type y or n.")


@parser.wrap()
def self_improve(cfg: SelfImproveConfig) -> None:
    init_logging()

    # `dataset_repo_id` must be a logical dataset id, not a filesystem path.
    # Use `--dataset_root` to control where the dataset is stored locally.
    if ":" in cfg.dataset_repo_id or "\\" in cfg.dataset_repo_id:
        raise ValueError(
            "Invalid `dataset_repo_id`. Expected something like 'namespace/repo_name' (e.g. 'yeyian/self_improve_pickplace'). "
            "Do NOT pass a Windows path here. If you want to control where it is stored, use `--dataset_root`."
        )

    now = dt.datetime.now()
    # Use microseconds to avoid collisions when restarting quickly after a crash.
    run_dir = cfg.train.output_root / f"{now:%Y-%m-%d_%H-%M-%S_%f}_self_improve"
    # If the folder still exists (e.g. copied/created manually), add a numeric suffix.
    if run_dir.exists():
        suffix = 1
        while (cfg.train.output_root / f"{now:%Y-%m-%d_%H-%M-%S_%f}_self_improve_{suffix}").exists():
            suffix += 1
        run_dir = cfg.train.output_root / f"{now:%Y-%m-%d_%H-%M-%S_%f}_self_improve_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    ds_base = _resolve_dataset_root(cfg, run_dir)
    ds_dir = ds_base / cfg.dataset_repo_id  # IMPORTANT: root must be the full dataset directory
    ds_base.mkdir(parents=True, exist_ok=True)

    logger.info(f"Self-improve run dir: {run_dir}")
    logger.info(f"Dataset: {cfg.dataset_repo_id} (root={ds_dir})")

    last_pretrained_dir: Path | None = None
    dataset_already_exists = ds_dir.exists()
    resume = dataset_already_exists

    for i in range(1, cfg.sessions + 1):
        if cfg.confirm_each_session and i > 1:
            if not _prompt_yes_no(f"Continue to session {i}?", default_yes=True):
                logger.info("Stopping self-improve loop (user requested).")
                return

        logger.info(f"=== Session {i}/{cfg.sessions} ===")

        # -------------------------
        # Step 1: Teleop demo
        # -------------------------
        logger.info(f"Step 1/3: Teleoperate {cfg.demos_per_session} demo episode(s).")
        demo_dataset_cfg = DatasetRecordConfig(
            repo_id=cfg.dataset_repo_id,
            root=ds_dir,
            fps=cfg.fps,
            episode_time_s=cfg.demo_episode_time_s,
            reset_time_s=0,
            num_episodes=cfg.demos_per_session,
            push_to_hub=False,
            single_task=cfg.task,
        )
        demo_record_cfg = RecordConfig(
            robot=cfg.robot,
            dataset=demo_dataset_cfg,
            teleop=cfg.teleop,
            policy=None,
            display_data=cfg.display_data,
            play_sounds=cfg.play_sounds,
            resume=resume,
        )
        record(demo_record_cfg)
        resume = True  # after first record, dataset exists

        # -------------------------
        # Step 2: short train
        # -------------------------
        logger.info("Step 2/3: Short training/fine-tune.")
        # Each session writes into a fresh output dir. IMPORTANT: `lerobot-train` expects
        # the output_dir to NOT exist (unless resume=true), so we must not create it here.
        run_dir.mkdir(parents=True, exist_ok=True)
        train_out = run_dir / f"train_session_{i:02d}"
        if train_out.exists():
            suffix = 1
            while (run_dir / f"train_session_{i:02d}_{suffix}").exists():
                suffix += 1
            train_out = run_dir / f"train_session_{i:02d}_{suffix}"

        policy_cfg = ACTConfig(device=cfg.train.device, push_to_hub=False)
        if last_pretrained_dir is not None:
            policy_cfg.pretrained_path = last_pretrained_dir

        train_cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id=cfg.dataset_repo_id,
                root=str(ds_dir),
                image_transforms=ImageTransformsConfig(enable=False),
            ),
            policy=policy_cfg,
            output_dir=train_out,
            job_name=f"self_improve_session_{i:02d}",
            steps=cfg.train.steps_per_session,
            log_freq=cfg.train.log_freq,
            eval_freq=0,
            save_freq=cfg.train.steps_per_session,
            wandb=WandBConfig(enable=False),
        )
        train(train_cfg)

        last_pretrained_dir = train_out / "checkpoints" / "last" / "pretrained_model"
        if not last_pretrained_dir.exists():
            raise FileNotFoundError(f"Expected pretrained model dir not found: {last_pretrained_dir}")

        # -------------------------
        # Step 3: policy trial + label
        # -------------------------
        logger.info("Step 3/3: Run 1 policy episode and label success/failure.")
        policy_dataset_cfg = DatasetRecordConfig(
            repo_id=cfg.dataset_repo_id,
            root=ds_dir,
            fps=cfg.fps,
            episode_time_s=cfg.policy_episode_time_s,
            reset_time_s=0,
            num_episodes=1,
            push_to_hub=False,
            single_task=cfg.task,
            prompt_success_label=True,
            discard_episode_on_failure=True,
        )
        policy_record_cfg = RecordConfig(
            robot=cfg.robot,
            dataset=policy_dataset_cfg,
            teleop=None,
            policy=ACTConfig(device=cfg.train.device, push_to_hub=False, pretrained_path=last_pretrained_dir),
            display_data=cfg.display_data,
            play_sounds=cfg.play_sounds,
            resume=True,
        )
        record(policy_record_cfg)

        logger.info(f"Session {i} complete. Current checkpoint: {last_pretrained_dir}")

    logger.info("Self-improve loop completed.")


def main() -> None:
    self_improve()


if __name__ == "__main__":
    main()


