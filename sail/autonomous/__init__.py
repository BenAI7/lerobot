"""
SAIL Autonomous Practice Module

Enables robot to practice tasks autonomously with VLM feedback.
"""

from sail.autonomous.practice_loop import (
    PracticeConfig,
    PracticeSession,
    run_autonomous_practice,
)

__all__ = [
    "PracticeConfig",
    "PracticeSession",
    "run_autonomous_practice",
]

