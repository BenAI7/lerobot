"""
SAIL Reward Models

VIP: Zero-shot progress from visual similarity
VLM: Success detection from vision-language model
"""

# Lazy imports to avoid slow loading
__all__ = [
    "VIPReward",
    "create_vip_reward",
    "VLMSuccessDetector",
    "create_vlm_detector",
    "RewardManager",
    "RewardSignals",
    "create_reward_manager",
]


def __getattr__(name):
    if name in ("VIPReward", "create_vip_reward"):
        from sail.rewards.vip import VIPReward, create_vip_reward
        return {"VIPReward": VIPReward, "create_vip_reward": create_vip_reward}[name]
    elif name in ("VLMSuccessDetector", "create_vlm_detector"):
        from sail.rewards.vlm_detector import VLMSuccessDetector, create_vlm_detector
        return {"VLMSuccessDetector": VLMSuccessDetector, "create_vlm_detector": create_vlm_detector}[name]
    elif name in ("RewardManager", "RewardSignals", "create_reward_manager"):
        from sail.rewards.reward_manager import RewardManager, RewardSignals, create_reward_manager
        return {"RewardManager": RewardManager, "RewardSignals": RewardSignals, "create_reward_manager": create_reward_manager}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

