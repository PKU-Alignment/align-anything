# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import Optional


def sel_metric(
    success: bool, optimal_episode_length: float, actual_episode_length: float
) -> Optional[float]:
    if not success:
        return 0.0
    elif optimal_episode_length < 0:
        return None
    elif optimal_episode_length == 0:
        if actual_episode_length == 0:
            return 1.0
        else:
            return 0.0
    else:
        travelled_distance = max(actual_episode_length, optimal_episode_length)
        return optimal_episode_length / travelled_distance
