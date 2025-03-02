# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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
