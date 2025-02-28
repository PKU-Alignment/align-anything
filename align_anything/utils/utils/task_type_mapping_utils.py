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


import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.task_specs import TaskSpec


def map_task_type(task_type: str) -> str:
    task_type_map = dict(SimpleExploreHouse="RoomVisit", ObjectNavOpenVocab="ObjectNavDescription")
    return task_type_map.get(task_type, task_type)  # or task_type


def inverse_map_task_type(task_type: str) -> str:
    task_type_map = dict(RoomVisit="SimpleExploreHouse", ObjectNavDescription="ObjectNavOpenVocab")
    return task_type_map.get(task_type) or task_type


def map_task_spec(task_spec: "TaskSpec") -> "TaskSpec":
    task_spec = copy.copy(task_spec)
    task_spec["task_type"] = map_task_type(task_spec["task_type"])
    return task_spec
