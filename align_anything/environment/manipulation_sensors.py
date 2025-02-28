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
import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super

from environment.stretch_controller import StretchController
from tasks import AbstractSPOCTask


class AnObjectIsInHand(Sensor):
    def __init__(self, uuid: str = "an_object_is_in_hand") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        objects_in_hand = env.get_held_objects()
        return np.array([len(objects_in_hand) > 0], dtype=np.int64)


class RelativeArmLocationMetadata(Sensor):
    def __init__(self, uuid: str = "relative_arm_location_metadata") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        full_pose = env.get_arm_proprioception()
        return np.array(full_pose, dtype=np.float64)


class TargetObjectWasPickedUp(Sensor):
    def __init__(self, uuid: str = "target_obj_was_pickedup") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        target_obj_in_hand = False
        if "synsets" in task.task_info:
            object_types = task.task_info["synsets"]
            object_ids = []
            for object_type in object_types:
                object_ids += task.task_info["synset_to_object_ids"][object_type]
            objects_in_hand = env.get_held_objects()
            target_obj_in_hand = len([x for x in objects_in_hand if x in object_ids]) > 0
        return np.array([target_obj_in_hand], dtype=np.int64)
