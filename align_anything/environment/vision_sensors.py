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
from typing import Any, Optional

import ai2thor
import ai2thor.controller
import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import EnvType, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from tasks import AbstractSPOCTask

from environment.stretch_controller import StretchController


class RawRGBSensorTHOR(Sensor):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(env, ai2thor.controller.Controller):
            return env.last_event.frame.copy()
        else:
            return env.current_frame.copy()


class RawManipulationStretchRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.manipulation_camera.copy()


class RawNavigationStretchRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.navigation_camera.copy()


class ReadyForDoneActionSensor(Sensor):
    def __init__(self, uuid: str = 'expert_done') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(task.successful_if_done(), dtype=np.float64)


class ReadyForSubDoneActionSensor(Sensor):
    def __init__(self, uuid: str = 'expert_subdone') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        output = False
        if task.get_current_room() not in task.seen_rooms:
            output = True
        return np.array(output, dtype=np.float64)
