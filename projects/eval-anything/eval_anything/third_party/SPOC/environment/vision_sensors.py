# Copyright 2024 Allen Institute for AI
# ==============================================================================

from typing import Any, Optional

import ai2thor
import ai2thor.controller
import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import EnvType, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super

from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.tasks import AbstractSafeTask


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
        task: AbstractSafeTask,
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
        task: AbstractSafeTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        output = False
        if task.get_current_room() not in task.seen_rooms:
            output = True
        return np.array(output, dtype=np.float64)
