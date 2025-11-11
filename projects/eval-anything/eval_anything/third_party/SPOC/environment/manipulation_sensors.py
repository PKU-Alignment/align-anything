# Copyright 2024 Allen Institute for AI
# ==============================================================================

import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super

from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.tasks import AbstractSafeTask


class AnObjectIsInHand(Sensor):
    def __init__(self, uuid: str = 'an_object_is_in_hand') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSafeTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        objects_in_hand = env.get_held_objects()
        return np.array([len(objects_in_hand) > 0], dtype=np.int64)


class RelativeArmLocationMetadata(Sensor):
    def __init__(self, uuid: str = 'relative_arm_location_metadata') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSafeTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        full_pose = env.get_arm_proprioception()
        return np.array(full_pose, dtype=np.float64)


class TargetObjectWasPickedUp(Sensor):
    def __init__(self, uuid: str = 'target_obj_was_pickedup') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(
        self,
        env: StretchController,
        task: AbstractSafeTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        target_obj_in_hand = False
        if 'synsets' in task.task_info:
            object_types = task.task_info['synsets']
            object_ids = []
            for object_type in object_types:
                object_ids += task.task_info['synset_to_object_ids'][object_type]
            objects_in_hand = env.get_held_objects()
            target_obj_in_hand = len([x for x in objects_in_hand if x in object_ids]) > 0
        return np.array([target_obj_in_hand], dtype=np.int64)
