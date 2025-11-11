# Copyright 2024 Allen Institute for AI
# ==============================================================================

import json
import random
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import gym
import numpy as np
import torch
from allenact.base_abstractions.sensor import Sensor, SubTaskType
from allenact.base_abstractions.task import EnvType
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask

from eval_anything.third_party.SPOC.environment.stretch_controller import StretchController
from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import (
    EMPTY_BBOX,
    EMPTY_DOUBLE_BBOX,
)
from eval_anything.third_party.SPOC.utils.string_utils import (
    convert_byte_to_string,
    convert_string_to_byte,
    json_templated_task_string,
)
from eval_anything.third_party.SPOC.utils.task_spec_to_instruction import best_lemma
from eval_anything.third_party.SPOC.utils.type_utils import get_task_relevant_synsets


if TYPE_CHECKING:
    from eval_anything.third_party.SPOC.tasks.abstract_task import AbstractSPOCTask
else:
    from typing import TypeVar

    AbstractSPOCTask = TypeVar('AbstractSPOCTask')


def get_best_of_two_bboxes(bbox_1, bbox_2):
    assert bbox_1.shape == bbox_2.shape
    B, T, dim = bbox_1.shape
    assert dim == 10
    size_target_obj_box_1 = bbox_1[:, :, 4]
    size_target_obj_box_2 = bbox_2[:, :, 4]

    box_2_is_bigger = size_target_obj_box_1 < size_target_obj_box_2
    bigger_box_obj = bbox_1.clone() if torch.is_tensor(bbox_1) else np.copy(bbox_1)
    bigger_box_obj[box_2_is_bigger] = bbox_2[box_2_is_bigger]

    size_receptacle_box_1 = bbox_1[:, :, 9]
    size_receptacle_box_2 = bbox_2[:, :, 9]

    box_2_is_bigger = size_receptacle_box_1 < size_receptacle_box_2
    bigger_box_rec = bbox_1.clone() if torch.is_tensor(bbox_1) else np.copy(bbox_1)
    bigger_box_rec[box_2_is_bigger] = bbox_2[box_2_is_bigger]

    bigger_box_obj[:, :, 5:9] = bigger_box_rec[:, :, 5:9]
    return bigger_box_obj


class LastActionSuccessSensor(Sensor):
    def __init__(self, uuid: str = 'last_action_success') -> None:
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
        return np.array([task.last_action_success], dtype=np.int64)


class LastActionIsRandomSensor(Sensor):
    def __init__(self, uuid: str = 'last_action_is_random') -> None:
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
        return np.array([task.last_action_random], dtype=np.int64)


class LastAgentLocationSensor(Sensor):
    def __init__(self, uuid: str = 'last_agent_location') -> None:
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
        agent_position_rotation = env.get_current_agent_full_pose()
        agent_position = agent_position_rotation['position']
        agent_rotation = agent_position_rotation['rotation']

        return np.array(
            [
                agent_position['x'],
                agent_position['y'],
                agent_position['z'],
                agent_rotation['x'],
                agent_rotation['y'],
                agent_rotation['z'],
            ],
            dtype=np.float64,
        )


class TaskTemplatedTextSpecSensor(Sensor):
    def __init__(
        self,
        uuid: str = 'templated_task_spec',
        str_max_len: Union[str, int] = 'adaptive',
    ) -> None:
        assert isinstance(str_max_len, int) or str_max_len == 'adaptive'
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.MultiDiscrete:
        if self.str_max_len == 'adaptive':
            return gym.spaces.MultiDiscrete([256] * 1)
        else:
            return gym.spaces.MultiDiscrete([256] * self.str_max_len)

    @staticmethod
    def encode_observation(task_info, str_max_len: Union[str, int]):
        task_string = json_templated_task_string(task_info)
        if str_max_len == 'adaptive':
            bytes = convert_string_to_byte(task_string, 2 * len(task_string))
            final_index = len(bytes) + 1
            for ind in reversed(range(len(bytes))):
                if bytes[ind] == 0:
                    final_index = ind
            return bytes[:final_index]
        elif isinstance(str_max_len, int):
            return convert_string_to_byte(task_string, str_max_len)
        else:
            raise NotImplementedError

    def get_observation(
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return self.encode_observation(task.task_info, self.str_max_len)


class TaskNaturalLanguageSpecSensor(Sensor):
    def __init__(
        self,
        uuid: str = 'task_natural_language_spec',
        str_max_len=1000,
        dynamic_instruction: bool = False,
    ) -> None:
        self.str_max_len = str_max_len
        self.dynamic_instruction = dynamic_instruction
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def dynamic_change_instruction(self, task: Optional[SubTaskType], goal: str) -> str:
        if hasattr(task, 'found_target_idx'):
            if self.dynamic_instruction:
                goal = goal.split(' a')[0]
                for i in range(len(task.task_info['synsets'])):
                    if i not in task.found_target_idx:
                        obj = best_lemma(task.task_info['synsets'][i])
                        if obj == 'apple':
                            goal += ' an ' + obj + ' and'
                        else:
                            goal += ' a ' + obj + ' and'
                goal = goal[:-4]
            else:
                goal = goal.split(', in that order')[0]
        return goal

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        natural_language_spec = task.task_info['natural_language_spec']
        natural_language_spec = self.dynamic_change_instruction(task, natural_language_spec)

        return convert_string_to_byte(natural_language_spec, self.str_max_len)


class HypotheticalTaskSuccessSensor(Sensor):
    def __init__(self, uuid: str = 'hypothetical_task_success') -> None:
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
        return np.array([task.successful_if_done(strict_success=True)], dtype=np.int64)


class MinimumTargetAlignmentSensor(Sensor):
    def __init__(self, uuid: str = 'minimum_visible_target_alignment') -> None:
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
        if 'synsets' not in task.task_info:
            return np.array([-1], dtype=np.float64)
        object_type = task.task_info['synsets'][0]
        visible_target_objects = [
            obj
            for obj in task.task_info['synset_to_object_ids'][object_type]
            if task.controller.object_is_visible_in_camera(
                obj, which_camera='nav', maximum_distance=2
            )
        ]
        visible_target_alignments = [
            abs(task.controller.get_agent_alignment_to_object(obj))
            for obj in visible_target_objects
        ]
        if len(visible_target_alignments) == 0:
            return np.array([-1], dtype=np.float64)
        else:
            return np.array([min(visible_target_alignments)], dtype=np.float64)


class Visible4mTargetCountSensor(Sensor):
    def __init__(self, uuid: str = 'visible_target_4m_count') -> None:
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
        if 'synsets' not in task.task_info:
            return np.array([0], dtype=np.float64)
        object_type = task.task_info['synsets'][0]
        visible_target_objects = [
            obj
            for obj in task.task_info['synset_to_object_ids'][object_type]
            if task.controller.object_is_visible_in_camera(
                obj, which_camera='nav', maximum_distance=4
            )
        ]
        return np.array([len(visible_target_objects)], dtype=np.int64)


class TaskRelevantObjectBBoxSensor(Sensor):
    def __init__(
        self,
        convert_to_pixel_coords: bool = True,
        which_camera: Literal['nav', 'manip'] = 'nav',
        uuid: str = 'task_relevant_object_bbox',
    ) -> None:
        self.convert_to_pixel_coords = convert_to_pixel_coords
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

        self.which_camera = which_camera

        self.task_relevant_oids = []
        self.task_relevant_synset_to_objects = {}
        self.oids_as_bytes = None
        self.synset_to_oids_as_bytes = None

    def _get_observation_space(self) -> gym.spaces.Dict:
        if self.convert_to_pixel_coords:
            return gym.spaces.Dict(
                spaces={
                    'oids_as_bytes': gym.spaces.MultiDiscrete([256] * 10),
                    'synset_to_oids_as_bytes': gym.spaces.MultiDiscrete([256] * 10),
                    'min_rows': gym.spaces.Box(low=-1, high=np.inf, shape=(10,), dtype=np.float32),
                    'max_rows': gym.spaces.Box(low=-1, high=np.inf, shape=(10,), dtype=np.float32),
                    'min_cols': gym.spaces.Box(low=-1, high=np.inf, shape=(10,), dtype=np.float32),
                    'max_cols': gym.spaces.Box(low=-1, high=np.inf, shape=(10,), dtype=np.float32),
                },
            )
        else:
            return gym.spaces.Dict(
                spaces={
                    'oids_as_bytes': gym.spaces.MultiDiscrete([256] * 10),
                    'synset_to_oids_as_bytes': gym.spaces.MultiDiscrete([256] * 10),
                    'min_xs': gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                    'max_xs': gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                    'min_ys': gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                    'max_ys': gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                },
            )

    @staticmethod
    def encode_json(json_serializable: Any):
        oids_json = json.dumps(json_serializable)

        encoded_bytes = convert_string_to_byte(oids_json, 2 * len(oids_json))
        final_index = len(encoded_bytes) + 1
        for ind in reversed(range(len(encoded_bytes))):
            if encoded_bytes[ind] == 0:
                final_index = ind
        return encoded_bytes[:final_index]

    def get_observation(
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if task.num_steps_taken() == 0:
            if hasattr(task, 'task_relevant_synset_to_oids'):
                self.task_relevant_synset_to_objects = task.task_relevant_synset_to_oids
                self.task_relevant_oids = task.task_relevant_oids
                self.oids_as_bytes = task.oids_as_bytes
                self.synset_to_oids_as_bytes = task.synset_to_oids_as_bytes
            else:
                task_relevant_synsets = get_task_relevant_synsets(task_spec=task.task_info)
                all_objects = env.get_objects()

                self.task_relevant_synset_to_objects = {}
                for synset in task_relevant_synsets:
                    self.task_relevant_synset_to_objects[synset] = env.get_all_objects_of_synset(
                        synset=synset, include_hyponyms=True, all_objs=all_objects
                    )

                self.task_relevant_oids = list(
                    sorted(
                        {
                            o['objectId']
                            for objs in self.task_relevant_synset_to_objects.values()
                            for o in objs
                        }
                    )
                )

                task_relevant_synset_to_oids = {
                    synset: [o['objectId'] for o in objs]
                    for synset, objs in self.task_relevant_synset_to_objects.items()
                }

                self.oids_as_bytes = self.encode_json(self.task_relevant_oids)
                self.synset_to_oids_as_bytes = self.encode_json(task_relevant_synset_to_oids)

                task.task_relevant_synset_to_oids = self.task_relevant_synset_to_objects
                task.task_relevant_oids = self.task_relevant_oids
                task.oids_as_bytes = self.oids_as_bytes
                task.synset_to_oids_as_bytes = self.synset_to_oids_as_bytes

        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        for oid in self.task_relevant_oids:
            min_x, min_y, max_x, max_y = -1, -1, -1, -1

            if task.controller.object_is_visible_in_camera(
                oid, which_camera=self.which_camera, maximum_distance=4
            ):
                points = env.get_approx_object_mask(
                    object_id=oid, which_camera=self.which_camera, divisions=7
                )

                if points is not None and len(points) != 0:
                    xs = [max(min(p['x'], 1), 0) for p in points]
                    ys = [max(min(p['y'], 1), 0) for p in points]
                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)

            min_xs.append(min_x)
            max_xs.append(max_x)
            min_ys.append(min_y)
            max_ys.append(max_y)

        if self.convert_to_pixel_coords:
            h, w = env.controller.last_event.frame.shape[:2]
            doesnt_have_value = ~np.array([min_x != -1 for min_x in min_xs], dtype=bool)
            min_cols = w * np.array(min_xs, dtype=np.float32)
            max_cols = w * np.array(max_xs, dtype=np.float32)
            max_rows = h * (1 - np.array(min_ys, dtype=np.float32))
            min_rows = h * (1 - np.array(max_ys, dtype=np.float32))

            hs, ws = env.navigation_camera.shape[:2]

            assert ws == w - 12 and hs == h
            min_cols = np.clip(min_cols - 6, a_min=0, a_max=ws - 1)
            max_cols = np.clip(max_cols - 6, a_min=0, a_max=ws - 1)

            min_cols[doesnt_have_value] = -1
            max_cols[doesnt_have_value] = -1
            max_rows[doesnt_have_value] = -1
            min_rows[doesnt_have_value] = -1
            return {
                'oids_as_bytes': self.oids_as_bytes,
                'synset_to_oids_as_bytes': self.synset_to_oids_as_bytes,
                'min_cols': min_cols.astype(int).astype(np.float32),
                'max_cols': max_cols.astype(int).astype(np.float32),
                'max_rows': max_rows.astype(int).astype(np.float32),
                'min_rows': min_rows.astype(int).astype(np.float32),
            }

        else:
            return {
                'oids_as_bytes': self.oids_as_bytes,
                'min_xs': np.array(min_xs, dtype=np.float32).astype(int).astype(np.float32),
                'max_xs': np.array(max_xs, dtype=np.float32).astype(int).astype(np.float32),
                'min_ys': np.array(min_ys, dtype=np.float32).astype(int).astype(np.float32),
                'max_ys': np.array(max_ys, dtype=np.float32).astype(int).astype(np.float32),
            }


class SlowAccurateObjectBBoxSensor(TaskRelevantObjectBBoxSensor):
    def __init__(
        self,
        convert_to_pixel_coords: bool = True,
        which_camera: Literal['nav', 'manip'] = 'nav',
        uuid: str = 'accurate_object_bbox',
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.convert_to_pixel_coords = convert_to_pixel_coords
        observation_space = self._get_observation_space()
        assert convert_to_pixel_coords is True

        self.which_camera = which_camera

        self.task_relevant_oids = []
        self.task_relevant_synset_to_objects = {}
        self.oids_as_bytes = None
        self.synset_to_oids_as_bytes = None

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if task.num_steps_taken() == 0:
            if hasattr(task, 'task_relevant_synset_to_oids'):
                self.task_relevant_synset_to_objects = task.task_relevant_synset_to_oids
                self.task_relevant_oids = task.task_relevant_oids
                self.oids_as_bytes = task.oids_as_bytes
                self.synset_to_oids_as_bytes = task.synset_to_oids_as_bytes
            else:
                task_relevant_synsets = get_task_relevant_synsets(task_spec=task.task_info)
                all_objects = env.get_objects()

                self.task_relevant_synset_to_objects = {}
                for synset in task_relevant_synsets:
                    self.task_relevant_synset_to_objects[synset] = env.get_all_objects_of_synset(
                        synset=synset, include_hyponyms=True, all_objs=all_objects
                    )

                self.task_relevant_oids = list(
                    sorted(
                        {
                            o['objectId']
                            for objs in self.task_relevant_synset_to_objects.values()
                            for o in objs
                        }
                    )
                )

                task_relevant_synset_to_oids = {
                    synset: [o['objectId'] for o in objs]
                    for synset, objs in self.task_relevant_synset_to_objects.items()
                }

                self.oids_as_bytes = self.encode_json(self.task_relevant_oids)
                self.synset_to_oids_as_bytes = self.encode_json(task_relevant_synset_to_oids)

                task.task_relevant_synset_to_oids = self.task_relevant_synset_to_objects
                task.task_relevant_oids = self.task_relevant_oids
                task.oids_as_bytes = self.oids_as_bytes
                task.synset_to_oids_as_bytes = self.synset_to_oids_as_bytes

        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        for oid in self.task_relevant_oids:
            min_x, min_y, max_x, max_y = -1, -1, -1, -1
            segm_mask = task.controller.get_segmentation_mask_of_object(
                oid, which_camera=self.which_camera
            )
            if np.any(segm_mask):
                min_x = np.min(np.where(segm_mask)[1])
                min_y = np.min(np.where(segm_mask)[0])
                max_x = np.max(np.where(segm_mask)[1])
                max_y = np.max(np.where(segm_mask)[0])

            min_xs.append(min_x)
            max_xs.append(max_x)
            min_ys.append(min_y)
            max_ys.append(max_y)

        min_cols = np.array(min_xs, dtype=np.float32).astype(int).astype(np.float32)
        max_cols = np.array(max_xs, dtype=np.float32).astype(int).astype(np.float32)
        max_rows = np.array(max_ys, dtype=np.float32).astype(int).astype(np.float32)
        min_rows = np.array(min_ys, dtype=np.float32).astype(int).astype(np.float32)

        return {
            'oids_as_bytes': self.oids_as_bytes,
            'synset_to_oids_as_bytes': self.synset_to_oids_as_bytes,
            'min_cols': min_cols,
            'max_cols': max_cols,
            'max_rows': max_rows,
            'min_rows': min_rows,
        }


class TaskRelevantObjectBBoxSensorOnlineEval(Sensor):
    def __init__(
        self,
        convert_to_pixel_coords: bool = True,
        which_camera: Literal['nav', 'manip'] = 'nav',
        uuid: str = 'task_relevant_object_bbox',
        original_sensor_to_use: Optional[type(Sensor)] = TaskRelevantObjectBBoxSensor,
    ) -> None:
        observation_space = gym.spaces.Discrete(3)
        super().__init__(**prepare_locals_for_super(locals()))
        self.sensor_to_use = original_sensor_to_use(
            convert_to_pixel_coords=convert_to_pixel_coords, which_camera=which_camera, uuid=uuid
        )

        self.convert_to_pixel_coords = convert_to_pixel_coords

        self.which_camera = which_camera

        self.task_relevant_oids = []
        self.task_relevant_synset_to_objects = {}
        self.oids_as_bytes = None
        self.synset_to_oids_as_bytes = None

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        observation_dict = self.sensor_to_use.get_observation(env, task, *args, **kwargs)

        num_boxes = observation_dict['min_cols'].shape[0]

        task_dict = task.task_info
        oids = eval(convert_byte_to_string(observation_dict['oids_as_bytes']))

        assert len(oids) == num_boxes, "Number of oids and boxes don't match"

        tgt_1_ids = []
        tgt_2_ids = []

        if 'broad_synset_to_object_ids' in task_dict:
            if task_dict['task_type'] == 'ObjectNavMulti':
                for idx in range(task.num_targets):
                    if idx not in task.found_target_idx:
                        for val in task.task_info['broad_synset_to_object_ids'][
                            task.task_info['synsets'][idx]
                        ]:
                            tgt_1_ids.append(val)
                # tgt_1_ids = [
                #     val
                #     for val in task.task_info["broad_synset_to_object_ids"][
                #         task.task_info["synsets"][task.current_target_idx]
                #     ]
                # ]
            else:
                tgt_1_ids = [val for val in task_dict['broad_synset_to_object_ids'].values()]
                tgt_1_ids = sum(tgt_1_ids, [])

        def parse_biggest_bbox(object_indices):
            object_indices = sorted(object_indices)
            if len(object_indices) == 0:  # both bbox_1 and bbox_2 need to have a default value
                return np.array(EMPTY_BBOX)
            x1 = observation_dict['min_cols'][object_indices]
            y1 = observation_dict['min_rows'][object_indices]
            x2 = observation_dict['max_cols'][object_indices]
            y2 = observation_dict['max_rows'][object_indices]
            area = (y2 - y1) * (x2 - x1)
            largest_area_oids = np.argmax(area, axis=0)
            bboxes = np.array(
                [
                    x1[largest_area_oids],
                    y1[largest_area_oids],
                    x2[largest_area_oids],
                    y2[largest_area_oids],
                    area[largest_area_oids],
                ]
            )
            bboxes[bboxes == -1] = 1000
            return bboxes

        bbox_1 = parse_biggest_bbox([oids.index(oid) for oid in tgt_1_ids])
        bbox_2 = parse_biggest_bbox([oids.index(oid) for oid in tgt_2_ids])
        bboxes_combined = np.concatenate([bbox_1, bbox_2], axis=0)
        bbox_to_return = bboxes_combined

        return bbox_to_return


class BestBboxSensorOnlineEval(Sensor):
    def __init__(
        self,
        convert_to_pixel_coords: bool = True,
        which_camera: Literal['nav', 'manip'] = 'nav',
        uuid: str = 'best_bbox',
        sensors_to_use: Optional[List[Sensor]] = None,
    ) -> None:
        observation_space = gym.spaces.Discrete(3)
        super().__init__(**prepare_locals_for_super(locals()))
        self.sensor_to_use = sensors_to_use or [TaskRelevantObjectBBoxSensorOnlineEval]

        self.convert_to_pixel_coords = convert_to_pixel_coords

        self.which_camera = which_camera

        self.task_relevant_oids = []
        self.task_relevant_synset_to_objects = {}
        self.oids_as_bytes = None
        self.synset_to_oids_as_bytes = None

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        bboxes = []
        for sensor in self.sensor_to_use:
            bboxes.append(
                sensor.get_observation(env, task, *args, **kwargs)[np.newaxis, np.newaxis, :]
            )
        assert len(self.sensor_to_use) == 2, 'can easily be extended to more than 2 sensors'
        bbox_to_return = get_best_of_two_bboxes(bboxes[0], bboxes[1])[0][0]
        return bbox_to_return


class MinL2TargetDistanceSensor(Sensor):
    def __init__(self, uuid: str = 'minimum_l2_target_distance') -> None:
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
        if not hasattr(task, 'min_l2_distance_to_target'):
            return np.array([-1], dtype=np.float64)
        return np.array([task.min_l2_distance_to_target()], dtype=np.float64)


class LastActionStrSensor(Sensor):
    def __init__(self, uuid: str = 'last_action_str', str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.last_taken_action_str, self.str_max_len)


class HouseNumberSensor(Sensor):
    def __init__(self, uuid: str = 'house_index') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(int(task.task_info['house_index']))


class GoalObjectTypeSensor(GoalObjectTypeThorSensor):
    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNaviThorGridTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        assert len(task.task_info['synsets']) == 1
        return self.object_type_to_ind[task.task_info['synsets'][0]]


class RoomsSeenSensor(Sensor):
    def __init__(self, uuid: str = 'rooms_seen') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(int(len(task.visited_and_left_rooms)))


class RoomCurrentSeenSensor(Sensor):
    def __init__(self, uuid: str = 'room_current_seen') -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(bool(task.get_current_room() in task.visited_and_left_rooms))


class CurrentAgentRoom(Sensor):
    def __init__(self, uuid: str = 'current_agent_room') -> None:
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
        try:
            room_num = int(task.get_current_room().replace('room|', ''))
        except ValueError:
            room_num = -1
        return np.array([room_num], dtype=np.int64)


class NumPixelsVisible(Sensor):
    def __init__(
        self, which_camera: Literal['nav', 'manip'], uuid: str = 'num_pixels_visible'
    ) -> None:
        uuid = f'{uuid}_{which_camera}'
        self.which_camera = which_camera
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
        num_pixels = 0

        if 'synsets' in task.task_info and len(task.task_info['synsets']) == 1:
            object_type = task.task_info['synsets'][0]
            object_ids = task.task_info['synset_to_object_ids'][object_type]
            num_pixels = 0

            visible_oids = env.get_visible_objects(
                which_camera=self.which_camera, maximum_distance=15
            )
            for oid in object_ids:
                if oid in visible_oids:
                    num_pixels += np.sum(
                        env.get_segmentation_mask_of_object(
                            object_id=oid, which_camera=self.which_camera
                        )
                    )

        return np.array([num_pixels], dtype=np.int64)


class TaskRelevantObjectBBoxSensorDummy(TaskRelevantObjectBBoxSensor):
    def __init__(
        self,
        which_camera: Literal['nav', 'manip'] = 'nav',
        convert_to_pixel_coords: bool = True,
        uuid: str = 'task_relevant_object_bbox',
        gpu_device=-1,
    ) -> None:
        super().__init__(
            which_camera=which_camera, convert_to_pixel_coords=convert_to_pixel_coords, uuid=uuid
        )

    def get_observation(self, env: StretchController, task: AbstractSPOCTask, *args, **kwargs):
        return np.array(EMPTY_DOUBLE_BBOX).astype(np.float32)


class TimeStepSensor(Sensor):
    def __init__(self, uuid: str = 'time_step', max_time_for_random_shift=0) -> None:
        observation_space = self._get_observation_space()
        self.max_time_for_random_shift = max_time_for_random_shift
        self.random_start = 0
        super().__init__(**prepare_locals_for_super(locals()))
        self._update = False

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def sample_random_start(self):
        self.random_start = random.randint(0, max(self.max_time_for_random_shift, 0))

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSPOCTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        steps = task.num_steps_taken()
        if self._update:
            steps += 1
        else:
            self._update = True
        if task.is_done():  # not increment at next episode start
            self._update = False
            self.sample_random_start()
        return np.array(self.random_start + int(steps), dtype=np.int64)


class TrajectorySensor(Sensor):
    def __init__(self, uuid: str = 'traj_index', max_idx: int = 4) -> None:
        observation_space = self._get_observation_space()
        self.curr_idx = 0
        self.max_idx = max_idx
        self._update = False
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
        if self._update:
            self.curr_idx += 1
            if self.curr_idx >= self.max_idx:
                self.curr_idx = 0
            self._update = False
        if task.is_done():  # update at next episode start
            self._update = True
        return np.array(self.curr_idx, dtype=np.int64)
