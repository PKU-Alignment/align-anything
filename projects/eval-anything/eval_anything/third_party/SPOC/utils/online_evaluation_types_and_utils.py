# Copyright 2024 Allen Institute for AI
# ==============================================================================

import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, TypedDict, Union

import numpy as np

from eval_anything.third_party.SPOC.utils.data_generation_utils.navigation_utils import (
    get_room_id_from_location,
)
from eval_anything.third_party.SPOC.utils.task_type_mapping_utils import (
    map_task_spec,
    map_task_type,
)
from eval_anything.third_party.SPOC.utils.type_utils import REGISTERED_TASK_PARAMS


if TYPE_CHECKING:
    from tasks.task_specs import TaskSpec


class EvalSample(TypedDict):
    task_type: str
    house_index: int
    natural_language_spec: str

    agent_starting_position: List[float]
    agent_y_rotation: float

    expert_length_bucket: Literal['long', 'medium', 'short']
    expert_length: int
    synsets: List[str]
    synset_to_object_ids: Dict[str, List[str]]
    broad_synset_to_object_ids: Dict[str, List[str]]
    extras: Dict[str, Any]
    task_path: str
    hypernyms: List[str]


class Observations(TypedDict):
    goal: str
    initial_agent_location: Union[np.ndarray, List[float]]  # 6 floats (xyz + 0rotation0)
    actions: List[str]
    time_ids: List[int]
    templated_task_type: str


class NormalizedEvalSample(TypedDict):
    task_type: str
    house_id: str

    sample_id: str

    sub_house_id: int
    needs_video: bool
    raw_navigation_camera: str
    sensors_path: str

    observations: Observations


def eval_sample_to_normalized_eval_sample(
    task_type: str, sample: EvalSample, index: int
) -> NormalizedEvalSample:
    if 'task_type' in sample:
        assert task_type == map_task_type(sample['task_type'])

    return NormalizedEvalSample(
        sample_id=f"task={task_type},house={sample['house_index']},sub_house_id={index}",
        house_id=str(sample['house_index']).zfill(6),
        task_type=task_type,
        sub_house_id=index,
        needs_video=False,
        raw_navigation_camera='',
        sensors_path='',
        observations=Observations(
            goal=sample['natural_language_spec'],
            initial_agent_location=np.array(
                sample['agent_starting_position'] + [0, sample['agent_y_rotation'], 0]
            ),
            actions=[],
            time_ids=[],
            templated_task_type=json.dumps(sample),
        ),
    )


def normalized_eval_sample_to_task_spec(s: NormalizedEvalSample) -> 'TaskSpec':
    templated_task_info = json.loads(s['observations']['templated_task_type'])
    task_spec = {
        'task_type': s['task_type'],
        'house_index': int(s['house_id']),
        'natural_language_spec': s['observations']['goal'],
        'agent_starting_position': s['observations']['initial_agent_location'][:3],
        'agent_y_rotation': float(s['observations']['initial_agent_location'][-2]),
        'eval_info': {
            'sample_id': s['sample_id'],
            'needs_video': s['needs_video'],
            **templated_task_info,
        },
    }

    task_spec = map_task_spec(task_spec)

    for key in REGISTERED_TASK_PARAMS.get(s['task_type']):
        try:
            task_spec[key] = templated_task_info[key]
        except KeyError:
            raise KeyError(
                f"Key {key} not found in {templated_task_info}, but is required by {s['task_type']}"
            )

    return task_spec


def calc_trajectory_room_visitation(room_poly_map, trajectory):
    visited_rooms = []
    for t in trajectory:
        room = get_room_id_from_location(room_poly_map, t.tolist())
        visited_rooms.append(room)
    visited_rooms = set(visited_rooms)
    percentage_visited = len(visited_rooms) / (len(room_poly_map) + 1e-9)
    total_visited = len(visited_rooms)
    return percentage_visited, total_visited
