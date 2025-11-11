# Copyright 2024 Allen Institute for AI
# ==============================================================================

import copy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from eval_anything.third_party.SPOC.tasks.task_specs import TaskSpec


def map_task_type(task_type: str) -> str:
    task_type_map = dict(SimpleExploreHouse='RoomVisit', ObjectNavOpenVocab='ObjectNavDescription')
    return task_type_map.get(task_type, task_type)  # or task_type


def inverse_map_task_type(task_type: str) -> str:
    task_type_map = dict(RoomVisit='SimpleExploreHouse', ObjectNavDescription='ObjectNavOpenVocab')
    return task_type_map.get(task_type) or task_type


def map_task_spec(task_spec: 'TaskSpec') -> 'TaskSpec':
    task_spec = copy.copy(task_spec)
    task_spec['task_type'] = map_task_type(task_spec['task_type'])
    return task_spec
