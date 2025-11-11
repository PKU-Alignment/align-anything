# Copyright 2024 Allen Institute for AI
# ==============================================================================

import abc
import multiprocessing as mp
import random
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
from torch.utils.data import Dataset

from eval_anything.third_party.SPOC.utils.online_evaluation_types_and_utils import (
    normalized_eval_sample_to_task_spec,
)
from eval_anything.third_party.SPOC.utils.task_type_mapping_utils import map_task_spec


class TaskSpec(TypedDict):
    task_type: str
    house_index: int
    natural_language_spec: str

    agent_starting_position: Union[np.ndarray, List[float]]  # xyz
    agent_y_rotation: float

    eval_info: Optional[Dict[str, Any]]


class TaskSpecDataset(Dataset[TaskSpec]):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> TaskSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> Union[int, float]:
        raise NotImplementedError


class TaskSpecSampler(abc.ABC):
    last_task_spec: Optional[TaskSpec]

    @abc.abstractmethod
    def next_task_spec(
        self, force_advance_scene: bool = False, house_index: Optional[int] = None
    ) -> TaskSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> Union[int, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def num_remaining(self) -> Union[int, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class TaskSpecSamplerDatasetWrapper(TaskSpecSampler):
    def __init__(self, task_spec_dataset: TaskSpecDataset):
        self.task_spec_dataset = task_spec_dataset
        self.dataset_iterator_index = -1
        self.last_task_spec: Optional[TaskSpec] = None

    def next_task_spec(
        self, force_advance_scene: bool = False, house_index: Optional[int] = None
    ) -> TaskSpec:
        assert (
            force_advance_scene is False and house_index is None
        ), 'force_advance_scene and house_index are not supported by TaskSpecSamplerDatasetWrapper.'
        self.dataset_iterator_index += 1
        self.last_task_spec = map_task_spec(self.task_spec_dataset[self.dataset_iterator_index])
        return self.last_task_spec

    def __len__(self) -> Union[int, float]:
        return len(self.task_spec_dataset)

    def num_remaining(self) -> Union[int, float]:
        return len(self.task_spec_dataset) - (self.dataset_iterator_index + 1)

    def reset(self):
        self.dataset_iterator_index = -1
        self.last_task_spec: Optional[TaskSpec] = None


class TaskSpecDatasetList(TaskSpecDataset):
    def __init__(self, task_specs: List[TaskSpec]) -> None:
        self.task_specs = task_specs

    def __getitem__(self, index: int) -> TaskSpec:
        return map_task_spec(self.task_specs[index])

    def __len__(self) -> Union[int, float]:
        return len(self.task_specs)


class TaskSpecDatasetInfiniteList(TaskSpecDataset):
    def __init__(self, task_specs: List[TaskSpec], shuffle: bool) -> None:
        self.shuffle = shuffle

        self.house_index_to_task_specs = {}

        task_specs = [map_task_spec(task_spec) for task_spec in task_specs]

        for task_spec in task_specs:
            if task_spec['house_index'] not in self.house_index_to_task_specs:
                self.house_index_to_task_specs[task_spec['house_index']] = []
            self.house_index_to_task_specs[task_spec['house_index']].append(task_spec)

        self.task_specs_full = task_specs
        self.task_specs = []
        self.last_task_spec = None
        self.last_index = -1

    def reset_task_specs(self):
        if self.shuffle:
            for l in self.house_index_to_task_specs.values():
                random.shuffle(l)

            specs_per_house = list(self.house_index_to_task_specs.values())
            random.shuffle(specs_per_house)
            self.task_specs = []
            for l in specs_per_house:
                self.task_specs.extend(l)
        else:
            self.task_specs = self.task_specs_full[:]

    def __getitem__(self, index: int) -> TaskSpec:
        if index not in [self.last_index + 1, self.last_index]:
            raise ValueError('TaskSpecDatasetInfiniteList can only be accessed sequentially')

        if index == self.last_index + 1:
            self.last_index = index

            if len(self.task_specs) == 0:
                self.reset_task_specs()

            self.last_task_spec = map_task_spec(self.task_specs.pop())

        return self.last_task_spec

    def __len__(self) -> Union[int, float]:
        return float('inf')


class TaskSpecSamplerInfiniteList(TaskSpecSampler):
    def __init__(
        self,
        house_index_to_task_specs: Dict[int, List[TaskSpec]],
        shuffle: bool,
        repeat_house_until_forced: bool,
    ) -> None:
        self.shuffle = shuffle
        self.repeat_house_until_forced = repeat_house_until_forced

        self.house_index_to_task_specs = {**house_index_to_task_specs}
        assert all(len(v) != 0 for v in self.house_index_to_task_specs.values())

        self.specs_for_current_house = []
        self.house_inds = []
        self.current_house_ind = None
        self.last_task_spec = None

    def reset_houses_inds_list(self):
        self.house_inds = list(self.house_index_to_task_specs.keys())

        if self.shuffle:
            random.shuffle(self.house_inds)

    def advance_house(self, force_advance_scene: bool, house_index: Optional[int]):
        if len(self.house_inds) == 0:
            self.reset_houses_inds_list()

        if house_index is not None:
            if house_index not in self.house_index_to_task_specs:
                raise ValueError(f'House index {house_index} not in `house_index_to_task_specs`')

            if house_index not in self.house_inds:
                self.reset_houses_inds_list()

            self.house_inds.remove(house_index)
            self.current_house_ind = house_index
        elif (
            force_advance_scene
            or self.current_house_ind is None
            or not self.repeat_house_until_forced
        ):
            self.current_house_ind = self.house_inds.pop()
        else:
            # If we're not being forced to advance, and we're repeating houses, then do nothing
            pass

        self.specs_for_current_house = [*self.house_index_to_task_specs[self.current_house_ind]]

        if self.shuffle:
            random.shuffle(self.specs_for_current_house)

    def next_task_spec(
        self, force_advance_scene: bool = False, house_index: Optional[int] = None
    ) -> TaskSpec:
        if force_advance_scene or len(self.specs_for_current_house) == 0 or house_index is not None:
            self.advance_house(force_advance_scene=force_advance_scene, house_index=house_index)

        self.last_task_spec = map_task_spec(self.specs_for_current_house.pop())
        return self.last_task_spec

    def __len__(self) -> Union[int, float]:
        return float('inf')

    def num_remaining(self) -> Union[int, float]:
        return float('inf')

    def reset(self):
        self.specs_for_current_house.clear()
        self.house_inds.clear()
        self.current_house_ind = None
        self.last_task_spec = None


class TaskSpecQueue(TaskSpecSampler):
    def __init__(self, queue: mp.Queue):
        self.queue = queue
        self.last_task_spec = None

    def next_task_spec(
        self, force_advance_scene: bool = False, house_index: Optional[int] = None
    ) -> TaskSpec:
        self.last_task_spec = normalized_eval_sample_to_task_spec(self.queue.get(timeout=5))
        return self.last_task_spec

    def __len__(self) -> Union[int, float]:
        return float('inf')

    def num_remaining(self) -> Union[int, float]:
        return float('inf')

    def reset(self):
        self.last_task_spec = None


class TaskSpecList(TaskSpecSampler):
    def __init__(self, tasksets: List):
        self.tasksets = tasksets
        self.last_task_spec = None
        self.processing_task_id = 0

    def next_task_spec(
        self, force_advance_scene: bool = False, house_index: Optional[int] = None
    ) -> TaskSpec:
        self.last_task_spec = normalized_eval_sample_to_task_spec(
            self.tasksets[self.processing_task_id]
        )
        self.processing_task_id += 1
        return self.last_task_spec

    def __len__(self) -> Union[int, float]:
        return float('inf')

    def num_remaining(self) -> Union[int, float]:
        return float('inf')

    def reset(self):
        self.last_task_spec = None
