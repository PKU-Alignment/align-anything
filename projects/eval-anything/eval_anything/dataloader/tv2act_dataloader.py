# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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

# from eval_anything.dataloader.base_dataloader import BaseDataLoader

import json
import multiprocessing as mp
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np

from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import (
    LazyJsonDataset,
    Metadataset,
    process_and_load_data,
    read_jsonlgz,
)


class TV2ACTDataLoader:

    def __init__(self, data_cfgs):
        self.data_cfgs = data_cfgs

    def load_task_dataset(self) -> List[InferenceInput]:
        """Load the houses dataset."""
        task_type = self.data_cfgs.task_type
        path = self.data_cfgs.dataset_path
        eval_set_size = self.data_cfgs.eval_set_size
        seed = self.data_cfgs.seed
        shuffle = self.data_cfgs.shuffle
        task_type = self.data_cfgs.task_type
        tasks = process_and_load_data(task_type, path)
        tasks = LazyJsonDataset(data=tasks, dataset='chores')
        samples: List[EvalSample] = tasks

        sample_ids = list(range(len(samples)))
        if shuffle:
            random.seed(seed)
            random.shuffle(sample_ids)
        if eval_set_size is not None:
            sample_ids = sample_ids[:eval_set_size]

        normalized_samples = [
            eval_sample_to_normalized_eval_sample(task_type=task_type, sample=samples[i], index=i)
            for i in sample_ids
        ]
        tasks = [
            NormalizedEvalSample(
                sample_id=f"task={task_type},house={samples[i]['house_index']},sub_house_id={i}",
                house_id=str(samples[i]['house_index']).zfill(6),
                task_type=task_type,
                sub_house_id=i,
                needs_video=False,
                raw_navigation_camera='',
                sensors_path='',
                observations=Observations(
                    goal=samples[i]['natural_language_spec'],
                    initial_agent_location=np.array(
                        samples[i]['agent_starting_position']
                        + [0, samples[i]['agent_y_rotation'], 0]
                    ),
                    actions=[],
                    time_ids=[],
                    templated_task_type=json.dumps(samples[i]),
                ),
            )
            for i in sample_ids
        ]
        return normalized_samples, tasks

    def load_house_assets(
        self,
        num_workers: int = 0,
        max_houses_per_split: Optional[Union[int, Dict[str, int]]] = None,
    ) -> Metadataset:
        path = self.data_cfgs.house_assets_path
        house_type = self.data_cfgs.house_type

        if not isinstance(max_houses_per_split, Dict):
            max_houses_per_split = (lambda x: defaultdict(lambda: x))(max_houses_per_split)

        if num_workers < 0:
            num_workers = mp.cpu_count()

        house_strs = []
        path = os.path.join(path, house_type + '.jsonl.gz')
        house_strs = read_jsonlgz(
            path=path,
        )
        dd = LazyJsonDataset(data=house_strs, dataset='procthor-100k-house')

        return dd


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
) -> InferenceInput:
    return InferenceInput(
        task=f"task={task_type},house={sample['house_index']},sub_house_id={index}",
        conversation='123123',
        metadata=NormalizedEvalSample(
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
        ),
    )
