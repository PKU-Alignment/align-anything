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


import gzip
import json
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import h5py
from tqdm import tqdm

from align_anything.utils.utils.string_utils import (
    convert_byte_to_string,
    get_natural_language_spec,
    json_templated_spec_to_dict,
)


def read_jsonlgz(path: str, max_lines: Optional[int] = None) -> List[bytes]:
    with gzip.open(path, 'r') as f:
        lines = []
        for line in tqdm(f, desc=f'Loading {path}'):
            lines.append(line)
            if max_lines is not None and len(lines) >= max_lines:
                break
    return lines


# create JsonType
JsonType = Union[str, bytes]


class LazyJsonDataset:
    """Lazily load a list of json data."""

    def __init__(self, data: List[JsonType]) -> None:
        """
        Inputs:
            data: a list of json documents
        """
        self.data = data
        self.cached_data: Dict[int, Union[List, Dict]] = {}

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        if index not in self.cached_data:
            self.cached_data[index] = json.loads(self.data[index])
        return self.cached_data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return 'LazyJsonDataset(num_samples={}, cached_samples={})'.format(
            len(self), len(self.cached_data)
        )

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            if i not in self.cached_data:
                self.cached_data[i] = json.loads(x)
            yield self.cached_data[i]

    def select(self, indices: Sequence[int]) -> 'LazyJsonDataset':
        """Return a new dataset containing only the given indices."""
        return LazyJsonDataset(
            data=[self.data[i] for i in indices],
        )


class LazyJsonHouses(LazyJsonDataset):
    """Lazily load the a list of json houses."""

    def __init__(self, data: List[JsonType]) -> None:
        super().__init__(data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return 'LazyJsonHouses(num_houses={}, cached_houses={})'.format(
            len(self), len(self.cached_data)
        )

    def select(self, indices: Sequence[int]) -> 'LazyJsonHouses':
        """Return a new dataset containing only the given indices."""
        return LazyJsonHouses(
            data=[self.data[i] for i in indices],
        )

    @staticmethod
    def from_jsonlgz(path: str, max_houses: Optional[int] = None) -> 'LazyJsonHouses':
        """Load the houses from a .jsonl.gz file."""
        return LazyJsonHouses(data=read_jsonlgz(path=path, max_lines=max_houses))

    @staticmethod
    def from_dir(
        house_dir: str,
        subset: Literal['train', 'val', 'test'],
        max_houses: Optional[int] = None,
    ) -> 'LazyJsonHouses':
        """Load the houses from a directory containing {subset}.jsonl.gz files."""
        return LazyJsonHouses.from_jsonlgz(
            path=os.path.join(house_dir, f'{subset}.jsonl.gz'),
            max_houses=max_houses,
        )


class LazyJsonTaskSpecs(LazyJsonDataset):
    """Lazily load a list of json task specs."""

    def __init__(self, data: List[JsonType]) -> None:
        super().__init__(data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return 'LazyJsonTaskSpecs(num_tasks={}, cached_tasks={})'.format(
            len(self), len(self.cached_data)
        )

    def select(self, indices: Sequence[int]) -> 'LazyJsonHouses':
        """Return a new dataset containing only the given indices."""
        return LazyJsonTaskSpecs(
            data=[self.data[i] for i in indices],
        )

    @staticmethod
    def from_jsonlgz(path: str, max_task_specs: Optional[int] = None) -> 'LazyJsonTaskSpecs':
        """Load the tasks from a .jsonl.gz file."""
        return LazyJsonTaskSpecs(data=read_jsonlgz(path=path, max_lines=max_task_specs))

    @staticmethod
    def from_dir(
        task_spec_dir: str,
        subset: Literal['train', 'val', 'test'],
        max_task_specs: Optional[int] = None,
    ) -> 'LazyJsonTaskSpecs':
        """Load the task specs from a directory containing {subset}.jsonl.gz files."""
        return LazyJsonTaskSpecs.from_jsonlgz(
            path=os.path.join(task_spec_dir, f'{subset}.jsonl.gz'),
            max_task_specs=max_task_specs,
        )


def load_hdf5_sensor(path):
    if not os.path.isfile(path):
        return []

    data = []
    with h5py.File(path, 'r') as d:
        for k in d.keys():
            j = json_templated_spec_to_dict(
                convert_byte_to_string(d[k]['templated_task_spec'][0, :])
            )
            j['house_index'] = int(d[k]['house_index'][0])
            last_agent_location = d[k]['last_agent_location'][0]
            j['agent_starting_position'] = [
                last_agent_location[0],
                last_agent_location[1],
                last_agent_location[2],
            ]
            j['agent_y_rotation'] = last_agent_location[4]
            j['natural_language_spec'] = get_natural_language_spec(j['task_type'], j)
            data.append(j)
    return data


class Hdf5TaskSpecs:
    """Load hdf5_sensors.hdf5 data stored as {dataset_dir}/*/hdf5_sensors.hdf5."""

    def __init__(
        self,
        subset_dir: str,
        data: Optional[List[Dict]] = None,
        proc_id: Optional[int] = None,
        total_procs: Optional[int] = None,
        max_house_id: Optional[int] = None,
        max_task_specs: Optional[int] = None,
    ) -> None:
        """
        Inputs:
            subset_dir: path to the directory containing subdirectories with hdf5_sensors.hdf5 files
        """
        self.subset_dir = subset_dir
        self.proc_id = proc_id if proc_id is not None else 0
        self.total_procs = total_procs if total_procs is not None else 1
        self.max_house_id = max_house_id

        if data is None:
            # subdirs are zfilled house ids
            subdirs = sorted(os.listdir(self.subset_dir))
            if self.max_house_id is not None:
                subdirs = [subdir for subdir in subdirs if int(subdir) < self.max_house_id]

            # select paths for the current process
            paths = [
                f'{self.subset_dir}/{subdir}/hdf5_sensors.hdf5'
                for i, subdir in enumerate(subdirs)
                if i % self.total_procs == self.proc_id
            ]
            self.data = self.read_hdf5_sensors(paths)
        else:
            self.data = data

        self.max_task_specs = max_task_specs if max_task_specs is not None else len(self.data)
        self.data = self.data[: self.max_task_specs]

    def read_hdf5_sensors(self, paths) -> List[Dict]:
        data = []
        desc = (
            f'[proc: {self.proc_id}/{self.total_procs}] '
            f'Loading hdf5_sensors.hdf5 files from {self.subset_dir}'
        )
        for path in tqdm(paths, desc=desc):
            data.extend(load_hdf5_sensor(path))

        return data

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        return self.data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return 'Hdf5TaskSpecs(num_samples={},proc_id={},total_procs={})'.format(
            len(self), self.proc_id, self.total_procs
        )

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            yield x

    def select(self, indices: Sequence[int]) -> 'Hdf5TaskSpecs':
        """Return a new dataset containing only the given indices."""
        return Hdf5TaskSpecs(
            subset_dir=self.subset_dir,
            data=[self.data[i] for i in indices],
            proc_id=self.proc_id,
            total_procs=self.total_procs,
        )

    def from_dataset_dir(
        dataset_dir: str,
        subset: Literal['train', 'val', 'test'],
        proc_id: Optional[int] = None,
        total_procs: Optional[int] = None,
        max_house_id: Optional[int] = None,
        max_task_specs: Optional[int] = None,
    ) -> 'Hdf5TaskSpecs':
        """Load the tasks from a directory containing {dataset_dir}/{subset}/*/hdf5_sensors.hdf5 files."""
        return Hdf5TaskSpecs(
            subset_dir=os.path.join(dataset_dir, subset),
            proc_id=proc_id,
            total_procs=total_procs,
            max_house_id=max_house_id,
            max_task_specs=max_task_specs,
        )


if __name__ == '__main__':
    from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR

    houses = LazyJsonHouses.from_dir(
        OBJAVERSE_HOUSES_DIR,
        subset='train',
        max_houses=10,
    )
    print(houses)
    # task_specs = LazyJsonTaskSpecs.from_dir(
    #     "/root/data/ObjectNavType_Poliformer",
    #     "train",
    #     max_task_specs=10,
    # )
    task_specs = Hdf5TaskSpecs.from_dataset_dir(
        '/root/vida_datasets/pointing_data/GoNearPoint', 'train', proc_id=2, total_procs=48
    )

    print(task_specs)

    import ipdb

    ipdb.set_trace()
