# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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


from typing import Any, List, Optional, Callable
import math
import random
import itertools

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, BatchSampler

__all__ = [
    'CombinedDataset',
    'DistributedCombinedDatasetBatchSampler',
]

class CombinedDataset(Dataset):
    def __init__(self, datasets) -> None:
        assert all(isinstance(d, type(datasets[0])) for d in datasets), "All datasets must be of the same type"
        self.datasets = datasets
        self.cumulative_lengths = [0] + list(itertools.accumulate(len(d) for d in datasets))

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return self.datasets[0].get_collator()

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int|List[int]) -> dict[str, torch.Tensor]:
        if isinstance(idx, int):
            for i in range(len(self.datasets)):
                if idx < self.cumulative_lengths[i + 1]:
                    return self.datasets[i][idx - self.cumulative_lengths[i]]
        elif isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
        
class DistributedCombinedDatasetBatchSampler(BatchSampler):
    def __init__(self, datasets: List[Dataset], batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, 
                 drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        
        self.epoch = 0
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.datasets = datasets
        self.drop_last = drop_last
        self.local_batch_size = batch_size
        self.num_replicas = num_replicas
        self.global_batch_size = batch_size * num_replicas
        self.dataset_lengths = [len(d) for d in datasets]
        
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            
        self.indices = self.combine_indices()
        self.num_samples = len(self.indices)
        self.total_size = len(self.indices) * self.num_replicas
        
        
    def combine_indices(self) -> List[List[int]]:
        indices_list = []
        start_index = 0
        for length in self.dataset_lengths:
            indices_dataset = []
            indices = list(range(start_index, start_index + length))
            start_index += length
            if self.shuffle:
                random.shuffle(indices)
            count = (length + self.global_batch_size - 1) // self.global_batch_size if not self.drop_last else length // self.global_batch_size
            
            for i in range(count):
                global_batch_indices = indices[i * self.global_batch_size: (i + 1) * self.global_batch_size]
                if len(global_batch_indices) < self.global_batch_size and not self.drop_last:
                    global_batch_indices += global_batch_indices[:self.global_batch_size - len(batch_indices)]
                if len(global_batch_indices) == self.global_batch_size:
                    indices_dataset.append(global_batch_indices)
                    
            if self.shuffle:
                random.shuffle(indices_dataset)
            indices_list.extend(indices_dataset)
        if self.shuffle:
            random.shuffle(indices_list)
        
        return indices_list


    def __iter__(self)-> iter:
        # subsample
        indices = []
        for _, global_indices in enumerate(self.indices):
            global_indice = [global_indices[i * self.local_batch_size:(i + 1) * self.local_batch_size] for i in range(self.num_replicas)]
            indices.append(global_indice[self.rank])
        assert len(indices) == self.num_samples, f"rank {self.rank}: the num_sample is {self.num_samples}, but the length of indices is {len(indices)}."
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
        
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
