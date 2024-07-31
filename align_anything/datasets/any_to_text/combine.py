
from typing import Any, List, Optional, Callable
import math
import random
import itertools

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, BatchSampler

__all__ = [
    'CombinedDataset',
    'CombinedDatasetBatchSampler',
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

class CombinedDatasetBatchSampler(BatchSampler):
    def __init__(self, datasets: List[Dataset], batch_size: int, shuffle: bool = True, 
                 seed: int = 0, drop_last: bool = False) -> None:
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle
        self.datasets = datasets
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.dataset_lengths = [len(d) for d in datasets]
        
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            
        self.indices = self.combine_indices()
        
    def combine_indices(self) -> List[List[List[int]]]:
        indices_list = []
        start_index = 0
        for length in self.dataset_lengths:
            indices_dataset = []
            indices = list(range(start_index, start_index + length))
            start_index += length
            if self.shuffle:
                random.shuffle(indices)
            count = (length + self.batch_size - 1) // self.batch_size if not self.drop_last else length // self.batch_size
            
            for i in range(count):
                batch_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
                if len(batch_indices) < self.batch_size and not self.drop_last:
                    batch_indices += batch_indices[:self.batch_size - len(batch_indices)]
                if len(batch_indices) == self.batch_size:
                    indices_dataset.append(batch_indices)
                    
            if self.shuffle:
                random.shuffle(indices_dataset)
            indices_list.append(indices_dataset)
            
        return indices_list

    def __iter__(self)-> iter:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
        
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
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.dataset_lengths = [len(d) for d in datasets]
        
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            
        self.indices_per_dataset = self.combine_indices()
        self.num_samples_per_dataset = []
        self.total_size_per_dataset = []
        
        for indices in self.indices_per_dataset:
            if self.drop_last and len(indices) % self.num_replicas != 0:  
                num_samples = math.ceil(
                    (len(indices) - self.num_replicas) / self.num_replicas  
                )
            else:
                num_samples = math.ceil(len(indices) / self.num_replicas)
            self.num_samples_per_dataset.append(num_samples)
            
            total_size = num_samples * self.num_replicas
            self.total_size_per_dataset.append(total_size)

            if not self.drop_last:
                padding_size = total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[:total_size]
            
            assert len(indices) == total_size
        
    def combine_indices(self) -> List[List[List[int]]]:
        indices_list = []
        start_index = 0
        for length in self.dataset_lengths:
            indices_dataset = []
            indices = list(range(start_index, start_index + length))
            start_index += length
            if self.shuffle:
                random.shuffle(indices)
            count = (length + self.batch_size - 1) // self.batch_size if not self.drop_last else length // self.batch_size
            
            for i in range(count):
                batch_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
                if len(batch_indices) < self.batch_size and not self.drop_last:
                    batch_indices += batch_indices[:self.batch_size - len(batch_indices)]
                if len(batch_indices) == self.batch_size:
                    indices_dataset.append(batch_indices)
                    
            if self.shuffle:
                random.shuffle(indices_dataset)
            indices_list.append(indices_dataset)
            
        return indices_list


    def __iter__(self)-> iter:
        # subsample
        indices = []
        for i, indices_dataset in enumerate(self.indices_per_dataset):
            indices += indices_dataset[self.rank:self.total_size_per_dataset[i]:self.num_replicas]
        assert len(indices) == sum(self.num_samples_per_dataset)
        return iter(indices)

    def __len__(self) -> int:
        return sum(self.num_samples_per_dataset)
        
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
