
from typing import Any, List, Optional, Callable
import math
import random
import itertools

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, BatchSampler

__all__ = [
    'CombinedDataset',
    'CombinedDatasetSampler'
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

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        for i in range(len(self.datasets)):
            if idx < self.cumulative_lengths[i+1]:
                return self.datasets[i][idx - self.cumulative_lengths[i]]

class CombinedDatasetBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, drop_last=False) -> None:
        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_lengths = [len(d) for d in datasets]

    def __iter__(self)-> iter:
        indices_list = []
        for length in self.dataset_lengths:
            indices = list(range(length))
            random.shuffle(indices)
            if self.drop_last:
                count = length // self.batch_size
            else:
                count = (length + self.batch_size - 1) // self.batch_size

            for i in range(count):
                batch_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    indices_list.append(batch_indices)
        random.shuffle(indices_list)
        
        for batch_indices in indices_list:
            yield batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return sum(length // self.batch_size for length in self.dataset_lengths)
        else:
            return sum((length + self.batch_size - 1) // self.batch_size for length in self.dataset_lengths)
        
class  DistributedCombinedDatasetBatchSampler(BatchSampler):
    def __init__(self, datasets: Dataset, batch_size: int, num_replicas: Optional[int] = None,
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
        self.num_replicas = num_replicas
        self.dataset_lengths = [len(d) for d in datasets]
        
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            
        self.indices = self.combine_indices()
        
        if self.drop_last and len(self.indices) % self.num_replicas != 0:  
            self.num_samples = math.ceil(
                (len(self.indices) - self.num_replicas) / self.num_replicas  
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)  
            
        self.total_size = self.num_samples * self.num_replicas

        if not self.drop_last:
            padding_size = self.total_size - len(self.indices)
            if padding_size <= len(self.indices):
                self.indices += self.indices[:padding_size]
            else:
                self.indices += (self.indices * math.ceil(padding_size / len(self.indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            self.indices = self.indices[:self.total_size]
            
        assert len(self.indices) == self.total_size
        
    def combine_indices(self) -> List[List[int]]:
        indices_list = []
        for length in self.dataset_lengths:
            indices = list(range(length))
            if self.shuffle:
                random.shuffle(indices)
            count = (length + self.batch_size - 1) // self.batch_size if not self.drop_last else length // self.batch_size
            
            for i in range(count):
                batch_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
                if len(batch_indices) < self.batch_size and not self.drop_last:
                    batch_indices += batch_indices[:self.batch_size - len(batch_indices)]
                if len(batch_indices) == self.batch_size:
                    indices_list.append(batch_indices)
                    
        if self.shuffle:
            random.shuffle(indices_list)
            
        return indices_list


    def __iter__(self)-> iter:
        # subsample
        indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(self.indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return sum(self.num_samples)
        
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
