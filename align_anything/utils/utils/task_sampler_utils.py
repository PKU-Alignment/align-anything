from typing import List, Optional, Tuple

from torch.distributions.utils import lazy_property
from tqdm import tqdm

from allenact.utils.system import get_logger
from utils.data_utils import LazyJsonHouses, LazyJsonTaskSpecs


class TaskSpecPartitioner:
    def __init__(
        self,
        task_specs: LazyJsonTaskSpecs,
        houses: LazyJsonHouses,
        process_ind: int,
        total_processes: int,
        max_houses: Optional[int] = None,
    ):
        self.task_specs = task_specs
        self.houses = houses
        self.process_ind = process_ind
        self.total_processes = total_processes
        self.max_houses = max_houses

        if self.total_processes > len(self.houses):
            raise RuntimeError(
                f"Cannot have `total_processes > len(houses)`"
                f" ({self.total_processes} > {len(self.houses)})."
            )
        elif len(self.houses) % self.total_processes != 0:
            if self.process_ind == 0:  # Only print warning once
                get_logger().warning(
                    f"Number of houses {len(self.houses)} is not cleanly divisible by "
                    f"the number of processes ({self.total_processes}). "
                    f"So, not all processes will be fed the same number of houses."
                )

    @lazy_property
    def house_inds_for_curr_process(self) -> List[int]:
        """Returns houses and house_inds for the current process"""
        desc = f"Selecting house indices for process {self.process_ind}"
        if self.max_houses is None:
            house_inds = [
                task_spec["house_index"] for task_spec in tqdm(self.task_specs, desc=desc)
            ]
        else:
            house_inds = [
                task_spec["house_index"]
                for task_spec in tqdm(self.task_specs, desc=desc)
                if task_spec["house_index"] < self.max_houses
            ]

        house_inds_for_curr_process = [
            ind for i, ind in enumerate(house_inds) if i % self.total_processes == self.process_ind
        ]

        return house_inds_for_curr_process

    @lazy_property
    def houses_for_curr_process(self) -> LazyJsonHouses:
        """Returns houses for the current process"""
        return self.houses.select(self.house_inds_for_curr_process)

    @lazy_property
    def task_specs_for_curr_process(self) -> LazyJsonTaskSpecs:
        """Returns tasks for the current process"""
        unique_house_inds = set(self.house_inds_for_curr_process)
        desc = f"Selecting task specs for process {self.process_ind}"
        task_specs_for_curr_process = [
            task_spec
            for task_spec in tqdm(self.task_specs, desc=desc)
            if task_spec["house_index"] in unique_house_inds
        ]
        return task_specs_for_curr_process
