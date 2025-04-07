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
"""vLLM worker wrapper for model weight updates."""

import logging

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.worker.worker import Worker


logger = logging.getLogger(__name__)


def get_physical_gpu_id():
    """Get the physical GPU ID."""
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.current_device()


class WorkerWrap(Worker):
    """Worker wrapper for vLLM."""

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend='nccl',
        use_ray=False,
    ):
        """Init torch process group for model weights update"""
        assert (
            torch.distributed.is_initialized()
        ), f'default torch process group must be initialized'
        assert group_name != '', f'group name must not be empty'

        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(
                world_size=world_size, rank=rank, backend=backend, group_name=group_name
            )
            self._model_update_group = group_name
        else:
            self._model_update_group = self._init_process_group(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                rank=rank,
            )
        self._model_update_with_ray = use_ray
        logger.info(
            f'init_process_group: master_address={master_address}, master_port={master_port}, '
            f'rank={rank}, world_size={world_size}, group_name={group_name}'
        )

    def _init_process_group(self, master_address, master_port, world_size, rank):
        """Initialize process group."""
        # Create a new process group
        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=rank,
            world_size=world_size,
        )
        pynccl = PyNcclCommunicator(pg, device=self.device)
        return pynccl

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            logger.info(f'update weight: {name}, dtype: {dtype}, shape: {shape}')

        assert (
            dtype == self.model_config.dtype
        ), f'mismatch dtype: src {dtype}, dst {self.model_config.dtype}'
        weight = torch.empty(shape, dtype=dtype, device='cuda')
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        del weight
