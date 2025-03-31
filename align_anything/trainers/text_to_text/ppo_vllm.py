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
"""Trainer for PPO training with vLLM generation."""


import argparse
import copy
import itertools
import os
import sys
from typing import Any

import ray

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from align_anything.configs.template import ChatTemplate
from align_anything.datasets import DummyDataset
from align_anything.utils.multi_process import is_main_process


from align_anything.datasets.text_to_text import (
    PromptOnlyBatch,
    PromptOnlyDataset,
    SupervisedDataset,
)

from align_anything.trainers.text_to_text.ppo import PPOTrainer
from align_anything.utils.vllm_utils.config import VLLMConfig

from align_anything.utils.device_utils import get_current_device, torch_gc, torch_set_device
from align_anything.utils.multi_process import (
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    read_cfgs,
    seed_everything,
    update_dict,
)

from align_anything.utils.vllm_utils.vllm_sampling import generate_with_vllm

from vllm.utils import get_ip, get_open_port


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)

class PPOVLLMTrainer(PPOTrainer):

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        # load training datasets
        self.prompt_only_dataloader, self.ptx_dataloader = (
            self.get_main_process_prompt_only_dataloader(PromptOnlyDataset, SupervisedDataset)
        )
        print('rank', dist.get_rank(), 'prompt_only_dataloader', len(self.prompt_only_dataloader))

    def init_models(self):
        # Call the original init_models method
        super().init_models()
        vllm_devices = os.environ.get('VLLM_DEVICES')
        num_actors = len(vllm_devices.split(','))
        current_rank = dist.get_rank()
        is_main_process = current_rank == 0
        # Initialize vLLM if enabled
        self.vllm_config = getattr(self.cfgs, 'vllm_cfgs', VLLMConfig())
        self.use_vllm = getattr(self.vllm_config, 'use_vllm', False)
        self.first_actor_init = True
        
        if self.use_vllm and is_main_process:
            try:
                import ray
                original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices
                from align_anything.utils.vllm_utils.vllm_engine import create_vllm_engines
                
                # Initialize Ray if not already initialized
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                
                # Create vLLM engines
                vllm_max_model_len = self.vllm_config.vllm_max_model_len or self.cfgs.model_cfgs.model_max_length
                
                self.vllm_engines = create_vllm_engines(
                    num_engines=self.vllm_config.vllm_num_engines,
                    tensor_parallel_size=self.vllm_config.vllm_tensor_parallel_size,
                    pretrain=self.cfgs.model_cfgs.actor_model_name_or_path,
                    seed=self.cfgs.train_cfgs.seed,
                    enable_prefix_caching=self.vllm_config.vllm_enable_prefix_caching,
                    enforce_eager=self.vllm_config.vllm_enforce_eager,
                    max_model_len=vllm_max_model_len,
                    num_total_actors=num_actors,
                    gpu_memory_utilization=self.vllm_config.vllm_gpu_memory_utilization,
                    vllm_enable_sleep=self.vllm_config.vllm_enable_sleep,
                )

                refs = [engine.init_process_group.remote(
                    master_address=get_ip(),
                    master_port=get_open_port(),
                    rank_offset=current_rank,
                    world_size=num_actors,
                    group_name="vllm_group",
                    backend="nccl",
                    use_ray=False,
                ) for engine in self.vllm_engines]
                ray.get(refs)

                if is_main_process:
                    print(f"Initialized {len(self.vllm_engines)} vLLM engines for accelerated sampling")
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
            except ImportError:
                print("vLLM or Ray not available. Falling back to standard generation.")
                self.use_vllm = False
                self.vllm_engines = None
        else:
            self.vllm_engines = None

    def actor_step(self, prompt_only_batch: PromptOnlyBatch):
        infer_batch = self.infer_batch(prompt_only_batch)
        actor_batch = copy.deepcopy(infer_batch)
        actor_batch['prompt_ids'] = prompt_only_batch['input_ids'].contiguous()

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        vllm_devices = os.environ.get('VLLM_DEVICES')
        os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices
        
        # Extract prompts from the batch
        input_ids = infer_batch.get('input_ids')
        batch_attention_mask = infer_batch.get('attention_mask')
        
        # Decode prompts
        prompts = []
        for i in range(input_ids.size(0)):
            mask = batch_attention_mask[i].bool()
            prompt_ids = input_ids[i, mask]
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            prompts.append(prompt)
        print('generating with vllm...')

        # Generate with vLLM on main process using VLLM_DEVICES
        sequences, attention_mask = generate_with_vllm(
            prompts=prompts,
            tokenizer=self.tokenizer,
            vllm_engines=self.vllm_engines,
            n_samples_per_prompt=1,
            temperature=self.cfgs.model_cfgs.temperature,
            top_p=self.cfgs.model_cfgs.top_p,
            top_k=-1,
            max_new_tokens=self.cfgs.model_cfgs.max_new_tokens,
            min_new_tokens=1,
            skip_special_tokens=False,
        )
        
        print(f"Successfully generated with vLLM on main process")
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        self.first_actor_init = False
        
        # Update actor batch with the broadcast results
        actor_batch['input_ids'] = sequences
        actor_batch['attention_mask'] = attention_mask
        
        return actor_batch

    def get_main_process_prompt_only_dataloader(
        self,
        train_data_dtype,
        ptx_data_dtype,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader]:
        """Get the dataloaders based on data_dtype."""
        formatter = self.processor if self.processor else self.tokenizer
        custom_formatter = (
            self.actor_model.apply_chat_template
            if hasattr(self.actor_model, 'apply_chat_template')
            else None
        )
        self.train_template = ChatTemplate(
            formatter, self.cfgs.data_cfgs.train_template, custom_formatter
        )
        self.eval_template = None

        if is_main_process():
            train_dataset = train_data_dtype(
                path=self.cfgs.data_cfgs.train_datasets,
                template=self.train_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name=self.cfgs.data_cfgs.train_name,
                size=self.cfgs.data_cfgs.train_size,
                split=self.cfgs.data_cfgs.train_split,
                data_files=self.cfgs.data_cfgs.train_data_files,
                optional_args=self.cfgs.data_cfgs.train_optional_args,
            )
            dataset_length = len(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.get_collator(),
                shuffle=True,
                batch_size=int(self.cfgs.train_cfgs.per_device_prompt_batch_size),
            )
        else:
            dataset_length = 0

        dist.barrier()
        torch.cuda.synchronize()

        # Broadcast the dataset length to all processes
        dataset_length_tensor = torch.tensor([dataset_length], dtype=torch.int64, device=get_current_device())
        dist.broadcast(dataset_length_tensor, src=0)
        dataset_length = dataset_length_tensor.item()
        if not is_main_process():
            # Create a dummy dataset with the received length
            train_dataloader = DataLoader(DummyDataset(dataset_length))

        # load ptx datasets
        self.use_ptx = self.cfgs.data_cfgs.ptx_datasets is not None
        if self.use_ptx:
            custom_formatter = (
                self.actor_model.apply_chat_template
                if hasattr(self.actor_model, 'apply_chat_template')
                else None
            )
            self.ptx_template = ChatTemplate(
                formatter, self.cfgs.data_cfgs.ptx_template, custom_formatter
            )
            ptx_dataset = ptx_data_dtype(
                path=self.cfgs.data_cfgs.ptx_datasets,
                template=self.ptx_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name=self.cfgs.data_cfgs.ptx_name,
                size=self.cfgs.data_cfgs.ptx_size,
                split=self.cfgs.data_cfgs.ptx_split,
                data_files=self.cfgs.data_cfgs.ptx_data_files,
                optional_args=self.cfgs.data_cfgs.ptx_optional_args,
            )
            ptx_dataloader = DataLoader(
                ptx_dataset,
                collate_fn=ptx_dataset.get_collator(),
                sampler=DistributedSampler(ptx_dataset, shuffle=True),
                batch_size=int(self.cfgs.train_cfgs.per_device_prompt_batch_size),
            )
        else:
            ptx_dataloader = DataLoader(DummyDataset(len(train_dataloader)))


        return train_dataloader, ptx_dataloader


    @torch.no_grad()
    def rollout(self, actor_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        # freeze the model for rolling out

        total_batch_size = actor_batch['input_ids'].size(0)
        micro_batch_size = int(self.cfgs.train_cfgs.per_device_train_batch_size)
        micro_inference_batches = []
        micro_training_batches = []
        mini_batch = {}
        for i in tqdm(range(0, total_batch_size, micro_batch_size), desc='Scoring batches and generating logprobs...', disable=not is_main_process()):

            mini_batch = {
                key: actor_batch[key][i : i + micro_batch_size] for key in actor_batch
            }
            # reward model and reward critic model scoring
            reward_batch = self.reward_model_step(mini_batch)
            # calculate the log probabilities
            logits = self.actor_model(**mini_batch).logits
            ref_logits = self.actor_reference_model(**mini_batch).logits
            log_probs = gather_log_probabilities(logits[:, :-1], mini_batch['input_ids'][:, 1:])
            ref_log_probs = gather_log_probabilities(
                ref_logits[:, :-1], mini_batch['input_ids'][:, 1:]
            )

            micro_training_batch = {}
            micro_training_batch['prompt_idx'] = mini_batch['prompt_ids'].size(-1) - 1
            micro_training_batch['log_probs'] = log_probs
            micro_training_batch['ref_log_probs'] = ref_log_probs
            micro_training_batch['reward'] = reward_batch['reward']
            micro_training_batch['reward_values'] = reward_batch['reward_values']

            mini_batch['input_ids'] = reward_batch['input_ids']
            # add rollout results to the batches
            micro_inference_batches.append(mini_batch)
            micro_training_batches.append(micro_training_batch)

        # unfreeze the model for training
        self.set_train()

        return micro_inference_batches, micro_training_batches


    def broadcast_batch(self, batch: dict[str, torch.Tensor], keys: list[str], dtypes: list[torch.dtype]) -> None:
        """Broadcast the batch to all processes."""
        for i, key in enumerate(keys):
            if is_main_process():
                tensor_shape = torch.tensor(batch[key].shape, dtype=torch.long, device=get_current_device())
            else:
                tensor_shape = torch.zeros(2, dtype=torch.long, device=get_current_device())

            dist.broadcast(tensor_shape, src=0)

            if not is_main_process():
                batch[key] = torch.zeros(tensor_shape.tolist(), dtype=dtypes[i], device=get_current_device())

            dist.broadcast(batch[key], src=0)

    def update_vllm_engines(self):
        """Update the weights of the vLLM engines."""
        torch_gc()
        model = self.actor_model.module
        count, num_params = 0, len(list(model.named_parameters()))
        if is_main_process():
            print(f"Updating vLLM engines...")
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param
            # Fire all vllm engines for broadcast
            if is_main_process():
                shape = param.shape if self.ds_train_cfgs['zero_optimization']['stage'] == 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)

        if is_main_process():
            print(f"Updated vLLM engines for generating next batch.")

        torch_gc()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        
    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')
        progress_bar = tqdm(
            total=self.total_training_steps,
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.eval()

        num_prompt_only_batches = len(self.prompt_only_dataloader) * self.cfgs.train_cfgs.per_device_prompt_batch_size
        num_ptx_batches = len(self.ptx_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches

        for epoch in range(int(self.cfgs.train_cfgs.epochs)):

            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.chain.from_iterable([self.ptx_dataloader] * num_ptx_replicas),
            ):
                self.set_train(mode=False)
                
                if is_main_process():
                    actor_batch = self.actor_step(prompt_only_batch)
                    print('Actor batch has been generated')
                else:
                    actor_batch = {
                        'prompt_ids': None,
                        'input_ids': None,
                        'attention_mask': None
                    }
                
                if dist.is_initialized():
                    
                    self.broadcast_batch(
                        actor_batch, 
                        ['prompt_ids', 'input_ids', 'attention_mask'], 
                        [torch.long, torch.long, torch.bool]
                    )
                
                dist.barrier()
                torch.cuda.synchronize()
                torch_gc()
                inference_batches, training_batches = self.rollout(actor_batch)
                if self.use_ptx:
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(inference_batches))]

                for _ in range(self.cfgs.train_cfgs.update_iters):
                    for idx in tqdm(range(len(inference_batches)), desc='Updating PPO with the scored batches...', disable=not is_main_process()):
                        inference_batch = inference_batches[idx]
                        training_batch = training_batches[idx]
                        ptx_batch = ptx_batches[idx]
                        rl_info = self.rl_step(inference_batch, training_batch)
                        torch_gc()

                        if self.use_ptx:
                            ptx_info = self.ptx_step(ptx_batch)
                            torch_gc()
                            self.logger.log(ptx_info, step=self.global_step)

                        # update the actor model weights for generating next batch
                        self.logger.log(rl_info, step=self.global_step)

                        save_interval = self.total_update_steps // self.cfgs.logger_cfgs.save_total_limit
                        self.global_step += 1
                        if self.global_step % save_interval == 0:
                            self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                            self.save(tag=self.global_step)
                            self.logger.print('Checkpoint saved.')

                self.update_vllm_engines()
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(reward {rl_info["train/reward"]:.4f})',
                )
                progress_bar.update(1)

def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'ppo_vllm')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    trainer = PPOVLLMTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
