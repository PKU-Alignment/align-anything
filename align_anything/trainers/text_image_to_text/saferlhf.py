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
"""Trainer for Safe RLHF training."""


import argparse
import copy
import itertools
import os
import sys
from collections import deque
from typing import Any

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_image_to_text import (
    PromptOnlyBatch,
    PromptOnlyDataset,
    SupervisedDataset,
)
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.text_to_text.ppo import PPOTrainer as PPOTextTrainer
from align_anything.utils.multi_process import (
    get_all_reduce_max,
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    batch_retokenize,
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    is_same_tokenizer,
    masked_mean,
    move_padding_left,
    prepare_ds_eval_cfgs,
    prepare_ds_train_cfgs,
    read_cfgs,
    remove_pad_tokens,
    seed_everything,
    update_dict,
)


class SafeRLHFVTrainer(PPOTextTrainer):
    """Trainer base class for Safe RLHF-V training."""

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.ds_eval_cfgs = prepare_ds_eval_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0

        self.init_check()
        dist.barrier()

        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}
        self.reward_infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}
        self.init_models_with_cost_model()
        if hasattr(self.actor_model, 'infer_batch'):
            self.infer_batch = self.actor_model.infer_batch
        if hasattr(self.reward_model, 'infer_batch'):
            self.reward_infer_batch = self.reward_model.infer_batch

        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

        self.kl_coeff = self.cfgs.train_cfgs.kl_coeff
        self.clip_range_ratio = self.cfgs.train_cfgs.clip_range_ratio
        self.clip_range_score = self.cfgs.train_cfgs.clip_range_score
        self.clip_range_value = self.cfgs.train_cfgs.clip_range_value
        self.ptx_coeff = self.cfgs.train_cfgs.ptx_coeff
        self.gamma = self.cfgs.train_cfgs.gamma
        self.gae_lambda = self.cfgs.train_cfgs.gae_lambda
        self.lambda_init = self.cfgs.train_cfgs.lambda_init
        self.lambda_max = self.cfgs.train_cfgs.lambda_max
        self.lambda_lr = self.cfgs.train_cfgs.lambda_lr
        self.lambda_update_delay_steps = self.cfgs.train_cfgs.lambda_update_delay_steps
        self.episode_cost_window_size = self.cfgs.train_cfgs.episode_cost_window_size
        self.threshold = self.cfgs.train_cfgs.threshold

        self.log_lambda = torch.nn.Parameter(
            torch.tensor(np.log(self.lambda_init), device=get_current_device()),
            requires_grad=True,
        )
        self.log_lambda_optimizer = torch.optim.SGD([self.log_lambda], lr=self.lambda_lr)
        self.log_lambda_max = np.log(self.lambda_max) if self.lambda_max else None
        self.episode_costs = deque(maxlen=self.episode_cost_window_size)

    def init_deepspeed_engines_with_cost_model(self) -> None:
        """Initialize DeepSpeed engines."""
        self.total_training_steps: int = (
            len(self.prompt_only_dataloader)
            * self.cfgs.train_cfgs.epochs
            * self.cfgs.train_cfgs.update_iters
            * self.cfgs.train_cfgs.per_device_prompt_batch_size
            // self.cfgs.train_cfgs.per_device_train_batch_size
        )
        # initialize the actor model engines
        actor_ds_cfgs = copy.deepcopy(self.ds_train_cfgs)
        actor_total_training_steps = self.total_training_steps
        if self.use_ptx:
            actor_ds_cfgs['train_batch_size'] *= 2
            actor_ds_cfgs['gradient_accumulation_steps'] *= 2
            actor_total_training_steps *= 2
        self.actor_model = self._init_train_deepspeed_engine(
            model=self.actor_model,
            weight_decay=self.cfgs.train_cfgs.actor_weight_decay,
            lr=self.cfgs.train_cfgs.actor_lr,
            lr_scheduler_type=self.cfgs.train_cfgs.actor_lr_scheduler_type,
            lr_warmup_ratio=self.cfgs.train_cfgs.actor_lr_warmup_ratio,
            total_training_steps=actor_total_training_steps,
            ds_cfgs=actor_ds_cfgs,
        )
        # initialize the actor reference model engines
        self.actor_reference_model = self._init_eval_deepspeed_engine(
            model=self.actor_reference_model,
            ds_cfgs=self.ds_eval_cfgs,
        )
        self.actor_reference_model.eval()
        # initialize the critic model engines
        self.reward_critic_model = self._init_train_deepspeed_engine(
            model=self.reward_critic_model,
            weight_decay=self.cfgs.train_cfgs.critic_weight_decay,
            lr=self.cfgs.train_cfgs.critic_lr,
            lr_scheduler_type=self.cfgs.train_cfgs.critic_lr_scheduler_type,
            lr_warmup_ratio=self.cfgs.train_cfgs.critic_lr_warmup_ratio,
            total_training_steps=self.total_training_steps,
            ds_cfgs=self.ds_train_cfgs,
        )
        self.cost_critic_model = self._init_train_deepspeed_engine(
            model=self.cost_critic_model,
            weight_decay=self.cfgs.train_cfgs.critic_weight_decay,
            lr=self.cfgs.train_cfgs.critic_lr,
            lr_scheduler_type=self.cfgs.train_cfgs.critic_lr_scheduler_type,
            lr_warmup_ratio=self.cfgs.train_cfgs.critic_lr_warmup_ratio,
            total_training_steps=self.total_training_steps,
            ds_cfgs=self.ds_train_cfgs,
        )
        self.reward_model = self._init_eval_deepspeed_engine(
            model=self.reward_model,
            ds_cfgs=self.ds_eval_cfgs,
        )
        self.cost_model = self._init_eval_deepspeed_engine(
            model=self.cost_model,
            ds_cfgs=self.ds_eval_cfgs,
        )
        self.cost_model.eval()
        self.reward_model.eval()
        self.cost_critic_model.train()
        # setup the gradient checkpointing
        if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
            self.actor_model.gradient_checkpointing_enable()
        if self.cfgs.train_cfgs.critic_gradient_checkpointing and not self.lora_enabled:
            self.reward_critic_model.gradient_checkpointing_enable()
        if self.cfgs.train_cfgs.critic_gradient_checkpointing and not self.lora_enabled:
            self.cost_critic_model.gradient_checkpointing_enable()

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines_with_cost_model()

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        # load training datasets
        self.prompt_only_dataloader, self.eval_dataloader, self.ptx_dataloader = (
            self.get_dataloaders(PromptOnlyDataset, PromptOnlyDataset, SupervisedDataset)
        )

    def init_models_with_cost_model(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        # loading actor model
        self.actor_model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        self.tokenizer.model_max_length = self.cfgs.model_cfgs.model_max_length
        # loading actor reference model
        self.actor_reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward model
        self.reward_model, self.reward_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward critic model
        self.reward_critic_model, self.reward_critic_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_critic_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        self.cost_model, self.cost_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.cost_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading cost critic model
        self.cost_critic_model, self.cost_critic_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.cost_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

        # initial checking
        if is_same_tokenizer(self.tokenizer, self.cost_tokenizer):
            self.reward_tokenizer = self.tokenizer
        if not is_same_tokenizer(self.tokenizer, self.reward_critic_tokenizer):
            raise ValueError(
                (
                    'Reward critic tokenizer must be the same as actor tokenizer. '
                    'Expected {0.__module__}.{0.__qualname__}(vocab_size={1}), '
                    'but got {2.__module__}.{2.__qualname__}(vocab_size={3}). '
                    'Please consider pass `--reward_critic_model_name_or_path` from the command line.'
                ).format(
                    type(self.tokenizer),
                    len(self.tokenizer),
                    type(self.reward_critic_tokenizer),
                    len(self.reward_critic_tokenizer),
                ),
            )

        # training setup
        self.reward_critic_tokenizer = self.tokenizer
        self.cost_tokenizer = self.tokenizer
        self.cost_critic_tokenizer = self.tokenizer
        self.generation_config = GenerationConfig(
            max_new_tokens=self.cfgs.model_cfgs.max_new_tokens,
            temperature=self.cfgs.model_cfgs.temperature,
            top_p=self.cfgs.model_cfgs.top_p,
            repetition_penalty=self.cfgs.model_cfgs.repetition_penalty,
            do_sample=True,
        )

    def actor_step(
        self, mini_prompt_only_batch: PromptOnlyBatch
    ) -> list[dict[str, Any], list[int]]:
        infer_batch = self.infer_batch(mini_prompt_only_batch)
        actor_batch = copy.deepcopy(infer_batch)
        sequences = self.actor_model.module.generate(
            **infer_batch,
            generation_config=self.generation_config,
            synced_gpus=True,
            do_sample=True,
        )
        sequences = move_padding_left(sequences.contiguous(), self.tokenizer.pad_token_id)
        attention_mask = sequences.not_equal(self.tokenizer.pad_token_id)
        actor_batch['input_ids'] = sequences
        actor_batch['attention_mask'] = attention_mask

        response_lens = []
        batch_size = sequences.size(0)
        for idx in range(batch_size):
            prompt_length = len(
                remove_pad_tokens(
                    mini_prompt_only_batch['input_ids'][idx].squeeze().tolist(),
                    self.tokenizer.pad_token_id,
                )
            )
            sequence_wo_pad = remove_pad_tokens(
                sequences[idx].squeeze().tolist(), self.tokenizer.pad_token_id
            )
            response = sequence_wo_pad[prompt_length:]
            response_lens.append(len(response))
        return actor_batch, response_lens

    def cost_model_step(self, actor_batch: PromptOnlyBatch) -> dict[str, Any]:
        cost_batch = copy.deepcopy(actor_batch)
        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                actor_batch['input_ids'],
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=get_current_device(),
            )
            cost_batch['input_ids'] = cost_tokenize_output['input_ids']
            cost_batch['attention_mask'] = cost_tokenize_output['attention_mask']

        cost_infer_batch = self.reward_infer_batch(cost_batch)
        cost_batch['cost'] = self.cost_model(**cost_infer_batch).end_scores.squeeze(dim=-1)
        critic_infer_batch = self.reward_infer_batch(actor_batch)
        scores = self.cost_critic_model(**critic_infer_batch).scores
        cost_batch['cost_values'] = scores.squeeze(dim=-1)[:, :-1]
        self.episode_costs.extend(cost_batch['cost'].tolist())

        return cost_batch

    @torch.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        # freeze the model for rolling out
        self.set_train(mode=False)

        micro_inference_batches = []
        micro_training_batches = []
        mini_batch = prompt_only_batch.copy()
        # actor generation
        actor_batch, response_lens = self.actor_step(mini_batch)
        # reward model and reward critic model scoring
        reward_batch = self.reward_model_step(actor_batch)
        cost_batch = self.cost_model_step(actor_batch)
        # calculate the log probabilities
        logits = self.actor_model(**actor_batch).logits
        ref_logits = self.actor_reference_model(**actor_batch).logits

        logprob_list = []
        ref_logprob_list = []
        reward_value_list = []
        cost_value_list = []

        batch_size = logits.size(0)

        for idx in range(batch_size):
            response_length = response_lens[idx]
            input_id = actor_batch['input_ids'][idx, 1:][-response_length:].unsqueeze(0)

            logit = logits[idx, :-1][-response_length:].unsqueeze(0)
            ref_logit = ref_logits[idx, :-1][-response_length:].unsqueeze(0)
            reward_value = reward_batch['reward_values'][idx][-response_length:].unsqueeze(0)
            cost_value = cost_batch['cost_values'][idx][-response_length:].unsqueeze(0)

            logprob = gather_log_probabilities(logit, input_id).squeeze()
            if logprob.dim() == 0:
                logprob = torch.cat([logprob.unsqueeze(0), logprob.new_zeros(2)])
            logprob_list.append(logprob)

            ref_logprob = gather_log_probabilities(ref_logit, input_id).squeeze()
            if ref_logprob.dim() == 0:
                ref_logprob = torch.cat([ref_logprob.unsqueeze(0), ref_logprob.new_zeros(2)])
            ref_logprob_list.append(ref_logprob)

            reward_value = reward_value.squeeze()
            if reward_value.dim() == 0:
                reward_value = torch.cat([reward_value.unsqueeze(0), reward_value.new_zeros(2)])
            reward_value_list.append(reward_value)

            cost_value = cost_value.squeeze()
            if cost_value.dim() == 0:
                cost_value = torch.cat([cost_value.unsqueeze(0), cost_value.new_zeros(2)])
            cost_value_list.append(cost_value)

        log_probs = torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        ref_log_probs = torch.nn.utils.rnn.pad_sequence(
            ref_logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        reward_values = torch.nn.utils.rnn.pad_sequence(
            reward_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        cost_values = torch.nn.utils.rnn.pad_sequence(
            cost_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        response_mask = (log_probs != 0).bool().to(logits.device)

        micro_training_batch = {}
        micro_training_batch['response_lens'] = response_lens
        micro_training_batch['log_probs'] = log_probs
        micro_training_batch['ref_log_probs'] = ref_log_probs
        micro_training_batch['reward'] = reward_batch['reward']
        micro_training_batch['reward_values'] = reward_values
        micro_training_batch['cost'] = cost_batch['cost']
        micro_training_batch['cost_values'] = cost_values
        micro_training_batch['response_mask'] = response_mask

        mini_batch['input_ids'] = reward_batch['input_ids']
        mini_batch['attention_mask'] = actor_batch['attention_mask']
        # add rollout results to the batches
        micro_inference_batches.append(mini_batch)
        micro_training_batches.append(micro_training_batch)

        # unfreeze the model for training
        self.set_train()

        return micro_inference_batches, micro_training_batches

    def actor_loss_fn_with_cost(
        self,
        log_probs: torch.Tensor,  # size = (B, L - S)
        old_log_probs: torch.Tensor,  # size = (B, L - S)
        reward_advantages: torch.Tensor,  # size = (B, L - S)
        cost_advantages: torch.Tensor,  # size = (B, L - S)
        mask: torch.BoolTensor,  # size = (B, L - S)
    ) -> torch.Tensor:  # size = ()
        # size = (B, L - S)
        multiplier = self.log_lambda.exp().item()
        advantages = (reward_advantages - multiplier * cost_advantages) / (1.0 + multiplier)
        ratios = torch.exp(log_probs - old_log_probs)
        surrogate1 = advantages * ratios
        surrogate2 = advantages * torch.clamp(
            ratios,
            1.0 - self.clip_range_ratio,
            1.0 + self.clip_range_ratio,
        )
        surrogate = torch.minimum(surrogate1, surrogate2)
        return -masked_mean(surrogate, mask)  # size = ()

    def add_kl_divergence_regularization_with_cost(
        self,
        reward: torch.Tensor,  # size = (B,)
        cost: torch.Tensor,  # size = (B,)
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # size = (B, L)
        end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)

        # size = (B, L)
        kl_divergence_estimate = log_probs - ref_log_probs
        kl_penalty_rewards = -self.kl_coeff * kl_divergence_estimate
        rewards = torch.scatter_add(
            kl_penalty_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        )
        costs = torch.scatter_add(
            -kl_penalty_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=cost.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        )
        return (
            torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score),
            torch.clamp(costs, min=-self.clip_range_score, max=self.clip_range_score),
        )

    def rl_step(
        self, inference_batch: dict[str, torch.Tensor], training_batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Perform a single update step with RL loss."""
        current_device = get_current_device()
        episode_cost = torch.tensor(self.episode_costs).mean().to(current_device)

        dist.reduce(episode_cost, dst=0, op=dist.ReduceOp.AVG)

        if is_main_process() and self.global_step >= self.lambda_update_delay_steps:
            lambda_loss = -(episode_cost - self.threshold) * self.log_lambda.exp()
            lambda_loss = torch.clamp(lambda_loss, min=-1e6, max=1e6)
            self.log_lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.log_lambda_optimizer.step()
            if self.log_lambda_max is not None:
                with torch.no_grad():
                    self.log_lambda.clamp_(max=self.log_lambda_max)
        dist.broadcast(self.log_lambda, src=0)
        response_lens = training_batch['response_lens']
        old_log_probs = training_batch['log_probs']
        ref_log_probs = training_batch['ref_log_probs']
        reward = training_batch['reward']
        cost = training_batch['cost']
        old_reward_values = training_batch['reward_values']
        old_cost_values = training_batch['cost_values']
        response_mask = training_batch['response_mask']

        input_ids = inference_batch['input_ids']

        sequence_mask = torch.ones_like(response_mask, dtype=torch.bool)
        batch_size = sequence_mask.size(0)
        new_size = min(sequence_mask.size(-1), old_reward_values.size(-1), old_cost_values.size(-1))

        sequence_mask = sequence_mask[:, :new_size]

        old_reward_values = old_reward_values[:, :new_size]
        old_cost_values = old_cost_values[:, :new_size]
        with torch.no_grad():
            old_rewards, old_costs = self.add_kl_divergence_regularization_with_cost(
                reward,
                cost,
                old_log_probs,
                ref_log_probs,
                sequence_mask,
            )

            reward_advantages, reward_returns = self.get_advantages_and_returns(
                old_reward_values,
                old_rewards,
                sequence_mask,
                start=0,
            )
            cost_advantages, cost_returns = self.get_advantages_and_returns(
                old_cost_values,
                old_costs,
                sequence_mask,
                start=0,
            )
        logits = self.actor_model(**inference_batch, use_cache=False).logits
        logprob_list = []

        for idx in range(batch_size):
            response_length = response_lens[idx]
            input_id = input_ids[idx, 1:][-response_length:].unsqueeze(0)
            logit = logits[idx, :-1][-response_length:].unsqueeze(0)

            logprob = gather_log_probabilities(logit, input_id).squeeze()
            if logprob.dim() == 0:
                logprob = torch.cat([logprob.unsqueeze(0), logprob.new_zeros(2)])
            logprob_list.append(logprob)

        log_probs = torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        actor_loss = self.actor_loss_fn_with_cost(
            log_probs,
            old_log_probs,
            reward_advantages,
            cost_advantages,
            sequence_mask,
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        raw_reward_values = self.reward_critic_model(**inference_batch).scores
        raw_reward_values = raw_reward_values.squeeze(dim=-1)[:, :-1]
        raw_cost_values = self.cost_critic_model(**inference_batch).scores
        raw_cost_values = raw_cost_values.squeeze(dim=-1)[:, :-1]

        reward_value_list = []
        cost_value_list = []

        for idx in range(batch_size):
            response_length = response_lens[idx]
            reward_value = raw_reward_values[idx][-response_length:].unsqueeze(0)
            reward_value = reward_value.squeeze()
            if reward_value.dim() == 0:
                reward_value = torch.cat([reward_value.unsqueeze(0), reward_value.new_zeros(2)])
            reward_value_list.append(reward_value)
        reward_values = torch.nn.utils.rnn.pad_sequence(
            reward_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)

        reward_critic_loss = self.critic_loss_fn(
            reward_values,
            old_reward_values,
            reward_returns,
            sequence_mask,
        )
        self.reward_critic_model.backward(reward_critic_loss)
        self.reward_critic_model.step()

        for idx in range(batch_size):
            response_length = response_lens[idx]
            cost_value = raw_cost_values[idx][-response_length:].unsqueeze(0)
            cost_value = cost_value.squeeze()
            if cost_value.dim() == 0:
                cost_value = torch.cat([cost_value.unsqueeze(0), cost_value.new_zeros(2)])
            cost_value_list.append(cost_value)
        cost_values = torch.nn.utils.rnn.pad_sequence(
            cost_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)

        cost_critic_loss = self.critic_loss_fn(
            cost_values,
            old_cost_values,
            cost_returns,
            sequence_mask,
        )
        self.cost_critic_model.backward(cost_critic_loss)
        self.cost_critic_model.step()

        with torch.no_grad():
            mask = sequence_mask
            kl_divergence = ((old_log_probs - ref_log_probs) * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            reward = reward.mean()
            cost = cost.mean()
            reward_with_kl_penalty = (old_rewards * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values, mask)

            cost_with_kl_penalty = (old_costs * mask).sum(dim=-1).mean()
            cost_advantage = masked_mean(cost_advantages, mask)
            cost_return = masked_mean(cost_returns, mask)
            cost_value = masked_mean(cost_values, mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            cost_critic_loss = get_all_reduce_mean(cost_critic_loss)
            reward = get_all_reduce_mean(reward)
            cost = get_all_reduce_mean(cost)
            reward_with_kl_penalty = get_all_reduce_mean(reward_with_kl_penalty)
            cost_with_kl_penalty = get_all_reduce_mean(cost_with_kl_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            cost_advantage = get_all_reduce_mean(cost_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            cost_return = get_all_reduce_mean(cost_return)
            reward_value = get_all_reduce_mean(reward_value)
            cost_value = get_all_reduce_mean(cost_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        return {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/cost_critic_loss': cost_critic_loss.item(),
            'train/reward': reward.item(),
            'train/cost': cost.item(),
            'train/log_lambda': self.log_lambda.item(),
            'train/lambda': self.log_lambda.exp().item(),
            'train/reward_with_kl_penalty': reward_with_kl_penalty.item(),
            'train/cost_with_kl_penalty': cost_with_kl_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/cost_advantage': cost_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/cost_return': cost_return.item(),
            'train/reward_value': reward_value.item(),
            'train/cost_value': cost_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/cost_critic_lr': self.cost_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }

    def add_kl_divergence_regularization(
        self,
        reward: torch.Tensor,  # size = (B,)
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
    ) -> torch.Tensor:  # size = (B, L)
        """Add KL divergence regularization on scalar rewards."""
        B, L = log_probs.size()
        end_index = (L - 1) * torch.ones((B,), dtype=torch.int64).to(reward.device)  # size = (B,)

        # size = (B, L)
        kl_divergence_estimate = log_probs - ref_log_probs
        kl_penalty_rewards = -self.kl_coeff * kl_divergence_estimate
        rewards = torch.scatter_add(
            kl_penalty_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        )
        return torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score)

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

        num_prompt_only_batches = len(self.prompt_only_dataloader)
        num_ptx_batches = len(self.ptx_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches
        for epoch in range(int(self.cfgs.train_cfgs.epochs)):
            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.chain.from_iterable([self.ptx_dataloader] * num_ptx_replicas),
            ):
                inference_batches, training_batches = self.rollout(prompt_only_batch)

                if self.use_ptx:
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(inference_batches))]
                torch.cuda.empty_cache()

                for _ in range(self.cfgs.train_cfgs.update_iters):
                    for inference_batch, training_batch, ptx_batch in zip(
                        inference_batches, training_batches, ptx_batches
                    ):
                        rl_info = self.rl_step(inference_batch, training_batch)

                        torch.cuda.empty_cache()
                        self.logger.log(rl_info, step=self.global_step)
                        if self.use_ptx:
                            ptx_info = self.ptx_step(ptx_batch)
                            torch.cuda.empty_cache()
                            self.logger.log(ptx_info, step=self.global_step)

                        self.global_step += 1
                        progress_bar.set_description(
                            f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                            f'(reward {rl_info["train/reward"]:.4f})',
                        )
                        progress_bar.update(1)

                        if self.global_step % self.cfgs.logger_cfgs.save_interval == 0:
                            self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                            self.save(tag=self.global_step)
                            self.logger.print('Checkpoint saved.')

                        if (
                            self.cfgs.data_cfgs.eval_datasets
                            and self.cfgs.train_cfgs.eval_strategy == 'steps'
                            and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                        ):
                            self.logger.print(
                                f'\n***** Evaluating at step {self.global_step} *****',
                            )
                            self.eval()

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.eval()

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        sequence_mask: torch.BoolTensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.size(-1)
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
        info: str | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        if info is None:
            self.save_transformers(model=model, tag=tag)
        else:
            self.save_transformers_with_info(model=model, tag=tag, info=info)


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_image_to_text', 'saferlhf')
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
    # finetune the model
    trainer = SafeRLHFVTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
