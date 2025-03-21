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
"""Trainer for GRPO Training"""


import argparse
import copy
import os
import sys

import deepspeed
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text import PromptOnlyDataset, SupervisedDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import RLTrainerBase
from align_anything.utils.device_utils import torch_set_device
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    batch_retokenize,
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    prepare_ds_eval_cfgs,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


class GRPOTrainer(RLTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.ds_eval_cfgs = prepare_ds_eval_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0

        self.init_check()
        dist.barrier()
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}
        dist.barrier()
        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

        self.beta = self.cfgs.train_cfgs.beta  # KL regularization coefficient
        self.num_generations = (
            self.cfgs.train_cfgs.num_generations
        )  # number of sequences generated for each prompt

    def init_check(self) -> None:
        super().init_check()
        if (
            self.cfgs.train_cfgs.per_device_prompt_batch_size
            % self.cfgs.train_cfgs.per_device_train_batch_size
            != 0
        ):
            raise ValueError('Every prompt batch size must be divisible by the micro-batch size.')

    def init_models(self) -> None:
        # DeepSpeed configuration, different from that in RLTrainerBase, we don't need critic model in GRPO
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)

        self.bnb_cfgs = self.cfgs.bnb_cfgs
        self.lora_cfgs = self.cfgs.lora_cfgs

        self.actor_model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

        self.actor_reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

        self.reward_model, self.reward_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

        self.generation_config = GenerationConfig(
            max_length=self.cfgs.model_cfgs.model_max_length,
            temperature=self.cfgs.model_cfgs.temperature,
            top_p=self.cfgs.model_cfgs.top_p,
            repetition_penalty=self.cfgs.model_cfgs.repetition_penalty,
            do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets"""
        self.prompt_only_dataloader, self.eval_dataloader, _ = self.get_dataloaders(
            PromptOnlyDataset, PromptOnlyDataset, SupervisedDataset
        )

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.actor_model.train()
            if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
                self.actor_model.gradient_checkpointing_enable()
        else:
            self.actor_model.eval()
            if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
                self.actor_model.gradient_checkpointing_disable()
        return

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        # different from that in RLTrainerBase, we don't need critic model in GRPO

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
        self.reward_model = self._init_eval_deepspeed_engine(
            model=self.reward_model,
            ds_cfgs=self.ds_eval_cfgs,
        )
        self.reward_model.eval()

        # load the checkpoint if specified
        if self.cfgs.train_cfgs.load_checkpoint:
            self.actor_model.load_checkpoint(load_dir=self.cfgs.model_cfgs.actor_model_name_or_path)
        # setup the gradient checkpointing
        if self.cfgs.train_cfgs.actor_gradient_checkpointing and not self.lora_enabled:
            self.actor_model.gradient_checkpointing_enable()

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """
        Compute the log-probabilities of the model on the specified tokens.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        logits = logits[:, -logits_to_keep:, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return per_token_logps  # shape: (B, logits_to_keep)

    def generate_completions(self, prompt_batch: dict) -> torch.Tensor:
        """
        Generate multiple completions based on the given prompt.
        Here, we set num_return_sequences = self.num_generations, which requires that each sample in the dataloader contains only one prompt,
        and then generate multiple completions for comparison within the group.
        """
        self.actor_model.eval()
        with torch.no_grad():
            sequences = self.actor_model.module.generate(
                **prompt_batch,
                generation_config=self.generation_config,
                num_return_sequences=self.cfgs.train_cfgs.num_generations,
                synced_gpus=True,
                do_sample=True,
            )
        return sequences  # shape: (B * num_generations, L_total)

    def compute_rewards(self, sequences: torch.Tensor, prompt_length: int) -> torch.Tensor:
        """
        Compute the rewards for the generated completions.
        """
        completions = sequences[:, prompt_length:]
        # generate mask: for each sequence, set the tokens after the first eos token to 0
        eos_token_id = self.tokenizer.eos_token_id
        completion_mask = torch.ones_like(completions)
        for i in range(completions.size(0)):
            eos_positions = (completions[i] == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = eos_positions[0].item()
                completion_mask[i, first_eos + 1 :] = 0
        completions_ids = completions * completion_mask
        reward_tokenize_output = batch_retokenize(
            completions_ids,
            src_tokenizer=self.tokenizer,
            dest_tokenizer=self.reward_tokenizer,
            skip_special_tokens=True,
            device=self.reward_model.device,
        )
        reward_inputs = {k: v.to(sequences.device) for k, v in reward_tokenize_output.items()}
        with torch.no_grad():
            rewards = self.reward_model(**reward_inputs).end_scores.squeeze(
                dim=-1
            )  # shape: (B*num_generations,
        return rewards

    def train_step(self, prompt_batch: dict) -> dict[str, float]:
        """Single training step"""
        device = self.actor_model.module.parameters().__next__().device
        prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}

        # record the original prompt length
        prompt_length = prompt_batch['input_ids'].size(1)

        # generate multiple completions (each prompt generates num_generations sequences)
        sequences = self.generate_completions(prompt_batch)  # shape: (B * num_generations, L_total)
        # restore train mode
        self.actor_model.train()

        # compute rewards
        rewards = self.compute_rewards(sequences, prompt_length)  # shape: (B * num_generations,)
        B = prompt_batch['input_ids'].size(0)
        G = self.num_generations
        rewards = rewards.view(B, G)
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True) + 1e-4
        advantages = (rewards - group_mean) / group_std  # shape: (B, G)
        advantages = advantages.view(-1, 1)

        # compute the attention mask of the generated sequences
        attention_mask = (sequences != self.tokenizer.pad_token_id).long()
        logits_to_keep = sequences.size(1) - prompt_length

        # compute the per-token log-probabilities of the actor_model on the generated sequences
        per_token_logps = self._get_per_token_logps(
            self.actor_model, sequences, attention_mask, logits_to_keep
        )
        # the log-probabilities of the reference model (no gradient)
        with torch.no_grad():
            ref_per_token_logps = self._get_per_token_logps(
                self.actor_reference_model, sequences, attention_mask, logits_to_keep
            )

        # compute the per-token KL divergence: KL = exp(ref - logp) - (ref - logp) - 1
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # expand the advantages to each token (assume the same advantages for all completion tokens)
        advantages_expanded = advantages.expand(-1, logits_to_keep)

        # formula: loss = - ( exp(logp - detach(logp)) * advantage - beta * KL )
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_expanded
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        # construct the completion mask: only count the loss for the valid tokens (not truncated by eos)
        completion_tokens = sequences[:, prompt_length:]
        eos_token_id = self.tokenizer.eos_token_id
        completion_mask = torch.ones_like(completion_tokens)
        for i in range(completion_tokens.size(0)):
            eos_positions = (completion_tokens[i] == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = eos_positions[0].item()
                completion_mask[i, first_eos + 1 :] = 0
        completion_mask = completion_mask.to(per_token_loss.dtype)

        # compute the total loss
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        self.actor_model.zero_grad()
        self.actor_model.backward(loss)
        self.actor_model.step()

        loss_val = get_all_reduce_mean(loss).item()
        avg_reward = get_all_reduce_mean(rewards.mean()).item()

        return {'train/loss': loss_val, 'train/reward': avg_reward}

    def train(self) -> None:
        """Training main loop"""
        self.logger.print('***** Running GRPO training *****')

        total_training_steps = self.total_training_steps
        progress_bar = tqdm(
            total=total_training_steps,
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        progress_bar.update(self.global_step)

        if self.cfgs.data_cfgs.eval_datasets:
            self.eval()

        remain_epoch = self.cfgs.train_cfgs.epochs - (
            self.global_step // len(self.prompt_only_dataloader)
        )

        start_batch_idx = self.global_step % len(self.prompt_only_dataloader)

        for epoch in range(int(remain_epoch)):
            for batch_idx, prompt_batch in enumerate(self.prompt_only_dataloader):
                if epoch == 0 and batch_idx < start_batch_idx:
                    continue

                train_info = self.train_step(prompt_batch)
                self.global_step += 1

                self.logger.log(train_info, step=self.global_step)
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} (reward {train_info['train/reward']:.4f})"
                )
                progress_bar.update(1)

                save_interval = (
                    self.cfgs.train_cfgs.epochs
                    * len(self.prompt_only_dataloader)
                    // self.cfgs.logger_cfgs.save_total_limit
                )
                if self.global_step % save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.save(tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.cfgs.data_cfgs.eval_datasets
                    and self.cfgs.train_cfgs.eval_strategy == 'steps'
                    and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.eval()

        self.save()

    def save(self, model: deepspeed.DeepSpeedEngine | None = None, tag: int | None = None) -> None:
        self.save_transformers(model=model, tag=tag)


def main():
    # initialize distributed training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read the default configuration from the yaml file
    task = os.path.join('text_to_text', 'grpo')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # read the custom configuration from the command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # initialize and start training the GRPO model
    trainer = GRPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
