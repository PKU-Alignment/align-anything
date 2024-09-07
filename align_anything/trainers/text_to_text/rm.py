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
"""Trainer for reward model training."""


import argparse
import os
import sys
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.preference import PreferenceBatch, PreferenceDataset
from align_anything.models.pretrained_model_with_value import load_pretrained_model_with_value_head
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    split_prompt_response,
    update_dict,
)


class RMTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the reward model trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0

        self.init_check()
        dist.barrier()
        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_check(self) -> None:
        """Initial configuration checking."""
        super().init_check()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs is not None and self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_train_cfgs)
        self.model, self.tokenizer, self.processor = load_pretrained_model_with_value_head(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.train_cfgs.trust_remote_code,
            modality='text',
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    def loss(
        self,
        batch: PreferenceBatch,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the reward model."""
        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = batch[
            'input_ids'
        ].chunk(chunks=2, dim=0)
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        output = self.model(
                input_ids=batch['input_ids'],
                attention_maks=batch['attention_maks'],
            )
        scores = output.scores
        end_scores = output.end_scores
        higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        higher_end_reward, lower_end_reward = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)

        loss = -F.logsigmoid(higher_end_reward - lower_end_reward).mean()

        if self.cfgs.train_cfgs.regularization > 0.0:
            loss = (
                loss
                + self.cfgs.train_cfgs.regularization
                * torch.stack([lower_end_reward, higher_end_reward]).square().mean()
            )

        accuracy = (higher_end_reward > lower_end_reward).float().mean()  # size = ()
        return {
            'loss': loss,  # size = ()
            'higher_end_reward': higher_end_reward,  # size = (B,)
            'lower_end_reward': lower_end_reward,  # size = (B,)
            'higher_rewards': higher_rewards,  # size = (B, L)
            'lower_rewards': lower_rewards,  # size = (B, L)
            'accuracy': accuracy,  # size = ()
        }

    def train_step(
        self,
        batch: PreferenceBatch,
    ) -> dict[str, Any]:
        """Perform a single training step."""
        loss_dict = self.loss(batch)
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        accuracy = loss_dict['accuracy']

        loss = get_all_reduce_mean(loss)
        accuracy = get_all_reduce_mean(accuracy)

        return {
            'train/loss': loss.item(),
            'train/accuracy': accuracy.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        self.logger.print('\n***** Evaluating at the beginning *****')
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
        num_correct_predictions = 0
        num_total_predictions = 0

        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )

        rewards = []
        batch = None
        for batch in eval_dataloader:
            output = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            end_scores = output.end_scores
            higher_end_rewards, lower_end_rewards = end_scores.squeeze(dim=-1).chunk(
                chunks=2, dim=0
            )
            batch_size = higher_end_rewards.size(0)
            num_correct_predictions += (higher_end_rewards > lower_end_rewards).sum()
            num_total_predictions += batch_size

            rewards.extend([higher_end_rewards, lower_end_rewards])

        if batch is None:
            self.logger.print('WARNING: `eval_dataloader` is empty.')
            return {}

        accuracy = num_correct_predictions / num_total_predictions
        accuracy = get_all_reduce_mean(accuracy)

        # Gather rewards from all devices for further analysis
        rewards = torch.cat(rewards, dim=0)
        if is_main_process():
            gathered_rewards = [torch.empty_like(rewards) for _ in range(dist.get_world_size())]
        else:
            gathered_rewards = []
        dist.gather(rewards, gathered_rewards, dst=0)
        if is_main_process():
            rewards = torch.cat(gathered_rewards, dim=0)

        self.model.train()
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Evaluation info
        info = {
            'eval/accuracy': accuracy.item(),
            'eval/reward_mean': rewards.mean().item(),
            'eval/reward_std': rewards.std().item(),
        }

        if is_main_process():
            # Print some examples from the last batch
            max_num_rows = 3
            (
                better_input_ids,  # size = (B, L)
                worse_input_ids,  # size = (B, L)
            ) = batch[
                'input_ids'
            ].chunk(chunks=2, dim=0)
            higher_reward_texts = self.tokenizer.batch_decode(
                better_input_ids[:max_num_rows],
                skip_special_tokens=True,
            )
            lower_reward_texts = self.tokenizer.batch_decode(
                worse_input_ids[:max_num_rows],
                skip_special_tokens=True,
            )

            h_prompts, h_responses = split_prompt_response(
                higher_reward_texts,
                split_token=self.eval_template.split_token,
            )
            l_prompts, l_responses = split_prompt_response(
                lower_reward_texts,
                split_token=self.eval_template.split_token,
            )
            assert h_prompts == l_prompts, 'prompts are not the same'
            h_rewards = [f'{reward:.6f}' for reward in higher_end_rewards.tolist()]
            l_rewards = [f'{reward:.6f}' for reward in lower_end_rewards.tolist()]

            title = ', '.join(
                f'{key.rpartition("/")[-1]} = {value:.6f}' for key, value in info.items()
            )
            self.logger.print_table(
                title=f'Evaluation: {title}',
                columns=[
                    'prompt',
                    'higher-reward response',
                    'reward',
                    'lower-reward response',
                    'reward',
                ],
                rows=tuple(
                    zip(
                        h_prompts[:max_num_rows],
                        h_responses[:max_num_rows],
                        h_rewards[:max_num_rows],
                        l_responses[:max_num_rows],
                        l_rewards[:max_num_rows],
                    ),
                ),
                max_num_rows=max_num_rows,
            )

        return info

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.cfgs.train_cfgs.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        for epoch in range(int(self.cfgs.train_cfgs.epochs)):
            self.model.train()

            for batch in self.train_dataloader:
                info = self.train_step(batch)
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                if self.global_step % self.cfgs.logger_cfgs.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.save(tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.cfgs.data_cfgs.eval_datasets
                    and self.cfgs.train_cfgs.eval_strategy == 'steps'
                    and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)
            self.model.tput_timer.update_epoch_count()

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'rm')
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
    trainer = RMTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
