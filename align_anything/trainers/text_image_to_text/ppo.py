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
"""Trainer for PPO training."""


import argparse
import os
import sys

import deepspeed
import torch
from transformers import GenerationConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_image_to_text import PromptOnlyDataset, SupervisedDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.models.pretrained_model_with_value import load_pretrained_model_with_value_head
from align_anything.trainers.text_to_text.ppo import PPOTrainer as PPOTextTrainer
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    is_same_tokenizer,
    read_cfgs,
    seed_everything,
    update_dict,
)


class PPOTrainer(PPOTextTrainer):  # pylint: disable=too-many-instance-attributes
    """Trainer base class for PPO training."""

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        # load training datasets
        self.prompt_only_dataloader, self.eval_dataloader, self.ptx_dataloader = (
            self.get_dataloaders(PromptOnlyDataset, PromptOnlyDataset, SupervisedDataset)
        )

    def init_models(self) -> None:
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
        )
        # loading actor reference model
        self.actor_reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
        )
        # loading reward model
        self.reward_model, self.reward_tokenizer, _ = load_pretrained_model_with_value_head(
            self.cfgs.model_cfgs.reward_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
        )
        # loading reward critic model
        self.reward_critic_model, self.reward_critic_tokenizer, _ = (
            load_pretrained_model_with_value_head(
                self.cfgs.model_cfgs.reward_critic_model_name_or_path,
                model_max_length=self.cfgs.model_cfgs.model_max_length,
                padding_side='left',
                trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            )
        )
        # initial checking
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
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


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'ppo')
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
    trainer = PPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
