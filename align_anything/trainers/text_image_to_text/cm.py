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

import deepspeed
import torch
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import torch.nn.functional as F

from align_anything.datasets.text_image_to_text.preference import PreferenceBatch_ours as PreferenceBatch
from align_anything.datasets.text_image_to_text.preference import PreferenceDataset_ours as PreferenceDataset
from align_anything.models.pretrained_model_with_value import load_pretrained_model_with_value_head
from align_anything.trainers.text_to_text.cm import CMTrainer as CMtextTrainer
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
)


class CMTrainer(CMtextTrainer):

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs is not None and self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_train_cfgs)
        self.model, self.tokenizer, self.processor = load_pretrained_model_with_value_head(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.train_cfgs.trust_remote_code,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            modality='text_image',
        )

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
        keys_to_remove = ['better_response_lens', 'worse_response_lens', 'response_lens']

        infer_batch = {key: value for key, value in batch.items() if key not in keys_to_remove}
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
        )

        scores = output.scores
        end_scores = output.end_scores
        higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        higher_end_reward, lower_end_reward = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        better_sign_list = torch.tensor(batch['is_better_safe']).to(higher_end_reward.device)
        worse_sign_list = torch.tensor(batch['is_worse_safe']).to(lower_end_reward.device)
        print('better_sign_list:', better_sign_list)
        print('worse_sign_list:', worse_sign_list)
        # exit()
        signed_higher_rewards = higher_end_reward * better_sign_list
        signed_lower_rewards = lower_end_reward * worse_sign_list


        print('higher_end_reward:', higher_end_reward)
        print('lower_end_reward:', lower_end_reward)
        print('signed_higher_rewards:', signed_higher_rewards)
        print('signed_lower_rewards:', signed_lower_rewards)
        #exit()
        cost = -F.logsigmoid(signed_higher_rewards).mean() - F.logsigmoid(signed_lower_rewards).mean()
        origin_loss = -F.logsigmoid(higher_end_reward - lower_end_reward).mean()
        loss = self.cfgs.train_cfgs.scale_coeff*cost + origin_loss
        #loss  = origin_loss
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



def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_image_to_text', 'cm')
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
    seed_everything(int(cfgs.train_cfgs.seed))

    # finetune the model
    trainer = CMTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    #trainer.eval()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
