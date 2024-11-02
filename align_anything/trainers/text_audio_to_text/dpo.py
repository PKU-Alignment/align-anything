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
"""Trainer for DPO training."""


import argparse
import os
import sys

import deepspeed
import torch
import torch.distributed
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.datasets.text_audio_to_text.preference import PreferenceDataset, PreferenceBatch
from align_anything.trainers.text_to_text.dpo import DPOTrainer as DPOtextTrainer
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
    gather_log_probabilities
)

def strip_pad(seq: torch.Tensor, pad_token_id: int):
    # remove the pad token in the tensor
    return seq[seq != pad_token_id]

class DPOTrainer(DPOtextTrainer):

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            padding_side='left',
            trust_remote_code=False,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_audio_proj=self.cfgs.train_cfgs.freeze_audio_proj,
            freeze_audio_tower=self.cfgs.train_cfgs.freeze_audio_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
        )
        self.reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            padding_side='left',
            trust_remote_code=self.cfgs.train_cfgs.trust_remote_code,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_audio_proj=self.cfgs.train_cfgs.freeze_audio_proj,
            freeze_audio_tower=self.cfgs.train_cfgs.freeze_audio_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
        )

    def compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        batch: PreferenceBatch,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        keys_to_remove = ['better_response_lens', 'worse_response_lens', 'response_lens']

        infer_batch = {key: value for key, value in batch.items() if key not in keys_to_remove}
        logits = model(**infer_batch).logits
        device = logits.device
        input_ids = batch['input_ids']
        batch_size = len(batch['response_lens'])
        logprob_list = []
        for idx in range(batch_size):
            response_length = batch['response_lens'][idx] # for the eos token
            logit = logits[idx][-response_length:].unsqueeze(0)
            input_id = input_ids[idx][-response_length:].unsqueeze(0)
            log_p = gather_log_probabilities(logit[:, :-1], input_id[:, 1:])
            logprob_list.append(log_p.squeeze(0))
        return torch.nn.utils.rnn.pad_sequence(logprob_list, batch_first=True, padding_value=0.).to(device)

    def loss(  # pylint: disable=too-many-locals
        self,
        batch: PreferenceBatch,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the DPO algorithm."""
        sequence_log_probs = self.compute_log_probs(
            self.model.module,
            batch,
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                batch,
            )
            ref_better_sequence_log_probs, ref_worse_sequence_log_probs = (
                ref_sequence_log_probs.chunk(chunks=2, dim=0)
            )

        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []

        better_input_ids, worse_input_ids = batch['input_ids'].chunk(chunks=2, dim=0)

        batch_size = better_input_ids.size(0)
        for i in range(batch_size):
            if torch.all(torch.eq(better_input_ids[i], worse_input_ids[i])).item():
                continue
            better_log_prob = better_sequence_log_probs[i, :].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, :].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, :].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, :].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob

            losses.append(
                -F.logsigmoid(
                    self.cfgs.train_cfgs.scale_coeff * (better_log_ratio - worse_log_ratio),
                ),
            )
            better_sample_rewards.append(
                self.cfgs.train_cfgs.scale_coeff * better_log_ratio.detach(),
            )
            worse_sample_rewards.append(self.cfgs.train_cfgs.scale_coeff * worse_log_ratio.detach())

        loss = torch.stack(losses).mean()  # size = ()
        better_sample_reward = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_reward = torch.stack(worse_sample_rewards)  # size = (B,)
        reward = better_sample_reward + worse_sample_reward  # size = (B,)
        reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean()  # size = ()
        reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)

        return {
            'loss': loss,
            'reward': reward,
            'better_sample_reward': better_sample_reward,
            'worse_sample_reward': worse_sample_reward,
            'reward_accuracy': reward_accuracy,
            'reward_margin': reward_margin,
        }


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_audio_to_text', 'dpo')
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
    trainer = DPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
