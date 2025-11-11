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
"""Trainer for DPO training."""


import argparse
import os
import sys

import deepspeed
import torch
import transformers
from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor

from align_anything.datasets.janus import PreferenceDataset, PreferenceBatch, PreferenceTokenizedDataset
from align_anything.trainers.text_to_text.dpo import DPOTrainer as DPOtextTrainer
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    gather_log_probabilities,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
)


transformers.logging.set_verbosity_info()

def strip_pad(seq: torch.Tensor, max_token_id: int):
    with torch.no_grad():
        # 使用torch.where创建索引张量
        indices = torch.where(seq < max_token_id)[0]
        
        # 检查索引张量是否为空
        if indices.numel() == 0:
            # 如果没有非填充标记，返回空张量
            return torch.tensor([], device=seq.device, dtype=seq.dtype)
        
        # 使用索引选择非填充标记
        return torch.index_select(seq, 0, indices)

class DPOTrainer(DPOtextTrainer):

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceTokenizedDataset, PreferenceTokenizedDataset
        ) # change to PreferenceDataset, PreferenceDataset in case of image input & text output

    def update_configs(self, model_config, args, fields):
        cross_update = lambda a, b, field_name: (
            setattr(b, field_name, getattr(a, field_name))
            if getattr(b, field_name, None) is None
            else setattr(a, field_name, getattr(b, field_name))
        )

        for f in fields:
            cross_update(model_config, args, f)

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        self.model = MultiModalityCausalLM.from_pretrained(
            self.cfgs.model_cfgs.model_name_or_path,
        ).to(get_current_device())

        self.reference_model = MultiModalityCausalLM.from_pretrained(
            self.cfgs.model_cfgs.model_name_or_path,
        ).to(get_current_device())

        if self.cfgs.train_cfgs.bf16:
            self.model = self.model.to(torch.bfloat16)
            self.reference_model = self.reference_model.to(torch.bfloat16)

        self.processor = VLChatProcessor.from_pretrained(
            self.cfgs.model_cfgs.model_name_or_path,
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.model_max_length = self.cfgs.model_cfgs.model_max_length
    
    def compute_log_probs(
        self,
        model: MultiModalityCausalLM,
        batch: PreferenceBatch,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = self.model.forward(**self.infer_batch(batch)).logits
        device = logits.device
        input_ids = batch['input_ids']
        # dim 0 is the batch size
        batch_size = input_ids.size(0)
        logprob_list = []
        for idx in range(batch_size):
            response_length = batch['input_ids'].size(1)
            print("input_ids[idx]", input_ids[idx])
            raw_input_id = strip_pad(input_ids[idx], 16384)
            logit = logits[idx][-response_length:].unsqueeze(0)
            input_id = raw_input_id[-response_length:].unsqueeze(0)
            print("logit.shape", logit.shape)
            print("input_id.shape", input_id.shape)
            log_p = gather_log_probabilities(logit[:, :-1], input_id[:, 1:])
            logprob_list.append(log_p.squeeze(0))
        return torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(device)

    def loss(self, batch: PreferenceBatch) -> dict[str, torch.Tensor]:
        """Loss function for preference learning."""
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
        
        ref_better_sequence_log_probs, ref_worse_sequence_log_probs = ref_sequence_log_probs.chunk(chunks=2, dim=0)

        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []

        batch_size = better_sequence_log_probs.size(0)
        for i in range(batch_size):
            better_log_prob = better_sequence_log_probs[i, :].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, :].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, :].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, :].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob
            
            losses.append(
                -torch.nn.functional.logsigmoid(better_log_ratio) - torch.nn.functional.logsigmoid(-worse_log_ratio)
            )
            better_sample_rewards.append(better_log_ratio)
            worse_sample_rewards.append(-worse_log_ratio)

        loss = torch.stack(losses).mean()
        better_sample_reward = torch.stack(better_sample_rewards)
        worse_sample_reward = torch.stack(worse_sample_rewards)
        reward = better_sample_reward + worse_sample_reward
        reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean()
        reward_margin = better_sample_reward - worse_sample_reward
        
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
    task = os.path.join('janus', 'dpo_gen')
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
