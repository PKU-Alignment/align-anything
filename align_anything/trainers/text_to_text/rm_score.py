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
"""Evaluator for reward model scoring."""


import argparse
import os
import sys
import json
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.supervised import SupervisedDataset
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


class RMScore(SupervisedTrainerBase):

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
        assert self.cfgs.train_cfgs.per_device_train_batch_size==1, 'per_device_train_batch_size must be 1'
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
            SupervisedDataset, SupervisedDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""

        self.model.eval()
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
        
        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )

        rewards = []
        prompts = []
        responses = []
        batch = None
        for batch in eval_dataloader:
            output = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            end_scores = output.end_scores
            rewards.append(end_scores)
        
            decoded_prompt_and_response = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            prompt, response = split_prompt_response(decoded_prompt_and_response, split_token=self.eval_template.split_token)
            prompts.append(prompt)
            responses.append(response)

        data_with_score = []
        for prompt, response, reward in zip(prompts, responses, rewards):
            data_with_score.append({
                'prompt': prompt[0].replace(self.eval_template.system_prompt, '').replace(self.eval_template.user_prompt.replace('{input} ', ''), '').strip(),
                'response': response[0],
                'score': reward.item(),
            })
        
        output_name = os.path.join(self.cfgs.logger_cfgs.output_dir, 'tmp', f'process_{dist.get_rank()}.json')
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        with open(output_name, 'w') as f:
            json.dump(data_with_score, f, indent=4)
            print(f'Saved {len(data_with_score)} samples to {output_name}')
        
        dist.barrier()

        if is_main_process():
            final_data_with_score = []
            for rank in range(dist.get_world_size()):
                output_name = os.path.join(self.cfgs.logger_cfgs.output_dir, 'tmp', f'process_{rank}.json')
                with open(output_name, 'r') as f:
                    final_data_with_score.extend(json.load(f))
                os.remove(output_name)
            os.rmdir(os.path.join(self.cfgs.logger_cfgs.output_dir, 'tmp'))
            output_name = os.path.join(self.cfgs.logger_cfgs.output_dir, 'eval_data_with_score.json')
            with open(output_name, 'w') as f:
                json.dump(final_data_with_score, f, indent=4)
            print(f'Saved {len(final_data_with_score)} samples to {output_name}')

def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'rm_score')
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
    evaluator = RMScore(cfgs=cfgs, ds_cfgs=ds_cfgs)
    evaluator.eval()


if __name__ == '__main__':
    sys.exit(main())
