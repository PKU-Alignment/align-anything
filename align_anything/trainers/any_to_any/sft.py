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
"""Trainer for supervised training."""


import argparse
import os
import sys

import deepspeed
import torch
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.any_to_any import SupervisedDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.text_to_text.sft import SupervisedTrainer as SupervisedtextTrainer
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
)

import transformers
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
from align_anything.models.modeling_emu3.mllm.processing_emu3 import Emu3Processor

transformers.logging.set_verbosity_info()

class SuperviseTrainer(SupervisedtextTrainer):

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            SupervisedDataset, SupervisedDataset
        )

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        self.model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            processor_name_or_path=self.cfgs.model_cfgs.processor_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=True,
        )
        processor_name_or_path = self.cfgs.model_cfgs.processor_name_or_path
        image_processor = AutoImageProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(processor_name_or_path, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.cfgs.model_cfgs.model_name_or_path, trust_remote_code=True)
        processor = Emu3Processor(
            image_processor,
            image_tokenizer,
            tokenizer,
        )
        self.processor = processor
        self.tokenizer = tokenizer


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('any_to_any', 'sft')
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
    trainer = SuperviseTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
