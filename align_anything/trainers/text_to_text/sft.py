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
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.supervised import SupervisedBatch, SupervisedDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import get_current_device, torch_gc, torch_set_device
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


class SupervisedTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the SFT trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}

        self.init_check()
        dist.barrier()
        self.init_models()
        if hasattr(self.model, 'infer_batch'):
            self.infer_batch = self.model.infer_batch
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
        self.bnb_cfgs = self.cfgs.bnb_cfgs
        self.lora_cfgs = self.cfgs.lora_cfgs
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=True,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            SupervisedDataset, SupervisedDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    def loss(self, sft_batch: SupervisedBatch) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs = self.model(**self.infer_batch(sft_batch))
        return {'loss': outputs.loss}

    def train_step(self, sft_batch: SupervisedBatch) -> dict[str, Any]:
        """Performs a single training step."""
        loss = self.loss(sft_batch)['loss']
        self.model.backward(loss)
        self.model.step()

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

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
        progress_bar.update(self.global_step)

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.log(self.eval(), step=0)

        remain_epoch = self.cfgs.train_cfgs.epochs - (
            self.global_step // len(self.train_dataloader)
        )

        start_batch_idx = self.global_step % len(self.train_dataloader)

        for epoch in range(int(remain_epoch)):
            self.model.train()
            progress_bar.set_description(
                f'Resuming from checkpoint {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
            )

            for batch_idx, batch in enumerate(self.train_dataloader):
                if epoch == 0 and batch_idx < start_batch_idx:
                    continue

                info = self.train_step(batch)
                torch_gc()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                save_interval = (
                    self.cfgs.train_cfgs.epochs
                    * len(self.train_dataloader)
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
                    self.logger.log(self.eval(), step=self.global_step)

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)
            self.model.tput_timer.update_epoch_count()

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}
        if not isinstance(self.eval_dataloader, dict):
            named_eval_dataloader = {self.cfgs.data_cfgs.eval_template: self.eval_dataloader}
        else:
            named_eval_dataloader = self.eval_dataloader

        self.model.eval()
        if self.cfgs.train_cfgs.gradient_checkpointing and not self.lora_enabled:
            self.model.gradient_checkpointing_disable()

        loss_logger = {}

        for template, raw_eval_dataloader in named_eval_dataloader.items():
            eval_dataloader = tqdm(
                raw_eval_dataloader,
                desc=f'Evaluating {template}',
                disable=not is_main_process(),
                position=1,
                leave=False,
            )
            batch = None
            eval_loss = []
            for batch in eval_dataloader:
                loss = self.loss(batch)['loss']
                eval_loss.append(loss.item())

                if batch is None:
                    self.logger.print(f'WARNING: `{template}` eval_dataloader is empty.')
                    return {}
            if len(eval_loss) > 0:
                loss_logger[f'eval/loss/{template}'] = sum(eval_loss) / len(eval_loss)

        self.model.train()
        if self.cfgs.train_cfgs.gradient_checkpointing and not self.lora_enabled:
            self.model.gradient_checkpointing_enable()
        return loss_logger

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
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'sft')
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
    trainer = SupervisedTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
