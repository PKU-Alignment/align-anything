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
from datetime import datetime
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import CONFIG_NAME, PreTrainedModel, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.supervised import SupervisedBatch, SupervisedDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.utils.logger import Logger
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    get_optimizer_grouped_parameters,
    namedtuple_to_dict,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


class SuperviseTrainer:

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the SFT trainer."""
        self.cfgs = cfgs
        self.ds_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
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
        return

    def init_logger(self) -> None:
        """Set logger."""
        logger_cfgs = self.cfgs.logger_cfgs
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.logger = Logger(
            log_type=logger_cfgs.log_type,
            log_dir=logger_cfgs.output_dir,
            log_project=logger_cfgs.log_project,
            log_run_name=f'{logger_cfgs.log_run_name}-{self.cfgs.data_cfgs.train_datasets}-{time}',
            config=namedtuple_to_dict(self.cfgs),
        )

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_cfgs is not None and self.ds_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_cfgs)
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=True,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        train_dataset = SupervisedDataset(
            path=self.cfgs.data_cfgs.train_datasets,
            template=self.cfgs.data_cfgs.train_template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            size=self.cfgs.data_cfgs.train_size,
            split=self.cfgs.data_cfgs.train_split,
            subset=self.cfgs.data_cfgs.train_subset,
            data_files=self.cfgs.data_cfgs.train_data_files,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.cfgs.train_cfgs.per_device_train_batch_size,
        )
        if self.cfgs.data_cfgs.eval_datasets:
            eval_dataset = SupervisedDataset(
                path=self.cfgs.data_cfgs.eval_datasets,
                template=self.cfgs.data_cfgs.eval_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                size=self.cfgs.data_cfgs.eval_size,
                split=self.cfgs.data_cfgs.eval_split,
                subset=self.cfgs.data_cfgs.eval_subset,
                data_files=self.cfgs.data_cfgs.eval_data_files,
            )
            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.cfgs.train_cfgs.per_device_train_batch_size,
            )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.cfgs.train_cfgs.gradient_accumulation_steps - 1
        ) // self.cfgs.train_cfgs.gradient_accumulation_steps
        total_training_steps = self.cfgs.train_cfgs.epochs * num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.cfgs.train_cfgs.weight_decay,
        )
        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.cfgs.train_cfgs.learning_rate,
            betas=self.cfgs.train_cfgs.adam_betas,
        )
        num_warmup_steps = int(self.cfgs.train_cfgs.lr_warmup_ratio * total_training_steps)
        lr_scheduler = get_scheduler(
            name=self.cfgs.train_cfgs.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            config=self.ds_cfgs,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def loss(self, sft_batch: SupervisedBatch) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs = self.model(**sft_batch)
        return {
            'loss': outputs.loss,
        }

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

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        for epoch in range(self.cfgs.train_cfgs.epochs):
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

            self.logger.print(f'\n***** Updating epoch counting at step {self.global_step} *****')
            self.model.tput_timer.update_epoch_count()

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

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

        batch = None
        eval_loss = []
        for batch in eval_dataloader:
            loss = self.loss(batch)['loss']
            eval_loss.append(loss.item())

        if batch is None:
            self.logger.print('WARNING: `eval_dataloader` is empty.')
            return {}

        self.model.train()
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        return {'eval/loss': sum(eval_loss) / len(eval_loss)}

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        dist.barrier()

        if model is None:
            model = self.model  # pylint: disable=no-member

        self.logger.print(f'Saving model to "{self.cfgs.logger_cfgs.output_dir}" ...')

        output_config_file = os.path.join(self.cfgs.logger_cfgs.output_dir, CONFIG_NAME)
        model_to_save: PreTrainedModel = getattr(model, 'module', model)

        if is_main_process():
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_pretrained(self.cfgs.logger_cfgs.output_dir)

        self.logger.print('Saving 16-bit model...')
        save_file_name = f'pytorch_model_{tag}.bin' if tag else 'pytorch_model.bin'
        model.save_16bit_model(self.cfgs.logger_cfgs.output_dir, save_filename=save_file_name)

        self.logger.print('Model saved!')


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task='sft')

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
