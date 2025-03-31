# Copyright 2024 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_video_to_action.supervised import ChoresMultitaskDataset
from align_anything.models.spoc_models.models.transformer_models import REGISTERED_MODELS
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import get_current_device, torch_set_device
from align_anything.utils.multi_process import is_main_process
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
        self.current_epoch = 0
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
        model, processor = REGISTERED_MODELS[self.cfgs.model_cfgs.model_architecture].build_model(
            model_version=self.cfgs.model_cfgs.model_version,
            input_sensors=self.cfgs.sensor_cfgs.input_sensors,
            loss=self.cfgs.train_cfgs.loss,
            data_augmentation=True,
            ckpt_pth=None,
        )
        self.model = model.cuda()
        self.processor = processor
        self.processor.device = torch.device('cuda')

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()

    def init_datasets(self) -> None:
        self.train_dataloader = self.get_dataloader('train')
        self.val_dataloader = self.get_dataloader('val')
        num_datasets = len(self.train_dataloader.dataset.dataset_names)
        self.max_samples = min(
            self.cfgs.data_cfgs.max_samples * num_datasets,
            len(self.train_dataloader.dataset),
        )

    @torch.no_grad()
    def val(self) -> dict[str, Any]:
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        val_loss = []
        loss_logger = {}
        val_dataloader = tqdm(
            self.val_dataloader,
            desc=f'Validation',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )
        for batch in val_dataloader:
            outputs, proc_batch = self.forward_batch(batch)
            for k, v in outputs.items():
                if 'loss' in k:
                    val_loss.append(v.item())
        loss_logger['val/loss'] = sum(val_loss) / len(val_loss)
        self.model.train()
        return loss_logger

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        dist.barrier()
        if model is None:
            model = self.model
        output_dir = os.path.join(self.cfgs.logger_cfgs.output_dir, f'slice_{tag or "end"}')
        os.makedirs(output_dir, exist_ok=True)
        self.logger.print(f'Saving model to "{output_dir}" ...')

        if not self.lora_enabled:
            self.logger.print('Saving 16-bit model...')
            zero_stage = self.ds_train_cfgs.get('zero_optimization', {}).get('stage', 0)
            if zero_stage >= 2:
                save_file_name = 'pytorch_model.bin'
                model.save_16bit_model(output_dir, save_filename=save_file_name)
                if self.cfgs.train_cfgs.save_checkpoint:
                    model.save_checkpoint(output_dir)
            else:
                if is_main_process():
                    model_to_save.save_pretrained(output_dir, is_main_process=True)
            self.logger.print('Checkpoint saved.')

    def on_train_epoch_start(self) -> None:
        prob_decay_size = (
            self.cfgs.data_cfgs.init_prob_sample_last_steps
            - self.cfgs.data_cfgs.final_prob_sample_last_steps
        ) / self.cfgs.train_cfgs.epochs
        current_prob = (
            self.cfgs.data_cfgs.init_prob_sample_last_steps - prob_decay_size * self.current_epoch
        )
        next_prob = self.cfgs.data_cfgs.init_prob_sample_last_steps - prob_decay_size * (
            self.current_epoch + 1
        )
        self.train_dataloader.dataset.init_prob_sample_last_steps(
            init_prob=current_prob,
            final_prob=next_prob,
            num_workers=self.cfgs.data_cfgs.num_workers,
            num_gpu_per_node=max(torch.cuda.device_count(), 1),
            num_node=1,
        )

    def forward_batch(self, batch):
        proc_batch = self.processor.process(batch)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(proc_batch)
        return outputs, proc_batch

    def train_step(self, batch):
        outputs, proc_batch = self.forward_batch(batch)
        losses = dict()
        total_loss = 0.0
        for k, v in outputs.items():
            if 'loss' in k:
                losses[f'train/{k}'] = v
                total_loss += v
        self.model.backward(total_loss)
        self.model.step()
        return losses

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        remain_epoch = self.cfgs.train_cfgs.epochs - (
            self.global_step // len(self.train_dataloader)
        )
        self.current_epoch = self.global_step // len(self.train_dataloader)
        progress_bar = tqdm(
            total=remain_epoch * len(self.train_dataloader),
            desc=f'Training {self.current_epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        progress_bar.update(self.global_step)
        if self.cfgs.data_cfgs.val_datasets:
            rst = self.val()
            self.logger.log(rst, step=self.global_step)
            self.logger.print(f'Validation loss: {rst["val/loss"]:.4f}')

        start_batch_idx = self.global_step % len(self.train_dataloader)

        for epoch in range(int(remain_epoch)):
            self.model.train()
            self.on_train_epoch_start()
            if self.current_epoch != 0:
                self.logger.print(
                    f'Resuming from checkpoint {self.current_epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                )
            for batch_idx, batch in enumerate(self.train_dataloader):
                if epoch == 0 and batch_idx < start_batch_idx:
                    continue

                info = self.train_step(batch)
                self.global_step += 1

                progress_bar.set_description(
                    f'Training {self.current_epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)
                info['train/epoch'] = self.global_step / len(self.train_dataloader)

                save_interval = (
                    self.cfgs.train_cfgs.epochs
                    * len(self.train_dataloader)
                    // self.cfgs.logger_cfgs.save_total_limit
                )
                if self.global_step % save_interval == 0:
                    self.save(tag=self.global_step)
                if (
                    self.cfgs.data_cfgs.val_datasets
                    and self.cfgs.train_cfgs.val_strategy == 'steps'
                    and self.global_step % self.cfgs.train_cfgs.val_interval == 0
                ):
                    self.logger.print(f'\n***** Validating at step {self.global_step} *****')
                    rst = self.val()
                    self.logger.log(rst, step=self.global_step)
                    self.logger.print(f'Validation loss: {rst["val/loss"]:.4f}')
            if self.cfgs.data_cfgs.val_datasets and self.cfgs.train_cfgs.val_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Validating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                rst = self.val()
                self.logger.log(rst, step=self.global_step)
                self.logger.print(f'Validation loss: {rst["val/loss"]:.4f}')
            self.model.tput_timer.update_epoch_count()

    def get_dataloader(self, subset: str):
        dataset = ChoresMultitaskDataset(
            base_data_dir=self.cfgs.data_cfgs.data_dir,
            dataset_names=self.cfgs.data_cfgs.dataset_task_type,
            subset=subset,
            max_samples=(
                self.cfgs.data_cfgs.max_samples
                if subset == 'train'
                else self.cfgs.data_cfgs.val_max_samples
            ),
            sliding_window=self.cfgs.data_cfgs.sliding_window,
            input_sensors=self.cfgs.sensor_cfgs.input_sensors,
            reduce_action_redundancy=(
                self.cfgs.data_cfgs.reduce_action_redundancy if subset == 'train' else False
            ),
        )
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=(
                self.cfgs.train_cfgs.per_device_train_batch_size
                if subset == 'train'
                else self.cfgs.train_cfgs.per_device_val_batch_size
            ),
            prefetch_factor=2,
            collate_fn=lambda batch: [sample for sample in batch if sample is not None],
            persistent_workers=False,
            pin_memory=True,
            num_workers=self.cfgs.data_cfgs.num_workers,
            sampler=sampler,
        )


if __name__ == '__main__':
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_video_to_action', 'sft')
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

    trainer = SupervisedTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()
