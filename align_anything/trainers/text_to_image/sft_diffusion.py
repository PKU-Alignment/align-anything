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
"""Trainer for diffusion SFT."""


import os
import sys
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from align_anything.datasets.text_to_image import SupervisedBatch, SupervisedDataset
from align_anything.models.pretrained_model import load_pretrained_image_diffusion_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import get_current_device, torch_gc, torch_set_device
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.process_image import get_image_processor
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    parse_unknown_args,
    prepare_accelerate_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


class SupervisedTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs) -> None:
        """Initialize the SFT diffusion trainer."""
        self.cfgs = cfgs
        self.muti_process_cfgs = prepare_accelerate_train_cfgs(custom_cfgs=cfgs.train_cfgs)
        self.global_step = 0
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.muti_process_cfgs['gradient_accumulation_steps'],
            mixed_precision=self.muti_process_cfgs['mixed_precision'],
        )
        self.dtype = torch.bfloat16 if self.muti_process_cfgs['mixed_precision'] else torch.float32
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

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
        self.model, self.vae, self.text_encoder, self.noise_scheduler, self.tokenizer = (
            load_pretrained_image_diffusion_models(
                self.cfgs.model_cfgs.model_name_or_path,
                trust_remote_code=True,
                freeze_unet=self.cfgs.train_cfgs.freeze_unet,
                lora_unet=self.cfgs.train_cfgs.lora_unet,
                dtype=self.dtype,
            )
        )
        self.processor = get_image_processor(resolution=int(self.cfgs.train_cfgs.resolution))

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            SupervisedDataset, SupervisedDataset
        )

    def init_engines(self) -> None:
        """Initialize Accelerate engines."""
        self.init_accelerate_engines()

    def loss(self, batch: SupervisedBatch) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        latents = self.vae.encode(batch['pixel_values'].to(self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
        )
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch['input_ids'], return_dict=False)[0]
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.noise_scheduler.config.prediction_type}'
            )

        model_pred = self.model(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[
            0
        ]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

        return {
            'loss': loss,
        }

    def train_step(self, batch: SupervisedBatch) -> dict[str, Any]:
        """Performs a single training step."""
        loss = self.loss(batch)['loss']
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.params_to_optimize, self.cfgs.train_cfgs.max_grad_norm
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return {
            'train/loss': loss.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
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

        for epoch in range(int(self.cfgs.train_cfgs.epochs)):
            self.model.train()

            for batch in self.train_dataloader:
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

    def save(self, tag: int | None = None) -> None:
        """Save the stable diffusion pipeline in Hugging Face format."""
        self.save_diffusers(tag=tag)


def main():
    # setup distribution training
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_image', 'sft')
    dict_cfgs, _ = read_cfgs(mode='train', task=task)
    unparsed_args = parse_unknown_args()
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = SupervisedTrainer(cfgs=cfgs)
    trainer.train()
    trainer.save()
    trainer.accelerator.end_training()


if __name__ == '__main__':
    sys.exit(main())
