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

import os
import sys
from typing import Any

import deepspeed
import torch
import torch.nn.functional as F

from accelerate import Accelerator

import torch.distributed as dist

from align_anything.datasets.text_to_image import AnyToImageDataset, AnyToImageBatch
from align_anything.models.pretrained_model import load_pretrained_diffusion_models
from align_anything.trainers.base import SupervisedTrainer
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
    parse_unknown_args,
    prepare_accelerate_train_cfgs
)

from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import is_compiled_module


class DiffusionTrainer(SupervisedTrainer):

    def __init__(self, cfgs) -> None:
        """Initialize the SFT trainer."""
        self.cfgs = cfgs
        self.muti_process_cfgs = prepare_accelerate_train_cfgs(custom_cfgs=cfgs.train_cfgs)
        self.global_step = 0
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.muti_process_cfgs['gradient_accumulation_steps'],
            mixed_precision=self.muti_process_cfgs['mixed_precision'],
        )
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
        return

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        self.model, self.vae, self.text_encoder, self.noise_scheduler, self.tokenizer = load_pretrained_diffusion_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=True,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(AnyToImageDataset, AnyToImageDataset)

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_accelerate_engines()

    def loss(self, sft_batch: AnyToImageBatch) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        latents = self.vae.encode(sft_batch["pixel_values"].to(self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        encoder_hidden_states = self.text_encoder(sft_batch["input_ids"], return_dict=False)[0]
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        model_pred = self.model(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return {
            'loss': loss,
        }

    def train_step(self, sft_batch: AnyToImageBatch) -> dict[str, Any]:
        """Performs a single training step."""
        loss = self.loss(sft_batch)['loss']
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfgs.train_cfgs.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return {
            'train/loss': loss.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.accelerator.wait_for_everyone()
        save_dir = os.path.join(self.cfgs.logger_cfgs.output_dir, f'epoch_{tag}')
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            model = model._orig_mod if is_compiled_module(model) else model
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.cfgs.model_cfgs.model_name_or_path,
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=model,
                revision=self.cfgs.train_cfgs.revision,
                variant=self.cfgs.train_cfgs.variant,
            )
            pipeline.save_pretrained(save_dir)

        self.logger.print('Model saved!')


def main():
    # setup distribution training
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    dict_cfgs, _ = read_cfgs(mode='train', task='sft')
    unparsed_args = parse_unknown_args()
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = DiffusionTrainer(cfgs=cfgs)
    trainer.train()
    trainer.save()
    trainer.accelerator.end_training()

if __name__ == '__main__':
    sys.exit(main())
