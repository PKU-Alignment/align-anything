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
"""Trainer for diffusion DPO."""


import os
import sys
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from align_anything.datasets.text_to_image import PreferenceBatch, PreferenceDataset
from align_anything.models.pretrained_model import load_pretrained_image_diffusion_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
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


class DPOTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs) -> None:
        """Initialize the DPO diffusion trainer.

        References:
            - Title: DiffusionModel Alignment Using Direct Preference Optimization
            - Authors: Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil
            Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, Nikhil Naik.
        """
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
        self.ref_model, _, _, _, _ = (
            load_pretrained_image_diffusion_models(
                self.cfgs.model_cfgs.model_name_or_path,
                trust_remote_code=True,
                freeze_unet=True,
                dtype=self.dtype,
            )
        )
        self.processor = get_image_processor(resolution=int(self.cfgs.train_cfgs.resolution))

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_engines(self) -> None:
        """Initialize Accelerate engines."""
        self.init_accelerate_engines()

    def loss(self, batch: PreferenceBatch) -> dict[str, torch.Tensor]:
        """Loss function for DPO finetuning."""
        pixel_values = batch['pixel_values'].to(dtype=self.dtype)
        feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))
        latents = []
        for i in range(0, feed_pixel_values.shape[0], self.cfgs.train_cfgs.vae_encode_batch_size):
            latents.append(
                self.vae.encode(
                    feed_pixel_values[i : i + self.cfgs.train_cfgs.vae_encode_batch_size]
                ).latent_dist.sample()
            )
        latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)
        batch_size = latents.shape[0] // 2

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
            dtype=torch.long,
        ).repeat(2)

        encoder_hidden_states = self.text_encoder(batch['input_ids'], attention_mask=None)[
            0
        ].repeat(2, 1, 1)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = self.model(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.noise_scheduler.config.prediction_type}'
            )

        model_losses = F.mse_loss(model_pred.float(), target.float(), reduction='none')
        model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
        model_losses_w, model_losses_l = model_losses.chunk(2)

        # For logging
        model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)

        # Reference model predictions
        if self.cfgs.train_cfgs.lora_unet:
            self.accelerator.unwrap_model(self.model).disable_adapters()
        with torch.no_grad():
            ref_preds = self.ref_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample.detach()
            ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction='none')
            ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

            ref_losses_w, ref_losses_l = ref_loss.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l

        if self.cfgs.train_cfgs.lora_unet:
            self.accelerator.unwrap_model(self.model).enable_adapters()

        logits = ref_diff - model_diff
        if self.cfgs.train_cfgs.loss_type == 'sigmoid':
            loss = -F.logsigmoid(self.cfgs.train_cfgs.beta_coeff * logits).mean()
        elif self.cfgs.train_cfgs.loss_type == 'hinge':
            loss = torch.relu(1 - self.cfgs.train_cfgs.beta_coeff * logits).mean()
        else:
            raise ValueError(f'Unknown loss type {self.cfgs.train_cfgs.loss_type}')

        implicit_acc = (logits > 0).sum().float() / logits.size(0)
        implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)

        return {
            'loss': loss,
            'reward_accuracy': implicit_acc,
            'model_diff': model_diff.mean().detach(),
            'ref_diff': ref_diff.mean().detach(),
        }

    def train_step(self, batch: PreferenceBatch) -> dict[str, Any]:
        """Performs a single training step."""
        loss_info = self.loss(batch)
        loss = loss_info['loss']
        reward_accuracy = loss_info['reward_accuracy']
        model_diff = loss_info['model_diff']
        ref_diff = loss_info['ref_diff']

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.params_to_optimize, self.cfgs.train_cfgs.max_grad_norm
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            loss = get_all_reduce_mean(loss)
            reward_accuracy = get_all_reduce_mean(reward_accuracy)
            model_diff = get_all_reduce_mean(model_diff)
            ref_diff = get_all_reduce_mean(ref_diff)

        return {
            'train/loss': loss.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
            'train/reward_accuracy': reward_accuracy.item(),
            'train/model_diff': model_diff.item(),
            'train/ref_diff': ref_diff.item(),
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
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(reward accuracy {info["train/reward_accuracy"]:.4f})',
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

    def save(self, tag: int | None = None) -> None:
        """Save the stable diffusion pipeline in Hugging Face format."""
        self.save_diffusers(tag=tag)


def main():
    # setup distribution training
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_image', 'dpo')
    dict_cfgs, _ = read_cfgs(mode='train', task=task)
    unparsed_args = parse_unknown_args()
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = DPOTrainer(cfgs=cfgs)
    trainer.train()
    trainer.save()
    trainer.accelerator.end_training()


if __name__ == '__main__':
    sys.exit(main())
