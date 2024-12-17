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
"""Trainer base for supervised training."""


import os
from abc import abstractmethod
from datetime import datetime
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam
from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import CONFIG_NAME, PreTrainedModel, get_scheduler

from align_anything.configs.template import ChatTemplate
from align_anything.datasets.any_to_text import (
    CombinedDataset,
    DistributedCombinedDatasetBatchSampler,
)
from align_anything.utils.logger import Logger
from align_anything.utils.multi_process import is_main_process
from align_anything.utils.tools import get_optimizer_grouped_parameters, namedtuple_to_dict


class SupervisedTrainerBase:

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize the SFT trainer."""
        self.cfgs = cfgs

    def init_check(self) -> None:
        """Initial configuration checking."""
        self.lora_enabled = False
        if self.cfgs.lora_cfgs and self.cfgs.lora_cfgs.use_lora:
            self.lora_enabled = True
            self.save_full_model = self.cfgs.lora_cfgs.save_full_model

    @abstractmethod
    def init_models(self) -> None:
        """Initialize model and tokenizer."""

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

    def get_dataloaders(self, train_data_dtype, eval_data_dtype) -> None:
        """Get the dataloaders based on data_dtype."""
        train_dataloader = []
        eval_dataloader = []
        if isinstance(self.cfgs.data_cfgs.train_datasets, str):
            formatter = self.processor if self.processor else self.tokenizer
            custom_formatter = (
                self.model.apply_chat_template
                if hasattr(self.model, 'apply_chat_template')
                else None
            )
            self.train_template = ChatTemplate(
                formatter, self.cfgs.data_cfgs.train_template, custom_formatter
            )
            train_dataset = train_data_dtype(
                path=self.cfgs.data_cfgs.train_datasets,
                template=self.train_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name=self.cfgs.data_cfgs.train_name,
                size=self.cfgs.data_cfgs.train_size,
                split=self.cfgs.data_cfgs.train_split,
                data_files=self.cfgs.data_cfgs.train_data_files,
                optional_args=self.cfgs.data_cfgs.train_optional_args,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.get_collator(),
                sampler=DistributedSampler(train_dataset, shuffle=True),
                batch_size=int(self.cfgs.train_cfgs.per_device_train_batch_size),
            )
        elif isinstance(self.cfgs.data_cfgs.train_datasets, list):
            train_datasets = []
            self.train_template = []
            for i in range(len(self.cfgs.data_cfgs.train_datasets)):
                formatter = self.processor if self.processor else self.tokenizer
                self.train_template.append(
                    ChatTemplate(formatter, self.cfgs.data_cfgs.train_template[i])
                )
                train_datasets.append(
                    train_data_dtype(
                        path=self.cfgs.data_cfgs.train_datasets[i],
                        template=self.train_template[i],
                        tokenizer=self.tokenizer,
                        processor=self.processor,
                        name=(
                            self.cfgs.data_cfgs.train_name[i]
                            if self.cfgs.data_cfgs.train_name
                            else None
                        ),
                        size=(
                            self.cfgs.data_cfgs.train_size[i]
                            if self.cfgs.data_cfgs.train_size
                            else None
                        ),
                        split=(
                            self.cfgs.data_cfgs.train_split[i]
                            if self.cfgs.data_cfgs.train_split
                            else None
                        ),
                        data_files=(
                            self.cfgs.data_cfgs.train_data_files[i]
                            if self.cfgs.data_cfgs.train_data_files
                            else None
                        ),
                        optional_args=(
                            self.cfgs.data_cfgs.train_optional_args[i]
                            if len(self.cfgs.data_cfgs.train_optional_args) > 0
                            else []
                        ),
                    )
                )
            combined_train_dataset = CombinedDataset(train_datasets)
            train_dataloader = DataLoader(
                combined_train_dataset,
                collate_fn=combined_train_dataset.get_collator(),
                batch_sampler=DistributedCombinedDatasetBatchSampler(
                    train_datasets,
                    shuffle=True,
                    drop_last=True,
                    batch_size=int(self.cfgs.train_cfgs.per_device_train_batch_size),
                ),
            )
        if isinstance(self.cfgs.data_cfgs.eval_datasets, str):
            formatter = self.processor if self.processor else self.tokenizer
            custom_formatter = (
                self.model.apply_chat_template
                if hasattr(self.model, 'apply_chat_template')
                else None
            )
            self.eval_template = ChatTemplate(
                formatter, self.cfgs.data_cfgs.eval_template, custom_formatter
            )
            eval_dataset = eval_data_dtype(
                path=self.cfgs.data_cfgs.eval_datasets,
                template=self.eval_template,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name=self.cfgs.data_cfgs.eval_name,
                split=self.cfgs.data_cfgs.eval_split,
                size=self.cfgs.data_cfgs.eval_size,
                data_files=self.cfgs.data_cfgs.eval_data_files,
                optional_args=self.cfgs.data_cfgs.eval_optional_args,
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=int(self.cfgs.train_cfgs.per_device_eval_batch_size),
            )
        elif isinstance(self.cfgs.data_cfgs.eval_datasets, list):
            eval_dataloader = {}
            self.eval_template = []
            for i in range(len(self.cfgs.data_cfgs.eval_datasets)):
                formatter = self.processor if self.processor else self.tokenizer
                self.eval_template.append(
                    ChatTemplate(formatter, self.cfgs.data_cfgs.eval_template[i])
                )
                eval_dataset = eval_data_dtype(
                    path=self.cfgs.data_cfgs.eval_datasets[i],
                    template=self.eval_template[i],
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    name=(
                        self.cfgs.data_cfgs.eval_name[i] if self.cfgs.data_cfgs.eval_name else None
                    ),
                    split=(
                        self.cfgs.data_cfgs.eval_split[i]
                        if self.cfgs.data_cfgs.eval_split
                        else None
                    ),
                    size=(
                        self.cfgs.data_cfgs.eval_size[i] if self.cfgs.data_cfgs.eval_size else None
                    ),
                    data_files=(
                        self.cfgs.data_cfgs.eval_data_files[i]
                        if self.cfgs.data_cfgs.eval_data_files
                        else None
                    ),
                    optional_args=(
                        self.cfgs.data_cfgs.eval_optional_args[i]
                        if len(self.cfgs.data_cfgs.eval_optional_args) > 0
                        else []
                    ),
                )
                raw_eval_dataloader = DataLoader(
                    eval_dataset,
                    collate_fn=eval_dataset.get_collator(),
                    sampler=DistributedSampler(eval_dataset, shuffle=True),
                    batch_size=int(self.cfgs.train_cfgs.per_device_eval_batch_size),
                )
                eval_dataloader[self.cfgs.data_cfgs.eval_template[i]] = raw_eval_dataloader

        return train_dataloader, eval_dataloader

    def init_deepspeed_engines(self) -> None:
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
            config=self.ds_train_cfgs,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )
        if self.cfgs.train_cfgs.gradient_checkpointing and not self.lora_enabled:
            self.model.gradient_checkpointing_enable()

    def init_accelerate_engines(self) -> None:
        """Initialize Accelerate engines."""
        num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.cfgs.train_cfgs.gradient_accumulation_steps - 1
        ) // self.cfgs.train_cfgs.gradient_accumulation_steps
        total_training_steps = (
            self.cfgs.train_cfgs.epochs
            * num_update_steps_per_epoch
            * self.cfgs.train_cfgs.gradient_accumulation_steps
        )
        self.params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.cfgs.train_cfgs.learning_rate,
            betas=self.cfgs.train_cfgs.adam_betas,
            eps=self.cfgs.train_cfgs.adam_epsilon,
        )

        num_warmup_steps = int(self.cfgs.train_cfgs.lr_warmup_ratio * total_training_steps)
        self.lr_scheduler = get_scheduler(
            name=self.cfgs.train_cfgs.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )
        if self.cfgs.train_cfgs.gradient_checkpointing and not self.lora_enabled:
            self.model.enable_gradient_checkpointing()
        if hasattr(self, 'ref_model') and self.ref_model is not None:
            self.model, self.ref_model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model,
                    self.ref_model,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                )
            )
            self.ref_model.eval()
        else:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            )

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
            self.model.tput_timer.update_epoch_count()

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        return {}

    def save_transformers(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save transformers model and tokenizer in Hugging Face format."""
        dist.barrier()

        if model is None:
            model = self.model  # pylint: disable=no-member

        output_dir = os.path.join(self.cfgs.logger_cfgs.output_dir, f'slice_{tag or "end"}')
        os.makedirs(output_dir, exist_ok=True)

        self.logger.print(f'Saving model to "{output_dir}" ...')

        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        model_to_save: PreTrainedModel = getattr(model, 'module', model)

        if is_main_process():
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_pretrained(output_dir)
            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

        if not self.lora_enabled:
            self.logger.print('Saving 16-bit model...')
            zero_stage = self.ds_train_cfgs.get('zero_optimization', {}).get('stage', 0)
            if zero_stage >= 2:
                save_file_name = 'pytorch_model.bin'
                model.save_16bit_model(output_dir, save_filename=save_file_name)
            else:
                if is_main_process():
                    model_to_save.save_pretrained(output_dir, is_main_process=True)
            self.logger.print('Model saved!')
        if self.lora_enabled and not self.lora_cfgs.save_full_model:
            self.logger.print('LoRA used. Saving model as LoRA adapters...')
            model.save_pretrained(output_dir)
            self.logger.print('Model saved!')
        if self.lora_enabled and self.lora_cfgs.save_full_model:
            self.logger.print('LoRA used. Saving full model...')
            model = model.module
            model_to_be_saved = model.merge_and_unload()
            model_to_be_saved.save_pretrained(output_dir)
            self.logger.print('Model saved!')

    def save_diffusers(
        self,
        tag: int | None = None,
    ) -> None:
        """Save the stable diffusion pipeline in Hugging Face format."""
        self.accelerator.wait_for_everyone()
        save_dir = os.path.join(self.cfgs.logger_cfgs.output_dir, f'epoch_{tag or "end"}')
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            if self.cfgs.train_cfgs.lora_unet:
                model = model.to(torch.float32)
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.cfgs.model_cfgs.model_name_or_path
                )
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(model)
                )
                LoraLoaderMixin.save_lora_weights(
                    save_directory=save_dir,
                    unet_lora_layers=unet_lora_state_dict,
                    text_encoder_lora_layers=None,
                )
            else:
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
