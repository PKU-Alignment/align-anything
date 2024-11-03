# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the tatsu-lab's stanford-alpaca library.
# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
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

from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Callable, Literal

import accelerate
import deepspeed
import torch
import torch.nn as nn
from accelerate.state import AcceleratorState
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, UNet3DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    ClapTextModelWithProjection,
    CLIPTextModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


try:
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    from transformers import is_deepspeed_zero3_enabled

from transformers.utils import ContextManagers

from align_anything.models.model_registry import AnyModel
from align_anything.utils.multi_process import get_current_device, is_main_process


DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'


# Reference: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
def resize_tokenizer_embedding(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    def verify_vocabulary_embedding_sizes(
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        format_message: Callable[[Any, Any], str],
    ) -> None:
        input_embeddings = model.get_input_embeddings()
        if (
            input_embeddings is not None
            and input_embeddings.num_embeddings != len(tokenizer)
            and is_main_process()
        ):
            warnings.warn(
                format_message(len(tokenizer), input_embeddings.num_embeddings),
                category=RuntimeWarning,
                stacklevel=3,
            )

    def init_new_embeddings(
        embeddings: nn.Embedding | nn.Linear | None,
        new_num_embeddings: int,
        num_new_embeddings: int,
    ) -> None:
        if embeddings is None:
            return

        params = [embeddings.weight, getattr(embeddings, 'bias', None)]
        context = (
            deepspeed.zero.GatheredParameters(params, modifier_rank=0)
            if is_deepspeed_zero3_enabled()
            else contextlib.nullcontext()
        )
        with context:
            for param in params:
                if param is None:
                    continue
                assert param.size(0) == new_num_embeddings
                param_data = param.data
                param_mean = param_data[:-num_new_embeddings].mean(dim=0, keepdim=True)
                param_data[-num_new_embeddings:] = param_mean

    verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            'The tokenizer vocabulary size ({}) is different from '
            'the model embedding size ({}) before resizing.'
        ).format,
    )

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_num_embeddings = len(tokenizer)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if num_new_tokens > 0:
        hf_device_map = getattr(model, 'hf_device_map', {})
        devices = {
            torch.device(device)
            for device in hf_device_map.values()
            if device not in {'cpu', 'disk'}
        }
        is_model_parallel = len(devices) > 1

        if not is_model_parallel:
            model.resize_token_embeddings(new_num_embeddings)
            init_new_embeddings(
                model.get_input_embeddings(),
                new_num_embeddings=new_num_embeddings,
                num_new_embeddings=num_new_tokens,
            )
            init_new_embeddings(
                model.get_output_embeddings(),
                new_num_embeddings=new_num_embeddings,
                num_new_embeddings=num_new_tokens,
            )

    verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            'The tokenizer vocabulary size ({}) is different from '
            'the model embedding size ({}) after resizing.'
        ).format,
    )


def load_pretrained_models(  # pylint: disable=too-many-arguments
    model_name_or_path: str | os.PathLike,
    model_max_length: int = 512,
    padding_side: Literal['left', 'right'] = 'right',
    auto_device_mapping: bool = False,
    freeze_vision_tower: bool = True,
    freeze_audio_tower: bool = True,
    freeze_mm_proj: bool = True,
    freeze_vision_proj: bool = True,
    freeze_audio_proj: bool = True,
    freeze_language_model: bool = False,
    dtype: torch.dtype | str | None = torch.bfloat16,
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    auto_model_args: tuple[Any, ...] = (),
    auto_model_kwargs: dict[str, Any] | None = None,
    auto_tokenizer_args: tuple[Any, ...] = (),
    auto_tokenizer_kwargs: dict[str, Any] | None = None,
    bnb_cfgs: dict[str, Any] | None = None,
    lora_cfgs: dict[str, Any] | None = None,
    processor_name_or_path: str | os.PathLike | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pre-trained model and tokenizer from a given path."""
    model_name_or_path = os.path.expanduser(model_name_or_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir is not None else None
    device_map = 'auto' if auto_device_mapping else None
    if auto_model_kwargs is None:
        auto_model_kwargs = {}
    if auto_tokenizer_kwargs is None:
        auto_tokenizer_kwargs = {}
        
    if bnb_cfgs and lora_cfgs:
        if bnb_cfgs.use_bnb:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=bnb_cfgs.load_in_4bit,
                load_in_8bit=bnb_cfgs.load_in_8bit,
                bnb_4bit_quant_type=bnb_cfgs.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_cfgs.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=bnb_cfgs.bnb_4bit_compute_dtype,
            )
        if lora_cfgs.use_lora:
            lora_config = LoraConfig(
                task_type=lora_cfgs.task_type,
                inference_mode=lora_cfgs.inference_mode,
                r=lora_cfgs.r,
                lora_alpha=lora_cfgs.lora_alpha,
                lora_dropout=lora_cfgs.lora_dropout,
                target_modules=lora_cfgs.target_modules,
            )
        if bnb_cfgs.use_bnb and not lora_cfgs.use_lora:
            raise ValueError(
                'bnb only is not compatible with deepspeeed, try working with peft + bnb + deepspeed by setting lora_cfgs.use_lora = True'
            )
        if lora_cfgs.use_lora and not bnb_cfgs.use_bnb:
            model = AnyModel.from_pretrained(
                model_name_or_path,
                *auto_model_args,
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **auto_model_kwargs,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        if lora_cfgs.use_lora and bnb_cfgs.use_bnb:
            model = AnyModel.from_pretrained(
                model_name_or_path,
                *auto_model_args,
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **auto_model_kwargs,
                quantization_config=quantization_config,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        if not lora_cfgs.use_lora and not bnb_cfgs.use_bnb:
            model = AnyModel.from_pretrained(
                model_name_or_path,
                *auto_model_args,
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **auto_model_kwargs,
            )
    else:
        model = AnyModel.from_pretrained(
            model_name_or_path,
            *auto_model_args,
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            **auto_model_kwargs,
        )
    if hasattr(model, 'audio_tower'):
        deepspeed.zero.register_external_parameter(model, model.audio_tower.embed_positions.weight)
    forbidden_modules = set()
    if freeze_vision_tower:
        forbidden_modules.add('vision_tower')
    if freeze_audio_tower:
        forbidden_modules.add('audio_tower')
    # attribute name of llava
    if freeze_mm_proj:
        forbidden_modules.add('multi_modal_projector')
    if freeze_vision_proj:
        forbidden_modules.add('image_projector')
    if freeze_audio_proj:
        forbidden_modules.add('audio_projector')
    if freeze_language_model:
        forbidden_modules.add('language_model')
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if dtype == torch.float32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        *auto_tokenizer_args,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
        **auto_tokenizer_kwargs,
    )

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        processor.tokenizer.padding_side = padding_side
        resize_tokenizer_embedding(tokenizer=processor.tokenizer, model=model)

        return model, processor.tokenizer, processor
    except Exception:
        processor = None
        resize_tokenizer_embedding(tokenizer=tokenizer, model=model)
        
        return model, tokenizer, processor


def load_pretrained_image_diffusion_models(  # pylint: disable=too-many-arguments
    model_name_or_path: str | os.PathLike,
    dtype: torch.dtype = torch.bfloat16,
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    non_ema_revision: str | None = None,
    variant: str | None = None,
    freeze_unet: bool = False,
    lora_unet: bool = False,
    lora_rank: int = 8,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pre-trained model and tokenizer from a given path."""
    model_name_or_path = os.path.expanduser(model_name_or_path)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder='scheduler')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        subfolder='tokenizer',
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            model_name_or_path,
            subfolder='text_encoder',
            revision=revision,
            variant=variant,
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder='vae',
            revision=revision,
            variant=variant,
            torch_dtype=dtype,
        )
    unet = UNet2DConditionModel.from_pretrained(
        model_name_or_path, subfolder='unet', revision=non_ema_revision, torch_dtype=dtype
    )
    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if freeze_unet:
        unet.requires_grad_(False)
    else:
        unet.train()
    current_device = get_current_device()
    text_encoder.to(current_device, dtype=dtype)
    vae.to(current_device, dtype=dtype)

    if lora_unet:
        # Set up LoRA.
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    resize_tokenizer_embedding(tokenizer=tokenizer, model=text_encoder)

    return unet, vae, text_encoder, noise_scheduler, tokenizer


def load_pretrained_video_diffusion_models(  # pylint: disable=too-many-arguments
    model_name_or_path: str | os.PathLike,
    dtype: torch.dtype = torch.bfloat16,
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    non_ema_revision: str | None = None,
    variant: str | None = None,
    freeze_unet: bool = False,
    lora_unet: bool = False,
    lora_rank: int = 8,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pre-trained model and tokenizer from a given path."""
    model_name_or_path = os.path.expanduser(model_name_or_path)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder='scheduler')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        subfolder='tokenizer',
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            model_name_or_path,
            subfolder='text_encoder',
            revision=revision,
            variant=variant,
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder='vae',
            revision=revision,
            variant=variant,
            torch_dtype=dtype,
        )
    unet = UNet3DConditionModel.from_pretrained(
        model_name_or_path, subfolder='unet', revision=non_ema_revision, torch_dtype=dtype
    )
    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if freeze_unet:
        unet.requires_grad_(False)
    else:
        unet.train()
    current_device = get_current_device()
    text_encoder.to(current_device, dtype=dtype)
    vae.to(current_device, dtype=dtype)

    if lora_unet:
        # Set up LoRA.
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    resize_tokenizer_embedding(tokenizer=tokenizer, model=text_encoder)

    return unet, vae, text_encoder, noise_scheduler, tokenizer


def load_pretrained_audio_diffusion_models(  # pylint: disable=too-many-arguments
    model_name_or_path: str | os.PathLike,
    dtype: torch.dtype = torch.bfloat16,
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    non_ema_revision: str | None = None,
    variant: str | None = None,
    freeze_unet: bool = False,
    lora_unet: bool = False,
    lora_rank: int = 8,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pre-trained model and tokenizer from a given path."""
    model_name_or_path = os.path.expanduser(model_name_or_path)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder='scheduler')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        subfolder='tokenizer',
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = ClapTextModelWithProjection.from_pretrained(
            model_name_or_path,
            subfolder='text_encoder',
            revision=revision,
            variant=variant,
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder='vae',
            revision=revision,
            variant=variant,
            torch_dtype=dtype,
        )
    unet = UNet2DConditionModel.from_pretrained(
        model_name_or_path, subfolder='unet', revision=non_ema_revision, torch_dtype=dtype
    )
    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if freeze_unet:
        unet.requires_grad_(False)
    else:
        unet.train()
    current_device = get_current_device()
    text_encoder.to(current_device, dtype=dtype)
    vae.to(current_device, dtype=dtype)

    if lora_unet:
        # Set up LoRA.
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    resize_tokenizer_embedding(tokenizer=tokenizer, model=text_encoder)

    return unet, vae, text_encoder, noise_scheduler, tokenizer
