# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the DAMO-NLP-SG's VideoLLaMA2 library.
# https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/model/__init__.py
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

import torch
import copy
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from .videollama2_mistral import Videollama2MistralForCausalLM
from .videollama2_mixtral import Videollama2MixtralForCausalLM
from .videollama2_qwen2 import Videollama2Qwen2ForCausalLM
from .videollama2_gemma2 import Videollama2Gemma2ForCausalLM
from .videollama2_phi3 import Videollama2Phi3ForCausalLM
from functools import partial
from .utils import process_video, tokenizer_multimodal_token, KeywordsStoppingCriteria

def model_init(model_path, **kwargs):
    tokenizer, model, processor = load_pretrained_model(model_path, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else 8
    processor = partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames)

    return model, processor, tokenizer

def mm_infer(image_or_video, instruct, model, tokenizer, **kwargs):
    tensor = image_or_video.half().cuda()
    tensor = [(tensor, 'video')]

    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': "<video>" + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = "<video>" + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = True
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    config = AutoConfig.from_pretrained(model_path)

    model_type = config.model_type

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

    if model_type in ['videollama2', 'videollama2_mistral']:
        model = Videollama2MistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    elif model_type in ['videollama2_mixtral']:
        model = Videollama2MixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    elif model_type in ['videollama2_qwen2']:
        model = Videollama2Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    elif model_type in ['videollama2_gemma2']:
        model = Videollama2Gemma2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    elif model_type in ['videollama2_phi3']:
        model = Videollama2Phi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    else:
        model = Videollama2MistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    processor = vision_tower.image_processor

    return tokenizer, model, processor