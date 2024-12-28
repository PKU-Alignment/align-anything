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

import json
import math

import numpy as np
from transformers import AutoModel, AutoProcessor


model = AutoModel.from_pretrained('PKU-Alignment/AnyRewardModel')
processor = AutoProcessor.from_pretrained('PKU-Alignment/AnyRewardModel')

model.eval()

user_prompt: str = 'USER: {input}'
assistant_prompt: str = '\nASSISTANT:\n{modality}{text_response}'


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def process_ia(prompt, image_path, audio_path):
    image_pixel_values = processor(data_paths=image_path, modality='image').pixel_values
    audio_pixel_values = processor(data_paths=audio_path, modality='audio').pixel_values

    text_input = processor(
        text=user_prompt.format(input=prompt)
        + assistant_prompt.format(modality='<image><audio>', text_response=''),
        modality='text',
    )
    return {
        'input_ids': text_input.input_ids,
        'attention_mask': text_input.attention_mask,
        'pixel_values_1': image_pixel_values.unsqueeze(0),
        'pixel_values_2': audio_pixel_values.unsqueeze(0),
        'modality': [['image', 'audio']],
    }


def process_ta(prompt, response, audio_path):
    audio_pixel_values = processor(data_paths=audio_path, modality='audio').pixel_values
    text_input = processor(
        text=user_prompt.format(input=prompt)
        + assistant_prompt.format(modality='<audio>', text_response=response),
        modality='text',
    )
    return {
        'input_ids': text_input.input_ids,
        'attention_mask': text_input.attention_mask,
        'pixel_values_1': audio_pixel_values.unsqueeze(0),
        'modality': [['audio', 'text']],
    }


def process_ti(prompt, response, image_path):
    image_pixel_values = processor(data_paths=image_path, modality='image').pixel_values
    text_input = processor(
        text=user_prompt.format(input=prompt)
        + assistant_prompt.format(modality='<image>', text_response=response),
        modality='text',
    )
    return {
        'input_ids': text_input.input_ids,
        'attention_mask': text_input.attention_mask,
        'pixel_values_1': image_pixel_values.unsqueeze(0),
        'modality': [['image', 'text']],
    }


def reward_eval_ia(example):
    image_path = example['response'][0]
    audio_path = example['response'][1]
    prompt = example['prompt']
    return sigmoid(
        model(**process_ia(prompt, image_path, audio_path)).end_scores.squeeze(dim=-1).item()
    )


def reward_eval_ta(example):
    response = example['response'][0]
    audio_path = example['response'][1]
    prompt = example['prompt']
    return sigmoid(
        model(**process_ta(prompt, response, audio_path)).scores.squeeze(dim=-1).mean().item()
    )


def reward_eval_ti(example):
    response = example['response'][0]
    image_path = example['response'][1]
    prompt = example['prompt']
    return sigmoid(
        model(**process_ti(prompt, response, image_path)).scores.squeeze(dim=-1).mean().item()
    )


def main():
    result_ia_file = 'example_ia.json'
    result_ta_file = 'example_ta.json'
    result_ti_file = 'example_ti.json'

    with open(result_ia_file) as f:
        result_ia = json.load(f)
    with open(result_ta_file) as f:
        result_ta = json.load(f)
    with open(result_ti_file) as f:
        result_ti = json.load(f)

    reward_ia = []
    reward_ta = []
    reward_ti = []
    for result in result_ia:
        reward_ia.append(reward_eval_ia(result))
    for result in result_ta:
        reward_ta.append(reward_eval_ta(result))
    for result in result_ti:
        reward_ti.append(reward_eval_ti(result))

    with open('reward_score.json', 'w') as f:
        json.dump(
            {
                'reward_ia': np.mean(reward_ia),
                'reward_ta': np.mean(reward_ta),
                'reward_ti': np.mean(reward_ti),
            },
            f,
        )


if __name__ == '__main__':
    main()
