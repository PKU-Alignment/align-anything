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
import os

import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from datasets import load_dataset


pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

USER_PROMPT_TEMPLATE = """\
USER:
You are about to receive a user instruction. Please convert the user instruction into an instruction to an image generation model. Make sure that the instruction for the image generation should be accurate and concise, no more than 40 words. You should only output the instruction prompt.

[User Instruction]
{prompt}
[/User Instruction]
ASSISTANT:
"""


def generate_image_instruction(prompt):
    """
    Implement the inference process of the image generation part of the model.
    """


def inference(data):
    """
    Implement the inference process of the audio generation part of the model.
    """
    prompt = USER_PROMPT_TEMPLATE.format(prompt=data['prompt'])
    image_instruction = generate_image_instruction(prompt)

    image = pipe(
        image_instruction,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator('cpu').manual_seed(0),
    ).images[0]
    return image


dataset = load_dataset(
    'PKU-Alignment/EvalAnything-InstructionFollowing',
    name='image_instruct',
    split='test',
    trust_remote_code=True,
)


os.makedirs('.cache/image_instruct', exist_ok=True)

results = []

for data in tqdm(dataset):
    image = inference(data)
    image.save(f".cache/image_instruct/{data['prompt_id']}.png")

    results.append(
        {
            'prompt_id': data['prompt_id'],
            'prompt': data['prompt'],
            'image_path': f".cache/image_instruct/{data['prompt_id']}.png",
        }
    )

with open('.cache/image_instruct/generated_results.json', 'w') as f:
    json.dump(results, f)
