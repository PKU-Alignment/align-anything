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
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm

from align_anything.utils.device_utils import get_current_device
from datasets import load_dataset


pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-2b', torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

USER_PROMPT_TEMPLATE = """\
You are about to receive a user instruction. Please convert the user instruction into an instruction to an video generation model. Make sure that the instruction for the video generation should be accurate and concise, no more than 40 words. You should only output the instruction prompt.

[User Instruction]
{prompt}
[/User Instruction]
"""


def generate_video_instruction(prompt):
    """
    Implement the inference process of the video generation part of the model.
    """


def inference(data):
    """
    Implement the inference process of the video generation part of the model.
    """
    prompt = USER_PROMPT_TEMPLATE.format(prompt=data['prompt'])
    video_instruction = generate_video_instruction(prompt)

    video = pipe(
        prompt=video_instruction,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device=get_current_device()).manual_seed(42),
    ).frames[0]

    return video


dataset = load_dataset(
    'PKU-Alignment/EvalAnything-InstructionFollowing',
    name='video_instruct',
    split='test',
    trust_remote_code=True,
)


os.makedirs('.cache/video_instruct', exist_ok=True)

results = {}

for data in tqdm(dataset):
    video = inference(data)
    export_to_video(video, f".cache/video_instruct/{data['prompt_id']}.mp4")

    results[data['prompt_id']] = f".cache/video_instruct/{data['prompt_id']}.mp4"

with open('.cache/video_instruct/generated_results.json', 'w') as f:
    json.dump(results, f)
