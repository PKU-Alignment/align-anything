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

import scipy
import torch
from diffusers import AudioLDM2Pipeline
from tqdm import tqdm

from datasets import load_dataset


repo_id = 'cvssp/audioldm2'
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

USER_PROMPT_TEMPLATE = """\
You are about to receive a user instruction. Please convert the user instruction into an instruction to an audio generation model. Make sure that the instruction for the audio generation should be accurate and concise, no more than 40 words. You should only output the instruction prompt.

[User Instruction]
{prompt}
[/User Instruction]
"""


def generate_audio_instruction(prompt):
    """
    Implement the inference process of the audio generation part of the model.
    """


def inference(data):
    """
    Implement the inference process of the audio generation part of the model.
    """
    prompt = USER_PROMPT_TEMPLATE.format(prompt=data['prompt'])
    audio_instruction = generate_audio_instruction(prompt)

    audio = pipe(audio_instruction, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
    return audio


dataset = load_dataset(
    'PKU-Alignment/EvalAnything-InstructionFollowing',
    name='audio_instruct',
    split='test',
    trust_remote_code=True,
)


os.makedirs('.cache/audio_instruct', exist_ok=True)

results = {}

for data in tqdm(dataset):
    audio = inference(data)
    scipy.io.wavfile.write(f".cache/audio_instruct/{data['prompt_id']}.wav", rate=16000, data=audio)

    results[data['prompt_id']] = f".cache/audio_instruct/{data['prompt_id']}.wav"

with open('.cache/audio_instruct/generated_results.json', 'w') as f:
    json.dump(results, f)
