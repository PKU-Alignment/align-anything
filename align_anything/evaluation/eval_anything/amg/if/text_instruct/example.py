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

from tqdm import tqdm

from datasets import load_dataset


def generate_text_instruction(prompt):
    """
    Implement the inference process of the text generation part of the model.
    """


def inference(data):
    """
    Implement the inference process of the text generation part of the model.
    """
    return generate_text_instruction(data['prompt'])


dataset = load_dataset(
    'PKU-Alignment/EvalAnything-InstructionFollowing',
    name='text_instruct',
    split='test',
    trust_remote_code=True,
)


os.makedirs('.cache/text_instruct', exist_ok=True)

results = []

for data in tqdm(dataset):
    response = inference(data)
    results.append({'prompt_id': data['prompt_id'], 'prompt': data['prompt'], 'response': response})

with open('.cache/text_instruct/generated_results.json', 'w') as f:
    json.dump(results, f)
