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

from __future__  import annotations
import json
import os
import argparse

import sys
from tqdm.auto import tqdm
from vllm import SamplingParams,LLM
PROMPT_USER: str = 'USER: ##Prompt: {prompt} ##Response: {response} Your critique and refinement:'
PROMPT_ASSISTANT: str = '\nASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_USER + PROMPT_ASSISTANT
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import requests


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )
    # Model
    parser.add_argument(
       '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
    )
    parser.add_argument(
        '--output_name',
        type=str,
        help='the name of the output json file',
        default=None,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Where to store the eval output.',
    )
    parser.add_argument(
        '--input_path',
        type=str,
        help='The path of the input json file.',
    )
    return parser.parse_args()


def generate_answer_by_vllm(problems: list[str], model_name_or_path:str) ->list[str]:
    samplingparams = SamplingParams(
        temperature = 1.0,
        repetition_penalty = 1.1,
        max_tokens = 2048,
        n=1,
    )
    
    llm = LLM(model=model_name_or_path, 
              gpu_memory_utilization=0.95, 
              swap_space=32, 
              trust_remote_code=True, 
              tensor_parallel_size=8)
    model_name = model_name_or_path.split('/')[-1]
    outputs = llm.generate(problems, samplingparams)
    answers = []
    
    for output, entry in tqdm(zip(outputs, problems)) :
        items = []
        for i in range(len(output.outputs)):
            item = {
                'from': model_name,
                'response': output.outputs[i].text.strip()
            }
            items.append(item)
        answers.append(items)
    
    return answers

def main() -> None:
    args = parse_arguments()
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = []
    for idx in range(len(data)):
        image_file = data[idx]['image']
        prompt = f"{data[idx]['prompt'].replace('<image>', ' image ').replace('<image>', ' image ')} <image>"
        response = data[idx]['output_text'].replace('<image>', ' image ').replace('<image>', ' image ')
        problem = {
            'prompt': PROMPT_INPUT.format(prompt=prompt, response=response),
            'multi_modal_data': {'image': Image.open(image_file)},
        }
        problems.append(problem)
        
    answers = generate_answer_by_vllm(problems, args.model_name_or_path)
    final_answer = []
    for idx in range(len(answers)):
        item = answers[idx]
        final_answer.append(item)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, answer in enumerate(answers):
        data[idx]['generated'] = answer
    if args.output_name is None:
        args.output_name = f"generated_{args.input_path.split('/')[-1]}_{args.model_name_or_path.split('/')[-1]}.json"
    else:
        args.output_name = f"{args.output_name}.json"
    output_file = os.path.join(args.output_dir, args.output_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
if __name__=='__main__':
    sys.exit(main())