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

import hashlib
import json
import os
import re
from argparse import ArgumentParser

from agent.actions.modality_generator import *
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--image_model_name_or_path', type=str, required=False)
    parser.add_argument('--audio_model_name_or_path', type=str, required=True)
    parser.add_argument('--video_model_name_or_path', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


def extract_instructions(text):
    response_pattern = r'\[\[response\]\]:\s*(.*?)(?=\n\n|\[\[image_instruction\]\])'
    image_pattern = r'\[\[image_instruction\]\]:\s*(.*?)(?=\n\n|\[\[audio_instruction\]\])'
    video_pattern = r'\[\[video_instruction\]\]:\s*(.*?)(?=\n\n|\[\[audio_instruction\]\])'
    audio_pattern = r'\[\[audio_instruction\]\]:\s*(.*?)(?=\n|\[/output_format\])'

    response = re.search(response_pattern, text, re.DOTALL)
    image_instruction = re.search(image_pattern, text, re.DOTALL)
    audio_instruction = re.search(audio_pattern, text, re.DOTALL)
    video_instruction = re.search(video_pattern, text, re.DOTALL)

    return {
        'response': response.group(1).strip() if response else None,
        'image_instruction': image_instruction.group(1).strip() if image_instruction else None,
        'audio_instruction': audio_instruction.group(1).strip() if audio_instruction else None,
        'video_instruction': video_instruction.group(1).strip() if video_instruction else None,
    }


def prepare_inputs(input_file: str, args):
    with open(input_file, encoding='utf-8') as f:
        data = json.load(f)
    inputs = []
    for item in data:
        instructions = extract_instructions(item['output'])
        if instructions['response'] is None or instructions['audio_instruction'] is None:
            continue
        if args.image_model_name_or_path is not None:
            if instructions['image_instruction'] is None:
                continue
            inputs.append(
                {
                    'instruction_uid': generate_hash_uid(item['prompt']),
                    'instruction': item['prompt'],
                    'text_response': instructions['response'],
                    'image_instruction': instructions['image_instruction'],
                    'audio_instruction': instructions['audio_instruction'],
                    'text_response_uid': generate_hash_uid(instructions['response']),
                }
            )
        if args.video_model_name_or_path is not None:
            if instructions['video_instruction'] is None:
                continue
            inputs.append(
                {
                    'instruction_uid': generate_hash_uid(item['prompt']),
                    'instruction': item['prompt'],
                    'text_response': instructions['response'],
                    'video_instruction': instructions['video_instruction'],
                    'audio_instruction': instructions['audio_instruction'],
                    'text_response_uid': generate_hash_uid(instructions['response']),
                }
            )

    print('len(raw_input): ', len(data))
    print('len(inputs): ', len(inputs))
    return inputs


def prepare_pipeline(
    image_model_name_or_path: str,
    video_model_name_or_path: str,
    audio_model_name_or_path: str,
    output_dir: str,
):
    generator = ModalityGenerator(
        image_model_path=image_model_name_or_path,
        video_model_path=video_model_name_or_path,
        audio_model_path=audio_model_name_or_path,
        output_dir=output_dir,
    )
    return generator


def generate(instruction: dict, generator):
    if 'image_instruction' in instruction.keys():
        return generator.generate(
            instruction_uid=instruction['instruction_uid'],
            image_prompt=instruction['image_instruction'],
            audio_prompt=instruction['audio_instruction'],
        )
    else:
        return generator.generate(
            instruction_uid=instruction['instruction_uid'],
            video_prompt=instruction['video_instruction'],
            audio_prompt=instruction['audio_instruction'],
        )


def generate_hash_uid(to_hash: dict | tuple | list | str) -> str:
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    return hash_object.hexdigest()


if __name__ == '__main__':
    args = parse_args()
    inputs = prepare_inputs(args.input_path, args)

    print(inputs[0])
    if args.image_model_name_or_path is not None:
        generator = prepare_pipeline(
            image_model_name_or_path=args.image_model_name_or_path,
            video_model_name_or_path=args.video_model_name_or_path,
            audio_model_name_or_path=args.audio_model_name_or_path,
            output_dir=args.output_dir,
        )
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for input in tqdm(inputs):
            save_path = generate(input, generator)
            output = {
                'uuid': input['instruction_uid'],
                'instruction': input['instruction'],
                'response': input['text_response'],
                'modality': {
                    'image': save_path['image_path'],
                    'video': None,
                    'audio': save_path['audio_path'],
                },
                'source': {
                    'text': 'gpt-4o',
                    'image': args.image_model_name_or_path,
                    'audio': args.audio_model_name_or_path,
                },
            }
            with open(os.path.join(args.output_dir, 'config.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
    elif args.video_model_name_or_path is not None:
        generator = prepare_pipeline(
            image_model_name_or_path=None,
            video_model_name_or_path=args.video_model_name_or_path,
            audio_model_name_or_path=args.audio_model_name_or_path,
            output_dir=args.output_dir,
        )
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for input in tqdm(inputs):
            save_path = generate(input, generator)
            output = {
                'uuid': input['instruction_uid'],
                'instruction': input['instruction'],
                'response': input['text_response'],
                'modality': {
                    'image': None,
                    'video': save_path['video_path'],
                    'audio': save_path['audio_path'],
                },
                'source': {
                    'text': 'gpt-4o',
                    'video': args.video_model_name_or_path,
                    'audio': args.audio_model_name_or_path,
                },
            }
            with open(os.path.join(args.output_dir, 'config.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
