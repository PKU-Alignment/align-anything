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

import numpy as np
from tqdm import tqdm

from datasets import load_dataset


def parse_modality(text):
    """
    Parse the generated text to determine which modalities are used

    Args:
        text (str): The generated text response

    Returns:
        str: Returns modality combination (one of 't','i','a','ti','ta','ia','tia')
        t: text, i: image, a: audio
    """
    # Initialize modality flags
    has_text = False
    has_image = False
    has_audio = False

    # Find content sections
    response_start = text.find('[[response]]:')
    image_start = text.find('[[image_instruction]]:')
    audio_start = text.find('[[audio_instruction]]:')

    # Check text response
    if response_start != -1:
        response_content = text[response_start:image_start].split(':')[1].strip()
        has_text = response_content and response_content.lower() != 'none'

    # Check image instruction
    if image_start != -1:
        image_content = text[image_start:audio_start].split(':')[1].strip()
        has_image = image_content and image_content.lower() != 'none'

    # Check audio instruction
    if audio_start != -1:
        audio_content = text[audio_start:].split(':')[1].strip()
        has_audio = audio_content and audio_content.lower() != 'none'

    # Return combination based on present modalities
    modalities = ''
    if has_text:
        modalities += 't'
    if has_image:
        modalities += 'i'
    if has_audio:
        modalities += 'a'

    return modalities if modalities != '' else 't'  # Default to 't' if no modalities detected


def generate(prompt):
    """
    The generate function should return a string of the response, where you should implement the inference logic.
    """


def inference(question):
    modality_selection_template = """
    You are now going to answer the user's question. Here are two tools for you to generate image and audio, you can generate two sets of instructions: one for an image generation model and one for an audio generation model. You should be free to choose whether or not to use the generator tool based on user instructions, and if you do, please make corresponding references and descriptions in your text to ensure consistency of information across modalities. If you think that no image or audio should be generated, please write None in the corresponding instruction section. Make sure:
    1.	Your text response should be clear and precise, referencing the generated image and audio content as much as possible in the answer.
    2.	If you think an image should be generated, the instructions for the image generation should be accurate and concise.
    3.	If you think an audio should be generated, the instructions for the audio generation should be accurate and concise, no more than 20 words.
    4.  Please analyze the user's instruction carefully, and please only generate other modal information if it can help you follow the user's instruction.

    Your output should follow the following format:
    [output_format]
    [[response]]: <response>
    [[image_instruction]]: <image_instruction>
    [[audio_instruction]]: <audio_instruction>
    [/output_format]

    [User question]
    {question}
    [/User question]
    """

    prompt = modality_selection_template.format(question=question)
    return generate(prompt)


def main():
    dataset = load_dataset('PKU-Alignment/EvalAnything-Selection_Synergy', trust_remote_code=True)

    save_dir = os.path.join('amg', 'selection')
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for entry in tqdm(dataset):
        instruction = entry['instruction']
        selection_distribution = entry['selection']

        response = inference(instruction)

        modality = parse_modality(response)

        results.append(
            {
                'instruction_uid': entry['instruction_uid'],
                'prompt': instruction,
                'response': response,
                'modality': modality,
                'score': selection_distribution[modality],
            }
        )

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump({'selection_score': np.mean(results['score']), 'results': results}, f, indent=4)


if __name__ == '__main__':
    main()
