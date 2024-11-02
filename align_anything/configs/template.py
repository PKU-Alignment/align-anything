# Copyright 2024 PKU-Alignment Team and LlamaFactory team. All Rights Reserved.
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


import io
import os
from abc import ABC
from typing import Any

import random
import requests
import librosa
import requests
import torch
import torchaudio
from PIL import Image
from torchvision.io import read_video

from align_anything.utils.template_registry import register_template


ALLOWED_ATTRIBUTES = ['split_token']
DEFAULT_SPLIT_TOKEN = 'ASSISTANT:'

def load_image(image_path: str):
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image
    except Exception:
        raise Exception(f"Error occurred when dealing with {image_path}")

def load_image_from_base64(base64_string):    
    image_stream = io.BytesIO(base64_string)
    image = Image.open(image_stream)
    
    return image

def insert_img_token(text, image):
    # do the same for worse
    if isinstance(image, str):
        decoded_images = [load_image(image)]
        num_images = 1
    elif isinstance(image, list):
        decoded_images = [load_image(img) for img in image]
        num_images = len(image)
    elif isinstance(image, Image.Image):
        decoded_images = [image]
        num_images = 1
    else:
        num_images = 0
        decoded_images = None
    
    processed_text = f"{'<image>' * num_images}{text}"
    return processed_text, decoded_images

def safe_add(list1, list2):
    if list1 is None and list2 is None:
        return []
    elif list1 is None:
        return list2.copy()
    elif list2 is None:
        return list1.copy()
    else:
        return list1 + list2

AUDIO_QUESTIONS = [
        "Summarize the audio's contents.<audio>",
        "Give an overview of what's in the audio.<audio>",
        "<audio>Detail the audio's subject matter.",
        "Explain the material covered in the audio.<audio>",
        "Outline the information in the audio.<audio>",
        "Break down the audio's key points.<audio>",
        "Describe the topics discussed in the audio.<audio>",
        "<audio>Highlight the main ideas in the audio.",
        "<audio>Recap the content of the audio.",
        "<audio>Provide a synopsis of the audio's content.",
        "<audio>Please recount what you listened to.",
        "Share the details of what reached your ears.<audio>",
        "Let me know the sounds you picked up.<audio>",
        "Could you describe the information you've heard?<audio>",
        "What did you catch from the conversation?<audio>",
        "<audio>Please inform me of the auditory information you've gathered.",
        "<audio>Relay the things you've heard, if you would.",
        "<audio>What have your ears caught wind of?",
        "I'm curious to know the reports you've heard.<audio>",
        "Let me in on the auditory details you're aware of.<audio>",
    ]

SPEECH_QUESTIONS = [
    "<audio>Could you please let me know the content of this speech?",
    "<audio>Can you tell me what this speech is about?",
    "<audio>Would you mind explaining the content of this speech?",
    "<audio>Please describe the content of this speech.",
    "I'd like to know the content of this speech.<audio>",
    "Can you inform me about the content of this speech?<audio>",
    "Could you summarize the content of this speech for me?<audio>",
    "What is the content of this speech, please?<audio>",
    "<audio>Could you provide details about the content of this speech?",
    "Please give me an overview of this speech's content.<audio>",
]

class Template(ABC):

    def __getattr__(self, name):
        if name in ALLOWED_ATTRIBUTES:
            return DEFAULT_SPLIT_TOKEN
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

@register_template('Dialogue')
class Dialogue(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    separator: str = ''

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        text = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=' '.join((raw_sample['instruction'], raw_sample['input'])))}"
            f"{self.assistant_prompt.format(output=raw_sample['output'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=' '.join((raw_sample['instruction'], raw_sample['input'])))}"
            f"{self.assistant_prompt.format(output='')}"
        )

        return_dict = {
            'text': text,
            'prompt': prompt,
        }
        return return_dict
    
@register_template('Aligner')
class Aligner(Template):
    system_prompt: str = ''
    user_prompt: str = '##QUESTION: {question} ##ANSWER: {answer} '
    assistant_prompt: str = '##CORRECTION: {correction}'
    separator: str = ''

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        
        text = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(question=raw_sample['question'], answer=raw_sample['answer'])}"
            f"{self.assistant_prompt.format(correction=raw_sample['correction'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(question=raw_sample['question'], answer=raw_sample['answer'])}"
            f"{self.assistant_prompt.format(correction='')}"
        )

        return_dict = {
            'text': text,
            'prompt': prompt,
        }
        return return_dict


@register_template('PKUSafeRLHF')
class PKUSafeRLHF(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = ''

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        text = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=raw_sample['prompt'])}"
            f"{self.assistant_prompt.format(output=raw_sample['answer'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=raw_sample['prompt'])}"
            f"{self.assistant_prompt.format(output='')}"
        )
        
        return {
            'text': text,
            'prompt': prompt,
        }

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        metrics = raw_sample['better_response_id']
        better_response = raw_sample[f'response_{int(metrics)}']
        worse_response = raw_sample[f'response_{1-int(metrics)}']
        prompt = raw_sample['prompt']

        formatted_better_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return False

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        return {'text': formatted_prompt}

@register_template('ShareGPT')
class ShareGPT:
    system_prompt: str = ''
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'
    end_token: str = '<|end_of_text|>'

    def format_supervised_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        raw_conversations = raw_sample['conversations'][:-2]
        last_conversations = raw_sample['conversations'][-2:]
        
        conversations = []
        for human, gpt in zip(raw_conversations[::2], raw_conversations[1::2]):
            conversations.append(f'{self.user_prompt.format(input=human["value"])}{self.assistant_prompt.format(output=gpt["value"])}')
        conversation = self.end_token.join(conversations)
        text = (
            f'{conversation}'
            f'{self.user_prompt.format(input=last_conversations[0]["value"])}'
            f"{self.assistant_prompt.format(output=last_conversations[1]['value'])}"
        )
        prompt = (
            f'{conversation}'
            f'{self.user_prompt.format(input=last_conversations[0]["value"])}'
            f"{self.assistant_prompt.format(output='')}"
        )
        
        return {
            'text': text,
            'prompt': prompt,
        }

    
@register_template('VQAv2')
class VQAv2:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = raw_sample['multiple_choice_answer']

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=answer)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        return {
            'text': text,
            'prompt': prompt,
            'image': raw_sample['image'],
        }

@register_template('ti2ti_preference')
class TI2TI_PREFERENCE:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        input_img = raw_sample['input_image']

        better_text = raw_sample['better_text']
        better_img = raw_sample['better_img']

        worse_text = raw_sample['worse_text']
        worse_img = raw_sample['worse_img']

        input_text_processed, input_images = insert_img_token(input_text, input_img)

        better_text_processed, better_images = insert_img_token(better_text, better_img)

        worse_text_processed, worse_images = insert_img_token(worse_text, worse_img)

        better_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f"{self.assistant_prompt.format(output=better_text_processed)}"
        )

        better_images_full = safe_add(input_images, better_images)

        worse_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f"{self.assistant_prompt.format(output=worse_text_processed)}"
        )

        worse_images_full = safe_add(input_images, worse_images)

        return {
            'better_text': better_text_full,
            'worse_text': worse_text_full,
            'better_images': better_images_full,
            'worse_images': worse_images_full,
        }

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        input_img = raw_sample['input_image']

        input_text_processed, input_images = insert_img_token(input_text, input_img)

        return {
            'text': input_text_processed,
            'image': input_images,
        }

@register_template('Chameleon_preference')
class CHAMELEON_PREFERENCE:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        input_img = raw_sample['input_image']
        
        better_text = raw_sample['better_text']
        better_img = raw_sample['better_img']
        
        worse_text = raw_sample['worse_text']
        worse_img = raw_sample['worse_img']
        
        input_text_processed, input_images = insert_img_token(input_text, input_img)
        
        better_text_processed, better_images = insert_img_token(better_text, better_img)
        
        worse_text_processed, worse_images = insert_img_token(worse_text, worse_img)
        
        better_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f"{self.assistant_prompt.format(output=better_text_processed)}"
        )
        
        better_images_full = safe_add(input_images, better_images)

        worse_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f"{self.assistant_prompt.format(output=worse_text_processed)}"
        )
        
        worse_images_full = safe_add(input_images, worse_images)
        
        return {
            'better_text': better_text_full,
            'worse_text': worse_text_full,
            'better_images': better_images_full,
            'worse_images': worse_images_full,
        }
        
    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        input_img = raw_sample['input_image']
        
        input_text_processed, input_images = insert_img_token(input_text, input_img)
        
        return {
            'text': input_text_processed,
            'image': input_images,
        }
        
@register_template('Any2Any')
class ANY2ANY(Template):

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        output_dict = raw_sample.copy()
        if 'input_image' in raw_sample and raw_sample['input_image'] is not None:
            output_dict['input_image'] = load_image(raw_sample['input_image'])
        if 'output_image' in raw_sample and raw_sample['output_image'] is not None:
            output_dict['output_image'] = load_image(raw_sample['output_image'])
        print(f"Get output dict: {output_dict}")
        return output_dict
        
@register_template('AA_textfeedback')
class AA_TF:
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: Judge the following two response of the same question and give a preference: \n ##Question: {input} \n ##Response 1: {response_1} \n ##Response 2: {response_2}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'
    
    def format_supervised_sample(self, raw_sample: dict[str, Any], path: str|None=None) -> dict[str, Any]:
        input_text = raw_sample['question']
        input_img = raw_sample['image_url']
        
        
        output_text_1 = raw_sample['response_1']
        output_img_1 = raw_sample['output_image_url_1']
        output_text_2 = raw_sample['response_2']
        output_img_2 = raw_sample['output_image_url_2']
        
        feedback = raw_sample['feedback']
        
        input_text_processed, input_images = insert_img_token(input_text, input_img)
        
        output_text_1_processed, output_images_1 = insert_img_token(output_text_1, output_img_1)
        output_text_2_processed, output_images_2 = insert_img_token(output_text_2, output_img_2)
        
        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed, response_1=output_text_1_processed, response_2=output_text_2_processed)}'
            f"{self.assistant_prompt.format(output=feedback)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed, response_1=output_text_1_processed, response_2=output_text_2_processed)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        
        input_images = safe_add(safe_add(input_images, output_images_1), output_images_2)
        
        return {
            'text': text,
            'prompt': prompt,
            'input_image': input_images,
            'image': input_images
        }
        
@register_template('spavl_ti2ti')
class TI2TI_SPAVL:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['question']
        input_img = raw_sample['image']
        
        better_text = raw_sample['chosen']
        better_img = None
        
        worse_text = raw_sample['rejected']
        worse_img = None
        
        input_text_processed, input_images = insert_img_token(input_text, input_img)
        
        better_text_processed, better_images = insert_img_token(better_text, better_img)
        
        worse_text_processed, worse_images = insert_img_token(worse_text, worse_img)
        
        better_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f"{self.assistant_prompt.format(output=better_text_processed)}"
        )
        
        better_images_full = safe_add(input_images, better_images)

        worse_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f"{self.assistant_prompt.format(output=worse_text_processed)}"
        )
        
        worse_images_full = safe_add(input_images, worse_images)
        
        return {
            'better_text': better_text_full,
            'worse_text': worse_text_full,
            'better_images': better_images_full,
            'worse_images': worse_images_full,
        }
    
    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return torch.equal(raw_sample['better_input_ids'], raw_sample['worse_input_ids'])
        
    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        input_img = raw_sample['input_image']
        
        input_text_processed, input_images = insert_img_token(input_text, input_img)
        
        return {
            'text': input_text_processed,
            'image': input_images,
        }
        
@register_template('PICKAPIC_TI2TI')
class Pickapic_TI2TI(TI2TI_PREFERENCE):
    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['caption']
        better_id = int(raw_sample['label_1'])
        worse_id = int(raw_sample['label_0'])

        raw_better_image = raw_sample[f'jpg_{better_id}']
        raw_worse_image = raw_sample[f'jpg_{worse_id}']
        better_image = Image.open(io.BytesIO(raw_better_image)).convert('RGB')
        worse_image = Image.open(io.BytesIO(raw_worse_image)).convert('RGB')
        raw_sample_upd = {
            "input_text": prompt,
            "input_image": [],
            "better_text": "",
            "better_img": better_image,
            "worse_text": "",
            "worse_img": worse_image
        }
        return super().format_example(raw_sample_upd)

@register_template('GQA')
class GQA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = raw_sample['answer']

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=answer)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        image_file = os.path.join(path, raw_sample['image_path'])
        return {
            'text': text,
            'prompt': prompt,
            'image': Image.open(image_file),
        }
    
@register_template('OK-VQA')
class OKVQA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = max(set(raw_sample['answers']), key=raw_sample['answers'].count)

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=answer)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        return {
            'text': text,
            'prompt': prompt,
            'image': raw_sample['image'],
        }
    
@register_template('A-OKVQA')
class AOKVQA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input} give me your rationales.'
    assistant_prompt: str = '\nASSISTANT: {output}, the rationales is that {rationales}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = raw_sample['choices'][raw_sample['correct_choice_idx']]
        rationales = " ".join(raw_sample['rationales'])

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=answer, rationales=rationales)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='', rationales='')}"
        )

        return {
            'text': text,
            'prompt': prompt,
            'image': raw_sample['image'],
        }

@register_template('OCRVQA')
class OCRVQA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image> According to the content of the pictures, answer the following questions in order.\n{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        questions = raw_sample['questions']
        answers = raw_sample['answers']
        question = '\n'.join(questions)
        answer = '\n'.join(answers)

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=answer)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        return {
            'text': text,
            'prompt': prompt,
            'image': raw_sample['image'],
        }
    
@register_template('VisualGenome')
class VisualGenome:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image> According to the content of the pictures, answer the following questions in order.\n{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        questions = raw_sample['questions']
        answers = raw_sample['answers']
        quetsion = '\n'.join(questions)
        answer = '\n'.join(answers)

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=quetsion)}'
            f"{self.assistant_prompt.format(output=answer)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=quetsion)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        return {
            'text': text,
            'prompt': prompt,
            'image': raw_sample['image'],
        }

@register_template('ShareGPT-4o')
class ShareGPT4o:
    system_prompt: str = ''
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        raw_conversations = raw_sample['conversations']
        text = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=raw_conversations[0]['value'])}"
            f"{self.assistant_prompt.format(output=raw_conversations[1]['value'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=raw_conversations[0]['value'])}"
            f"{self.assistant_prompt.format(output='')}"
        )
        
        image_file = os.path.join(path, 'mnt/petrelfs/wangwenhai/workspace_cef/4o/image', raw_sample['image'])
        return {
            'text': text,
            'prompt': prompt,
            'image': Image.open(image_file),
        }

    
@register_template('Llava')
class Llava:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        raw_conversations = raw_sample['conversations']
        raw_prompt = raw_conversations[0]['value'].replace('<image>\n', '').replace('\n<image>', '')

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=raw_prompt)}'
            f"{self.assistant_prompt.format(output=raw_conversations[1]['value'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=raw_prompt)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        image_file = f"http://images.cocodataset.org/val2017/{raw_sample['image']}"

        return {
            'text': text,
            'prompt': prompt,
            'image': load_image(image_file),
        }

@register_template('Llava_Local')
class Llava_Local:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        raw_conversations = raw_sample['conversations']
        raw_prompt = raw_conversations[0]['value'].replace('<image>\n', '').replace('\n<image>', '')

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=raw_prompt)}'
            f"{self.assistant_prompt.format(output=raw_conversations[1]['value'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=raw_prompt)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        image_file = raw_sample['image']

        return {
            'text': text,
            'prompt': prompt,
            'image': load_image(image_file),
        }

@register_template('Llava-CC3M')
class Llava_CC3M:
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    
    def format_supervised_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        raw_conversations = raw_sample['conversations']
        question = raw_conversations[0]['value']
        answer = raw_conversations[1]['value']
        image = raw_sample['image']

        text = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=answer)}"
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        image_file = os.path.join(path, 'images', image)

        return {
            'text': text,
            'prompt': prompt,
            'image': Image.open(image_file),
        }

@register_template('AudioCaps')
class AudioCaps:
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        caption = raw_sample['caption']
        audiocap_path = raw_sample['audiocap_path']
        question = random.choice(AUDIO_QUESTIONS)
        
        text = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=caption)}"
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        audio, sample_rate = torchaudio.load(os.path.join(path, f"{audiocap_path}"))
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        return {
            'text': text,
            'prompt': prompt,
            'audio': audio.squeeze().tolist(),
            'sampling_rate': sample_rate
        }

@register_template('LibriSpeech')
class LibriSpeech:
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        caption = raw_sample['text'].lower()
        question = random.choice(SPEECH_QUESTIONS)
        
        text = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=caption)}"
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        audio = raw_sample['audio']['array']
        sampling_rate = raw_sample['audio']['sampling_rate']
        return {
            'text': text,
            'prompt': prompt,
            'audio': audio,
            'sampling_rate': sampling_rate
        }


@register_template('AudioSet')
class AudioSet:
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    def format_sample(self, raw_sample: dict[str, Any], path: str=None) -> dict[str, Any]:
        caption = f"The content of audio is {', '.join(raw_sample['captions'])}."
        question = random.choice(AUDIO_QUESTIONS)
        
        text = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output=caption)}"
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        audio, sample_rate = torchaudio.load(os.path.join(path, raw_sample["audio"]))
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        return {
            'text': text,
            'prompt': prompt,
            'audio': audio.squeeze().tolist(),
            'sampling_rate': sample_rate
        }

@register_template('DiffusionDB')
class DiffusionDB:
    system_prompt: str = ''
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = ' ASSISTANT:{output}'
    split_token: str = ' ASSISTANT:'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:

        text = raw_sample['prompt']
        return {
            'text': text,
            'prompt': text,
            'image': raw_sample['image'].convert('RGB'),
        }
        
@register_template('ti2ti')
class TI2TI:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        output_text = raw_sample['output_text']
        input_img = raw_sample['input_image']
        output_img = raw_sample['output_image']

        if isinstance(input_img, str):
            input_images = [load_image(input_img)]
            num_imput_img = 1
        elif isinstance(input_img, list):
            input_images = [load_image(img) for img in input_img]
            num_input_img = len(input_img)
        else:
            raise ValueError("input_image must be either a string or a list of strings")


        input_text = f"{'<image>' * num_imput_img}{input_text}"

        # do the same for output
        if isinstance(output_img, str):
            output_images = [load_image(output_img)]
            num_output_img = 1

        elif isinstance(output_img, list):
            output_images = [load_image(img) for img in output_img]
            num_output_img = len(output_img)
        else:
            raise ValueError("output_image must be either a string or a list of strings")

        output_text = f"{output_text}{'<image>' * num_output_img}"

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f"{self.assistant_prompt.format(output=output_text)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        return {
            'text': text,
            'prompt': prompt,
            'images': input_images + output_images,
        }

@register_template('Chameleon')
class CHAMELEON:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['input_text']
        output_text = raw_sample['output_text']
        input_img = raw_sample['input_image']
        output_img = raw_sample['output_image']
        
        if isinstance(input_img, str):
            input_images = [load_image(input_img)]
            num_imput_img = 1
        elif isinstance(input_img, list):
            input_images = [load_image(img) for img in input_img]
        else:
            raise ValueError("input_image must be either a string or a list of strings")
        
        
        input_text = f"{'<image>' * num_imput_img}{input_text}"
        
        # do the same for output
        if isinstance(output_img, str):
            output_images = [load_image(output_img)]
            num_output_img = 1
            
        elif isinstance(output_img, list):
            output_images = [load_image(img) for img in output_img]
            num_output_img = len(output_img)
        else:
            raise ValueError("output_image must be either a string or a list of strings")
        
        output_text = f"{output_text}{'<image>' * num_output_img}"
        
        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f"{self.assistant_prompt.format(output=output_text)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        
        return {
            'text': text,
            'prompt': prompt,
            'images': input_images + output_images,
        }
        
@register_template('ANYTHING_TI2TI')
class ANYTHING_TI2TI:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        input_text = raw_sample['question']
        output_text = raw_sample['response']
        input_img = raw_sample['image_url']
        output_img = raw_sample['output_image_url']
        
        if isinstance(input_img, str):
            input_images = [load_image(input_img)]
            num_imput_img = 1
        elif isinstance(input_img, list):
            input_images = [load_image(img) for img in input_img]
            num_input_img = len(input_img)
        else:
            raise ValueError("input_image must be either a string or a list of strings")
        
        
        input_text = f"{'<image>' * num_imput_img}{input_text}"
        
        # do the same for output
        if isinstance(output_img, str):
            output_images = [load_image(output_img)]
            num_output_img = 1
            
        elif isinstance(output_img, list):
            output_images = [load_image(img) for img in output_img]
            num_output_img = len(output_img)
        else:
            raise ValueError("output_image must be either a string or a list of strings")
        
        output_text = f"{output_text}{'<image>' * num_output_img}"
        
        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f"{self.assistant_prompt.format(output=output_text)}"
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        
        return {
            'text': text,
            'prompt': prompt,
            'input_image': input_images,
            'image': input_images + output_images,
        }
        
    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return torch.equal(raw_sample['better_input_ids'], raw_sample['worse_input_ids'])

@register_template('RLAIFV')
class RLAIFV:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['chosen']
        worse_response = raw_sample['rejected']
        prompt = raw_sample['question']
        image = raw_sample['image']

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
        )
        formatted_better_output = (
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'prompt': formatted_prompt,
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return raw_sample['chosen'] == raw_sample['rejected']

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question']
        image = raw_sample['image']

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        return {
            'text': formatted_prompt,
            'image': image,
        }


@register_template('SPA_VL')
class SPA_VL:
    system_prompt: str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    user_prompt: str = 'USER: \n<image> {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['chosen']
        worse_response = raw_sample['rejected']
        prompt = raw_sample['question']
        image = raw_sample['image']

        
        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
        )
        formatted_better_output = (
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.assistant_prompt.format(output=worse_response)}'
        )
        image = image.convert('RGBA')

        return {
            'prompt': formatted_prompt,
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return raw_sample['chosen'] == raw_sample['rejected']

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
        image = raw_sample['image']

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )
        image = image.convert('RGBA')

        return {
            'text': formatted_prompt,
            'image': image,
        }


@register_template('Pickapic')
class Pickapic:

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['caption']
        better_id = int(raw_sample['label_1'])
        worse_id = int(raw_sample['label_0'])

        raw_better_image = raw_sample[f'jpg_{better_id}']
        raw_worse_image = raw_sample[f'jpg_{worse_id}']

        better_image = Image.open(io.BytesIO(raw_better_image)).convert('RGB')
        worse_image = Image.open(io.BytesIO(raw_worse_image)).convert('RGB')

        return {
            'prompt': prompt,
            'better_image': better_image,
            'worse_image': worse_image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_id = float(raw_sample['label_0'])
        if better_id == 0.5:
            return True
        else:
            return False


@register_template('Webvid')
class Webvid:
    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        video, _, _ = read_video(os.path.join(path, raw_sample['video_path']))
        return {
            'prompt': raw_sample['caption'],
            'video': video.squeeze(0),
        }


@register_template('SafeSora')
class SafeSora:
    def format_preference_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        prompt = raw_sample['prompt_text']

        better_id = None
        worse_id = None
        if raw_sample['helpfulness'] == 'video_0':
            better_id = 'video_0'
            worse_id = 'video_1'
        else:
            better_id = 'video_1'
            worse_id = 'video_0'

        raw_better_video = raw_sample[better_id]['video_path']
        raw_worse_video = raw_sample[worse_id]['video_path']

        better_video, _, _ = read_video(os.path.join(path, raw_better_video))
        worse_video, _, _ = read_video(os.path.join(path, raw_worse_video))

        return {
            'prompt': prompt,
            'better_video': better_video.squeeze(0),
            'worse_video': worse_video.squeeze(0),
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return False


@register_template('WavCaps')
class WavCaps:
    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        caption = raw_sample['answer']
        audio = raw_sample['context']['array']
        sampling_rate = raw_sample['context']['sampling_rate']
        return {
            'prompt': caption,
            'audio': audio,
            'sampling_rate': sampling_rate,
        }


@register_template('SOMOS')
class SOMOS:
    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        better_audio_path = os.path.join(path, raw_sample['better_data_path'])
        worse_audio_path = os.path.join(path, raw_sample['worse_data_path'])

        better_audio, _ = librosa.load(better_audio_path, sr=None)
        worse_audio, _ = librosa.load(worse_audio_path, sr=None)

        return {
            'prompt': prompt,
            'better_audio': better_audio,
            'worse_audio': worse_audio,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return False

@register_template('Qwen2-VL')
class QWEN2VL:
    system_prompt: str = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    split_token: str = '\n'
    separator: str = 'assistant\n'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        output = raw_sample['output']

        return_text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f"{self.assistant_prompt.format(output=output)}"
        )

        return_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        video_info = raw_sample['video_path']
        if isinstance(video_info, str):
            video_info = [video_info]

        return_dict = {
            "text": return_text,
            "prompt": return_prompt,
            "image": [],
            "video": video_info,
        }
        return return_dict

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        better_output = raw_sample['better_output']
        worse_output = raw_sample['worse_output']

        return_better_text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f"{self.assistant_prompt.format(output=better_output)}"
        )

        return_worse_text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f"{self.assistant_prompt.format(output=worse_output)}"
        )

        video_info = raw_sample['video_path']
        if isinstance(video_info, str):
            video_info = [video_info]

        return_dict = {
            "better_text": return_better_text,
            "worse_text": return_worse_text,
            "image": [],
            "video": video_info,
        }
        return return_dict 
    
    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        if raw_sample['better_output'] == raw_sample['worse_output']:
            return True
        else:
            return False

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']

        return_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f"{self.assistant_prompt.format(output='')}"    
        )

        video_info = raw_sample['video_path']
        if isinstance(video_info, str):
            video_info = [video_info]

        return {
            'text': return_prompt,
            'image': [],
            'video': video_info,
        }

@register_template('Alpaca')
class Alpaca(Dialogue):
    system_prompt: str = 'Below is an instruction that describes a task. '
    user_prompt: str = '### Instruction:\n{input}\n\n'
    assistant_prompt: str = '### Response:\n{output}'


@register_template('Aquila')
class Aquila(Dialogue):
    system_prompt: str = (
        'A chat between a curious human and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    )
    user_prompt: str = 'Human: {input}'
    assistant_prompt: str = '###Assistant:{output}'
    separator: str = '###'


@register_template('Atom')
class Atom(Dialogue):
    system_prompt: str = ''
    user_prompt: str = '<bos>Human: {input}\n<eos>'
    assistant_prompt: str = '<bos>Assistant:{output}'
    separator: str = ''


@register_template('Baichuan')
class Baichuan(Dialogue):
    system_prompt: str = ''
    user_prompt: str = '<reserved_102>{input}'
    assistant_prompt: str = '<reserved_103>{output}'
    separator: str = ''


@register_template('Baichuan2')
class Baichuan2(Dialogue):
    system_prompt: str = ''
    user_prompt: str = '<reserved_106>{input}'
    assistant_prompt: str = '<reserved_107>{output}'
    separator: str = ''


@register_template('Belle')
class Belle(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = 'Human: {input}'
    assistant_prompt: str = '\n\nBelle: {output}'
    separator: str = '\n\n'


@register_template('Bluelm')
class Bluelm(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = 'Human: {input}'
    assistant_prompt: str = '\n\nBelle: {output}'
    separator: str = ''


@register_template('Breeze')
class Breeze(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = '[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@register_template('Chatglm2')
class Chatglm2(Dialogue):
    system_prompt: str = '[gMASK]<sop>'
    user_prompt: str = '[Round 0]\n\n问：{input}'
    assistant_prompt: str = '\n\n答：{output}'
    separator: str = '\n\n'


@register_template('Chatglm3')
class Chatglm3(Dialogue):
    system_prompt: str = '[gMASK]<sop><|system|>\n'
    user_prompt: str = '{input}'
    assistant_prompt: str = '{output}'
    separator: str = ''


@register_template('Chatml')
class Chatml(Dialogue):
    system_prompt: str = '<|im_start|>system\n<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = ''


@register_template('Chatml_de')
class Chatml_de(Dialogue):
    system_prompt: str = 'Du bist ein freundlicher und hilfsbereiter KI-Assistent.'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '\n'


@register_template('Codegeex2')
class Codegeex2(Dialogue):
    system_prompt: str = '[gMASK]<sop>'
    user_prompt: str = '{input}'
    assistant_prompt: str = '{output}'
    separator: str = ''


@register_template('Codegeex4')
class Codegeex2(Dialogue):
    system_prompt: str = (
        '[gMASK]<sop><|system|>\n你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'
    )
    user_prompt: str = '<|user|>\n{input}'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = ''


@register_template('Cohere')
class Cohere(Dialogue):
    system_prompt: str = '<bos><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>'
    user_prompt: str = '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{input}<|END_OF_TURN_TOKEN|>'
    assistant_prompt: str = '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{output}'
    separator: str = ''


@register_template('Cpm')
class Cpm(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = '<用户>{input}'
    assistant_prompt: str = '<AI>{output}'
    separator: str = ''


@register_template('Dbrx')
class Dbrx(Dialogue):
    system_prompt: str = (
        '<|im_start|>system\nYou are DBRX, created by Databricks. You were last updated in December 2023. '
        'You answer questions based on information available up to that point.\n'
        'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough '
        'responses to more complex and open-ended questions.\nYou assist with various tasks, '
        'from writing to coding (using markdown for code blocks — remember to use ``` with '
        'code, JSON, and tables).\n(You do not have real-time data access or code execution '
        'capabilities. You avoid stereotyping and provide balanced perspectives on '
        'controversial topics. You do not provide song lyrics, poems, or news articles and '
        'do not divulge details of your training data.)\nThis is your system prompt, '
        'guiding your responses. Do not reference it, just respond to the user. If you find '
        'yourself talking about this message, stop. You should be responding appropriately '
        'and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION '
        "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.<|im_end|>\n"
    )
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '\n'


@register_template('Deepseek')
class Deepseek(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = 'User: {input}\n'
    assistant_prompt: str = '\nAssistant:{output}'
    separator: str = ''


@register_template('Deepseekcoder')
class Deepseekcoder(Dialogue):
    system_prompt: str = (
        '<bos>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n'
    )
    user_prompt: str = '### Instruction:\n{input}\n'
    assistant_prompt: str = '### Response:{output}'
    separator: str = '\n'


@register_template('Dolphin')
class Dolphin(Dialogue):
    system_prompt: str = '<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>'
    assistant_prompt: str = '\n<|im_start|>assistant\n{output}'
    separator: str = '\n'


@register_template('Falcon')
class Falcon(Dialogue):
    system_prompt: str = ''
    user_prompt: str = 'User: {input}\n'
    assistant_prompt: str = 'Falcon:{output}'
    separator: str = '\n'


@register_template('Gemma')
class Gemma(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = '<start_of_turn>user\n{input}<end_of_turn>\n'
    assistant_prompt: str = '<start_of_turn>model\n{output}'
    separator: str = '<end_of_turn>\n'


@register_template('Glm4')
class Glm4(Dialogue):
    system_prompt: str = '[gMASK]<sop><|system|>\n'
    user_prompt: str = '<|user|>\n{input}'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = ''


@register_template('Intern')
class Intern(Dialogue):
    system_prompt: str = '<bos><|System|>:\n'
    user_prompt: str = '<|User|>:{input}\n'
    assistant_prompt: str = '<|Bot|>:{output}'
    separator: str = '<eoa>\n'


@register_template('Intern2')
class Intern2(Dialogue):
    system_prompt: str = '<bos><|im_start|>system\n{input}<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '<|im_end|>\n'


@register_template('Llama2')
class Llama2(Dialogue):
    system_prompt: str = '<<SYS>>\n\n<</SYS>>\n\n'
    user_prompt: str = '<bos>[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@register_template('Llama2_zh')
class Llama2_zh(Dialogue):
    system_prompt: str = (
        '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
    )
    user_prompt: str = '<bos>[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@register_template('Llama2_hf')
class Llama2_hf(Dialogue):
    system_prompt: str = (
        "<<SYS>>\n        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    )
    user_prompt: str = '<bos>[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@register_template('Llama3')
class Llama3(Dialogue):
    system_prompt: str = '<bos><|start_header_id|>system<|end_header_id|>\n\n'
    user_prompt: str = '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
    assistant_prompt: str = '<|start_header_id|>assistant<|end_header_id|>\n\n{output}'
    separator: str = ''


@register_template('Llama3_hf')
class Llama3_hf(Dialogue):
    system_prompt: str = (
        "<bos><|start_header_id|>system<|end_header_id|>\n\n    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    user_prompt: str = '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
    assistant_prompt: str = '<|start_header_id|>assistant<|end_header_id|>\n\n{output}'
    separator: str = ''


@register_template('Mistral')
class Mistral(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = '[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@register_template('Olmo')
class Olmo(Dialogue):
    system_prompt: str = '<eos>'
    user_prompt: str = '<|user|>\n{input}'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = ''


@register_template('Openchat')
class Openchat(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = 'GPT4 Correct User: {input}<eos>'
    assistant_prompt: str = 'GPT4 Correct Assistant:{output}'
    separator: str = ''


@register_template('Openchat3')
class Openchat3(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = '<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{input}<|eot_id|>'
    assistant_prompt: str = '<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n'
    separator: str = ''


@register_template('Orion')
class Orion(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = 'Human: {input}\n'
    assistant_prompt: str = '\nAssistant: <eos>{output}'
    separator: str = ''


@register_template('Phi')
class Phi(Dialogue):
    system_prompt: str = '<bos>'
    user_prompt: str = '<|user|>\n{input}<|end|>'
    assistant_prompt: str = '\n<|assistant|>\n{output}'
    separator: str = '\n'


@register_template('Qwen')
class Qwen(Dialogue):
    system_prompt: str = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>'
    assistant_prompt: str = '\n<|im_start|>assistant\n{output}'
    separator: str = '\n'


@register_template('Solar')
class Solar(Dialogue):
    system_prompt: str = '### System:\n\n\n'
    user_prompt: str = '### User:\n{input}\n'
    assistant_prompt: str = '\n### Assistant:\n{output}'
    separator: str = ''


@register_template('Starchat')
class Starchat(Dialogue):
    system_prompt: str = '<|system|>\n<|end|>\n'
    user_prompt: str = '<|user|>\n{input}<|end|>'
    assistant_prompt: str = '\n<|assistant|>{output}'
    separator: str = '\n'


@register_template('Telechat')
class Telechat(Dialogue):
    system_prompt: str = r'<\_system><\_end>'
    user_prompt: str = '<_user>{input}'
    assistant_prompt: str = '<_bot>{output}'
    separator: str = ''


@register_template('Xuanyuan')
class Xuanyuan(Dialogue):
    system_prompt: str = (
        '以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n'
    )
    user_prompt: str = 'Human: {input} '
    assistant_prompt: str = 'Assistant:{output}'
    separator: str = ''


@register_template('Xverse')
class Xverse(Dialogue):
    system_prompt: str = ''
    user_prompt: str = 'Human: {input}\n'
    assistant_prompt: str = '\nAssistant: {output}'
    separator: str = ''


@register_template('Yayi')
class Yayi(Dialogue):
    system_prompt: str = (
        "<|System|>\nYou are a helpful, respectful and honest assistant named YaYi developed by Beijing Wenge Technology Co.,Ltd. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n"
    )
    user_prompt: str = '<|Human|>\n{input}\n'
    assistant_prompt: str = '\n<|YaYi|>:{output}'
    separator: str = '\n\n'


@register_template('Yi')
class Yi(Dialogue):
    system_prompt: str = '<|im_start|>system\n<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '\n'


@register_template('Yi_vl')
class Yi_vl(Dialogue):
    system_prompt: str = (
        'This is a chat between an inquisitive human and an AI assistant. '
        "Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    )
    user_prompt: str = '### Human: {input}\n'
    assistant_prompt: str = '### Assistant:{output}'
    separator: str = '\n'


@register_template('Yuan')
class Yuan(Dialogue):
    system_prompt: str = ''
    user_prompt: str = '{input}<sep>'
    assistant_prompt: str = '{output}'
    separator: str = '\n'


@register_template('Zephyr')
class Zephyr(Dialogue):
    system_prompt: str = '<|system|>\nYou are Zephyr, a helpful assistant.'
    user_prompt: str = '<|user|>\n{input}<eos>'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = '\n'


@register_template('Ziya')
class Zephyr(Dialogue):
    system_prompt: str = ''
    user_prompt: str = '<human>:{input}\n'
    assistant_prompt: str = '<bot>:{output}'
    separator: str = '\n'

@register_template('OpenAQA')
class OpenAQA:
    system_prompt: str = 'You are a helpful assistant.'
    split_token: str = '<|im_end|>\n<|im_start|>assistant\n'
    separator: str = '<|im_end|>\n<|im_start|>assistant\n'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['instruction']
        audio_url = raw_sample['audio_id']
        response = raw_sample['output']

        conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": response},
        ]

        return {
            'conversation': conversation,
            'prompt': conversation[:-1],
        }
    
@register_template('RLHFAQA')
class RLHFAQA:
    system_prompt: str = 'You are a helpful assistant.'
    split_token: str = 'assistant\n'
    separator: str = 'assistant\n'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        raw_input = raw_sample['raw_input']
        better_id = raw_sample['overall_response']

        if int(better_id) == 1:
            better_response = raw_input['output']
            worse_response = raw_input['reject_answer']
        elif int(better_id) == 2:
            better_response = raw_input['reject_answer']
            worse_response = raw_input['output']
        else:
            raise RuntimeError(f'Expect better_id is type `int`, but got: {better_id}')
        prompt = raw_input['prompt']
        audio_url = raw_input['audio_url']

        better_conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": better_response},
        ]
        
        worse_conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": worse_response},
        ]

        return {
            'prompt': better_conversation[:-1],
            'better_conversation': better_conversation,
            'worse_conversation': worse_conversation,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        raw_input = raw_sample['raw_input']
        return raw_input['output']==raw_input['reject_answer']

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['raw_input']['prompt']
        audio_url = raw_sample['raw_input']['audio_url']

        conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
        ]

        return {'conversation': conversation}
    
@register_template('AA_TI2T_Local')
class AA_TI2T_Local:
    system_prompt: str = ""
    user_prompt: str = 'USER: {input}\n<image>'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id==1 else 1

        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['question']
        image_path = raw_sample['image']
        image = load_image(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
        )
        formatted_better_output = (
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'prompt': formatted_prompt,
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['overall_response'])
        if better_id not in [1, 2]:
            return True
        worse_id = 2 if better_id==1 else 1
        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']

        return better_response == worse_response

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        image_path = raw_sample['image']
        image = load_image(image_path).convert('RGBA')

        return {
            'text': formatted_prompt,
            'image': image,
        }
    
    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        image_path = raw_sample['image']
        image = load_image(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )
        formatted_answer = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=answer)}'
        )

        return {
            'text': formatted_answer,
            'prompt': formatted_prompt,
            'image': image,
        }

@register_template('AA_TI2T')
class AA_TI2T:
    system_prompt: str = ""
    user_prompt: str = 'USER: {input}\n<image>'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id==1 else 1

        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['question']
        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
        )
        formatted_better_output = (
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'prompt': formatted_prompt,
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id==1 else 1
        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']

        return better_response == worse_response

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        return {
            'text': formatted_prompt,
            'image': image,
        }
    
    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )
        formatted_answer = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=answer)}'
        )

        return {
            'text': formatted_answer,
            'prompt': formatted_prompt,
            'image': image,
        }

@register_template('AA_TI2T_Critique')
class AA_TI2T_Critique:
    system_prompt: str = ""
    user_prompt: str = 'USER: {input}\n<image>'
    user_prompt_wo_image: str = 'USER: {input}'
    critique_prompt: str = 'Please provide a critique of the response.'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        return {
            'text': formatted_prompt,
            'image': image,
        }
    

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        critique = raw_sample['critique']
        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=answer)}'
            f'{self.user_prompt_wo_image.format(input=self.critique_prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )
        formatted_answer = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=answer)}'
            f'{self.user_prompt_wo_image.format(input=self.critique_prompt)}'
            f'{self.assistant_prompt.format(output=critique)}'
        )

        return {
            'text': formatted_answer,
            'prompt': formatted_prompt,
            'image': image,
        }

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['refinement'].replace('<image>', '')
        worse_response = raw_sample['response'].replace('<image>', '')
        prompt = raw_sample['prompt'].replace('<image>', '')
        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
        )
        formatted_better_output = (
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'prompt': formatted_prompt,
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_response = raw_sample['refinement']
        worse_response = raw_sample['response']

        return better_response == worse_response

@register_template('LLAMA_3_2')
class LLAMA_3_2:
    system_prompt: str = ''
    user_prompt: str = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{input}<|eot_id|>'
    assistant_prompt: str = '<|start_header_id|>assistant<|end_header_id|>\n{output}'
    split_token: str = '<|start_header_id|>assistant<|end_header_id|>'
    separator: str = '###'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question']
        answer = raw_sample['answer']
        image_path = raw_sample['image']
        image = load_image_from_base64(image_path).convert('RGBA')

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )
        formatted_answer = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=answer)}'
        )

        return {
            'text': formatted_answer,
            'prompt': formatted_prompt,
            'image': image,
        }

@register_template('Qwen2Audio')
class Qwen2Audio:
    system_prompt: str = 'You are a helpful assistant.'
    user_prompt: str = '<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant{output}'
    split_token: str = '<|im_end|>\n<|im_start|>assistant\n'
    separator: str = '<|im_end|>\n<|im_start|>assistant\n'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = ' '.join((raw_sample['instruction'], raw_sample['input']))
        audio_url = raw_sample['audio_path']
        response = raw_sample['output']

        conversation = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": response},
        ]

        return {
            'conversation': conversation,
            'prompt': conversation[:-1],
        }

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id==1 else 1
        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['prompt']
        audio_url = raw_sample['audio_path']

        better_conversation = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": better_response},
        ]

        worse_conversation = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": worse_response},
        ]

        formatted_prompt = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
        ]

        return {
            'prompt': formatted_prompt,
            'better_conversation': better_conversation,
            'worse_conversation': worse_conversation,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id==1 else 1
        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']

        return better_response==worse_response

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        audio_url = raw_sample['audio_path']

        conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
        ]

        return {'conversation': conversation}
    
@register_template('Qwen2AudioCritique')
class Qwen2AudioCritique:
    system_prompt: str = 'You are a helpful assistant.'
    user_prompt: str = '<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant{output}'
    split_token: str = '<|im_end|>\n<|im_start|>assistant\n'
    separator: str = '<|im_end|>\n<|im_start|>assistant\n'

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        audio_url = raw_sample['audio_path']
        response = raw_sample['response']
        critique = raw_sample['critique']
        critique_prompt = 'Please provide the ##Critique and ##Refinement.'
        conversation = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": response},
            {'role': 'user', 'content': [{"type": "text", "text": critique_prompt}]},
            {"role": "assistant", "content": critique},
        ]

        return {
            'conversation': conversation,
            'prompt': conversation[:-1],
        }


    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['refinement']
        worse_response = raw_sample['response']
        prompt = raw_sample['prompt']
        audio_url = raw_sample['audio_path']

        better_conversation = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": better_response},
        ]

        worse_conversation = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            {"role": "assistant", "content": worse_response},
        ]

        formatted_prompt = [
            {
                "role": "system", "content": self.system_prompt
            },
            {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
        ]

        return {
            'prompt': formatted_prompt,
            'better_conversation': better_conversation,
            'worse_conversation': worse_conversation,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_response = raw_sample['refinement']
        worse_response = raw_sample['response']

        return better_response==worse_response