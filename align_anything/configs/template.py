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
from abc import ABC, abstractmethod
from typing import Any

import librosa
import requests
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
    except Exception as e:
        print(f"Error occured when dealing with {image_path}")
        raise Exception

class Template(ABC):
    @abstractmethod
    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        pass

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

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
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

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        
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

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
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


@register_template('LLAVA')
class LLAVA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
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


@register_template('DiffusionDB')
class DiffusionDB:
    system_prompt: str = ''
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = ' ASSISTANT:{output}'
    split_token: str = ' ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:

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
        
@register_template('ANYTHING_TI2TI')
class ANYTHING_TI2TI:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
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

@register_template('RLAIFV')
class RLAIFV:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['chosen']
        worse_response = raw_sample['rejected']
        prompt = raw_sample['question']
        image = raw_sample['image']

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
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return False

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
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['chosen']
        worse_response = raw_sample['rejected']
        prompt = raw_sample['question']
        image = raw_sample['image']

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
        image = image.convert('RGBA')

        return {
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
            'image': image,
        }

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return False

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question']
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

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
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
    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
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
