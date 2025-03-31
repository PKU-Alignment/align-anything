# Copyright 2025 PKU-Alignment Team and LlamaFactory team. All Rights Reserved.
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
import random
from typing import Any

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
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image
    except Exception:
        raise Exception(f'Error occurred when dealing with {image_path}')


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
    'Explain the material covered in the audio.<audio>',
    'Outline the information in the audio.<audio>',
    "Break down the audio's key points.<audio>",
    'Describe the topics discussed in the audio.<audio>',
    '<audio>Highlight the main ideas in the audio.',
    '<audio>Recap the content of the audio.',
    "<audio>Provide a synopsis of the audio's content.",
    '<audio>Please recount what you listened to.',
    'Share the details of what reached your ears.<audio>',
    'Let me know the sounds you picked up.<audio>',
    "Could you describe the information you've heard?<audio>",
    'What did you catch from the conversation?<audio>',
    "<audio>Please inform me of the auditory information you've gathered.",
    "<audio>Relay the things you've heard, if you would.",
    '<audio>What have your ears caught wind of?',
    "I'm curious to know the reports you've heard.<audio>",
    "Let me in on the auditory details you're aware of.<audio>",
]

SPEECH_QUESTIONS = [
    '<audio>Could you please let me know the content of this speech?',
    '<audio>Can you tell me what this speech is about?',
    '<audio>Would you mind explaining the content of this speech?',
    '<audio>Please describe the content of this speech.',
    "I'd like to know the content of this speech.<audio>",
    'Can you inform me about the content of this speech?<audio>',
    'Could you summarize the content of this speech for me?<audio>',
    'What is the content of this speech, please?<audio>',
    '<audio>Could you provide details about the content of this speech?',
    "Please give me an overview of this speech's content.<audio>",
]


class BaseFormatter:
    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        return True

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict]:
        """Format the sample for supervised training.

        Args:
            raw_sample (dict[str, Any]): The raw sample from the dataset.

        Example:
            >>> self.format_supervised_sample({'instruction': 'Write a story', 'output': 'Once upon a time, there was a cat.'})
            ([{'role': 'user', 'content': 'Write a story'}, {'role': 'assistant', 'content': 'Once upon a time, there was a cat.'}], '')

        Returns:
            tuple[list[dict[str, Any]], dict]: The formatted sample.
        """
        return [], {}

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
        """Format the sample for preference training.

        Args:
            raw_sample (dict[str, Any]): The raw sample from the dataset.

        Returns:
            tuple[list[dict[str, Any]], list[dict[str, Any]], dict]: The formatted sample.
        """
        return [], [], {}

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict]:
        """Format the sample for prompt-only training, e.g., PPO.

        Args:
            raw_sample (dict[str, Any]): The raw sample from the dataset.

        Returns:
            tuple[list[dict[str, Any]], dict]: The formatted sample.
        """
        return [], {}

    def format_unmatched_supervised_sample(
        self, raw_sample_for_prompt: dict[str, Any], raw_sample_for_response: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict]:
        """Format the sample for unmatched supervised training, e.g., KTO.

        Args:
            raw_sample_for_prompt (dict[str, Any]): The raw sample for prompt from the dataset.
            raw_sample_for_response (dict[str, Any]): The raw sample for response from the dataset.

        Returns:
            tuple[list[dict[str, Any]], dict]: The formatted sample.
        """
        return [], {}

        
@register_template('Alpaca')
class Alpaca(BaseFormatter):

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = ' '.join((raw_sample['instruction'], raw_sample['input']))
        response = raw_sample['output']
        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]},
        ], {}


@register_template('PKUSafeRLHF')
class PKUSafeRLHF(BaseFormatter):
    system_prompt: str = ''

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        metrics = raw_sample['better_response_id']
        better_response = raw_sample[f'response_{int(metrics)}']
        worse_response = raw_sample[f'response_{1-int(metrics)}']
        prompt = raw_sample['prompt']

        better_conversation = [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': better_response}]},
        ]

        worse_conversation = [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': worse_response}]},
        ]

        meta_info = {
            'better_response': better_response,
            'worse_response': worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = raw_sample['prompt']
        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
        ], {}

    def format_unmatched_supervised_sample(
        self, raw_sample_for_prompt: dict[str, Any], raw_sample_for_response: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = raw_sample_for_prompt['prompt']
        response = raw_sample_for_response['response_1']
        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]},
        ], {}


@register_template('Aligner')
class Aligner(BaseFormatter):
    system_prompt: str = ''

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:

        prompt = '##QUESTION: ' + raw_sample['question'] + ' ##ANSWER: ' + raw_sample['answer']
        text = '##CORRECTION: ' + raw_sample['correction']

        return [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': text}]},
        ], {}


@register_template('O1_T2T')
class O1_T2T(BaseFormatter):
    system_prompt: str = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_tokens = os.environ.get('O1_SPECIAL_TOKENS')
        if self.special_tokens is None:
            raise ValueError('O1_SPECIAL_TOKENS is not set')
        if self.special_tokens.startswith('[') and self.special_tokens.endswith(']'):
            self.special_tokens = self.special_tokens[1:-1].split(',')
            self.special_tokens = [
                token.strip().strip('"').strip("'") for token in self.special_tokens
            ]
        else:
            raise ValueError('O1_SPECIAL_TOKENS must be a list of strings')
        if len(self.special_tokens) < 3:
            raise ValueError('O1_SPECIAL_TOKENS must contain at least three tokens')

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        prompt = raw_sample['prompt']
        thoughts = ''
        answer = raw_sample['answer']
        for thought in raw_sample['thoughts']:
            if 'title' in thought.keys():
                thoughts += (
                    f"**{thought['title']}**\n{thought['content']}\n{self.special_tokens[1]}\n"
                )
            else:
                thoughts += f"{thought['content']}\n{self.special_tokens[1]}\n"
        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': f'{self.special_tokens[0]}{thoughts}{self.special_tokens[2]}{answer}',
                    }
                ],
            },
        ], {}


@register_template('AA_T2T')
class AA_T2T(BaseFormatter):
    system_prompt: str = ''

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']

        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {}


@register_template('Math-Zero-RL')
class Math_Zero_RL(BaseFormatter):
    # NOTE you should add the system prompt in these prompt templates
    system_prompt: str = (
        'You are a helpful assistant good at solving math problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.'
    )

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if 'prompt' in raw_sample:
            prompt = raw_sample['prompt']
        elif 'question' in raw_sample:
            prompt = raw_sample['question']
        else:
            raise ValueError(
                'Prompt Preparation Error: prompt or question is not found in the raw_sample'
            )
        answer = raw_sample['answer']

        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': answer},
        ], {}

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if 'prompt' in raw_sample:
            prompt = raw_sample['prompt']
        elif 'question' in raw_sample:
            prompt = raw_sample['question']
        else:
            raise ValueError(
                'Prompt Preparation Error: prompt or question is not found in the raw_sample'
            )
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ], {}


@register_template('TLDR')
class TLDR(BaseFormatter):
    system_prompt: str = ''
    summary_prompt: list[str] = [
        'Please summarize the following text: ',
        'Please give a concise summary of the following text: ',
        'Please provide a brief summary of the following text: ',
        'Please summarize the following text in a few sentences: ',
        'I need a summary of the following text: ',
        'Could you please provide a summary of the following text? ',
        "I'm looking for a summary of the following text: ",
        'Please give me a summary of the following text: ',
        'I need a summary of the following text: ',
        'Could you summarize the following text for me? ',
        "I'm looking for a summary of the following text: ",
        'Please provide a summary of the following text: ',
        "Here's the text I need summarized: ",
        'Here is the text I need summarized: ',
    ]

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = random.choice(self.summary_prompt) + raw_sample['content']
        answer = raw_sample['summary']

        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {}


@register_template('GSM8K')
class GSM8K(BaseFormatter):
    system_prompt: str = ''

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['question']
        answer = raw_sample['answer']

        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {}


@register_template('AA_TI2T')
class AA_TI2T(BaseFormatter):
    system_prompt: str = ''

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id == 1 else 1

        if better_id not in [1, 2] or worse_id not in [1, 2]:
            return [], [], {}

        raw_better_response = raw_sample[f'response_{better_id}']
        raw_worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['question']
        image_element = raw_sample['image']
        assert isinstance(
            image_element, (bytes, Image.Image)
        ), "raw_sample['image'] must be bytes or PIL.Image.Image type"
        image = (
            Image.open(io.BytesIO(image_element)).convert('RGBA')
            if isinstance(image_element, bytes)
            else image_element.convert('RGBA')
        )
        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        meta_info = {
            'image': image,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['question']
        image = raw_sample['image'].convert('RGBA')

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'image': image}

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        image = raw_sample['image'].convert('RGBA')

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {'image': image}

    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id == 1 else 1
        return better_id in [1, 2] and worse_id in [1, 2]


@register_template('AA_TA2T')
class AA_TA2T(BaseFormatter):
    system_prompt: str = 'You are a helpful assistant.'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id == 1 else 1
        better_response = raw_sample[f'response_{better_id}']
        worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['prompt']

        better_conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': 'placeholder'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': better_response}]},
        ]

        worse_conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': 'placeholder'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': worse_response}]},
        ]
        raw_audio, raw_sr = (
            raw_sample['audio_path']['array'],
            raw_sample['audio_path']['sampling_rate'],
        )
        audio = librosa.resample(raw_audio, orig_sr=raw_sr, target_sr=16000)

        meta_info = {
            'audios': [audio],
            'better_response': better_response,
            'worse_response': worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_id = int(raw_sample['overall_response'])
        better_response = raw_sample[f'response_{better_id}']
        prompt = raw_sample['prompt']

        better_conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': 'placeholder'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': better_response}]},
        ]

        raw_audio, raw_sr = (
            raw_sample['audio_path']['array'],
            raw_sample['audio_path']['sampling_rate'],
        )
        audio = librosa.resample(raw_audio, orig_sr=raw_sr, target_sr=16000)

        meta_info = {'audios': [audio]}

        return better_conversation, meta_info

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']

        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': 'placeholder'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ]

        raw_audio, raw_sr = (
            raw_sample['audio_path']['array'],
            raw_sample['audio_path']['sampling_rate'],
        )
        audio = librosa.resample(raw_audio, orig_sr=raw_sr, target_sr=16000)

        return conversation, {'audios': [audio]}


@register_template('AA_TA2T_LLF')
class AA_TA2T_LLF(BaseFormatter):
    system_prompt: str = 'You are a helpful assistant.'

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        better_response = raw_sample['refinement']
        worse_response = raw_sample['response']
        prompt = raw_sample['prompt']
        audio = raw_sample['audio']

        better_conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': audio},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': better_response},
        ]

        worse_conversation = [
            {'role': 'system', 'content': self.system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': audio},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': worse_response},
        ]

        meta_info = {
            'audio_path': audio,
            'better_response': better_response,
            'worse_response': worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        audio = raw_sample['audio']

        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': audio},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ]

        return conversation, {'audio_path': audio}


@register_template('AA_TI2T_LLF')
class AA_TI2T_LLF(BaseFormatter):
    system_prompt: str = ''

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        raw_better_response = raw_sample['refinement']
        raw_worse_response = raw_sample['response']
        prompt = raw_sample['prompt']
        image = load_image_from_base64(raw_sample['image']).convert('RGBA')
        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        meta_info = {
            'image': image,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt']
        image = load_image_from_base64(raw_sample['image']).convert('RGBA')

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'image': image}


@register_template('AA_TV2T')
class AA_TV2T(BaseFormatter):
    system_prompt: str = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_video_path = os.environ.get('ROOT_VIDEO_PATH')
        if self.root_video_path is None:
            raise ValueError('ROOT_VIDEO_PATH is not set')

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt'].replace('<video>', '').strip()
        video_path = self.root_video_path + raw_sample['video_path'].replace('./', '/')
        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'video': video_path}

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id == 1 else 1

        if better_id not in [1, 2] or worse_id not in [1, 2]:
            return [], [], {}

        raw_better_response = raw_sample[f'response_{better_id}']
        raw_worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['prompt'].replace('<video>', '').strip()
        video_path = self.root_video_path + raw_sample['video_path'].replace('./', '/')
        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        meta_info = {
            'video': video_path,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        better_id = int(raw_sample['overall_response'])
        worse_id = 2 if better_id == 1 else 1

        if better_id not in [1, 2] or worse_id not in [1, 2]:
            return [], [], {}

        raw_better_response = raw_sample[f'response_{better_id}']
        prompt = raw_sample['prompt'].replace('<video>', '').strip()
        video_path = self.root_video_path + raw_sample['video_path'].replace('./', '/')
        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]

        meta_info = {
            'video': video_path,
        }

        return better_conversation, meta_info


@register_template('DiffusionDB')
class DiffusionDB:
    system_prompt: str = ''

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        multi_modal_info = {'image': raw_sample['image'].convert('RGB')}
        return raw_sample['prompt'], multi_modal_info


@register_template('DiffusionDBCanny')
class DiffusionDBCanny:
    system_prompt: str = ''

    def format_diffusion_supervised_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        multi_modal_info = {'image': raw_sample['image'].convert('RGB')}
        return raw_sample['text'], multi_modal_info


@register_template('Pickapic')
class Pickapic:

    def format_diffusion_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['caption']
        better_id = int(raw_sample['label_1'])
        worse_id = int(raw_sample['label_0'])

        raw_better_image = raw_sample[f'jpg_{better_id}']
        raw_worse_image = raw_sample[f'jpg_{worse_id}']

        better_image = Image.open(io.BytesIO(raw_better_image)).convert('RGB')
        worse_image = Image.open(io.BytesIO(raw_worse_image)).convert('RGB')

        multi_modal_info = {
            'better_image': better_image,
            'worse_image': worse_image,
        }

        return prompt, multi_modal_info

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_id = float(raw_sample['label_0'])
        if better_id == 0.5:
            return True
        else:
            return False


@register_template('WavCaps')
class WavCaps:

    def format_diffusion_supervised_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        caption = raw_sample['answer']
        audio = raw_sample['context']['array']
        sampling_rate = raw_sample['context']['sampling_rate']
        multi_modal_info = {
            'audio': audio,
            'sampling_rate': sampling_rate,
        }
        return caption, multi_modal_info


@register_template('AA_T2A')
class AA_T2A(BaseFormatter):

    def format_diffusion_preference_sample(self, raw_sample: dict[str, Any]) -> tuple[str, Any]:
        better_id = int(raw_sample['overall_audio'])
        worse_id = 2 if better_id == 1 else 1
        better_audio = raw_sample[f'response_{better_id}']
        worse_audio = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['prompt']

        multi_modal_info = {
            'better_audio': better_audio,
            'worse_audio': worse_audio,
        }
        return prompt, multi_modal_info

    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['overall_audio'])
        worse_id = 2 if better_id == 1 else 1
        return better_id in [1, 2] and worse_id in [1, 2]

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['overall_audio'])
        worse_id = 2 if better_id == 1 else 1
        return better_id == worse_id


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
            f'{self.assistant_prompt.format(output=better_text_processed)}'
        )

        better_images_full = safe_add(input_images, better_images)

        worse_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f'{self.assistant_prompt.format(output=worse_text_processed)}'
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
            f'{self.assistant_prompt.format(output=better_text_processed)}'
        )

        better_images_full = safe_add(input_images, better_images)

        worse_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f'{self.assistant_prompt.format(output=worse_text_processed)}'
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
class ANY2ANY:

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        output_dict = raw_sample.copy()
        if 'input_image' in raw_sample and raw_sample['input_image'] is not None:
            output_dict['input_image'] = load_image(raw_sample['input_image'])
        if 'output_image' in raw_sample and raw_sample['output_image'] is not None:
            output_dict['output_image'] = load_image(raw_sample['output_image'])
        print(f'Get output dict: {output_dict}')
        return output_dict


@register_template('AA_textfeedback')
class AA_TF:
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = (
        'USER: Judge the following two response of the same question and give a preference: \n ##Question: {input} \n ##Response 1: {response_1} \n ##Response 2: {response_2}'
    )
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    def format_supervised_sample(
        self, raw_sample: dict[str, Any], path: str | None = None
    ) -> dict[str, Any]:
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
            f'{self.assistant_prompt.format(output=feedback)}'
        )

        prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed, response_1=output_text_1_processed, response_2=output_text_2_processed)}'
            f"{self.assistant_prompt.format(output='')}"
        )

        input_images = safe_add(safe_add(input_images, output_images_1), output_images_2)

        return {'text': text, 'prompt': prompt, 'input_image': input_images, 'image': input_images}


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
            f'{self.assistant_prompt.format(output=better_text_processed)}'
        )

        better_images_full = safe_add(input_images, better_images)

        worse_text_full = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text_processed)}'
            f'{self.assistant_prompt.format(output=worse_text_processed)}'
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
            'input_text': prompt,
            'input_image': [],
            'better_text': '',
            'better_img': better_image,
            'worse_text': '',
            'worse_img': worse_image,
        }
        return super().format_example(raw_sample_upd)


@register_template('GQA')
class GQA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = raw_sample['answer']

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=answer)}'
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

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = max(set(raw_sample['answers']), key=raw_sample['answers'].count)

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=answer)}'
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

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        question = raw_sample['question']
        answer = raw_sample['choices'][raw_sample['correct_choice_idx']]
        rationales = ' '.join(raw_sample['rationales'])

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=answer, rationales=rationales)}'
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
    user_prompt: str = (
        'USER: \n<image> According to the content of the pictures, answer the following questions in order.\n{input}'
    )
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        questions = raw_sample['questions']
        answers = raw_sample['answers']
        question = '\n'.join(questions)
        answer = '\n'.join(answers)

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=answer)}'
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
    user_prompt: str = (
        'USER: \n<image> According to the content of the pictures, answer the following questions in order.\n{input}'
    )
    assistant_prompt: str = '\nASSISTANT: {output}'
    split_token: str = 'ASSISTANT:'

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        questions = raw_sample['questions']
        answers = raw_sample['answers']
        quetsion = '\n'.join(questions)
        answer = '\n'.join(answers)

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=quetsion)}'
            f'{self.assistant_prompt.format(output=answer)}'
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

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
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

        image_file = os.path.join(
            path, 'mnt/petrelfs/wangwenhai/workspace_cef/4o/image', raw_sample['image']
        )
        return {
            'text': text,
            'prompt': prompt,
            'image': Image.open(image_file),
        }


@register_template('AudioCaps')
class AudioCaps:
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        caption = raw_sample['caption']
        audiocap_path = raw_sample['audiocap_path']
        question = random.choice(AUDIO_QUESTIONS)

        text = (
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=caption)}'
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        audio, sample_rate = torchaudio.load(os.path.join(path, f'{audiocap_path}'))
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        return {
            'text': text,
            'prompt': prompt,
            'audio': audio.squeeze().tolist(),
            'sampling_rate': sample_rate,
        }


@register_template('LibriSpeech')
class LibriSpeech:
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        caption = raw_sample['text'].lower()
        question = random.choice(SPEECH_QUESTIONS)

        text = (
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=caption)}'
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        audio = raw_sample['audio']['array']
        sampling_rate = raw_sample['audio']['sampling_rate']
        return {'text': text, 'prompt': prompt, 'audio': audio, 'sampling_rate': sampling_rate}


@register_template('AudioSet')
class AudioSet:
    user_prompt: str = 'USER: {input}'
    assistant_prompt: str = '\nASSISTANT: {output}'

    def format_sample(self, raw_sample: dict[str, Any], path: str = None) -> dict[str, Any]:
        caption = f"The content of audio is {', '.join(raw_sample['captions'])}."
        question = random.choice(AUDIO_QUESTIONS)

        text = (
            f'{self.user_prompt.format(input=question)}'
            f'{self.assistant_prompt.format(output=caption)}'
        )

        prompt = (
            f'{self.user_prompt.format(input=question)}'
            f"{self.assistant_prompt.format(output='')}"
        )
        audio, sample_rate = torchaudio.load(os.path.join(path, raw_sample['audio']))
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        return {
            'text': text,
            'prompt': prompt,
            'audio': audio.squeeze().tolist(),
            'sampling_rate': sample_rate,
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
            len(input_img)
        else:
            raise ValueError('input_image must be either a string or a list of strings')

        input_text = f"{'<image>' * num_imput_img}{input_text}"

        # do the same for output
        if isinstance(output_img, str):
            output_images = [load_image(output_img)]
            num_output_img = 1

        elif isinstance(output_img, list):
            output_images = [load_image(img) for img in output_img]
            num_output_img = len(output_img)
        else:
            raise ValueError('output_image must be either a string or a list of strings')

        output_text = f"{output_text}{'<image>' * num_output_img}"

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f'{self.assistant_prompt.format(output=output_text)}'
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
            raise ValueError('input_image must be either a string or a list of strings')

        input_text = f"{'<image>' * num_imput_img}{input_text}"

        # do the same for output
        if isinstance(output_img, str):
            output_images = [load_image(output_img)]
            num_output_img = 1

        elif isinstance(output_img, list):
            output_images = [load_image(img) for img in output_img]
            num_output_img = len(output_img)
        else:
            raise ValueError('output_image must be either a string or a list of strings')

        output_text = f"{output_text}{'<image>' * num_output_img}"

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f'{self.assistant_prompt.format(output=output_text)}'
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
            len(input_img)
        else:
            raise ValueError('input_image must be either a string or a list of strings')

        input_text = f"{'<image>' * num_imput_img}{input_text}"

        # do the same for output
        if isinstance(output_img, str):
            output_images = [load_image(output_img)]
            num_output_img = 1

        elif isinstance(output_img, list):
            output_images = [load_image(img) for img in output_img]
            num_output_img = len(output_img)
        else:
            raise ValueError('output_image must be either a string or a list of strings')

        output_text = f"{output_text}{'<image>' * num_output_img}"

        text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=input_text)}'
            f'{self.assistant_prompt.format(output=output_text)}'
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

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        raw_better_response = raw_sample['chosen']
        raw_worse_response = raw_sample['rejected']
        prompt = raw_sample['question']
        image = raw_sample['image'].convert('RGBA')

        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        meta_info = {
            'image': image,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return raw_sample['chosen'] == raw_sample['rejected']

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['question']
        image = raw_sample['image'].convert('RGBA')

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'image': image}


@register_template('SPA_VL')
class SPA_VL:
    system_prompt: str = (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        raw_better_response = raw_sample['chosen']
        raw_worse_response = raw_sample['rejected']
        prompt = raw_sample['question']
        image = raw_sample['image'].convert('RGBA')

        better_conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        meta_info = {
            'image': image,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def check_equal(self, raw_sample: dict[str, Any]) -> bool:
        return raw_sample['chosen'] == raw_sample['rejected']

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            raw_sample['question']
            .replace('<image>\n', '')
            .replace('\n<image>', '')
            .replace('<image>', '')
        )
        image = raw_sample['image'].convert('RGBA')

        return [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'image': image}


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
    def format_preference_sample(
        self, raw_sample: dict[str, Any], path: str = None
    ) -> dict[str, Any]:
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
            f'{self.assistant_prompt.format(output=output)}'
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
            'text': return_text,
            'prompt': return_prompt,
            'image': [],
            'video': video_info,
        }
        return return_dict

    def format_preference_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']
        better_output = raw_sample['better_output']
        worse_output = raw_sample['worse_output']

        return_better_text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=better_output)}'
        )

        return_worse_text = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=worse_output)}'
        )

        video_info = raw_sample['video_path']
        if isinstance(video_info, str):
            video_info = [video_info]

        return_dict = {
            'better_text': return_better_text,
            'worse_text': return_worse_text,
            'image': [],
            'video': video_info,
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
            {
                'role': 'user',
                'content': [
                    {'type': 'audio', 'audio_url': audio_url},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': response},
        ]

        return {
            'conversation': conversation,
            'prompt': conversation[:-1],
        }


@register_template('SafeRLHF_V_Reward')
class SafeRLHF_V_Reward(BaseFormatter):
    system_prompt: str = ''

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        better_id = int(raw_sample['more_helpful_response_id'])
        worse_id = 2 if better_id == 1 else 1

        if better_id not in [1, 2] or worse_id not in [1, 2]:
            return [], [], {}

        raw_better_response = raw_sample[f'response_{better_id}']
        raw_worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['question']
        image = raw_sample['image']
        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        meta_info = {
            'image': image,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
        }

        return better_conversation, worse_conversation, meta_info

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['question']
        image = raw_sample['image']

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'image': image}

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        image = raw_sample['image']

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {'image': image}

    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['more_helpful_response_id'])
        worse_id = 2 if better_id == 1 else 1
        return better_id in [1, 2] and worse_id in [1, 2]


@register_template('SafeRLHF_V_Cost')
class SafeRLHF_V_Cost(BaseFormatter):
    system_prompt: str = ''

    def format_preference_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        worse_id = int(raw_sample['safer_response_id'])
        better_id = 2 if worse_id == 1 else 1

        if better_id not in [1, 2] or worse_id not in [1, 2]:
            return [], [], {}

        raw_better_response = raw_sample[f'response_{better_id}']
        raw_worse_response = raw_sample[f'response_{worse_id}']
        prompt = raw_sample['question']
        image = raw_sample['image']
        better_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_better_response}]},
        ]
        worse_conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': raw_worse_response}]},
        ]

        is_better_safe = raw_sample[f'response_{better_id}_harmless_rate'] * (-1)
        is_worse_safe = raw_sample[f'response_{worse_id}_harmless_rate'] * (-1)

        meta_info = {
            'image': image,
            'better_response': raw_better_response,
            'worse_response': raw_worse_response,
            'is_better_safe': is_better_safe,
            'is_worse_safe': is_worse_safe,
        }

        return better_conversation, worse_conversation, meta_info

    def format_prompt_only_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['question']
        image = raw_sample['image']

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ], {'image': image}

    def format_supervised_sample(
        self, raw_sample: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = raw_sample['prompt']
        answer = raw_sample['response']
        image = raw_sample['image']

        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt},
                ],
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {'image': image}

    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        better_id = int(raw_sample['safer_response_id'])
        worse_id = 2 if better_id == 1 else 1
        return better_id in [1, 2] and worse_id in [1, 2]


@register_template('LLaVA_Pretrain')
class LLaVA_Pretrain(BaseFormatter):
    system_prompt: str = ''

    def format_supervised_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        conversation = raw_sample['conversations']
        for message in conversation:
            if message['from'] == 'human':
                prompt = message['value']
            elif message['from'] == 'gpt':
                answer = message['value']
        coco_data_dir = os.environ['COCO_DATA_DIR']
        image_path = os.path.join(coco_data_dir, raw_sample['image'])
        image = Image.open(image_path).convert('RGBA')
        return [
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]},
        ], {'image': image}
