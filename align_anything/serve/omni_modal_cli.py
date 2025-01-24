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
"""Command line interface for interacting with a multi-modal model."""


import argparse
import os

import gradio as gr
import librosa
import torch
from PIL import Image

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.utils.process_minicpmo import get_video_chunk_content


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXAMPLES = [
    {
        'files': [os.path.join(CURRENT_DIR, 'examples/logo.jpg')],
        'text': 'What is the university of this logo?',
    },
]

AUDIO_EXAMPLES = [
    {
        'files': [os.path.join(CURRENT_DIR, 'examples/drum.wav')],
        'text': 'What is the emotion of this drumbeat like?',
    },
]

VIDEO_EXAMPLES = [
    {'files': [os.path.join(CURRENT_DIR, 'examples/baby.mp4')], 'text': 'What is the video about?'},
]

AUDIO_FILES = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
IMAGE_FILES = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.ico', '.webp']
VIDEO_FILES = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']


def satisfy_modal(info: str):
    for file in AUDIO_FILES:
        if info.endswith(file):
            return 'audio'
    for file in IMAGE_FILES:
        if info.endswith(file):
            return 'image'
    for file in VIDEO_FILES:
        if info.endswith(file):
            return 'video'


def multi_modal_conversation(question: str, multi_modal_info: list):
    content = [question]
    for info in multi_modal_info:
        modality = satisfy_modal(info)
        if modality == 'audio':
            audio, _ = librosa.load(info, sr=16000, mono=True)
            content.append(audio)
        elif modality == 'image':
            content.append(Image.open(info))
        elif modality == 'video':
            content.append(get_video_chunk_content(info))
    return [{'role': 'user', 'content': content}]


def text_conversation(text: str, role: str = 'user'):
    return [{'role': role, 'content': [{'type': 'text', 'text': text}]}]


def question_answering(message: dict, history: list):
    system_message = model.get_sys_prompt(mode='omni', language='en')
    conversation = [system_message]
    for i, past_message in enumerate(history):
        if isinstance(past_message, str):
            conversation.extend(text_conversation(past_message))
        elif isinstance(past_message, dict):
            if past_message['role'] == 'user':
                # judge whether the next message is a multi-modal message
                if isinstance(past_message['content'], str):
                    if i + 1 < len(history) and isinstance(history[i + 1]['content'], tuple):
                        conversation.extend(
                            multi_modal_conversation(
                                past_message['content'], list(history[i + 1]['content'])
                            )
                        )
                    elif i - 1 >= 0 and isinstance(history[i - 1]['content'], tuple):
                        conversation.extend(
                            multi_modal_conversation(
                                past_message['content'], list(history[i - 1]['content'])
                            )
                        )
                    else:
                        conversation.extend(text_conversation(past_message['content']))
            elif past_message['role'] == 'assistant':
                conversation.extend(text_conversation(past_message['content'], 'assistant'))

    if len(message['files']) == 0:
        current_question = message['text']
        conversation.extend(text_conversation(current_question))
    else:
        current_question = message['text']
        current_multi_modal_info = message['files']
        conversation.extend(
            multi_modal_conversation(current_question, list(current_multi_modal_info))
        )

    res = model.chat(
        msgs=conversation,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.5,
        max_new_tokens=4096,
        omni_input=True,  # please set omni_input=True when omni inference
        use_tts_template=True,
        max_slice_nums=1,
        use_image_id=False,
        return_dict=True,
    )
    return res.text


if __name__ == '__main__':
    # Define the Gradio interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    os.environ['MODEL_NAME_OR_PATH'] = model_name_or_path
    global processor, model, tokenizer, chat_template
    model, tokenizer, processor = load_pretrained_models(
        model_name_or_path,
        dtype=torch.float16,
        trust_remote_code=True,
        auto_model_kwargs={'init_vision': True, 'init_audio': True, 'init_tts': True},
    )
    model = model.eval().cuda()

    examples = IMAGE_EXAMPLES + AUDIO_EXAMPLES + VIDEO_EXAMPLES
    iface = gr.ChatInterface(
        fn=question_answering,
        type='messages',
        multimodal=True,
        title='Align Anything Omni-Modal CLI',
        description='Upload multiple modalities info and ask a question related. The AI will try to answer it.',
        examples=examples,
        theme=gr.themes.Ocean(
            text_size='lg',
            spacing_size='lg',
            radius_size='lg',
        ),
    )

    iface.launch(share=True)
