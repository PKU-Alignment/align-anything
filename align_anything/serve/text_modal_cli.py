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
"""Command line interface for interacting with a text-modal model."""


import argparse
import os

import gradio as gr
import torch

from align_anything.models.pretrained_model import load_pretrained_models


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def text_conversation(text: str, role: str = 'user'):
    return [{'role': role, 'content': text}]


def question_answering(message: dict, history: list):
    conversation = []

    for past_message in history:
        if isinstance(past_message, str):
            conversation.extend(text_conversation(past_message))
        elif isinstance(past_message, dict):
            if past_message['role'] == 'user':
                conversation.extend(text_conversation(past_message['content']))
            elif past_message['role'] == 'assistant':
                conversation.extend(text_conversation(past_message['content'], 'assistant'))

    current_question = message['text']
    conversation.extend(text_conversation(current_question))
    res = model.chat(messages=conversation, tokenizer=tokenizer)
    return res


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
    )
    model = model.eval().cuda()

    iface = gr.ChatInterface(
        fn=question_answering,
        type='messages',
        multimodal=True,
        title='Align Anything Text-Modal CLI',
        description='Ask a question related. The AI will try to answer it.',
        theme=gr.themes.Ocean(
            text_size='lg',
            spacing_size='lg',
            radius_size='lg',
        ),
    )

    iface.launch(share=True)
