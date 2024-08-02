# Copyright 2024 PKU-Alignment Team and Haotian Liu. All Rights Reserved.
#
# This code is inspired by the LLaVA library.
# https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/gradio_web_server.py
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
# ========================================================================================================
import argparse
import base64
import dataclasses
import datetime
import hashlib
import json
import os
import time
from io import BytesIO
from typing import List

import gradio as gr
import requests
from PIL import Image

from align_anything.configs.template import *
from align_anything.utils.logger import Logger
from align_anything.utils.template_registry import get_template_class


server_error_msg = '**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**'
moderation_msg = 'YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN.'

LOGDIR = '.'


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    roles: List[str]
    messages: List[List[str]]
    offset: int
    template: Template = Dialogue()

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        ret = self.template.system_prompt
        if len(messages) > 0:
            messages = self.messages.copy()
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                        if role == 'user':
                            ret += self.template.user_prompt.format(
                                input=message.replace('<image>', '')
                            )
                        elif role == 'assistant':
                            ret += self.template.assistant_prompt.format(output=message)
                        else:
                            raise ValueError(f'Invalid role: {role}')
                    else:
                        if role == 'user':
                            user_prompt = self.template.user_prompt.format(input=message)
                            ret += user_prompt.replace('<image>', '')
                        elif role == 'assistant':
                            ret += self.template.assistant_prompt.format(output=message)
                        else:
                            raise ValueError(f'Invalid role: {role}')
                else:
                    if role == 'user':
                        ret += self.template.user_prompt.format(input='')
                    elif role == 'assistant':
                        ret += self.template.assistant_prompt.format(output='')
                    else:
                        raise ValueError(f'Invalid role: {role}')
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(
        self,
        image,
        image_process_mode,
        return_pil=False,
        image_format='PNG',
        max_len=1344,
        min_len=672,
    ):
        if image_process_mode == 'Pad':

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ['Default', 'Crop']:
            pass
        elif image_process_mode == 'Resize':
            image = image.resize((336, 336))
        else:
            raise ValueError(f'Invalid image_process_mode: {image_process_mode}')
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    # TODO:adjust this function to video and audio.
    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, 'Default', return_pil=False, image_format='JPEG'
                    )
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, 'Default', return_pil=False, image_format='JPEG'
                    )
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret[-1][-1] = msg
                else:
                    ret[-1][-1] = msg
        return ret

    def copy(self) -> 'Conversation':
        return Conversation(
            roles=self.roles,
            messages=self.messages.copy(),
            offset=self.offset,
            template=self.template,
            skip_next=self.skip_next,
        )

    def dict(self) -> dict:
        return {
            'roles': self.roles,
            'messages': [[x, y[0] if isinstance(y, tuple) else y] for x, y in self.messages],
            'offset': self.offset,
            'skip_next': self.skip_next,
        }


default_conversation = Conversation(('user', 'assistant'), [], 0, LLAVA)

logger = Logger()

headers = {'User-Agent': 'LLaVA Client'}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = 'https://api.openai.com/v1/moderations'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
    }
    text = text.replace('\n', '')
    data = '{' + '"input": ' + f'"{text}"' + '}'
    data = data.encode('utf-8')
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()['results'][0]['flagged']
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f'{t.year}-{t.month:02d}-{t.day:02d}-conv.json')
    return name


def get_model_list():
    ret = requests.post(args.controller_url + '/refresh_all_workers')
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + '/list_models')
    model_names = ret.json()['model_names']
    model_templates = ret.json()['model_templates']
    logger.print('Models: ')
    for model_name in model_names:
        logger.print(f'{model_name} ')
    return model_names, model_templates


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.print(f'load_demo. ip: {request.client.host}. params: {url_params}')

    dropdown_update = gr.Dropdown(visible=True)
    if 'model' in url_params:
        model = url_params['model']
        if model in models.keys():
            dropdown_update = gr.Dropdown(value=model, visible=True)
    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.print(f'load_demo. ip: {request.client.host}')
    model_names, model_templates = get_model_list()
    models = dict(zip(model_names, model_templates))
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=model_names, value=model_names[0] if len(model_names) > 0 else ''
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), 'a') as fout:
        data = {
            'tstamp': round(time.time(), 4),
            'type': vote_type,
            'model': model_selector,
            'state': state.dict(),
            'ip': request.client.host,
        }
        fout.write(json.dumps(data) + '\n')


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.print(f'upvote. ip: {request.client.host}')
    vote_last_response(state, 'upvote', model_selector, request)
    return ('',) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.print(f'downvote. ip: {request.client.host}')
    vote_last_response(state, 'downvote', model_selector, request)
    return ('',) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.print(f'flag. ip: {request.client.host}')
    vote_last_response(state, 'flag', model_selector, request)
    return ('',) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.print(f'regenerate. ip: {request.client.host}')
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), '', None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.print(f'clear_history. ip: {request.client.host}')
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), '', None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, videobox, audiobox, request: gr.Request):
    logger.print(f'add_text. ip: {request.client.host}. len: {len(text)}')
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), '', None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), '', None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.print(f'http_bot. ip: {request.client.host}')
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        model_names, model_templates = get_model_list()
        models = dict(zip(model_names, model_templates))
        template_name = models[model_name]
        template = get_template_class(template_name)
        new_state = Conversation(('user', 'assistant'), [], 0, template)
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + '/get_worker_address', json={'model': model_name})
    worker_addr = ret.json()['address']
    logger.print(f'model_name: {model_name}, worker_addr: {worker_addr}')

    # No available worker
    if worker_addr == '':
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, 'serve_images', f'{t.year}-{t.month:02d}-{t.day:02d}', f'{hash}.jpg'
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        'model': model_name,
        'prompt': prompt,
        'temperature': float(temperature),
        'top_p': float(top_p),
        'max_new_tokens': min(int(max_new_tokens), 1536),
        'stop': state.template.separator,
        'images': f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.print(f'==== request ====\n{pload}')

    pload['images'] = state.get_images()

    state.messages[-1][-1] = '▌'
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + '/worker_generate_stream',
            headers=headers,
            json=pload,
            stream=True,
            timeout=10,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b'\0'):
            if chunk:
                data = json.loads(chunk.decode())
                if data['error_code'] == 0:
                    output = data['text'][len(prompt) :].strip()
                    state.messages[-1][-1] = output + '▌'
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data['text'] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.print(f'{output}')

    with open(get_conv_log_filename(), 'a') as fout:
        data = {
            'tstamp': round(finish_tstamp, 4),
            'type': 'chat',
            'model': model_name,
            'start': round(start_tstamp, 4),
            'finish': round(finish_tstamp, 4),
            'state': state.dict(),
            'images': all_image_hash,
            'ip': request.client.host,
        }
        fout.write(json.dumps(data) + '\n')


title_markdown = """# Align-Anything
[[Project Page](https://align-anything.com)] [[Code](https://github.com/PKU-Alignment/align-anything)]
"""

tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


learn_more_markdown = """
### License
Copyright 2024 PKU-Alignment Team and Haotian Liu. All Rights Reserved.

This code is inspired by the LLaVA library.
https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/gradio_web_server.py

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(
        show_label=False, placeholder='Enter text and press ENTER', container=False
    )
    with gr.Blocks(title='align-anything', theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=3):
                if not embed_mode:
                    with gr.Row():
                        gr.Image(
                            value=f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/assets/logo.jpg'
                        )
                        gr.Markdown(title_markdown)
                with gr.Row(elem_id='model_selector_row'):
                    model_selector = gr.Dropdown(
                        choices=model_names,
                        value=model_names[0] if len(model_names) > 0 else '',
                        interactive=True,
                        show_label=False,
                        container=False,
                    )

                imagebox = gr.Image(type='pil')
                image_process_mode = gr.Radio(
                    ['Crop', 'Resize', 'Pad', 'Default'],
                    value='Default',
                    label='Preprocess for non-square image',
                    visible=False,
                )

                with gr.Column(scale=3):
                    videobox = gr.Video()

                with gr.Column(scale=3):
                    audiobox = gr.Audio(type='filepath')

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))

                with gr.Accordion('Parameters', open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label='Temperature',
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label='Top P',
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label='Max output tokens',
                    )

            with gr.Column(scale=8):
                gr.Examples(
                    examples=[
                        [f'{cur_dir}/examples/PKU.jpg', 'What is great about this image?'],
                        [
                            f'{cur_dir}/examples/boya.jpg',
                            'What are the things I should pay attention to when I visit here?',
                        ],
                    ],
                    inputs=[imagebox, textbox, audiobox, videobox],
                )
                chatbot = gr.Chatbot(
                    elem_id='chatbot',
                    label='Chatbot',
                    height=650,
                    layout='panel',
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value='Send', variant='primary')
                with gr.Row(elem_id='buttons') as button_row:
                    upvote_btn = gr.Button(value='👍  Upvote', interactive=False)
                    downvote_btn = gr.Button(value='👎  Downvote', interactive=False)
                    flag_btn = gr.Button(value='⚠️  Flag', interactive=False)
                    regenerate_btn = gr.Button(value='🔄  Regenerate', interactive=False)
                    clear_btn = gr.Button(value='🗑️  Clear', interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )

        regenerate_btn.click(
            regenerate, [state, image_process_mode], [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        )

        clear_btn.click(
            clear_history, None, [state, chatbot, textbox, imagebox] + btn_list, queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode, videobox, audiobox],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode, videobox, audiobox],
            [state, chatbot, textbox, imagebox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        )

        if args.model_list_mode == 'once':
            demo.load(load_demo, [url_params], [state, model_selector], js=get_window_url_params)
        elif args.model_list_mode == 'reload':
            demo.load(load_demo_refresh_model_list, None, [state, model_selector], queue=False)
        else:
            raise ValueError(f'Unknown model list mode: {args.model_list_mode}')

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int)
    parser.add_argument('--controller-url', type=str, default='http://localhost:10000')
    parser.add_argument('--concurrency-count', type=int, default=16)
    parser.add_argument('--model-list-mode', type=str, default='once', choices=['once', 'reload'])
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--moderate', action='store_true')
    parser.add_argument('--embed', action='store_true')
    args = parser.parse_args()
    logger.print(f'args: {args}')

    model_names, model_templates = get_model_list()
    models = dict(zip(model_names, model_templates))
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[
            f'{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/assets/logo.jpg'
        ],
    )
