# Copyright 2024 PKU-Alignment Team and Haotian Liu. All Rights Reserved.
#
# This code is inspired by the LLaVA library.
# https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/model_worker.py
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
import asyncio
import base64
import json
import threading
import time
import uuid
from functools import partial
from io import BytesIO
from threading import Thread

import requests
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from PIL import Image
from transformers import TextIteratorStreamer

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.utils.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    WORKER_HEART_BEAT_INTERVAL,
)
from align_anything.utils.logger import Logger


server_error_msg = '**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**'


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return 'None'
    return f'Semaphore(value={semaphore._value}, locked={semaphore.locked()})'


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = Logger()
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_path,
        model_name,
        device,
        is_multimodal,
        template=None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.context_len = 2048
        self.template = template
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split('/')
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + '_' + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.print(f'Loading the model {self.model_name} on worker {worker_id} ...')
        self.model, self.tokenizer, self.image_processor = load_pretrained_models(
            model_path, auto_device_mapping=True, padding_side='right', trust_remote_code=True
        )
        self.is_multimodal = is_multimodal

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.print('Register to controller')

        url = self.controller_addr + '/register_worker'
        data = {
            'worker_name': self.worker_addr,
            'check_heart_beat': True,
            'worker_status': self.get_status(),
            'template': self.template,
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.print(
            f'Send heart beat. Models: {[self.model_name]}. '
            f'Semaphore: {pretty_print_semaphore(model_semaphore)}. '
            f'global_counter: {global_counter}'
        )

        url = self.controller_addr + '/receive_heart_beat'

        while True:
            try:
                ret = requests.post(
                    url,
                    json={'worker_name': self.worker_addr, 'queue_length': self.get_queue_length()},
                    timeout=5,
                )
                exist = ret.json()['exist']
                break
            except requests.exceptions.RequestException as e:
                logger.error(f'heart beat error: {e}')
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)
            )

    def get_status(self):
        return {
            'model_names': [self.model_name],
            'speed': 1,
            'queue_length': self.get_queue_length(),
            'template': self.template,
        }

    # TODO:change this
    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params['prompt']
        ori_prompt = prompt
        images = params.get('images', None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError(
                    'Number of images does not match number of <image> tokens in prompt'
                )

            images = [load_image_from_base64(image) for image in images]

            if type(images) is list:
                for image in images:
                    images = image
            else:
                images = images

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        else:
            images = None

        temperature = float(params.get('temperature', 0.6))
        top_p = float(params.get('top_p', 0.9))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get('max_new_tokens', 256)), 1024)
        stop_str = params.get('stop', None)
        do_sample = True if temperature > 0.001 else False

        inputs = image_processor(prompt, images, return_tensors='pt')
        if images is None:
            inputs = dict(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
            )
        else:
            inputs = dict(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                pixel_values=inputs['pixel_values'].to(self.device),
            )
        keywords = [stop_str]
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
        )

        max_new_tokens = min(
            max_new_tokens, max_context_length - inputs['input_ids'].shape[-1] - num_image_tokens
        )

        if max_new_tokens < 1:
            yield json.dumps(
                {
                    'text': ori_prompt
                    + 'Exceeds max token length. Please start a new conversation, thanks.',
                    'error_code': 0,
                }
            ).encode() + b'\0'
            return

        thread = Thread(
            target=model.generate,
            kwargs=dict(
                **inputs,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
            ),
        )
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[: -len(stop_str)]
            yield json.dumps({'text': generated_text, 'error_code': 0}).encode() + b'\0'

    def generate_stream_gate(self, params):
        try:
            yield from self.generate_stream(params)
        except ValueError as e:
            print('Caught ValueError:', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'
        except torch.cuda.CudaError as e:
            print('Caught torch.cuda.CudaError:', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'
        except Exception as e:
            print('Caught Unknown Error', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post('/worker_generate_stream')
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post('/worker_get_status')
async def get_status(request: Request):
    return worker.get_status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=21002)
    parser.add_argument('--worker-address', type=str, default='http://localhost:21002')
    parser.add_argument('--controller-address', type=str, default='http://localhost:21001')
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit-model-concurrency', type=int, default=5)
    parser.add_argument('--stream-interval', type=int, default=1)
    parser.add_argument('--no-register', action='store_true')
    parser.add_argument('--is-multimodal', required=True, type=bool)
    parser.add_argument('--template', type=str, default='Dialogue')
    args = parser.parse_args()
    logger.print(f'args: {args}')

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_name,
        args.device,
        args.is_multimodal,
        args.template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
