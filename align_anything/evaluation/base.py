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
import random
from abc import abstractmethod
from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.evaluation.dis_utils import *
from align_anything.models.pretrained_model import load_pretrained_models
from datasets import DatasetDict, load_dataset


ACTION_GENERATION = 'generation'
ACTION_LOGITS = 'logits'
ACTION_PPL = 'ppl'


class BaseEvaluator:

    action_map = {
        ACTION_LOGITS: 'choice_logits',
        ACTION_PPL: 'choice_ppl',
        ACTION_GENERATION: 'generation',
    }

    def __init__(self, cfgs, ds_cfgs):
        self.ds_cfgs = ds_cfgs
        self.eval_cfgs, self.data_cfgs, self.model_cfgs = (
            cfgs.eval_cfgs,
            cfgs.data_cfgs,
            cfgs.model_cfgs,
        )

        self.action = self.eval_cfgs.action
        self.num_shot = self.eval_cfgs.n_shot
        self.device = (
            self.eval_cfgs.device
            if self.eval_cfgs.device
            else 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.output_dir = self.eval_cfgs.output_dir
        self.cache_dir = os.path.join(self.output_dir, '.cache')
        self.temperature = self.eval_cfgs.temperature if self.eval_cfgs.temperature else 0.7
        self.generate_config = (
            self.eval_cfgs.generate_config if self.eval_cfgs.generate_config else {}
        )

        self.batch_size = self.eval_cfgs.batch_size if self.eval_cfgs.batch_size else 1
        assert self.batch_size == 1, 'Current version only supports batch_size=1'

        self.split = self.data_cfgs.split
        self.task_dir = self.data_cfgs.task_dir
        self.candidate_labels = self.data_cfgs.candidate_labels

        self.model_id = self.model_cfgs.model_id
        self.max_length = self.model_cfgs.max_length
        self.max_new_tokens = self.model_cfgs.max_new_tokens

        self.task2details = {}
        self.task2correction = {}
        self.details_filename = f'{self.model_id}_details.jsonl'
        self.results_filename = f'{self.model_id}_results.json'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self.task_names = self.get_task_names()

        self.init_model()

    def init_model(self) -> None:
        if self.ds_cfgs is not None and self.ds_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_cfgs)

        if self.ds_cfgs is not None:
            self.model, self.tokenizer, self.processor = load_pretrained_models(
                self.model_cfgs.model_name_or_path,
                model_max_length=self.model_cfgs.model_max_length,
                padding_side='right',
                trust_remote_code=self.model_cfgs.trust_remote_code,
            )
            self.model, *_ = deepspeed.initialize(model=self.model, config=self.ds_cfgs)
        else:
            self.model, self.tokenizer, self.processor = load_pretrained_models(
                self.model_cfgs.model_name_or_path,
                model_max_length=self.model_cfgs.model_max_length,
                padding_side='right',
                auto_device_mapping=True,
                trust_remote_code=self.model_cfgs.trust_remote_code,
            )
        self.model.eval()

    def load_dataset(self, task_name: str) -> DatasetDict:
        return load_dataset(self.task_dir, task_name)

    def eval(self) -> None:
        local_task2details = {}
        local_task2correction = {}
        for name in self.task_names:
            task2details, correction = self.eval_task(name, self.split)

            self.update_results(task2details)

            local_task2details.update(task2details)
            local_task2correction.update(correction)

        self.save_to_cache(local_task2details, f'{self.model_id}_details.jsonl')
        self.save_to_cache(local_task2correction, f'{self.model_id}_correction.jsonl')
        dist.barrier()

        if is_main_process():
            self.load_cache()
            self.calculate_results()

    def load_cache(self) -> None:
        for line in open(
            os.path.join(self.cache_dir, f'{self.model_id}_details.jsonl'), encoding='utf-8'
        ):
            task2details = json.loads(line)
            if len(self.task2details) == 0:
                self.task2details = task2details
            else:
                for k, v in task2details.items():
                    self.task2details[k].extend(v)

        for line in open(
            os.path.join(self.cache_dir, f'{self.model_id}_correction.jsonl'), encoding='utf-8'
        ):
            task2correction = json.loads(line)
            if len(self.task2correction) == 0:
                self.task2correction = task2correction
            else:
                for k, v in task2correction.items():
                    self.task2correction[k].extend(v)

    def save_to_cache(self, data: Dict[str, Any], filename: str) -> None:
        with open(os.path.join(self.cache_dir, filename), 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def update_results(self, task2details: Dict[str, Dict[str, Any]]) -> None:
        file_path = os.path.join(self.output_dir, self.details_filename)

        with open(file_path, 'a', encoding='utf-8') as file:
            for k, v in task2details.items():
                json_record = json.dumps({k: v}, ensure_ascii=False)
                file.write(json_record + '\n')

    def eval_task(self, task_name: str, split='val') -> Dict[str, Dict[str, Any]]:
        dataset = self.load_dataset(task_name)
        self.set_fewshot_dataset(dataset)

        task_details, task_correction = [], []

        def collate_fn(batch):
            preprocessed = [self.preproccess(data) for data in batch]
            keys = preprocessed[0].keys()
            return {key: [data[key] for data in preprocessed] for key in keys}

        sampler = DistributedSampler(dataset[split]) if torch.distributed.is_initialized() else None
        dataloader = DataLoader(
            dataset[split], sampler=sampler, batch_size=self.batch_size, collate_fn=collate_fn
        )

        for batch in tqdm(dataloader, desc=f'Evaluating task {task_name}'):
            details, correction = self.eval_instance(batch)
            task_details.extend(details)
            task_correction.extend(correction)

        return {task_name: task_details}, {task_name: task_correction}

    def eval_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        details, correction = [], []
        preds, infos = self.predict(instance)

        for prompt, answer, pred, info in zip(
            instance['prompts'], instance['answers'], preds, infos
        ):
            is_correct = self.is_correct(pred, answer)

            detail = {
                'prompt': prompt,
                'pred': pred,
                'answer': answer,
                'is_correct': is_correct,
                **info,
            }

            details.append(detail)
            correction.append(is_correct)
        return details, correction

    @torch.no_grad()
    def predict(self, inputs: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.action is None:
            raise ValueError(
                f'Action is not set, please set it in yaml, all actions are: {ACTION_LOGITS}, {ACTION_PPL}, {ACTION_GENERATION}'
            )
        action_func_name = self.action_map.get(self.action)
        if action_func_name is None:
            raise ValueError(f"Action '{self.action}' is not supported")
        action_func = getattr(self, action_func_name)
        return action_func(inputs)

    def choice_logits(self, inputs: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        # TODO: add support for multiple prompts
        logits = self.model(**inputs['inputs'][0]).logits[:, -1].flatten()

        candidate_logits = torch.tensor(
            [logits[self.tokenizer(label).input_ids[-1]] for label in self.candidate_labels]
        ).to(torch.float32)
        probs = torch.nn.functional.softmax(candidate_logits, dim=-1).cpu().numpy()

        pred_index = probs.argmax()
        pred = self.candidate_labels[pred_index]

        info = {
            'probs': [f'{prob: .4f}' for prob in probs],
            'logits': [f'{logit: .4f}' for logit in candidate_logits.cpu().numpy()],
        }

        return [pred], [info]

    def generation(self, inputs: Dict[str, Any]):
        return self._generation(inputs)

    def _generation(self, inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        inputs = inputs['inputs'][0]
        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, **self.generate_config
        )
        outputs = outputs[:, inputs['input_ids'].shape[1] :]
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self.parser_response(response)], [{'response': response}]

    def choice_ppl(self, inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # TODO: add support for multiple prompts
        inputs = inputs['inputs'][0]

        def score(ctx):
            input_ids = ctx['input_ids']
            target_ids = input_ids.clone()
            loss = self.model(input_ids, labels=target_ids).loss.to(torch.float32)
            return loss.detach().cpu().numpy()

        candidate_scores = [score(ctx) for ctx in inputs]
        pred = 'ABCDEFG'[np.argmin(candidate_scores)]
        info = {'candidate_scores': [f'{x: .4f}' for x in candidate_scores]}
        return [pred], [info]

    def parser_response(self, response: List[str]) -> str:
        response = response[0].strip()
        assert isinstance(response, str)

        for x in response.split('\n'):
            x = x.strip()
            if len(x) == 1:
                return x

    def is_correct(self, pred: str, answer: str) -> bool:
        return pred == answer

    def calculate_results(self) -> None:
        result = {'average': -1}
        total_correct = 0
        total_length = 0
        for task, correction in self.task2correction.items():
            result[task] = sum(correction) / len(correction) if len(correction) > 0 else -1
            total_correct += sum(correction)
            total_length += len(correction)
        result['average'] = total_correct / total_length

        with open(os.path.join(self.output_dir, self.results_filename), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = self.processor(prompts, return_tensors='pt').to(self.device)
        answers = self.get_answer(data)

        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }

    @abstractmethod
    def get_task_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def build_example_prompt(self, data, with_answer: bool = True):
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_answer(self, data):
        raise NotImplementedError
