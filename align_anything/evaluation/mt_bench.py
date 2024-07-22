# Copyright 2024 PKU-Alignment Team and FastChat Team. All Rights Reserved.
#
# This code is inspired by the FastChat.
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_judgment.py
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

import argparse
import glob
import json
import os
import random
import time
from typing import Any, Dict, List

import deepspeed
import numpy as np
import shortuuid
import torch
from tqdm import tqdm
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.evaluation.dis_utils import *
from align_anything.evaluation.evaluator_registry import register_evaluator
from align_anything.evaluation.utils import load_jsonl, make_judge, make_match, play_a_match
from align_anything.models.pretrained_model import load_pretrained_models


@register_evaluator('mt_bench')
class MTBench:
    temperature_config = {
        'writing': 0.7,
        'roleplay': 0.7,
        'extraction': 0.0,
        'math': 0.0,
        'coding': 0.0,
        'reasoning': 0.0,
        'stem': 0.1,
        'humanities': 0.1,
        'arena-hard-200': 0.0,
    }

    def __init__(self, cfgs, ds_cfgs) -> None:
        self.ds_cfgs = ds_cfgs
        self.eval_cfgs, self.data_cfgs, self.model_cfgs = (
            cfgs.eval_cfgs,
            cfgs.data_cfgs,
            cfgs.model_cfgs,
        )

        self.model_id = self.model_cfgs.model_id
        self.temperature = self.eval_cfgs.temperature if self.eval_cfgs.temperature else 0.7
        self.seed = self.eval_cfgs.seed if self.eval_cfgs.seed else 3407
        self.judge_model = self.eval_cfgs.judge_model

        self.questions_file = f'{self.data_cfgs.task_dir}/{self.data_cfgs.task}/question.jsonl'
        self.judge_prompts_file = (
            f'{self.data_cfgs.task_dir}/{self.data_cfgs.task}/judge_prompts.jsonl'
        )
        self.answers_dir = f'{self.eval_cfgs.output_dir}/{self.model_id}/model_answers'
        self.answers_file = f'{self.answers_dir}/{self.model_id}.jsonl'
        self.references_dir = f'{self.data_cfgs.task_dir}/{self.data_cfgs.task}/reference_answer'
        self.output_dir = f'{self.eval_cfgs.output_dir}/model_judgment'

        self.NEED_REF_CATS = ['math', 'reasoning', 'coding', 'arena-hard-200']

        self.init_model()
        self.load_dataset()

    def init_model(self) -> None:
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.model_cfgs.model_name_or_path,
            model_max_length=self.model_cfgs.model_max_length,
            padding_side='right',
            auto_device_mapping=True,
            trust_remote_code=self.model_cfgs.trust_remote_code,
        )
        self.model.eval()

    def load_dataset(self):
        self.questions = load_jsonl(self.questions_file)
        judge_prompts = load_jsonl(self.judge_prompts_file)
        self.judge_prompts = {}
        for judge_prompt in judge_prompts:
            self.judge_prompts[judge_prompt['name']] = judge_prompt

        os.makedirs(os.path.dirname(self.answers_file), exist_ok=True)
        random.shuffle(self.questions)

    def load_model_answers(self, answer_dir: str):
        filenames = glob.glob(os.path.join(answer_dir, '*.jsonl'))
        filenames.sort()
        model_answers = {}
        for filename in filenames:
            model_name = os.path.basename(filename)[:-6]
            answer = {}
            with open(filename) as fin:
                for line in fin:
                    line = json.loads(line)
                    answer[line['question_id']] = line
            model_answers[model_name] = answer
        return model_answers

    def format_messages(
        self, question: str, conversations: List[Dict[str, str]], **kwargs
    ) -> List[Dict[str, str]]:
        messages = []
        for k in range(len(conversations)):
            messages.append({'role': 'user', 'content': conversations[k]['input']})
            messages.append({'role': 'assistant', 'content': conversations[k]['output']})
        messages.append({'role': 'user', 'content': question})
        return messages

    def message_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ''
        last_message = None
        if len(messages) % 2 == 1:
            last_message = messages[-1]
            messages = messages[:-1]
        for i in range(0, len(messages), 2):
            user_message = messages[i]
            assistant_message = messages[i + 1]
            prompt += f"USER: {user_message.get('content', '')} ASSISTANT:{assistant_message.get('content', '')}\n"
        if last_message is not None:
            prompt += f"USER: {last_message.get('content', '')} ASSISTANT:"
        return prompt

    def generate_answers(self):

        for question_obj in tqdm(self.questions):
            if question_obj['category'] in self.temperature_config:
                self.temperature = self.temperature_config[question_obj['category']]
            choices = []
            conversations = []
            for i in range(self.eval_cfgs.num_choices):
                torch.manual_seed(i + self.seed)
                turns = []
                for j in range(len(question_obj['turns'])):
                    question = question_obj['turns'][j]
                    messages = self.format_messages(
                        question,
                        conversations,
                    )

                    prompt = self.message_to_prompt(messages)

                    output = self.generation(prompt)
                    turns.append(output)
                    conversations.append(
                        {
                            'input': question,
                            'output': output,
                        }
                    )

                choices.append({'index': i, 'turns': turns})

            with open(os.path.expanduser(self.answers_file), 'a') as fout:
                answer_json = {
                    'question_id': question_obj['question_id'],
                    'answer_id': shortuuid.uuid(),
                    'model_id': self.model_id,
                    'choices': choices,
                    'tstamp': time.time(),
                }
                fout.write(json.dumps(answer_json) + '\n')

    @torch.no_grad()
    def generation(self, prompt: str) -> str:
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            max_new_tokens=self.model_cfgs.max_new_token,
            pad_token_id=self.tokenizer.eos_token_id,
        )[:, inputs['input_ids'].shape[1] :]

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output

    def generate_judgements(self):
        # Load answers
        model_answers = self.load_model_answers(self.answers_dir)
        refer_answers = self.load_model_answers(self.references_dir)

        judges = make_judge(self.judge_model, self.judge_prompts)

        question_math = [q for q in self.questions if q['category'] in self.NEED_REF_CATS]
        question_default = [q for q in self.questions if q['category'] not in self.NEED_REF_CATS]

        # Make matches
        matches = []
        matches += make_match(
            question_default,
            self.model_id,
            model_answers,
            judges['default'],
        )
        matches += make_match(
            question_math,
            self.model_id,
            model_answers,
            judges['math'],
            refer_answers,
        )
        matches += make_match(
            question_default,
            self.model_id,
            model_answers,
            judges['default-mt'],
            multi_turn=True,
        )
        matches += make_match(
            question_math,
            self.model_id,
            model_answers,
            judges['math-mt'],
            refer_answers,
            multi_turn=True,
        )

        results = []
        output_file_path = os.path.join(self.output_dir, f'{self.judge_model}.jsonl')
        for match in tqdm(matches):
            result = play_a_match(match, output_file=output_file_path)
            results.append(result)

        score = {}
        score['avg'] = []
        for result in results:
            question_type = result['judge'][1]
            if question_type not in score:
                score[question_type] = []
            score[question_type].append(result['score'])
            score['avg'].append(result['score'])

        score_results = {
            q_type: {'mean': np.mean(scores), 'std': np.std(scores)}
            for q_type, scores in score.items()
        }

        with open(os.path.join(self.output_dir, 'score.json'), 'w') as fout:
            json.dump(score_results, fout, indent=4)

    def eval(self):
        self.generate_answers()
        self.generate_judgements()
