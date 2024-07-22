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

import ast
import dataclasses
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import openai


# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = '$ERROR$'

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ['math', 'reasoning', 'coding', 'arena-hard-200']

# Extract scores from judgments
two_score_pattern = re.compile(r'\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]')
two_score_pattern_backup = re.compile(r'\[(\d+\.?\d*),\s?(\d+\.?\d*)\]')
one_score_pattern = re.compile(r'\[\[(\d+\.?\d*)\]\]')
one_score_pattern_backup = re.compile(r'\[(\d+\.?\d*)\]')


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: Dict[str, Any]
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: Dict[str, Any]
    model: str
    answer: Dict[str, Any]
    judge: Judge
    ref_answer: Optional[Dict[str, Any]] = None
    multi_turn: bool = False


def make_judge(judge_model: str, judge_prompts: Dict[str, Dict[str, Any]]) -> Dict[str, Judge]:
    return {
        'default': Judge(judge_model, judge_prompts['single-v1']),
        'math': Judge(judge_model, judge_prompts['single-math-v1'], ref_based=True),
        'default-mt': Judge(judge_model, judge_prompts['single-v1-multi-turn'], multi_turn=True),
        'math-mt': Judge(
            judge_model, judge_prompts['single-math-v1-multi-turn'], ref_based=True, multi_turn=True
        ),
    }


def make_match(
    questions: List[Dict[str, Any]],
    model: str,
    model_answers: Dict[str, Dict[str, Any]],
    judge: Judge,
    ref_answers: Optional[Dict[str, Dict[str, Any]]] = None,
    multi_turn: bool = False,
) -> List[MatchSingle]:
    matches = []
    for question in questions:
        if multi_turn and len(question['turns']) != 2:
            continue
        question_id = question['question_id']
        model_answer = model_answers[model][question_id]
        ref_answer = ref_answers[judge.model_name][question_id] if ref_answers else None
        matches.append(
            MatchSingle(
                question, model, model_answer, judge, ref_answer=ref_answer, multi_turn=multi_turn
            )
        )
    return matches


def run_judge(
    question: Dict[str, Any],
    answer: Dict[str, Any],
    judge: Judge,
    ref_answer: Optional[Dict[str, Any]] = None,
    multi_turn: bool = False,
) -> Tuple[int, str, str]:
    kwargs = {}
    model = judge.model_name

    if ref_answer:
        kwargs['ref_answer_1'] = ref_answer['choices'][0]['turns'][0]
        if multi_turn:
            kwargs['ref_answer_2'] = ref_answer['choices'][0]['turns'][1]

    if multi_turn:
        user_prompt = judge.prompt_template['prompt_template'].format(
            question_1=question['turns'][0],
            question_2=question['turns'][1],
            answer_1=answer['choices'][0]['turns'][0],
            answer_2=answer['choices'][0]['turns'][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template['prompt_template'].format(
            question=question['turns'][0],
            answer=answer['choices'][0]['turns'][0],
            **kwargs,
        )

    system_prompt = judge.prompt_template['system_prompt']
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    judgment = chat_completion_openai(model, messages, temperature=0, max_tokens=2048)

    rating = -1
    if judge.prompt_template['output_format'] == '[[rating]]':
        match = re.search(one_score_pattern, judgment) or re.search(
            one_score_pattern_backup, judgment
        )
        if match:
            rating = ast.literal_eval(match.groups()[0])
    else:
        raise ValueError(f"invalid output format: {judge.prompt_template['output_format']}")

    return rating, user_prompt, judgment


def play_a_match(match: MatchSingle, output_file: Optional[str] = None) -> Dict[str, Any]:
    score, user_prompt, judgment = run_judge(
        match.question, match.answer, match.judge, match.ref_answer, multi_turn=match.multi_turn
    )

    result = {
        'question_id': match.question['question_id'],
        'model': match.model,
        'judge': (match.judge.model_name, match.judge.prompt_template['name']),
        'user_prompt': user_prompt,
        'judgment': judgment,
        'score': score,
        'turn': 1 if not match.multi_turn else 2,
        'tstamp': time.time(),
    }

    print(
        f"question: {result['question_id']}, turn: {result['turn']}, model: {result['model']}, "
        f"score: {result['score']}, judge: {result['judge']}"
    )

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a') as fout:
            json.dump(result, fout)
            fout.write('\n')

    return result


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def chat_completion_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    api_dict: Optional[Dict[str, str]] = None,
) -> str:
    if api_dict:
        openai.api_base = api_dict.get('api_base', openai.api_base)
        openai.api_key = api_dict.get('api_key', openai.api_key)

    for _ in range(API_MAX_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f'Exception: {e}')
            time.sleep(API_RETRY_SLEEP)

    return API_ERROR_OUTPUT
