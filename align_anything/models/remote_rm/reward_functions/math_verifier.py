# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# This code is inspired by the lmm-r1 library.
# https://github.com/TideDra/lmm-r1/tree/main/openrlhf/models/remote_rm/math_verifier.py
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

import random
import re
from typing import List, Optional

from flask import jsonify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


format_pattern = r'^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z'


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count('<think>')
    answer_count = content.count('<answer>')
    return (
        bool(re.match(format_pattern, content, re.DOTALL))
        and think_count == 1
        and answer_count == 1
    )


def verify_acc(response, golden_response):
    """
    Verify if the response is correct
    We require the answer to be provided in correct latex (no malformed operators)
    """
    gold_parsed = parse(
        golden_response,
        extraction_mode='first_match',
        extraction_config=[LatexExtractionConfig()],
    )
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        new_response = answer_match.group(1).strip()
    else:
        # If the answer is not provided or not in the correct format, we reward 0
        return 0.0
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)

        answer_parsed = parse(
            new_response,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,  # NOTE change
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensure boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,  # NOTE change
                )
            ],
            extraction_mode='first_match',
        )
        # If the content is the same as the real value, then reward 1, otherwise reward 0
        try:
            reward = float(verify(answer_parsed, gold_parsed))
            print('****' * 60, answer_parsed, gold_parsed, reward, '****' * 60)
        except Exception as e:
            # reward 1.0 to skip this example
            reward = 1.0
            print('Verification failed: ', e)
    else:
        # If the real answer is not parsable, we reward 1 to skip this example
        reward = 1.0
        print('Parsing real answer failed: ', golden_response)

    return reward


def math_verifier_reward_function(
    prompts: List[str], responses: List[str], golden_responses: Optional[List[str]] = None
) -> List[float]:
    """
    Math verifier reward function, evaluate the accuracy of the answer

    Args:
        prompts: List of math problems
        responses: List of model answers
        golden_responses: Optional list of golden responses
    Returns:
        List of reward scores for each (prompt, response) pair
    """
    rewards = []
    format_rewards = []
    acc_rewards = []
    for prompt, response, golden_response in zip(prompts, responses, golden_responses):
        if prompt is None:
            return jsonify({'error': f'problem not found from {prompt}'}), 400
        if golden_response is None:
            return jsonify({'error': f'golden response not found from {prompt}'}), 400
        # TODO: processing the error code 400

        format_reward = float(verify_format(response))
        acc_reward = float(verify_acc(response, golden_response))
        rewards.append(0.5 * format_reward + acc_reward)
        format_rewards.append(format_reward)
        acc_rewards.append(acc_reward)

        do_print = random.randint(1, 10) == 1
        if do_print:
            info = f'Query: {prompt}\n\nAnswer: {golden_response}\n\nResponse: {response}\n\nFormat Reward: {format_reward}\n\nAcc Reward: {acc_reward}\n\n'
            info = re.sub(r'<\|.*?\|>', '', info)
            print(info)
    return rewards
