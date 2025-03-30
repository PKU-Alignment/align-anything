# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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


from typing import List, Optional


def example_math_reward_function(
    prompts: List[str], responses: List[str], golden_responses: Optional[List[str]] = None
) -> List[float]:
    """
    Math reward function, evaluate the accuracy of the answer

    Args:
        prompts: List of math problems
        responses: List of model answers

    Returns:
        List of reward scores
    """
    rewards = []

    for prompt, response in zip(prompts, responses):
        # Simple example: check if the answer contains numbers and math symbols
        has_numbers = any(char.isdigit() for char in response)
        has_math_symbols = any(symbol in response for symbol in ['+', '-', '*', '/', '=', '<', '>'])

        if has_numbers and has_math_symbols:
            # There should be more complex logic to verify the correctness of the math answer
            # You can refer to the math_verifier.py in lmm-r1 for implementation
            reward = 0.8
        else:
            reward = 0.2

        rewards.append(reward)

    return rewards


def example_coding_reward_function(
    prompts: List[str], responses: List[str], golden_responses: Optional[List[str]] = None
) -> List[float]:
    """
    Coding reward function, evaluate the quality of the code

    Args:
        prompts: List of coding problems
        responses: List of model answers

    Returns:
        List of reward scores
    """
    rewards = []

    for prompt, response in zip(prompts, responses):
        # Simple example: check if the answer contains code features
        code_features = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ', 'while ']
        has_code = any(feature in response for feature in code_features)

        # Check code format
        has_indentation = '    ' in response or '\t' in response

        if has_code and has_indentation:
            reward = 0.9
        elif has_code:
            reward = 0.6
        else:
            reward = 0.3

        rewards.append(reward)

    return rewards


def example_safety_reward_function(
    prompts: List[str], responses: List[str], golden_responses: Optional[List[str]] = None
) -> List[float]:
    """
    Safety assessment reward function, check if the answer contains harmful content

    Args:
        prompts: List of prompts
        responses: List of model answers

    Returns:
        List of reward scores (safe answers get higher scores)
    """
    rewards = []

    # Example harmful words list (should be more comprehensive in actual applications)
    harmful_words = ['violence', 'attack', 'hacker', 'illegal', 'harm', 'danger', 'weapon']

    for prompt, response in zip(prompts, responses):
        # Check if the answer contains harmful words
        contains_harmful = any(word in response for word in harmful_words)

        # Answers containing harmful content get lower scores
        reward = 0.2 if contains_harmful else 1.0
        rewards.append(reward)

    return rewards
