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

# ref: https://github.com/agiresearch/MoralBench/blob/main/main.py
import collections
import json
import os

from openai import OpenAI


def read_prompt(file_path):
    prompt = ''
    with open(file_path) as f:
        prompt = f.readlines()
    prompt = '\n'.join(prompt)
    return prompt


def LLM_response(target_folder, question):
    api_key = os.getenv('API_KEY')
    api_base = os.getenv('API_BASE')
    client = OpenAI(api_key=api_key, base_url=api_base)
    systemPrompt = read_prompt('./benchmarks/text_to_text/MoralBench/template/moral_system.txt')
    # 6_concepts QFT_30 6_concepts_compare QFT_30_compare
    userPrompt = read_prompt(
        f'./benchmarks/text_to_text/MoralBench/questions/{target_folder}/{question}.txt'
    )  # 6_concepts QFT_30
    print('The current the question is:\n', userPrompt)
    messages = [
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': userPrompt},
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model='gpt-4',  # gpt-3.5-turbo,gpt-4
    )
    return chat_completion.choices[0].message.content


def print_fancy_header():
    # Define the header message
    header_message = 'Welcome to the Large Language Model Moral Test'

    top_bottom_border = '=' * 80
    side_borders = '|' + ' ' * (len(top_bottom_border) - 2) + '|'
    message_length = len(header_message)
    left_padding = (len(top_bottom_border) - message_length) // 2 - 1
    right_padding = len(top_bottom_border) - left_padding - message_length - 2
    centered_message = f"|{' ' * left_padding}{header_message}{' ' * right_padding}|"

    print(top_bottom_border)
    print(side_borders)
    print(centered_message)
    print(side_borders)
    print(top_bottom_border)


def get_all_files(path):
    files = []
    entries = os.listdir(path)

    for entry in entries:
        if entry.endswith('txt'):
            files.append(entry)

    return files


def eval_MoralBench():
    total_score = 0
    cur_score = 0
    concepts_score = collections.defaultdict(float)

    print_fancy_header()
    # MFQ_30, 6_concepts,MFQ_30_compare, 6_concepts_compare
    target_folder = 'MFQ_30_compare'
    # get the question answers
    ans = {}
    with open(f'./benchmarks/text_to_text/MoralBench/answers/{target_folder}.json') as json_file:
        ans = json.load(json_file)

    questions = get_all_files(f'./benchmarks/text_to_text/MoralBench/questions/{target_folder}/')
    # questions = ["care_1.txt"]
    for question in questions:
        response = LLM_response(target_folder, question[:-4])
        print(f'The answer of the Large Language Model is:\n {response} \n')

        score = ans[question[:-4]][response[0]]
        print('The current score is: ', score)
        cur_score += score
        total_score += 4
        concepts_score[question[:-6]] += score
        print(f'The total score is: {cur_score:.1f}/{total_score:.1f}')
    concepts = ['harm', 'fairness', 'ingroup', 'authority', 'purity', 'liberty']
    for key in concepts:
        print(f'The concepts {key} score is: {concepts_score[key]:.1f}')
