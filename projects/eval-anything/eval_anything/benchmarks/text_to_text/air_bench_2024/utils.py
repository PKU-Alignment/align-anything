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

# ref: https://github.com/stanford-crfm/air-bench-2024/blob/main/evaluation/utils.py
# ref: https://github.com/stanford-crfm/air-bench-2024/blob/main/evaluation/pipeline1_step2_QA_eval.ipynb
import os
import random
import time

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


def extract_content(tag, text):
    start_idx = text.find(tag)
    # if start_idx == -1:
    #     return None
    start_of_content = start_idx + len(tag)
    if tag == '##the_score: ':
        end_idx = text.find('\n', start_of_content)
    else:
        end_idx = text.find('.\n##', start_of_content)
    if end_idx == -1:
        content = text[start_of_content:].strip()
    else:
        content = text[start_of_content:end_idx].strip()
    return content


def sample_row(raw_data, sample_num):
    sample_row_list = []
    for l2 in range(1, 17):

        l2_specific_rows = []
        for i, data in enumerate(raw_data):
            cate_idx, l2_name, l3_name, l4_name, prompt = data.values()
            l2_index = int(cate_idx.split('.')[0])
            if l2_index == l2:
                l2_specific_rows.append((cate_idx, l2_name, l3_name, l4_name, prompt))

        l2_specific_sampled_row = random.sample(l2_specific_rows, sample_num)
        sample_row_list.extend(l2_specific_sampled_row)

    return sample_row_list


def gpt_eval(questions, responses, cate_idxs):
    api_key = os.getenv('API_KEY')
    api_base = os.getenv('API_BASE')
    client = OpenAI(api_key=api_key, base_url=api_base)
    judge_prompt_list = load_dataset('stanford-crfm/air-bench-2024', 'judge_prompts', split='test')

    def get_judge_prompt(idx):
        for item in judge_prompt_list:
            if item['cate-idx'] == idx:
                return item['judge_prompt']
        raise ValueError(f'No judge prompt found for index {idx}')

    def judger(cate_index, QApairs):
        reasons = []
        scores = []
        for QApair in QApairs:
            while True:
                try:
                    judge_prompt = get_judge_prompt(cate_index)

                    response = client.chat.completions.create(
                        model='gpt-4o',
                        messages=[
                            {
                                'role': 'user',
                                'content': judge_prompt.replace('{{QUESTION}}', QApair[0]).replace(
                                    '{{ANSWER}}', QApair[1]
                                ),
                            }
                        ],
                        temperature=0,
                        max_tokens=512,
                    )

                    reason = extract_content(
                        '##short_reasoning: ', response.choices[0].message.content
                    )
                    score = float(
                        extract_content('##the_score: ', response.choices[0].message.content)
                    )
                    reasons.append(reason)
                    scores.append(score)
                    break

                except Exception as err:
                    print('Exception occurs when calling Chatgpt for judge:', err)
                    print('Will sleep for ten seconds before retry...')
                    time.sleep(10)

        return (reasons, scores)

    eval_reasons = []
    eval_scores = []
    for question, response, cate_idx in tqdm(
        zip(questions, responses, cate_idxs), total=len(questions), desc='Running Evaluation'
    ):
        reasons, scores = judger(cate_idx, (question, response))
        eval_reasons.append(reasons[0])
        eval_scores.append(scores[0])

    return eval_reasons, eval_scores
