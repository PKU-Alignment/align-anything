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

import re

import gpt
from amu_eval_prompt import (
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_IF,
    SYSTEM_PROMPT_SAFETY,
    USER_PROMPT,
)


def post_process(response: str):
    pattern = r'\[\[(\d+)\]\]'
    match = re.search(pattern, response)
    if match:
        score = match.group(1)
        try:
            return {'score': int(score), 'response': response}
        except ValueError:
            return {'score': None, 'response': response}
    else:
        return {'score': None, 'response': response}


def hash_checker(result: dict):
    return result['score'] is not None


def gpt_eval(eval_data, cache_dir='./cache'):
    eval_results = []

    empty_index = list(range(len(eval_data)))
    max_repeat = 3
    while len(empty_index) > 0 and max_repeat > 0:
        print(f'Remaining repeat times: {max_repeat}')
        print(f'Remaining captions to be annotated: {len(empty_index)}')
        system_contents = []
        for i in empty_index:
            if eval_data[i]['evaluation_dimension'] == 'instruction-following':
                system_contents.append(SYSTEM_PROMPT_IF)
            elif eval_data[i]['evaluation_dimension'] == 'safety':
                system_contents.append(SYSTEM_PROMPT_SAFETY)
            else:
                system_contents.append(SYSTEM_PROMPT_DEFAULT)
        user_contents = []
        user_input = USER_PROMPT
        for i in empty_index:
            user_contents.append(
                user_input.format(
                    reference_0=eval_data[i]['annotations'][0]['answer'],
                    image_keywords_0=' '.join(eval_data[i]['annotations'][0]['keywords']['visual']),
                    audio_keywords_0=' '.join(
                        eval_data[i]['annotations'][0]['keywords']['auditory']
                    ),
                    reference_1=eval_data[i]['annotations'][1]['answer'],
                    image_keywords_1=' '.join(eval_data[i]['annotations'][1]['keywords']['visual']),
                    audio_keywords_1=' '.join(
                        eval_data[i]['annotations'][1]['keywords']['auditory']
                    ),
                    reference_2=eval_data[i]['annotations'][2]['answer'],
                    image_keywords_2=' '.join(eval_data[i]['annotations'][2]['keywords']['visual']),
                    audio_keywords_2=' '.join(
                        eval_data[i]['annotations'][2]['keywords']['auditory']
                    ),
                    reference_3=eval_data[i]['annotations'][3]['answer'],
                    image_keywords_3=' '.join(eval_data[i]['annotations'][3]['keywords']['visual']),
                    audio_keywords_3=' '.join(
                        eval_data[i]['annotations'][3]['keywords']['auditory']
                    ),
                    reference_4=eval_data[i]['annotations'][4]['answer'],
                    image_keywords_4=' '.join(eval_data[i]['annotations'][4]['keywords']['visual']),
                    audio_keywords_4=' '.join(
                        eval_data[i]['annotations'][4]['keywords']['auditory']
                    ),
                    reference_5=eval_data[i]['annotations'][5]['answer'],
                    image_keywords_5=' '.join(eval_data[i]['annotations'][5]['keywords']['visual']),
                    audio_keywords_5=' '.join(
                        eval_data[i]['annotations'][5]['keywords']['auditory']
                    ),
                    reference_6=eval_data[i]['annotations'][6]['answer'],
                    image_keywords_6=' '.join(eval_data[i]['annotations'][6]['keywords']['visual']),
                    audio_keywords_6=' '.join(
                        eval_data[i]['annotations'][6]['keywords']['auditory']
                    ),
                    reference_7=eval_data[i]['annotations'][7]['answer'],
                    image_keywords_7=' '.join(eval_data[i]['annotations'][7]['keywords']['visual']),
                    audio_keywords_7=' '.join(
                        eval_data[i]['annotations'][7]['keywords']['auditory']
                    ),
                    reference_8=eval_data[i]['annotations'][8]['answer'],
                    image_keywords_8=' '.join(eval_data[i]['annotations'][8]['keywords']['visual']),
                    audio_keywords_8=' '.join(
                        eval_data[i]['annotations'][8]['keywords']['auditory']
                    ),
                    reference_9=eval_data[i]['annotations'][9]['answer'],
                    image_keywords_9=' '.join(eval_data[i]['annotations'][9]['keywords']['visual']),
                    audio_keywords_9=' '.join(
                        eval_data[i]['annotations'][9]['keywords']['auditory']
                    ),
                    response=eval_data[i]['response'],
                )
            )
        assert len(system_contents) == len(user_contents)

        results = gpt.api(
            system_contents,
            user_contents,
            num_workers=50,
            post_process=post_process,
            hash_checker=hash_checker,
            cache_dir=cache_dir,
        )
        for index, result in zip(empty_index, results):
            eval_results.append(
                {
                    'question_id': eval_data[index]['question_id'],
                    'evaluation_dimension': eval_data[index]['evaluation_dimension'],
                    'score': result['score'],
                    'response': result['response'],
                }
            )
        empty_index = [i for i in empty_index if eval_results[i].get('score') is None]
        max_repeat -= 1

    return eval_results
