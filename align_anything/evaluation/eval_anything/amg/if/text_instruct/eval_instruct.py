import json
import re

import gpt
import numpy as np
from eval_prompt import TEXT_EVALUATE_SYSTEM_PROMPT, TEXT_EVALUATE_USER_PROMPT


def post_process(response: str):
    score_pattern = r'<Score>: \[\[(.*)\]\]'
    explanation_pattern = r'<Explanation>: \[\[(.*)\]\]'
    score_match = re.search(score_pattern, response)
    explanation_match = re.search(explanation_pattern, response)
    if score_match and explanation_match:
        score = score_match.group(1)
        explanation = explanation_match.group(1)
        return {'score': int(score), 'explanation': explanation, 'response': response}
    else:
        return {'score': None, 'explanation': None, 'response': response}


def hash_checker(result: dict):
    return result['score'] is not None and result['explanation'] is not None


def gpt_eval(eval_data, cache_dir='./cache'):
    eval_results = []

    empty_index = list(range(len(eval_data)))
    max_repeat = 3
    while len(empty_index) > 0 and max_repeat > 0:
        print(f'Remaining repeat times: {max_repeat}')
        print(f'Remaining captions to be annotated: {len(empty_index)}')
        system_contents = [TEXT_EVALUATE_SYSTEM_PROMPT] * len(empty_index)
        user_contents = []
        for i in empty_index:
            user_contents.append(
                TEXT_EVALUATE_USER_PROMPT.format(
                    prompt=eval_data[i]['prompt'], response=eval_data[i]['response']
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
                    'prompt_id': eval_data[index]['prompt_id'],
                    'score': result['score'],
                    'explanation': result['explanation'],
                    'response': result['response'],
                }
            )
        empty_index = [i for i in empty_index if eval_results[i].get('score') is None]
        max_repeat -= 1

    return eval_results


with open('.cache/text_instruct/generated_results.json') as f:
    eval_data = json.load(f)

eval_results = gpt_eval(eval_data)

score = np.mean([result['score'] for result in eval_results])

with open('.cache/text_instruct/eval_results.json', 'w') as f:
    json.dump({'score': score, 'eval_results': eval_results}, f)
