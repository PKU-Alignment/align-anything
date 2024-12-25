import json
import os
from copy import deepcopy

import numpy as np
from gpt_eval_amu import gpt_eval
from tqdm import tqdm

from datasets import load_dataset


def inference(question, visual_modality_path, auditory_modality_path):
    """
    The inference function should return a string of the response, where you should implement the inference logic.
    """


def main():
    dataset = load_dataset(
        'PKU-Alignment/EvalAnything-AMU', name='image', split='test', trust_remote_code=True
    )

    catagories = ['perception', 'reasoning', 'instruction-following', 'safety']

    save_dir = os.path.join('amu')
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for item in tqdm(dataset):
        result_item = inference(
            item['question'],
            item['visual_modality_path'],
            item['auditory_modality_path'],
        )

        result = deepcopy(item)
        result['response'] = result_item
        results.append(result)

    eval_results = gpt_eval(results)

    amu_score = {}
    for category in catagories:
        amu_score[category] = [
            result['score'] for result in eval_results if result['evaluation_dimension'] == category
        ]

    for category in catagories:
        amu_score[category] = np.mean(amu_score[category])
    amu_score['all'] = np.mean([score for score in amu_score.values()])

    with open(os.path.join(save_dir, 'amu_score.json'), 'w') as f:
        json.dump(amu_score, f, indent=4)


if __name__ == '__main__':
    main()
