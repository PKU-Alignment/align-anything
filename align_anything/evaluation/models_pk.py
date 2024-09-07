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

import os
import json
import csv
import argparse
from typing import Union, Dict, List

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--benchmark",
        "-b",
        default=None,
        help="Benchmarks that support pk: ARC, BBH, Belebele, CMMLU, GSM8K, HumanEval, MMLU, MMLUPRO, mt-bench, PAWS-X, RACE, TruthfulQA, MME, MMBench, MMMU, POPE, MMVet, MathVista, MM-SafetyBench, TextVQA, VizWizVQA, SPA-VL, A-OKVQA, llava-bench-in-the-wild, llava-bench-coco, ScienceQA, MMStar, HPSv2, ImageRewardDB",
        choices=[
            "ARC", "BBH", "Belebele", "CMMLU", "GSM8K", "HumanEval",
            "MMLU", "MMLUPRO", "mt_bench", "PAWS-X", "RACE", "TruthfulQA",
            "MME", "MMBench", "MMMU", "POPE", "MMVet", "MathVista",
            "MM-SafetyBench", "TextVQA", "VizWizVQA", "SPA-VL",
            "A-OKVQA", "llava-bench-in-the-wild", "llava-bench-coco",
            "ScienceQA", "MMStar", "HPSv2", "ImageRewardDB"
        ],
    )
    parser.add_argument(
        "--model_1",
        required=True,
        help="ID of the first model to compare"
    )
    parser.add_argument(
        "--model_2",
        required=True,
        help="ID of the second model to compare"
    )
    args = parser.parse_args()
    return args

def load_results(file_path: str) -> Dict[str, List[int]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def models_pk(model_1: str, model_2: str) -> None:
    model_1_data = load_results(f'{model_1}_result.json')
    model_2_data = load_results(f'{model_2}_result.json')
    
    model_1_results = model_1_data[model_1]
    model_2_results = model_2_data[model_2]
    
    model_1_win, model_1_tie, model_1_lose = 0, 0, 0
    model_2_win, model_2_tie, model_2_lose = 0, 0, 0
    
    for res_1, res_2 in zip(model_1_results, model_2_results):
        if res_1 == res_2:
            model_1_tie += 1
            model_2_tie += 1
        elif res_1 > res_2:
            model_1_win += 1
            model_2_lose += 1
        else:
            model_1_lose += 1
            model_2_win += 1
            
    total_matches_model_1 = model_1_win + model_1_tie + model_1_lose
    model_1_win_rate = model_1_win / total_matches_model_1 if total_matches_model_1 > 0 else 0

    results = {
        model_1: [model_1_win, model_1_tie, model_1_lose],
        model_2: [model_2_win, model_2_tie, model_2_lose],
        model_1 + "_win_rate": [model_1_win_rate],
    }
    
    output_file = f'{model_1}_vs_{model_2}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model', 'win', 'tie', 'lose'])
        for model, counts in results.items():
            writer.writerow([model] + counts)

def main(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()
    folder_path = f'./benchmarks/{args.benchmark}'
    os.chdir(folder_path)
    models_pk(args.model_1, args.model_2)

if __name__ == "__main__":
    main()
