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
import sys
import json
import argparse
from typing import Union
import subprocess
from align_anything.evaluation.eval_logger import EvalLogger
from datetime import datetime
import uuid

eval_logger = EvalLogger('Align-Anything-Evaluation')

def get_uuid():
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    unique_id = str(uuid.uuid4())

    return f"{current_time}_{unique_id}"
    
def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--chat_template", default="", help="Chat template id of your model, details can be refered in `align-anything/align_anything/configs/template.py`.")
    parser.add_argument(
        "--benchmark",
        "-b",
        default=None,
        help="The benchmark you want to test on. Choices: ARC, BBH, Belebele, CMMLU, GSM8K, HumanEval, MMLU, MMLUPRO, mt-bench, PAWS-X, RACE, TruthfulQA, MME, MMBench, MMMU, POPE, MMVet, MathVista, MM-SafetyBench, TextVQA, VizWizVQA, SPA-VL, A-OKVQA, llava-bench-in-the-wild, llava-bench-coco, ScienceQA, MMStar, L-Eval, LongBench, AGIEval, HPSv2, ImageRewardDB, MSCOCO, ChronoMagicBench, AudioCaps, MVBench, VideoMME",
        choices=[
            "ARC", "BBH", "Belebele", "CMMLU", "GSM8K", "HumanEval",
            "MMLU", "MMLUPRO", "mt_bench", "PAWS-X", "RACE", "TruthfulQA",
            "MME", "MMBench", "MMMU", "POPE", "MMVet", "MathVista",
            "MM-SafetyBench", "TextVQA", "VizWizVQA", "SPA-VL",
            "A-OKVQA", "llava-bench-in-the-wild", "llava-bench-coco",
            "ScienceQA", "MMStar", "L-Eval", "LongBench", "AGIEval",
            "HPSv2", "MSCOCO", "ImageRewardDB", "ChronoMagicBench", "AudioCaps",
            "MVBench", "VideoMME", "AIRBench"
        ],
    )
    parser.add_argument(
        "--model_id",
        default="",
        help="Unique identifier for the model, used to track and distinguish model evaluations.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="",
        help="The local path or hugggingface link of model",
    )
    parser.add_argument(
        "--n_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--chain_of_thought",
        type=bool,
        default=False,
        help="If True, chain-of-thought will be implemented during generation",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default=1,
        help="Batch size for generation (when using deepspeed backend).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        metavar="= [DIR]",
        help="Path for saving output and log files",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--generation_backend",
        "-g",
        default="vLLM",
        required=True,
        help="vLLM or Deepspeed.",
    )
    args = parser.parse_args()
    return args

def save_result(model_id, output_dir, result_dir):
    results = []
    for filename in os.listdir(result_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(result_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    score = item.get('score')
                    if isinstance(score, bool):
                        result = 1 if score else 0
                    elif isinstance(score, int):
                        result = score
                    else:
                        result = 0
                    results.append(result)
    result_dict = {model_id: results}
    output_file_path = os.path.join(os.getcwd(), output_dir, f'{model_id}_result.json')
    
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, indent=4, ensure_ascii=False)
    
    print(f'Results saved to {output_file_path}')

def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()
    eval_logger.log_dir = args.output_dir

    if len(sys.argv) == 1:
        print("┌─────────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                            │")
        print("│ `align-anything-eval --model_path llava-hf/llava-1.5-7b-hf --benchmark MME`     │")
        print("│ More default configs can be refered in `align-anything/align_anything/configs`  │")
        print("└─────────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    folder_path = './benchmarks/'
    subfolder = args.benchmark
    args.generation_backend = args.generation_backend.lower()
    selected_subfolder_path = os.path.join(folder_path, subfolder)

    run_benchmark(selected_subfolder_path, args)

def run_benchmark(file_path, args):
    uuid = get_uuid()
    try:
        file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        if args.generation_backend == 'vllm':
            if 'vllm_eval.py' not in file_names:
                eval_logger.log('warning', 'vLLM backend is not support for this benchmark.')
                if 'ds_evaluate.py' in file_names:
                    eval_logger.log('info', 'Generating responses using Deepspeed backend.')
                    args.generation_backend = 'deepspeed'
                elif 'eval.py' in file_names:
                    eval_logger.log('info', 'Generating responses using Non-accelerating backend')
                    args.generation_backend = 'none'
            else:
                eval_logger.log('info', 'Generating responses using vLLM backend.')
        elif args.generation_backend == 'deepspeed':
            if 'ds_evaluate.py' not in file_names:
                eval_logger.log('warning', 'Deepspeed backend is not support for this benchmark.')
                if 'vllm_eval.py' in file_names:
                    eval_logger.log('info', 'Generating responses using vLLM backend.')
                    args.generation_backend = 'vllm'
                elif 'eval.py' in file_names:
                    eval_logger.log('info', 'Generating responses using Non-accelerating backend')
                    args.generation_backend = 'none'
            else:
                eval_logger.log('info', 'Generating responses using Deepspeed backend')
        else:
            if 'eval.py' not in file_names:
                eval_logger.log('warning', 'Non-accelerating backend is not support for this benchmark.')
                if 'vllm_eval.py' in file_names:
                    eval_logger.log('info', 'Generating responses using vLLM backend.')
                    args.generation_backend = 'vllm'
                elif 'ds_evaluate.py' in file_names:
                    eval_logger.log('info', 'Generating responses using Deepspeed backend.')
                    args.generation_backend = 'deepspeed'
            else:
                eval_logger.log('info', 'Generating responses using Non-accelerating backend')
        
        args_list = []
        args_list.append(f"--uuid")
        args_list.append(uuid)
        for key, value in vars(args).items():
            if isinstance(value, bool):
                if value:
                    args_list.append(f"--{key}")
            elif value is not None:
                args_list.append(f"--{key}")
                args_list.append(str(value))
        
        command = f"bash eval.sh {' '.join(args_list)}"
        os.system(command)
        os.chdir(file_path)

        result_dir = os.path.join(vars(args)['output_dir'], uuid)
        if os.path.exists(result_dir):
            save_result(vars(args)['model_id'], args.output_dir, result_dir)

        print(f"{file_path} executed successfully with arguments {args}.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {file_path}: {e}")

if __name__ == "__main__":

    cli_evaluate()
