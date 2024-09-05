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
    parser.add_argument(
        "--benchmark",
        "-b",
        default=None,
        help="The benchmark you want to test on. Choices: HPSv2, ImageRewardDB, TIFAv1.0",
        choices=[
            "HPSv2", "ImageRewardDB", "TIFAv1.0"
        ],
    )
    parser.add_argument(
        "--model_id1",
        default="",
        help="Unique identifier for the model1, used to track and distinguish model evaluations.",
    )
    parser.add_argument(
        "--model_id2",
        default="",
        help="Unique identifier for the model2, used to track and distinguish model evaluations.",
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
        "--generation_output",
        "-g",
        required=True,
        help="Generation output directory",
    )
    args = parser.parse_args()
    return args

def save_result(model_id, result_dir):
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
                    elif isinstance(score, int) or isinstance(score, float):
                        result = score
                    else:
                        result = 0
                    results.append(result)
    result_dict = {model_id: results}
    output_file_path = os.path.join(os.getcwd(), f'{model_id}_result.json')
    
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, indent=4, ensure_ascii=False)

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
    selected_subfolder_path = os.path.join(folder_path, subfolder)

    run_benchmark(selected_subfolder_path, args)

def seperate_file(args, uuid):
    gen_dir = f"./generation_output/{uuid}"
    os.makedirs(gen_dir, exist_ok=True)
    base_dir = os.getcwd()
    gen_dir1 = os.path.join(base_dir, gen_dir, 'gen1.json')
    gen_dir2 = os.path.join(base_dir, gen_dir, 'gen2.json')

    with open(args.generation_output, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data1 = [{"prompt": item["prompt"], "image": item["image1"]} for item in data]
    data2 = [{"prompt": item["prompt"], "image": item["image2"]} for item in data]
    with open(gen_dir1, "w") as file1:
        json.dump(data1, file1, indent=4)
    with open(gen_dir2, "w") as file2:
        json.dump(data2, file2, indent=4)
        
    return [gen_dir1, gen_dir2]
        
def run_benchmark(file_path, args):
    uuid = get_uuid()
    gen_dir = seperate_file(args, uuid)
    base_dir = os.getcwd()
    
    try:
        eval_logger.log('info', 'Generating responses using Non-accelerating backend')
        for i in range(2):
            os.chdir(base_dir)
            args_list = []
            args_list.append(f"--uuid")
            args_list.append(uuid)
            for key, value in vars(args).items():
                if isinstance(value, bool):
                    if value:
                        args_list.append(f"--{key}")
                elif value is not None:
                    if 'model_id' in key:
                        if key == f'model_id{i+1}':
                            args_list.append(f"--model_id")
                            args_list.append(str(value))
                    elif 'generation_output' in key:
                        args_list.append(f"--generation_output")
                        args_list.append(str(gen_dir[i]))
                    else:
                        args_list.append(f"--{key}")
                        args_list.append(str(value))
        
            command = f"bash eval_only.sh {' '.join(args_list)}"
            os.system(command)
            os.chdir(file_path)

            result_dir = os.path.join(vars(args)['output_dir'], uuid, vars(args)[f'model_id{i+1}'])
            if os.path.exists(result_dir):
                save_result(vars(args)[f'model_id{i+1}'], result_dir)

            print(f"{file_path} executed successfully with arguments {args}.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {file_path}: {e}")

if __name__ == "__main__":

    cli_evaluate()
