import importlib
import os
import yaml
import sys
import json

import traceback
import argparse
import numpy as np
import datetime

import warnings
import traceback

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from pathlib import Path
from typing import Union
import hashlib
import subprocess
from align_anything.evaluation.eval_logger import EvalLogger

eval_logger = EvalLogger('Align-Anything-Evaluation')


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--chat_template", default="", help="Chat template id of your model, details can be refered in `align-anything/align_anything/configs/template.py`.")
    parser.add_argument(
        "--benchmark",
        "-b",
        default=None,
        help="The benchmark you want to test on",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task list of the benchmark you want to test on",
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
    parser.add_argument("--batch_size", type=str, default=1)
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


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()
    eval_logger.log_dir = args.output_dir

    if len(sys.argv) == 1:
        print("┌─────────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                            │")
        print("│ `align-anything-eval --model_path liuhaotian/llava-v1.6-7b --benchmark MME`     │")
        print("│ More default configs can be refered in `align-anything/align_anything/configs`  │")
        print("└─────────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    folder_path = './benchmarks/'
    subfolder = args.benchmark
    args.generation_backend = args.generation_backend.lower()
    selected_subfolder_path = os.path.join(folder_path, subfolder)

    run_benchmark(selected_subfolder_path, args)


def run_benchmark(file_path, args):
    try:
        file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        if args.generation_backend == 'vllm':
            if 'vllm_eval.py' not in file_names:
                eval_logger.log('warning', 'vLLM backend is not support for this benchmark.')
                if 'ds_eval.py' in file_names:
                    eval_logger.log('info', 'Generating responses using Deepspeed backend.')
                    args.generation_backend = 'deepspeed'
            else:
                eval_logger.log('info', 'Generating responses using vLLM backend.')
        else:
            if 'ds_infer.py' not in file_names:
                eval_logger.log('warning', 'Deepspeed backend is not support for this benchmark.')
                if 'vllm_eval.py' in file_names:
                    eval_logger.log('info', 'Generating responses using vLLM backend.')
                    args.generation_backend = 'vllm'
            else:
                eval_logger.log('info', 'Generating responses using Deepspeed backend')
        
        sh_file_path = os.path.join(file_path, "eval.sh")
        args_list = []
        for key, value in vars(args).items():
            if isinstance(value, bool):
                if value:
                    args_list.append(f"--{key}")
            elif value is not None:
                args_list.append(f"--{key}")
                args_list.append(str(value))

        command = f"sh {sh_file_path} {' '.join(args_list)}"
        os.system(command)
        print(f"{file_path} executed successfully with arguments {args}.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {file_path}: {e}")

if __name__ == "__main__":
    cli_evaluate()
