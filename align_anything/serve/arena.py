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

import argparse

import torch
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from align_anything.serve.cli import main as cli_main
from align_anything.utils.tools import str2bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Talking to two models in cli mode.')
    parser.add_argument(
        '--red_corner_model_name_or_path',
        type=str,
        required=True,
        help='Path to the model checkpoint or its name.',
    )
    parser.add_argument(
        '--blue_corner_model_name_or_path',
        type=str,
        required=True,
        help='Path to the model checkpoint or its name.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='The value used to module the next token probabilities.',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length of generation.',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help=(
            'If set to float < 1, only the smallest set of most probable tokens with '
            'probabilities that add up to`top_p` or higher are kept for generation.'
        ),
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Whether to stream the output.',
        default=False,
    )
    parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    parser.add_argument(
        '--tf32',
        type=str2bool,
        default=None,
        help='Whether to use tf32 mix precision.',
    )
    parser.add_argument(
        '--template',
        type=str,
        default='Dialogue',
        help='Model template',
    )
    parser.add_argument(
        '--vlm',
        type=str,
        default=False,
        help='Whether to use VLM model',
    )

    args = parser.parse_args()
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error(
            'bf16 precision is not supported on this GPU. '
            'Please disable `--bf16` flag or use another precision flag (e.g., `--fp16`).',
        )
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    return args


def main() -> None:
    args = parse_arguments()
    args.model_name_or_path = [
        args.red_corner_model_name_or_path,
        args.blue_corner_model_name_or_path,
    ]
    return cli_main(args)


if __name__ == '__main__':
    main()
