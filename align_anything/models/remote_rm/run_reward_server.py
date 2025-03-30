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


"""
Start rule-based reward server

Usage:
    python run_reward_server.py --reward-type math --port 6000
"""

import argparse

from align_anything.models.remote_rm.reward_functions import *
from align_anything.models.remote_rm.reward_server import start_server


reward_functions = {
    'example_math': example_math_reward_function,
    'example_coding': example_coding_reward_function,
    'example_safety': example_safety_reward_function,
    'math_verifier': math_verifier_reward_function,
}


def main():

    parser = argparse.ArgumentParser(description='Start rule-based reward server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host address')
    parser.add_argument('--port', type=int, default=6000, help='Server port')
    parser.add_argument(
        '--reward-type',
        type=str,
        choices=reward_functions.keys(),
        default='example_math',
        help='The type of reward function to use',
    )
    parser.add_argument('--dataset', type=str, default=None, help='Golden dataset path')

    args = parser.parse_args()

    # NOTE for debug, use default reward function to calculate length of responses
    reward_func = (
        reward_functions[args.reward_type] if args.reward_type in reward_functions else None
    )

    if reward_func is None:
        print(f'Using default reward function for debug')
        print(f"Available reward types: {', '.join(reward_functions.keys())}")

    print(f'Using reward function: {args.reward_type}')

    # Start server
    start_server(host=args.host, port=args.port, reward_func=reward_func, dataset_path=args.dataset)


if __name__ == '__main__':
    main()
