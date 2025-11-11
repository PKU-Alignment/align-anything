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

import argparse

from eval_anything.pipeline.base_task import BaseTask
from eval_anything.utils.utils import parse_unknown_args


# import eval_anything.benchmarks.text_to_text.DecodingTrust.eval
# print("DecodingTrust module imported!")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--eval_info', type=str, required=True, default='evaluate.yaml', help="YAML filename for evaluation configurations in eval-anything/configs/.")
    parser.add_argument(
        '--eval_info',
        type=str,
        required=False,
        default='evaluate.yaml',
        help='YAML filename for evaluation configurations in eval-anything/configs/.',
    )
    known_args, unparsed_args = parser.parse_known_args()
    unparsed_args = parse_unknown_args(unparsed_args)
    task = BaseTask(known_args.eval_info, **unparsed_args)
    task.iterate_run()


if __name__ == '__main__':
    main()
