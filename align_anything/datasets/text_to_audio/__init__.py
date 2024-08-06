#!/usr/bin/env bash
#
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
"""Dataset classes for text to audio training."""

<<<<<<<< HEAD:align_anything/evaluation/benchmarks/ARC/eval.sh
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      output="$2"
      shift 2
      ;;
    --generation_backend)
      backend="$2"
      shift 2
      ;;
    -g)
      backend="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [ "$backend" = "vllm" ]; then
  python vllm_eval.py \
    --output_dir "$output"
else
  deepspeed \
    --module ds_infer \
    --output_dir $output
  python ds_eval.py \
    --output_dir "$output"
fi

rm -rf .cache
rm -rf __pycache__
========

from align_anything.datasets.text_to_audio.preference import *
from align_anything.datasets.text_to_audio.supervised import *
>>>>>>>> upstream/dev-eval-2:align_anything/datasets/text_to_audio/__init__.py
