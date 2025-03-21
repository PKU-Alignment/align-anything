#!/usr/bin/env bash
#
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


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

# [Need] Your OpenAI API key
export OPENAI_API_KEY=''

# [Optional] If you want to use your own API,
# you can set the following environment variables
export OPENAI_API_BASE=""
export OPENAI_API_BASE_URL=""

BENCHMARKS=("llava-bench-coco") # evaluation benchmarks
OUTPUT_DIR="../output/evaluation" # output dir
GENERATION_BACKEND="vLLM" # generation backend
MODEL_ID="llava-1.5-7b-hf" # model's unique id for logging, you can use any name you like
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model name or path
CHAT_TEMPLATE="Llava" # model template

for BENCHMARK in "${BENCHMARKS[@]}"; do
    python __main__.py \
        --benchmark ${BENCHMARK} \
        --output_dir ${OUTPUT_DIR} \
        --generation_backend ${GENERATION_BACKEND} \
        --model_id ${MODEL_ID} \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --chat_template ${CHAT_TEMPLATE}
done
