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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

BENCHMARKS=("")
OUTPUT_DIR=""
GENERATION_BACKEND=""
MODEL_IDS=("" "")
MODEL_NAME_OR_PATHS=("" "")
CHAT_TEMPLATES=("" "")

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "Processing benchmark: ${BENCHMARK}"
    
    for i in "${!MODEL_IDS[@]}"; do
        MODEL_ID=${MODEL_IDS[$i]}
        MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATHS[$i]}
        CHAT_TEMPLATE=${CHAT_TEMPLATES[$i]}
        
        echo "Running model ${MODEL_ID} for benchmark ${BENCHMARK}"
        python __main__.py \
            --benchmark ${BENCHMARK} \
            --output_dir ${OUTPUT_DIR} \
            --generation_backend ${GENERATION_BACKEND} \
            --model_id ${MODEL_ID} \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --chat_template ${CHAT_TEMPLATE}
    done

    python models_pk.py --benchmark ${BENCHMARK} \
                        --model_1 "${MODEL_IDS[0]}" \
                        --model_2 "${MODEL_IDS[1]}"
done