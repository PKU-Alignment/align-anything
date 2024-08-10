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

cd ../align_anything/evaluation

BENCHMARK=""
OUTPUT_DIR=""
GENERATION_BACKEND=""

python __main__.py \
    --benchmark ${BENCHMARK} \
    --output_dir ${OUTPUT_DIR} \
    --generation_backend ${GENERATION_BACKEND}