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


# Initialize variables
MODEL_NAME_OR_PATH="/data/hantao/models/Emu3-Chat"
PROCESSOR_NAME_OR_PATH="/data/hantao/models/Emu3-VisionTokenizer"
TRAIN_DATASETS="/data/hantao/align-anything/data/any2any"
TRAIN_DATA_FILE="example.json"
OUTPUT_DIR="/data/hantao/align-anything/outputs/any2any/debug"

export WANDB_API_KEY="547f38af44135ca76a4f4eed9c8d135532da4960"
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.any_to_any.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--processor_name_or_path ${PROCESSOR_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
    --train_data_file ${TRAIN_DATA_FILE} \
    --train_template Any2Any \
    --train_split train \
	--output_dir ${OUTPUT_DIR}