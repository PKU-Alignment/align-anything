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


MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

TRAIN_DATASETS='PKU-Alignment/BeaverTails-V' # dataset path
TRAIN_TEMPLATE="SafeRLHF_V_Cost" # dataset template
TRAIN_NAME="animal_abuse" # dataset name
TRAIN_SPLIT="train" # split the dataset

EVAL_DATASETS="PKU-Alignment/BeaverTails-V"
EVAL_TEMPLATE="SafeRLHF_V_Cost"
EVAL_NAME="dangerous_behavior"
EVAL_SPLIT="evaluation"

OUTPUT_DIR="../outputs/llava_cm" # output dir
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.text_image_to_text.cost_model \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_template ${TRAIN_TEMPLATE} \
    --train_name ${TRAIN_NAME} \
    --train_split ${TRAIN_SPLIT} \
    --eval_datasets ${EVAL_DATASETS} \
    --eval_template ${EVAL_TEMPLATE} \
    --eval_name ${EVAL_NAME} \
    --eval_split ${EVAL_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --save_interval 1000 \
    --epochs 3
