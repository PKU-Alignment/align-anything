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


MODEL_NAME_OR_PATH="Qwen/Qwen2-Audio-7B-Instruct" # model path

TRAIN_DATASETS="PKU-Alignment/align-anything" # dataset path
TRAIN_TEMPLATE="AA_TA2T" # dataset template
TRAIN_NAME="text-audio-to-text" # dataset name
TRAIN_SPLIT="train" # split the dataset

EVAL_DATASETS="PKU-Alignment/align-anything" # dataset path
EVAL_TEMPLATE="AA_TA2T" # dataset template
EVAL_NAME="text-audio-to-text" # dataset name
EVAL_SPLIT="val" # split the dataset

OUTPUT_DIR="../outputs/qwen2_audio_rm" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_audio_to_text.rm \
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
     --save_total_limit 3 \
     --epochs 2
