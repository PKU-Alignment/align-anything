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


MODEL_NAME_OR_PATH="openbmb/MiniCPM-o-2_6" # model path
export MODEL_NAME_OR_PATH

TRAIN_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K" # dataset path
TRAIN_TEMPLATE="AA_TI2T" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../outputs/minicpmo_sft_vision" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_image_to_text.sft \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --trust_remote_code True \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_split ${TRAIN_SPLIT} \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 3 \
     --epochs 5
