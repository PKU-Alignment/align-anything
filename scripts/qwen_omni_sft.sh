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


MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Omni-7B" # model path

TRAIN_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K" # dataset path
TRAIN_TEMPLATE="Qwen_Omni_TI2T" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../output/qwen_omni_sft" # output dir

# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.qwen_omni.ti2t_sft \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 2 \
     --per_device_train_batch_size 1 \
     --train_size 10 \
     --epochs 3
