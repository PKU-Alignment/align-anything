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

# This script is used to train the LLaVA model
# based on the cookbook: cookbooks/en(or zh)/modality_scaling.ipynb

# Initialize variables
MODEL_NAME_OR_PATH="/PATH/TO/YOUR/MLLMs"
TRAIN_DATASETS="/PATH/TO/YOUR/liuhaotian/LLaVA-Pretrain"
TRAIN_TEMPLATE="LLaVA_Pretrain" # dataset template
TRAIN_DATA_FILES="blip_laion_cc_sbu_558k.json"

OUTPUT_DIR="../outputs/llava_step1" # output dir

# For wandb online logging
export WANDB_API_KEY=""

export COCO_DATA_DIR="/PATH/TO/YOUR/COCO_DATA_DIR" # the dir where you save the unzipped COCO images

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
        --master_port ${MASTER_PORT} \
        --module align_anything.trainers.text_image_to_text.sft \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_split train \
        --train_data_files ${TRAIN_DATA_FILES} \
        --output_dir ${OUTPUT_DIR} \
        --save_total_limit 3 \
        --freeze_vision_tower True \
        --freeze_mm_proj False \
        --freeze_language_model True \
        --epochs 1 \
        --ds_cfgs ds_z2_config.json \
        --learning_rate 1.e-3 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 32
