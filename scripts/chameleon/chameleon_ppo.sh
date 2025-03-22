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

ACTOR_MODEL_NAME_OR_PATH="PKU-Alignment/AA-chameleon-7b-base" # model path
REWARD_MODEL_NAME_OR_PATH="../outputs/chameleon_rm" # model path
CRITIC_MODEL_NAME_OR_PATH="../outputs/chameleon_rm" # model path

TRAIN_DATASETS="" # dataset path
TRAIN_DATA_FILES="" # dataset name

PTX_DATASETS="" # dataset path
PTX_DATA_FILES="" # dataset name

OUTPUT_DIR="../outputs/chameleon_ppo" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_image_to_text_image.ppo \
     --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
     --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
     --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_data_files ${TRAIN_DATA_FILES} \
     --ptx_datasets ${PTX_DATASETS} \
     --ptx_data_files ${PTX_DATA_FILES} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 3 \
     --epochs 2
