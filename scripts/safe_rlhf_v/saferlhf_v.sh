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


ACTOR_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path
REWARD_MODEL_NAME_OR_PATH="../outputs/llava_rm" # model path
CRITIC_MODEL_NAME_OR_PATH="../outputs/llava_rm" # model path
COST_MODEL_NAME_OR_PATH="../outputs/llava_cm" # model path

TRAIN_DATASETS='PKU-Alignment/BeaverTails-V' # dataset path
TRAIN_TEMPLATE="SafeRLHF_V_Reward" # dataset template
TRAIN_NAME="animal_abuse" # dataset name
TRAIN_SPLIT="train" # split the dataset

PTX_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K"
PTX_TEMPLATE="AA_TI2T"
PTX_SPLIT="train"

OUTPUT_DIR="../outputs/saferlhf_v" # output dir
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
    --module align_anything.trainers.text_image_to_text.saferlhf \
    --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
    --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
    --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
    --cost_model_name_or_path ${COST_MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_template  ${TRAIN_TEMPLATE} \
    --train_name ${TRAIN_NAME} \
    --train_split ${TRAIN_SPLIT} \
    --ptx_datasets ${PTX_DATASETS} \
    --ptx_template ${PTX_TEMPLATE} \
    --ptx_split ${PTX_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --save_interval 1000 \
    --epochs 2
