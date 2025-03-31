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


ACTOR_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct" # model path
CRITIC_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct" # model path


TRAIN_DATASETS="../align_anything/models/remote_rm/math_verify_dataset/mathvl_345_example.json" # dataset path

TRAIN_TEMPLATE="Math-Zero-RL" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../output/qwen_2_5_vl_ppo_remote_rm" # output dir


# For wandb online logging
export WANDB_API_KEY=""
export REMOTE_RM_URL="http://127.0.0.1:6000/get_reward"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.ppo_remote_rm \
     --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
     --remote_rm_url ${REMOTE_RM_URL} \
     --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 3 \
     --epochs 2
