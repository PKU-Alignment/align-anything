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

# NOTE need to start the remote rm server first
bash start_remote_rm.sh

# NOTE need to change the model path
ACTOR_MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct" # actor model path
CRITIC_MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct" # critic model path

TRAIN_DATASETS="../align_anything/models/remote_rm/math_verify_dataset/mathvl_345_example.json" # dataset path
TRAIN_TEMPLATE="Math-Zero-RL" # math zero rlhf dataset template, note that for math zero rl, you are recommended to expand token length to longer length such as 18000
TRAIN_SPLIT="train" # split the input dataset

PTX_DATASETS="tatsu-lab/alpaca" # sft dataset path
PTX_TEMPLATE="Alpaca" # sft dataset template
PTX_SPLIT="train" # split the sft dataset

OUTPUT_DIR="../output/llama_ppo_remote_rm" # output dir
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
  --train_split ${TRAIN_SPLIT} \
  --train_template ${TRAIN_TEMPLATE} \
  --ptx_split ${PTX_SPLIT} \
  --ptx_datasets ${PTX_DATASETS} \
  --ptx_template ${PTX_TEMPLATE} \
  --output_dir ${OUTPUT_DIR}
