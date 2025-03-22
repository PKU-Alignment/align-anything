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


ACTOR_MODEL_NAME_OR_PATH="facebook/opt-125m" # actor model path
CRITIC_MODEL_NAME_OR_PATH=${REWARD_MODEL_NAME_OR_PATH:-"facebook/opt-125m"} # please set the reward model path
REWARD_MODEL_NAME_OR_PATH=${REWARD_MODEL_NAME_OR_PATH:-"facebook/opt-125m"} # please set the reward model path

TRAIN_DATASETS="../assets/text_to_text/preference" # rlhf dataset path
TRAIN_TEMPLATE="PKUSafeRLHF" # rlhf dataset template
TRAIN_SPLIT="train" # split the rlhf dataset

PTX_DATASETS="../assets/text_to_text/supervised" # sft dataset path
PTX_TEMPLATE="Alpaca" # sft dataset template
PTX_SPLIT="train" # split the sft dataset

OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR

if [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "OUTPUT_ROOT_DIR is not set"
    OUTPUT_ROOT_DIR="../outputs"
fi

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/opt_125m_ppo" # output dir
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.text_to_text.ppo \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
  --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
  --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
  --train_datasets ${TRAIN_DATASETS} \
  --train_split ${TRAIN_SPLIT} \
  --train_template ${TRAIN_TEMPLATE} \
  --ptx_split ${PTX_SPLIT} \
  --ptx_datasets ${PTX_DATASETS} \
  --ptx_template ${PTX_TEMPLATE} \
  --output_dir ${OUTPUT_DIR} \
  --epochs 1
