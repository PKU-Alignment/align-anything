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
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export CUDA_HOME=""


# You can replace it with a local model path
MODEL_NAME_OR_PATH=""
# You can replace it with a local model path
ACTOR_MODEL_NAME_OR_PATH=""
# You can replace it with a local model path
CRITIC_MODEL_NAME_OR_PATH=""
# You can replace it with a local model path
REWARD_MODEL_NAME_OR_PATH=""
COST_MODEL_NAME_OR_PATH=""

TRAIN_DATASETS=''
TRAIN_TEMPLATE="SPA_VL_ours" # dataset template
TRAIN_SPLIT="train" # split the dataset

# PTX_DATASETS=""
PTX_DATASETS="PKU-Alignment/align-anything"
PTX_SUBSET="text-image-to-text" # dataset name
PTX_TEMPLATE="AA_TI2T"
PTX_SPLIT="train"

# You can replace it with a new path with correct permission
OUTPUT_DIR=""
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh
# Execute deepspeed command
deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.text_image_to_text.ppo_lag \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
  --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
  --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
  --cost_model_name_or_path ${COST_MODEL_NAME_OR_PATH} \
  --train_datasets ${TRAIN_DATASETS} \
	--train_template  ${TRAIN_TEMPLATE} \
  --train_split ${TRAIN_SPLIT} \
  --ptx_datasets ${PTX_DATASETS} \
  --ptx_template ${PTX_TEMPLATE} \
  --ptx_subset ${PTX_SUBSET} \
  --ptx_split ${PTX_SPLIT} \
  --output_dir ${OUTPUT_DIR}