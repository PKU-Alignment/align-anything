# Copyright 2025 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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

export HOME_PREFIX=/path/to/your/data

export OBJAVERSE_HOUSES_BASE_DIR=${HOME_PREFIX}/houses/objaverse_houses
export OBJAVERSE_HOUSES_DIR=${HOME_PREFIX}/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_BASE_DIR=${HOME_PREFIX}/assets/objaverse_houses
export OBJAVERSE_DATA_DIR=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28/annotations.json.gz
export WANDB_DIR=${HOME_PREFIX}/align-anything/wandb
export LONG_ACTION_NAME=1

python -m align_anything.trainers.text_video_to_action.training.offline.train_pl \
 --max_samples 10000000 \
 --eval_max_samples 100 \
 --eval_every 300 \
 --model_version small_3 \
 --sliding_window 100 \
 --per_gpu_batch 8 \
 --lr 0.0002 \
 --data_dir ${HOME_PREFIX}/data/ \
 --dataset_version CHORES \
 --model EarlyFusionCnnTransformer \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 --precision 16-mixed \
 --resume_local \
 --output_dir il_ckpt \
 --loss action \
 --max_epochs 400
