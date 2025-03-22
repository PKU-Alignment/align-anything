# Copyright 2024 Allen Institute for AI

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


export HOME_PREFIX=/path/to/objaverse/data

export OBJAVERSE_HOUSES_DIR=${HOME_PREFIX}/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_DIR=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28/annotations.json.gz
export LONG_ACTION_NAME=1

OUTPUT_DIR="../outputs/spoc_sft"

source ./scripts/setup.sh

deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_video_to_action.sft \
