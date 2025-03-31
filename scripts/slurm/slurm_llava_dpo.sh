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

#SBATCH --job-name=llava_dpo
#SBATCH --output=/path/to/your/align-anything/llava_dpo_%j.log
#SBATCH --error=/path/to/your/align-anything/llava_dpo_%j.log
#SBATCH --partition=your_partition_name
#SBATCH --account=your_account_name
#SBATCH --nodes=1
#SBATCH --gres=gpu:8


# Run the script
srun  llava_dpo.sh
