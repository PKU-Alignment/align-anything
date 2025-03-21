# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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


import os

import torch.nn as nn
from allenact.base_abstractions.misc import ActorCriticOutput


def download_from_wandb_to_directory(run, filename, custom_directory, force_download=False):
    print(f'Downloading file {filename} from wandb run: {run.id} to {custom_directory}')
    ckpt = run.file(filename)
    if ckpt.size == 0:
        print(f'{filename} either does not exist in wandb run: {run.id} or has 0 size')
        return None

    info = ckpt.download(root=custom_directory, replace=force_download, exist_ok=True)
    return info.name


def log_ac_return(ac: ActorCriticOutput, task_id_obs):
    os.makedirs('output/ac-data/', exist_ok=True)
    assert len(task_id_obs.shape) == 3

    for i in range(len(task_id_obs[0])):
        task_id = ''.join([chr(int(k)) for k in task_id_obs[0, i] if chr(int(k)) != ' '])

        with open(f'output/ac-data/{task_id}.txt', 'a') as f:
            estimated_value = ac.values[0, i].item()
            policy = nn.functional.softmax(ac.distributions.logits[0, i]).tolist()
            f.write(','.join(map(str, policy + [estimated_value])) + '\n')
