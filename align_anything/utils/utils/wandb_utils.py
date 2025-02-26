import os

import torch.nn as nn
from allenact.base_abstractions.misc import ActorCriticOutput


def download_from_wandb_to_directory(run, filename, custom_directory, force_download=False):
    print(f"Downloading file {filename} from wandb run: {run.id} to {custom_directory}")
    ckpt = run.file(filename)
    if ckpt.size == 0:
        print(f"{filename} either does not exist in wandb run: {run.id} or has 0 size")
        return None

    info = ckpt.download(root=custom_directory, replace=force_download, exist_ok=True)
    return info.name


def log_ac_return(ac: ActorCriticOutput, task_id_obs):
    os.makedirs("output/ac-data/", exist_ok=True)
    assert len(task_id_obs.shape) == 3

    for i in range(len(task_id_obs[0])):
        task_id = "".join([chr(int(k)) for k in task_id_obs[0, i] if chr(int(k)) != " "])

        with open(f"output/ac-data/{task_id}.txt", "a") as f:
            estimated_value = ac.values[0, i].item()
            policy = nn.functional.softmax(ac.distributions.logits[0, i]).tolist()
            f.write(",".join(map(str, policy + [estimated_value])) + "\n")
