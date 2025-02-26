import os

import torch


def load_pl_ckpt_allenact(model, ckpt_pth, ckpt_prefix="model.", verbose=False):
    print(f"Loading ckpt {ckpt_pth} using ckpt_prefix='{ckpt_prefix}' ...")
    ckpt_state_dict = torch.load(ckpt_pth, map_location="cpu")["state_dict"]

    # init new state dict with curr state dict
    new_state_dict = model.state_dict()

    ckpt_state_dict = {
        k.replace(
            "actor.weight",
            "actor.linear.weight",
        ): v
        for k, v in ckpt_state_dict.items()
    }

    ckpt_state_dict = {
        k.replace(
            "actor.bias",
            "actor.linear.bias",
        ): v
        for k, v in ckpt_state_dict.items()
    }

    params_to_load = [k for k in new_state_dict.keys() if ckpt_prefix + k in ckpt_state_dict]
    for k in params_to_load:
        new_state_dict[k] = ckpt_state_dict[ckpt_prefix + k]

    model.load_state_dict(new_state_dict)
    if verbose:
        print("-" * 80)
        print(
            "Parameters found and loaded from the checkpoint:",
            params_to_load,
        )
        print("-" * 80)

    params_in_model_not_in_ckpt = [
        k for k in new_state_dict.keys() if ckpt_prefix + k not in ckpt_state_dict
    ]
    if len(params_in_model_not_in_ckpt) > 0:
        print("-" * 80)
        print(
            "Parameters that are present in model but not present in checkpoint:",
            params_in_model_not_in_ckpt,
        )
        print("-" * 80)

    params_in_ckpt_not_in_model = [
        k[len(ckpt_prefix):]
        for k in ckpt_state_dict
        if k[len(ckpt_prefix):] not in new_state_dict and "visual_encoder.image_encoder.model" not in k  # we do not consider DINO weights
    ]
    if len(params_in_ckpt_not_in_model):
        print("-" * 80)
        print(
            f"Parameters present in checkpoint not present in model (removing ckpt_prefix='{ckpt_prefix}'):",
            params_in_ckpt_not_in_model,
        )
        print("-" * 80)


def load_pl_ckpt(model, ckpt_pth, ckpt_prefix="model.", verbose=False):
    print(f"Loading ckpt {ckpt_pth} using ckpt_prefix='{ckpt_prefix}' ...")
    ckpt_state_dict = torch.load(ckpt_pth, map_location="cpu")["state_dict"]

    # init new state dict with curr state dict
    new_state_dict = model.state_dict()

    params_to_load = [k for k in new_state_dict.keys() if ckpt_prefix + k in ckpt_state_dict]
    for k in params_to_load:
        new_state_dict[k] = ckpt_state_dict[ckpt_prefix + k]

    model.load_state_dict(new_state_dict)
    if verbose:
        print("-" * 80)
        print(
            "Parameters found and loaded from the checkpoint:",
            params_to_load,
        )
        print("-" * 80)

    params_in_model_not_in_ckpt = [
        k for k in new_state_dict.keys() if ckpt_prefix + k not in ckpt_state_dict
    ]
    if len(params_in_model_not_in_ckpt) > 0:
        print("-" * 80)
        print(
            "Parameters that are present in model but not present in checkpoint:",
            params_in_model_not_in_ckpt,
        )
        print("-" * 80)

    params_in_ckpt_not_in_model = [
        k[len(ckpt_prefix) :]
        for k in ckpt_state_dict
        if k[len(ckpt_prefix) :] not in new_state_dict
    ]
    if len(params_in_ckpt_not_in_model):
        print("-" * 80)
        print(
            f"Parameters present in checkpoint not present in model (removing ckpt_prefix='{ckpt_prefix}'):",
            params_in_ckpt_not_in_model,
        )
        print("-" * 80)


def get_latest_local_ckpt_pth(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        return None

    ckpts = sorted(ckpts, key=lambda x: int(x.split(".")[0].split("=")[-1]))
    return os.path.join(ckpt_dir, ckpts[-1])
