# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the tgxs002's HPSv2 library.
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
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

import torch
from PIL import Image
from align_anything.models.clip import create_model_and_transforms, get_tokenizer
import warnings
from typing import Union, List
import huggingface_hub

warnings.filterwarnings("ignore", category=UserWarning)

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

hps_version_map = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}

def initialize_model():
    if not model_dict:
        model, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            device=device,
            output_dict=True,
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

def get_score(img_paths: Union[List[list], List[str], List[Image.Image]], prompts: List[str], cp: str = None, hps_version: str = "v2.0") -> List[float]:
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    if len(img_paths) != len(prompts):
        raise ValueError("The number of images and prompts must be the same.")

    results = []

    for img_path, prompt in zip(img_paths, prompts):
        if isinstance(img_path, list):
            result = []
            for one_img_path in img_path:
                with torch.no_grad():
                    if isinstance(one_img_path, str):
                        image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
                    elif isinstance(one_img_path, Image.Image):
                        image = preprocess_val(one_img_path).unsqueeze(0).to(device=device, non_blocking=True)
                    else:
                        raise TypeError('The type of parameter img_path is illegal.')
                    text = tokenizer([prompt]).to(device=device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        outputs = model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T

                        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                result.append(hps_score[0])    
            results.append(result[0]) 
        elif isinstance(img_path, str):
            with torch.no_grad():
                image = preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=device, non_blocking=True)
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            results.append(hps_score[0]) 
        elif isinstance(img_path, Image.Image):
            with torch.no_grad():
                image = preprocess_val(img_path).unsqueeze(0).to(device=device, non_blocking=True)
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            results.append(hps_score[0]) 
        else:
            raise TypeError('The type of parameter img_path is illegal.')

    return results

def get_score_single(img_path: Union[list, str, Image.Image], prompt: str, cp: str = None, hps_version: str = "v2.0") -> list:
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    
    
    if isinstance(img_path, list):
        result = []
        for one_img_path in img_path:
            with torch.no_grad():
                if isinstance(one_img_path, str):
                    image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
                elif isinstance(one_img_path, Image.Image):
                    image = preprocess_val(one_img_path).unsqueeze(0).to(device=device, non_blocking=True)
                else:
                    raise TypeError('The type of parameter img_path is illegal.')
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])    
        return result[0]
    elif isinstance(img_path, str):
        with torch.no_grad():
            image = preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return hps_score[0]
    elif isinstance(img_path, Image.Image):
        with torch.no_grad():
            image = preprocess_val(img_path).unsqueeze(0).to(device=device, non_blocking=True)
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return hps_score[0]
    else:
        raise TypeError('The type of parameter img_path is illegal.')