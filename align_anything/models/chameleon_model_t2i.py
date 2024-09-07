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

import os
import sys
import torch
import numpy as np
from typing import Literal, Optional
from typing import List, Literal, Optional, Tuple, Union
from transformers import ChameleonForConditionalGeneration, ChameleonModel, ChameleonProcessor, set_seed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmsg', 'integrations'))

class ChameleonTextToImagePipeline:
    def __init__(
        self,
        model_name_or_path: str,
        inference_mode: str = "image-only",
        max_new_tokens: int = 2400,
        fast: bool = False,
        model_cache_dir: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.inference_mode = inference_mode
        self.max_new_tokens = max_new_tokens
        self.fast = fast
        self.model_cache_dir = model_cache_dir
        self.seed = seed
        self.device = device
        self.init_model()
        
    def init_model(self) -> None:
        self.model = ChameleonForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=self.model_cache_dir,
        ).to(self.device)
        if self.seed is not None:
            set_seed(self.seed)
        torch.set_printoptions(threshold=10_000)

        self.processor = ChameleonProcessor.from_pretrained(
            self.model_name_or_path,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=self.model_cache_dir,
        )
        
    def build_response_from_segments(
        self,
        model: Union["ChameleonModel", "ChameleonForConditionalGeneration"],
        processor: "ChameleonProcessor",
        segments: List[Tuple[Literal["text", "image"], List[int]]],
        image_path: Optional[str] = None,
    ):
        text_tokens_list = [
            token_ids for modality, token_ids in segments if modality == "text"
        ]
        image_tokens_list = [
            token_ids[:1024]
            if len(token_ids) > 1024
            else [1] * (1024 - len(token_ids)) + token_ids
            for modality, token_ids in segments
            if modality == "image"
        ]

        text_str_list = processor.batch_decode(text_tokens_list, skip_special_tokens=True)
        image_tokens_tensor = torch.tensor(image_tokens_list, device=model.device)

        try:
            pixel_values = model.decode_image_tokens(image_tokens_tensor)
            images = processor.postprocess_pixel_values(
                pixel_values.float().detach().cpu().numpy()
            )
        except:
            images = None

        response = {"text": "", "images": []}
        for modality, _ in segments:
            if modality == "text":
                response["text"] += text_str_list.pop(0)
            elif modality == "image":
                if images is None:
                    continue
                response["text"] += "<image>"
                image = images.pop(0)

                image.save(image_path)

                response["images"].append(image_path)
        return response

    def split_tokens_into_segments_by_modality(
        self,
        token_ids: "np.ndarray",
        boi_token_id: int,
        eoi_token_id: int,
        validate: bool = False,
    ) -> List[Tuple[Literal["text", "image"], List[int]]]:
        segments: List[Tuple[Literal["text", "image"], List[int]]] = []
        curr_sequence: List[int] = []
        modality: Literal["text", "image"] = "text"
        
        for idx, token_id in enumerate(token_ids):
            if token_id == boi_token_id:
                if validate and modality == "image":
                    raise ValueError(
                        f"Invalid token sequence: sequence has duplicate image generation start token."
                    )
                if idx > 0:
                    segments.append((modality, curr_sequence))
                    curr_sequence = []
                modality = "image"
                continue
            elif token_id == eoi_token_id:
                if validate and modality == "text":
                    raise ValueError(
                        f"Invalid token sequence: sequence has image generation end token without start token."
                    )
                segments.append((modality, curr_sequence))
                modality = "text"
                curr_sequence = []
                continue
            curr_sequence.append(token_id)
            
        if curr_sequence:
            if modality == "text":
                segments.append(("text", curr_sequence))
            else:
                if validate:
                    raise ValueError(
                        f"Invalid token sequence: sequence has image generation start token without end token."
                    )
                segments.append(("image", curr_sequence))
        return segments

    def postprocess_token_sequence(
        self,
        token_ids: "np.ndarray",
        model: Union["ChameleonModel", "ChameleonForConditionalGeneration"],
        processor: "ChameleonProcessor",
        image_path: Optional[str] = None,
        validate: bool = True,
    ):
        segments = self.split_tokens_into_segments_by_modality(
            token_ids,
            model.vocabulary_mapping.boi_token_id,
            model.vocabulary_mapping.eoi_token_id,
            validate=validate,
        )
        return self.build_response_from_segments(model, processor, segments, image_path)

    def generation(
        self,
        prompt: str,
        image_path: str = ".",
    ) -> str:
        prompt = f"Generate an image according to the following instruction: {prompt}"
        inputs = self.processor(prompt, return_tensors="pt").to(
            self.model.device, dtype=self.model.dtype
        )
        with torch.inference_mode():
            output_token_ids_batch = self.model.generate(
                **inputs,
                multimodal_generation_mode=self.inference_mode,
                max_new_tokens=self.max_new_tokens,
                do_sample=True
            )
        output_token_ids_batch = output_token_ids_batch.to(dtype=inputs["input_ids"].dtype).detach().cpu().numpy()
        response_token_ids = output_token_ids_batch[0][len(inputs["input_ids"][0]) :]

        response = self.postprocess_token_sequence(
            response_token_ids, self.model, self.processor, image_path, validate=True
        )
        response['prompt'] = prompt