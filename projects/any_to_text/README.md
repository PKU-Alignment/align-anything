# Any to Text Model Doc for Align-Anything

## Build a custom multimodal model for alignment research

This document provides a set of scripts for building an Any-to-Text multimodal language model using a pure text model and modality encoder. It is designed to facilitate researchers in constructing their own multimodal language models for alignment research.

## Usage Examples

The files `build_llama_vision.py` and `build_llama_vision_audio.py` serve as example scripts for building the Image+Text -> Text model and the Image+Text+Audio -> Text model, respectively. To obtain an initialized vision-text large language model, run:
```
python build_llama_vision.py
```
To obtain an initialized vision-audio-text large language model, run:
```
python build_llama_vision_audio.py
```
The default base large language model is `meta-llama/Meta-Llama-3.1-8B-Instruct`, the vision encoder is `openai/clip-vit-large-patch14-336`, and the audio encoder is `laion/clap-htsat-fuse`.

## Training

When training a multimodal large language model, it is recommended to divide the process into two stages:

- Stage 1: Train the projector between the Encoder and the Large Language Model. For example, in training a vision-text model, the training script is as follows:
```
# Initialize variables
MODEL_NAME_OR_PATH=""
TRAIN_DATASETS=""
OUTPUT_DIR=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_to_text.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--output_dir ${OUTPUT_DIR} \
	--freeze_vision_tower True \
	--freeze_mm_proj False \
	--freeze_language_model True \
```

- Stage 2: Jointly train the projector and the Large Language Model. For example, in training a vision-text model, the training script is as follows:
```
# Initialize variables
MODEL_NAME_OR_PATH=""
TRAIN_DATASETS=""
OUTPUT_DIR=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_to_text.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--output_dir ${OUTPUT_DIR} \
	--freeze_vision_tower True \
	--freeze_mm_proj False \
	--freeze_language_model False \
```
Considering that Stage 2 will require training on multiple datasets, Align-Anything supports joint training across multiple datasets. For example, in training a vision-text model, the training script is as follows:
```
# Initialize variables
TRAIN_DATASETS="\
liuhaotian/LLaVA-Instruct-150K,\
HuggingFaceM4/A-OKVQA,\
HuggingFaceM4/VQAv2,\
"

TRAIN_TEMPLATE="\
LLAVA,\
A-OKVQA,\
VQAv2,\
"

TRAIN_SPLIT="\
train,\
train,\
train,\
"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_to_text.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
  	--train_split ${TRAIN_SPLIT} \
	--train_template ${TRAIN_TEMPLATE} \
	--output_dir ${OUTPUT_DIR} \
	--freeze_vision_tower True \
	--freeze_mm_proj False \
	--freeze_language_model False \
```

**Note:** If you train your image-audio-text multimodal language model across multiple datasets in deepspeed Zero3, you shall set `prefetch_bucket_size` to `0` in configuration file `ds_z3_config.json`. More details could refer to [issue](https://github.com/microsoft/DeepSpeed/issues/5828)

## Model

### Model Weights

- [Image-Text Multimodal Language Model](https://huggingface.co/PKU-Alignment/llama3.1-8b-instruct-vision)
- [Image-Audio-Text Multimodal Language Model](https://huggingface.co/PKU-Alignment/llama3.1-8b-vision-audio)

### Model Evaluation

#### Image-Text Multimodal Language Model:

After training with `liuhaotian/LLaVA-Instruct-150K`,`OpenGVLab/ShareGPT-4o`,
`HuggingFaceM4/A-OKVQA`,`Multimodal-Fatima/OK-VQA_train`,`howard-hou/OCR-VQA`, `HuggingFaceM4/VQAv2`, some of the evaluation results for our image-text multimodal language model are as follows:

|mme cognition score|mme perception score|ok vqa|mmbench en|
|:---:|:---:|:---:|:---:|
|259.64|1191.75|36.7|48.88|

#### Image-Audio-Text Multimodal Language Model:

The image-audio-text language model is currently still in development.

### Model Usage

#### Image-Text Multimodal Language Model:
```
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)
from PIL import Image

path = <path_to_model_dir>
processor = AutoProcessor.from_pretrained(path)
model = LlavaForConditionalGeneration.from_pretrained(path)

prompt = "<|start_header_id|>user<|end_header_id|>: <image> Give an overview of what's in the image.\n<|start_header_id|>assistant<|end_header_id|>: "
image_path = "align-anything/assets/test_image.webp"
image = Image.open(image_path)

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

#### Image-Audio-Text Multimodal Language Model:
```
from align_anything.models.llama_vision_audio_model import (
    LlamaVisionAudioForConditionalGeneration,
    LlamaVisionAudioProcessor,
)
import torch
import torchaudio
from PIL import Image

path = <path_to_model_dir>
processor = LlamaVisionAudioProcessor.from_pretrained(path)
model = LlamaVisionAudioForConditionalGeneration.from_pretrained(path)

prompt = "<|start_header_id|>user<|end_header_id|>: Where is the capital of China?\n<|start_header_id|>assistant<|end_header_id|>: "

inputs = processor(text=prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0], skip_special_tokens=True))

prompt = "<|start_header_id|>user<|end_header_id|>: Summarize the audio's contents.<audio>\n<|start_header_id|>assistant<|end_header_id|>: "

audio_path = "align-anything/assets/test_audio.wav"
audio, _ = torchaudio.load(audio_path)
if audio.shape[0] == 2:
    audio = audio.mean(dim=0, keepdim=True)
audio = audio.squeeze().tolist()

inputs = processor(text=prompt, raw_speech=audio, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0], skip_special_tokens=False))

prompt = "<|start_header_id|>user<|end_header_id|>: <image> Give an overview of what's in the image.\n<|start_header_id|>assistant<|end_header_id|>: "
image_path = "align-anything/assets/test_image.webp"
image = Image.open(image_path)

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0], skip_special_tokens=True))
```