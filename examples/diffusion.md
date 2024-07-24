# Diffusion Training Scripts

We provide examples of various scripts for fine-tuning based on the [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) series models as follows. You can execute these commands in the `./scripts` directory to start the corresponding training.

## Supervised Fine-Tuning

### Text to Image

**Note:** The complete Diffusion [DB dataset](https://huggingface.co/datasets/poloclub/diffusiondb) is very large, so we only provide a script with a small sample by passing `--train_optional_args large_random_1k` here. If you encounter any issues while training on the full dataset, please feel free to discuss them in the Issue section.

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="CompVis/stable-diffusion-v1-4"
# You can replace it with a local dataset path
TRAIN_DATASETS="poloclub/diffusiondb"
# You can replace it with a new path
OUTPUT_DIR="../output/sft_diffusion"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute accelerate command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_image.sft_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
        --train_template DiffusionDB \
        --train_optional_args large_random_1k \
	--train_split train \
	--output_dir ${OUTPUT_DIR}
```

### Text to Video

**Note:** In the original data of [Webvid-10M](https://huggingface.co/datasets/TempoFunk/webvid-10M), videos exist in the form of URLs. The current version does not support online downloading of corresponding videos. Before training, you need to manually download the corresponding videos via URL and save them locally, organizing them in the following format.

```json
{
    "video_path": "<Your_video_path>",
    "caption": "<The_caption_corresponding_to_the_video>"
}
```

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="ali-vilab/text-to-video-ms-1.7b"
# You can replace it with a local dataset path
TRAIN_DATASETS="TempoFunk/webvid-10M"
# You can replace it with a new path
OUTPUT_DIR="../output/sft_diffusion"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute accelerate command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_video.sft_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
        --train_template Webvid \
	--train_split train \
	--output_dir ${OUTPUT_DIR}
```

### Text to Audio

**Note:** WavCaps cannot be used directly after downloading and needs to be unpacked and organized according to the [Hugging Face tutorial](https://huggingface.co/datasets/cvssp/WavCaps). The organization format can refer to [wavcaps_test](https://huggingface.co/datasets/AudioLLMs/wavcaps_test)

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="cvssp/audioldm-s-full-v2"
# You can replace it with a local dataset path
TRAIN_DATASETS="cvssp/WavCaps"
# You can replace it with a new path
OUTPUT_DIR="../output/sft_diffusion"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute accelerate command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_audio.sft_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
        --train_template WavCaps \
	--train_split train \
	--output_dir ${OUTPUT_DIR}
```

## DPO Training

### Text to Image

**Note:** We conducted code testing on a small subset of the [Pickapic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) dataset. However, since the full Pickapic dataset is too large, we chose a smaller version for users to run quickly here.

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="CompVis/stable-diffusion-v1-4"
# You can replace it with a local dataset path
TRAIN_DATASETS="kashif/pickascore"
# You can replace it with a new path
OUTPUT_DIR="../output/dpo_diffusion"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute accelerate command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_image.dpo_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
    --train_template Pickapic \
	--train_split validation \
	--output_dir ${OUTPUT_DIR}
```

### Text to Video

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="ali-vilab/text-to-video-ms-1.7b"
# You can replace it with a local dataset path
TRAIN_DATASETS="PKU-Alignment/SafeSora"
# You can replace it with a new path
OUTPUT_DIR="../output/dpo_diffusion"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute accelerate command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_video.dpo_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
        --train_template SafeSora \
	--train_split train \
	--output_dir ${OUTPUT_DIR}
```

### Text to Audio

**Note:** The SOMOS dataset is not available on Hugging Face currently. You need to download MOS data from [Zenodo](https://zenodo.org/records/7378801) and then obtain audio files and their corresponding captions based on the content in the SOMOS dataset. Please organize all files in the following format.
```json
{
    "better_data_path": "<Your_audio_path>",
    "worse_data_path": "<Your_audio_path>",
    "prompt": "<The_caption_corresponding_to_the_audios>"
}
```

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="cvssp/audioldm-s-full-v2"
# You can replace it with a local dataset path
TRAIN_DATASETS="<Your_local_SOMOS_path>"
# You can replace it with a new path
OUTPUT_DIR="../output/dpo_diffusion"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute accelerate command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_audio.dpo_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
        --train_template SOMOS \
	--train_split train \
	--output_dir ${OUTPUT_DIR}
```
