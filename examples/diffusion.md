# Llava Training Scripts

We provide examples of various scripts for fine-tuning based on the [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) series models as follows. You can execute these commands in the `./scripts` directory to start the corresponding training.

## Supervised Fine-Tuning

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

# Execute deepspeed command
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

## DPO Training

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
# You can replace it with a local dataset path
TRAIN_DATASETS="openbmb/RLAIF-V-Dataset"
# You can replace it with a new path with correct permission
OUTPUT_DIR="../output/dpo"
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_to_text.dpo \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--train_template RLAIFV \
	--train_split train \
	--freeze_mm_proj True \
	--freeze_vision_tower False \
	--output_dir ${OUTPUT_DIR}
```
