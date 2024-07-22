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

# Source the setup script
source ./setup.sh

# Execute deepspeed command
accelerate launch  \
	--multi_gpu \
	--module align_anything.trainers.text_to_image.dpo_diffusion \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
    --train_template Pickapic \
	--train_split validation \
	--output_dir ${OUTPUT_DIR}
```
