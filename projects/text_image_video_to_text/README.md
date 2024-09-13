# Qwen2-VL Model Doc for Align-Anything

## Environment setup

Currently, the Transformer in pypi is not updated to support the latest Qwen2-VL model, so we need to install the forked stable version of the repo by running:
pip install git+https://github.com/huggingface/transformers.git

## Model

We use the `Qwen2-VL-Chat` model for the text-image-video-to-text task. You can download the model from [here](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), or directly use the model id `Qwen/Qwen2-VL-7B-Instruct` in the script.

## Training

### Model SFT

Add a script named `sft_text_image_video_to_text.sh` under the `scripts` file like this:

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
# You can replace it with a local dataset path
TRAIN_DATASETS="../../data"
TRAIN_DATA_FILE="text_image_video_to_text.json"
# You can replace it with a new path
OUTPUT_DIR="../outputs/sft_qwen2_vl"
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_video_to_text.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--train_data_files ${TRAIN_DATA_FILE} \
	--train_split 'train' \
	--output_dir ${OUTPUT_DIR} \
	--train_template TIV2T \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 2 \
	--ddp_timeout 1800000000 # this is a must for qwen2-vl training
```

and set up the correct model path and dataset path, then run:

```bash
bash scripts/sft_text_image_video_to_text.sh
```

### Reward Model Training

Add a script named `rm_text_image_video_to_text.sh` under the `scripts` file like this:

```bash
# Initialize variables
MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
# You can replace it with a local dataset path
TRAIN_DATASETS="path/to/dataset"
# You can replace it with a new path
OUTPUT_DIR="../outputs/rm_text_image_video_to_text"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_video_to_text.rm \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
    --train_template TIV2T_preference \
	--output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--ddp_timeout 1800000000 # this is a must for qwen2-vl training
```

and set up the correct model path and dataset path, then run:

```bash
bash scripts/rm_text_image_video_to_text.sh
```

### Model PPO

Add a script named `ppo_text_image_video_to_text.sh` under the `scripts` file like this:

```bash
# Initialize variables
ACTOR_MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-7B-Instruct"
# You can replace it with a local model path
CRITIC_MODEL_NAME_OR_PATH=""
REWARD_MODEL_NAME_OR_PATH=""
# You can replace it with a local dataset path
TRAIN_DATASETS=""
PTX_DATASETS=""
# You can replace it with a new path
OUTPUT_DIR=""
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_video_to_text.ppo \
	--actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
	--reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
	--reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--train_template TIV2T_preference \
	--train_data_files ${TRAIN_PT_NAME} \
	--train_split 'train' \
	--ptx_datasets ${PTX_DATASETS} \
	--ptx_template TIV2T \
    --ddp_timeout 1800000000 # this is a must for qwen2-vl training
```

and set up the correct model path and dataset path, then run:

```bash
bash scripts/ppo_text_image_video_to_text.sh
```