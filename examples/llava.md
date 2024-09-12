# LLAVA Training Scripts

We provide examples of various scripts for fine-tuning based on the [LLAVA](https://huggingface.co/llava-hf) series models as follows. You can execute these commands in the `./scripts` directory to start the corresponding training.

**Note:** The [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) dataset uses remote links to access images. However, due to the server hosting these images occasionally stopping its response, this can lead to unstable training. We strongly recommend that you download the [coco2017 dataset](http://images.cocodataset.org/zips/train2017.zip) locally here, then update the image paths in llava_instruct_150k.json to your local paths, and then use the `LLAVA_Local` template.

## Supervised Fine-Tuning

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
# You can replace it with a local dataset path
TRAIN_DATASETS="liuhaotian/LLaVA-Instruct-150K"
# You can replace it with a new path
OUTPUT_DIR="../output/sft"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_to_text.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--output_dir ${OUTPUT_DIR} \
  	--train_split train \
	--train_template LLAVA \
	--train_data_files llava_instruct_150k.json
```

## Reward Model Training

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
# You can replace it with a local dataset path
TRAIN_DATASETS="sqrti/SPA-VL"
# You can replace it with a local dataset path
EVAL_DATASETS="sqrti/SPA-VL"
# You can replace it with a new path with correct permission
OUTPUT_DIR="../output/rm"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_image_to_text.rm \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--eval_datasets ${EVAL_DATASETS} \
	--output_dir ${OUTPUT_DIR} \
  	--train_split train \
	--eval_split train \
	--train_template SPA_VL \
	--eval_template SPA_VL
```

## DPO Training

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
# You can replace it with a local dataset path
TRAIN_DATASETS="sqrti/SPA-VL"
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
	--train_template SPA_VL \
	--train_split train \
	--freeze_mm_proj True \
	--freeze_vision_tower False \
	--output_dir ${OUTPUT_DIR}
```

## PPO Training

```bash
# You can replace it with a local model path
ACTOR_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
# You can replace it with a local model path
CRITIC_MODEL_NAME_OR_PATH="PATH_TO_YOUR_CRITIC_MODEL"
# You can replace it with a local model path
REWARD_MODEL_NAME_OR_PATH="PATH_TO_YOUR_REWARD_MODEL"
# You can replace it with a local dataset path
TRAIN_DATASETS="sqrti/SPA-VL"
# You can replace it with a local dataset path
PTX_DATASETS="liuhaotian/LLaVA-Instruct-150K"
# You can replace it with a new path with correct permission
OUTPUT_DIR="../output/ppo"
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.text_image_to_text.ppo \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
  --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
  --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
  --train_datasets ${TRAIN_DATASETS} \
  --train_split train \
  --ptx_split train \
  --train_template SPA_VL \
  --ptx_datasets ${PTX_DATASETS} \
  --ptx_template LLAVA \
  --ptx_data_files llava_instruct_150k.json \
  --freeze_mm_proj True \
  --freeze_vision_tower False \
  --output_dir ${OUTPUT_DIR}
```
