# Llama Training Scripts

We provide examples of various scripts for fine-tuning based on the [Llama](https://llama.meta.com/) series models (or corresponding SFT version, Alpaca-reproduced) as follows. You can execute these commands in the `./scripts` directory to start the corresponding training.

## Supervised Fine-Tuning

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B"
# You can replace it with a local dataset path
TRAIN_DATASETS="tatsu-lab/alpaca"
# You can replace it with a new path with correct permission
OUTPUT_DIR="../output/sft"
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
 --master_port ${MASTER_PORT} \
 --module align_anything.trainers.text_to_text.sft \
 --model_name_or_path ${MODEL_NAME_OR_PATH} \
 --train_datasets ${TRAIN_DATASETS} \
 --output_dir ${OUTPUT_DIR} \
 --train_template Dialogue \
 --train_split train
```

## DPO Training

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced"
# You can replace it with a local dataset path
TRAIN_DATASETS="PKU-Alignment/PKU-SafeRLHF"
# You can replace it with a new path with correct permission
OUTPUT_DIR="../output/dpo"
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.text_to_text.dpo \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
  	--train_template PKUSafeRLHF \
	--train_split train \
	--output_dir ${OUTPUT_DIR}
```

## PPO Training

```bash
# You can replace it with a local model path
ACTOR_MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced"
# You can replace it with a local model path
CRITIC_MODEL_NAME_OR_PATH="PATH_TO_YOUR_CRITIC_MODEL"
# You can replace it with a local model path
REWARD_MODEL_NAME_OR_PATH="PATH_TO_YOUR_REWARD_MODEL"
# You can replace it with a local dataset path
TRAIN_DATASETS="PKU-Alignment/PKU-SafeRLHF"
# You can replace it with a local dataset path
PTX_DATASETS="tatsu-lab/alpaca"
# You can replace it with a new path with correct permission
OUTPUT_DIR="../output/ppo"
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.text_to_text.ppo \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
  --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
  --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
  --train_datasets ${TRAIN_DATASETS} \
  --train_split train \
  --train_template PKUSafeRLHF \
  --ptx_split train \
  --ptx_datasets ${PTX_DATASETS} \
  --ptx_template Dialogue \
  --output_dir ${OUTPUT_DIR}
```
