MODEL_NAME_OR_PATH="openbmb/MiniCPM-V" # model path

TRAIN_DATASETS="PKU-Alignment/align-anything" # dataset path
TRAIN_TEMPLATE="AA_TI2T" # dataset template
TRAIN_NAME="text-image-to-text" # dataset name
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../outputs/minicpmv_dpo" # output dir

# Important: set the zero stage file smaller than 3
# since the model does not support ZeRO stage 3
# sees: https://github.com/OpenBMB/MiniCPM-V/issues/458
export ZERO_STAGE_FILE="ds_z2_config.json"
# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_image_to_text.dpo \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --trust_remote_code True \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_name ${TRAIN_NAME} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 600 \
     --epochs 5
