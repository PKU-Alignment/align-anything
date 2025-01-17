MODEL_NAME_OR_PATH="openbmb/MiniCPM-o-2_6" # model path
export MODEL_NAME_OR_PATH

TRAIN_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K" # dataset path\
TRAIN_TEMPLATE="AA_TI2T" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../outputs/minicpmo_dpo" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_image_to_text.sft \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --trust_remote_code True \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 600 \
     --epochs 5
