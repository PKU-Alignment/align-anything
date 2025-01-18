MODEL_NAME_OR_PATH="openbmb/MiniCPM-o-2_6" # model path
export MODEL_NAME_OR_PATH

TRAIN_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K" # dataset path\
TRAIN_TEMPLATE="AA_TI2T" # dataset template
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../outputs/minicpmo_sft" # output dir

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
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 20000 \
     --epochs 5