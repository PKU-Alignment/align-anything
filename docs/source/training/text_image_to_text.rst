Text Image to Text Scripts
==========================

We provide examples of various scripts for finetuning based on the
`PKU-Alignment/Align-Anything-TI2T-Instruction-100K <https://huggingface.co/datasets/PKU-Alignment/Align-Anything-TI2T-Instruction-100K>`__ and `PKU-Alignment/align-anything-400k <https://huggingface.co/datasets/PKU-Alignment/align-anything-400k>`__ series datasets as follows. You can execute these commands
under the ``./scripts`` to start the corresponding training.

Supervised Fine-Tuning
----------------------

.. code:: bash

   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

   TRAIN_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K" # dataset path
   TRAIN_TEMPLATE="AA_TI2T" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../outputs/sft_llava" # output dir

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
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR} \
        --save_interval 1200 \
        --epochs 3

Reward Model Training
---------------------

.. code:: bash

   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

   TRAIN_DATASETS="PKU-Alignment/align-anything-400k" # dataset path
   TRAIN_TEMPLATE="AA_TI2T" # dataset template
   TRAIN_NAME="text-image-to-text" # dataset name
   TRAIN_SPLIT="train" # split the dataset

   EVAL_DATASETS="PKU-Alignment/align-anything-400k" # dataset path
   EVAL_TEMPLATE="AA_TI2T" # dataset template
   EVAL_NAME="text-image-to-text" # dataset name
   EVAL_SPLIT="val" # split the dataset

   OUTPUT_DIR="../outputs/rm_llava" # output dir

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
        --train_template ${TRAIN_TEMPLATE} \
        --train_name ${TRAIN_NAME} \
        --train_split ${TRAIN_SPLIT} \
        --eval_datasets ${EVAL_DATASETS} \
        --eval_template ${EVAL_TEMPLATE} \
        --eval_name ${EVAL_NAME} \
        --eval_split ${EVAL_SPLIT} \
        --output_dir ${OUTPUT_DIR} \
        --save_interval 1200 \
        --epochs 5


DPO Training
------------

.. code:: bash

   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

   TRAIN_DATASETS="PKU-Alignment/align-anything-400k" # dataset path
   TRAIN_TEMPLATE="AA_TI2T" # dataset template
   TRAIN_NAME="text-image-to-text" # dataset name
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../outputs/dpo_llava" # output dir

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
        --train_template ${TRAIN_TEMPLATE} \
        --train_name ${TRAIN_NAME} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR} \
        --save_interval 1200 \
        --epochs 5

.. warning::

    PPO may encounter errors when calling the ``model.generate`` method due to an excessively high version of transformers. We are working on resolving this issue. For now, you can temporarily avoid the error by running ``pip install transformers==4.41.2``.

PPO Training
------------

.. code:: bash

    ACTOR_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path
    REWARD_MODEL_NAME_OR_PATH="../outputs/rm_llava" # model path
    CRITIC_MODEL_NAME_OR_PATH="../outputs/rm_llava" # model path

    TRAIN_DATASETS="PKU-Alignment/align-anything-400k" # dataset path
    TRAIN_TEMPLATE="AA_TI2T" # dataset template
    TRAIN_NAME="text-image-to-text" # dataset name
    TRAIN_SPLIT="train" # split the dataset

    PTX_DATASETS="PKU-Alignment/Align-Anything-TI2T-Instruction-100K"
    PTX_TEMPLATE="AA_TI2T"
    PTX_SPLIT="train"

    OUTPUT_DIR="../outputs/ppo_ti2t" # output dir
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
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --train_name ${TRAIN_NAME} \
        --ptx_datasets ${PTX_DATASETS} \
        --ptx_template ${PTX_TEMPLATE} \
        --ptx_split ${PTX_SPLIT} \
        --output_dir ${OUTPUT_DIR} \
        --save_interval 1200 \
        --epochs 5
