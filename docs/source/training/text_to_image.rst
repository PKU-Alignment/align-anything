Text to Image Scripts
=====================

We provide examples of various scripts for fine-tuning based on the `Stable Diffusion <https://huggingface.co/CompVis/stable-diffusion-v1-4>`__ series models as follows. You can execute these commands in the ``./scripts`` to start the corresponding training.

Supervised Fine-Tuning
----------------------

.. note::

    The complete Diffusion `DB dataset <https://huggingface.co/datasets/poloclub/diffusiondb>`__ is very large, so we only provide a script with a small sample by passing ``--train_optional_args large_random_1k`` here. If you encounter any issues while training on the full dataset, please feel free to discuss them in the Issue section.

.. code:: bash

   MODEL_NAME_OR_PATH="CompVis/stable-diffusion-v1-4" # model path

   TRAIN_DATASETS="poloclub/diffusiondb" # dataset path
   TRAIN_TEMPLATE="DiffusionDB" # dataset template
   TRAIN_SPLIT="train" # split the dataset
   TRAIN_NAME="large_random_1k" # optional dataset arguments

   OUTPUT_DIR="../output/sft_diffusion" # output dir

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
        --train_template ${TRAIN_TEMPLATE} \
        --train_name ${TRAIN_NAME} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}

DPO Training
------------

.. note::

    We conducted code testing on a small subset of the `Pickapic <https://huggingface.co/datasets/yuvalkirstain/pickapic_v1>`__ dataset. However, since the full Pickapic dataset is too large, we chose a smaller version for users to run quickly here.

.. code:: bash

   MODEL_NAME_OR_PATH="CompVis/stable-diffusion-v1-4" # model path

   TRAIN_DATASETS="kashif/pickascore" # dataset path
   TRAIN_TEMPLATE="Pickapic" # dataset template
   TRAIN_SPLIT="validation" # split the dataset

   OUTPUT_DIR="../output/dpo_diffusion" # output dir

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
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}