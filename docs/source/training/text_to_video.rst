Text to Video Scripts
=====================

We provide examples of various scripts for fine-tuning based on the `ali-vilab/text-to-video-ms-1.7b <https://huggingface.co/ali-vilab/text-to-video-ms-1.7b>`__ model as follows. You can execute these commands in the ``./scripts`` to start the corresponding training.

Supervised Fine-Tuning
----------------------

.. note::

    In the original data of `Webvid-10M <https://huggingface.co/datasets/TempoFunk/webvid-10M>`__, videos exist in the form of URLs. The current version does not support online downloading of corresponding videos. Before training, you need to manually download the corresponding videos via URL and save them locally, organizing them in the following format.

    .. code:: json

        {
            "video_path": "<Your_video_path>",
            "caption": "<The_caption_corresponding_to_the_video>"
        }

.. code:: bash

   MODEL_NAME_OR_PATH="ali-vilab/text-to-video-ms-1.7b" # model path

   TRAIN_DATASETS="TempoFunk/webvid-10M" # dataset path
   TRAIN_TEMPLATE="Webvid" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/sft_diffusion" # output dir

   # For wandb online logging
   export WANDB_API_KEY=""

   # Source the setup script
   source ./setup.sh

    # Execute accelerate command
    accelerate launch  \
        --multi_gpu \
        --module align_anything.trainers.text_to_video.sft_diffusion \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}

DPO Training
------------

.. code:: bash

   MODEL_NAME_OR_PATH="ali-vilab/text-to-video-ms-1.7b" # model path

   TRAIN_DATASETS="PKU-Alignment/SafeSora" # dataset path
   TRAIN_TEMPLATE="SafeSora" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/dpo_diffusion" # output dir

   # For wandb online logging
   export WANDB_API_KEY=""

   # Source the setup script
   source ./setup.sh

    # Execute accelerate command
    accelerate launch  \
        --multi_gpu \
        --module align_anything.trainers.text_to_video.dpo_diffusion \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}