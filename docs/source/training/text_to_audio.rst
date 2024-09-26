Text to Audio Scripts
=====================

We provide examples of various scripts for fine-tuning based on the `cvssp/audioldm-s-full-v2 <https://huggingface.co/cvssp/audioldm-s-full-v2>`__ series models as follows. You can execute these commands in the ``./scripts`` to start the corresponding training.

Supervised Fine-Tuning
----------------------

.. note::

    WavCaps cannot be used directly after downloading and needs to be unpacked and organized according to the `Hugging Face tutorial <https://huggingface.co/datasets/cvssp/WavCaps>`__. The organization format can refer to `wavcaps_test <https://huggingface.co/datasets/AudioLLMs/wavcaps_test>`__

.. code:: bash

   MODEL_NAME_OR_PATH="cvssp/audioldm-s-full-v2" # model path

   TRAIN_DATASETS="cvssp/WavCaps" # dataset path
   TRAIN_TEMPLATE="WavCaps" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/sft_diffusion" # output dir

   # For wandb online logging
   export WANDB_API_KEY=""

   # Source the setup script
   source ./setup.sh

    # Execute accelerate command
    accelerate launch  \
        --multi_gpu \
        --module align_anything.trainers.text_to_audio.sft_diffusion \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}

DPO Training
------------

.. note::

    The SOMOS dataset is not available on Hugging Face currently. You need to download MOS data from `Zenodo <https://zenodo.org/records/7378801>`__ and then obtain audio files and their corresponding captions based on the content in the SOMOS dataset. Please organize all files in the following format.

    .. code:: json

        {
            "better_data_path": "<Your_audio_path>",
            "worse_data_path": "<Your_audio_path>",
            "prompt": "<The_caption_corresponding_to_the_audios>"
        }


.. code:: bash

   MODEL_NAME_OR_PATH="cvssp/audioldm-s-full-v2" # model path

   TRAIN_DATASETS="<Your_local_SOMOS_path>" # dataset path
   TRAIN_TEMPLATE="SOMOS" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/dpo_diffusion" # output dir

   # For wandb online logging
   export WANDB_API_KEY=""

   # Source the setup script
   source ./setup.sh

    # Execute accelerate command
    accelerate launch  \
        --multi_gpu \
        --module align_anything.trainers.text_to_audio.dpo_diffusion \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}