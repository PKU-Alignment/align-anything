Text Image to Text Scripts
==========================

We provide examples of various scripts for finetuning based on the
`LLaVA <https://huggingface.co/llava-hf>`__ series models as follows. You can execute these commands
under the ``./scripts`` to start the corresponding training.

Supervised Fine-Tuning
----------------------

.. warning::

    The `LLaVA-Instruct-150K <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>`__ dataset uses remote links to access images on `coco 2017 <https://cocodataset.org/#home>`__. However, due to the server hosting these images occasionally stopping its response, this can lead to unstable training. We strongly recommend that you download the `coco2017 dataset <http://images.cocodataset.org/zips/train2017.zip>`__ locally here, then update the image paths in ``llava_instruct_150k.json`` to your local paths, and then use the ``Llava_Local`` template.

.. code:: bash

   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

   TRAIN_DATASETS="liuhaotian/LLaVA-Instruct-150K" # dataset path
   TRAIN_TEMPLATE="Llava" # dataset template
   TRAIN_SPLIT="train" # split the dataset
   TRAIN_DATA_FILES="llava_instruct_150k.json" # specify the training data files

   OUTPUT_DIR="../output/sft" # output dir

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
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --train_data_files ${TRAIN_DATA_FILES}

Reward Model Training
---------------------

.. code:: bash

   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

   TRAIN_DATASETS="sqrti/SPA-VL" # dataset path
   TRAIN_TEMPLATE="SPA_VL" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/rm" # output dir

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
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}

DPO Training
------------

.. code:: bash

   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

   TRAIN_DATASETS="sqrti/SPA-VL" # dataset path
   TRAIN_TEMPLATE="SPA_VL" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/dpo" # output dir

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
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}

.. warning::

    PPO may encounter errors when calling the ``model.generate`` method due to an excessively high version of transformers. We are working on resolving this issue. For now, you can temporarily avoid the error by running ``pip install transformers==4.41.2``.

PPO Training
------------

.. code:: bash

   ACTOR_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # actor model path
   CRITIC_MODEL_NAME_OR_PATH="PATH_TO_YOUR_REWARD_MODEL" # critic model path
   REWARD_MODEL_NAME_OR_PATH="PATH_TO_YOUR_REWARD_MODEL" # reward model path
   
   TRAIN_DATASETS="sqrti/SPA-VL" # rlhf dataset path
   TRAIN_TEMPLATE="SPA_VL" # rlhf dataset template
   TRAIN_SPLIT="train" # split the rlhf dataset

   OUTPUT_DIR="../output/ppo" # output dir
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
     --train_split ${TRAIN_SPLIT} \
     --train_template ${TRAIN_TEMPLATE} \
     --output_dir ${OUTPUT_DIR}