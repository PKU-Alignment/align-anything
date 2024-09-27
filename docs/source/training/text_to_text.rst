Text to Text Scripts
====================

We provide examples of various scripts for finetuning based on the
`Llama <https://llama.meta.com/>`__ series models (or corresponding SFT
version, *i.e.*, `PKU-Alignment/alpaca-7b-reproduced <https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced>`__) as follows. You can execute these commands
under the ``./scripts`` to start the corresponding training.

.. note::

    For supervised finetuning and reward modeling, we always use pretrained model, *e.g.*, ``meta-llama/Meta-Llama-3-8B``

Supervised FineTuning
----------------------

.. code:: bash

   MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B" # model path

   TRAIN_DATASETS="tatsu-lab/alpaca" # dataset path
   TRAIN_TEMPLATE="Dialogue" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/sft" # output dir

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
    --train_template ${TRAIN_TEMPLATE} \
    --train_split ${TRAIN_SPLIT}

Reward Model Training
---------------------

.. code:: bash

   MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B" # model path

   TRAIN_DATASETS="PKU-Alignment/PKU-SafeRLHF" # dataset path
   TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/rm" # output dir

   # For wandb online logging
   export WANDB_API_KEY=""

   # Source the setup script
   source ./setup.sh

   # Execute deepspeed command
   deepspeed \
   	--master_port ${MASTER_PORT} \
   	--module align_anything.trainers.text_to_text.rm \
   	--model_name_or_path ${MODEL_NAME_OR_PATH} \
   	--train_datasets ${TRAIN_DATASETS} \
   	--train_template ${TRAIN_TEMPLATE} \
   	--train_split ${TRAIN_SPLIT} \
   	--output_dir ${OUTPUT_DIR}

.. note::

    For RLHF (DPO, PPO, ORPO, *etc.*) finetuning, we always use model has been supervised finetuned, *e.g.*, ``PKU-Alignment/alpaca-7b-reproduced``

DPO Training
------------

.. code:: bash

   MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced" # model path

   TRAIN_DATASETS="PKU-Alignment/PKU-SafeRLHF" # dataset path
   TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
   TRAIN_SPLIT="train" # split the dataset

   OUTPUT_DIR="../output/dpo" # output dir

   # For wandb online logging
   export WANDB_API_KEY=""

   # Source the setup script
   source ./setup.sh

   # Execute deepspeed command
   deepspeed \
   	--master_port ${MASTER_PORT} \
   	--module align_anything.trainers.text_to_text.dpo \
   	--model_name_or_path ${MODEL_NAME_OR_PATH} \
   	--train_template ${TRAIN_TEMPLATE} \
   	--train_datasets ${TRAIN_DATASETS} \
   	--train_split ${TRAIN_SPLIT} \
   	--output_dir ${OUTPUT_DIR}

.. warning::

    PPO may encounter errors when calling the ``model.generate`` method due to an excessively high version of transformers. We are working on resolving this issue. For now, you can temporarily avoid the error by running ``pip install transformers==4.41.2``.

PPO Training
------------

.. code:: bash

   ACTOR_MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced" # actor model path
   CRITIC_MODEL_NAME_OR_PATH="PKU-Alignment/beaver-7b-v1.0-reward" # critic model path
   REWARD_MODEL_NAME_OR_PATH="PKU-Alignment/beaver-7b-v1.0-reward" # reward model path
   
   TRAIN_DATASETS="PKU-Alignment/PKU-SafeRLHF" # rlhf dataset path
   TRAIN_TEMPLATE="PKUSafeRLHF" # rlhf dataset template
   TRAIN_SPLIT="train" # split the rlhf dataset

   PTX_DATASETS="tatsu-lab/alpaca" # sft dataset path
   PTX_TEMPLATE="Dialogue" # sft dataset template
   PTX_SPLIT="train" # split the sft dataset

   OUTPUT_DIR="../output/ppo" # output dir
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
     --train_split ${TRAIN_SPLIT} \
     --train_template ${TRAIN_TEMPLATE} \
     --ptx_split ${PTX_SPLIT} \
     --ptx_datasets ${PTX_DATASETS} \
     --ptx_template ${PTX_TEMPLATE} \
     --output_dir ${OUTPUT_DIR}