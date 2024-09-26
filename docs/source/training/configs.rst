Training Configurations
=======================

Align-Anything supports a diverse interface for hyperparameter settings.
We categorize these parameters into the following major categories:
``train_cfgs``, ``data_cfgs``, ``model_cfgs``, ``logger_cfgs``,
``lora_cfgs``, and ``bnb_cfgs``. Their meanings are as follows:

-  ``train_cfgs``: Configuration settings related to the training
   process, such as batch size, learning rate, number of epochs, etc.
-  ``data_cfgs``: Configuration settings related to data handling,
   including data paths, preprocessing options, augmentation methods,
   etc.
-  ``model_cfgs``: Configuration settings related to the model
   architecture, such as the type of model, layer configurations,
   activation functions, etc.
-  ``logger_cfgs``: Configuration settings related to logging, which
   includes the frequency of logging, log file paths, types of
   information to be logged, etc.
-  ``lora_cfgs``: Configuration settings specific to Low-Rank Adaptation
   (LoRA) techniques, which may include rank sizes, fine-tuning
   strategies, etc.
-  ``bnb_cfgs``: Configuration settings related to bits-and-bytes (or
   similar quantization schemes), which might involve precision levels,
   quantization methods, etc.

All training-related configurations are located under ``align_anything/configs/train``, and they are categorized according to different modalities. Here is a simple example:

.. code-block:: yaml

    # The training configurations
    train_cfgs:
        # The deepspeed configuration
        ds_cfgs: ds_z3_config.json
        # Number of training epochs
        epochs: 3
        # Seed for random number generator
        seed: 42
        ...
    # The datasets configurations
    data_cfgs:
        # Datasets to use for training
        train_datasets: null
        # The format template for training
        train_template: null
        ...
    # The logging configurations
    logger_cfgs:
        # Type of logging to use, choosing from [wandb, tensorboard]
        log_type: wandb
        # Project name for logging
        log_project: align-anything
        ...
    # The LoRA configurations
    lora_cfgs:
        # Whether to use LoRA
        use_lora: False
        # Task type for LoRA configuration
        task_type: TaskType.CAUSAL_LM
        ...
    # The QLoRA configurations
    bnb_cfgs:
        # Whether to use BNB(For QLoRA)
        use_bnb: False
        # Whether to use 4-bit quantization
        load_in_4bit: True
        ...

In addition, to facilitate your code debugging, we also support passing parameters via the command line. You just need to specify the specific parameter names. For example:

.. code-block:: bash

    MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path
    TRAIN_DATASETS="sqrti/SPA-VL" # dataset path
    TRAIN_TEMPLATE="SPA_VL" # dataset template
    TRAIN_SPLIT="train" # split the dataset
    OUTPUT_DIR="../output/dpo" # output dir
    export WANDB_API_KEY="YOUR_WANDB_KEY" # wandb logging

    source ./setup.sh # source the setup script

    export CUDA_HOME=$CONDA_PREFIX # replace it with your CUDA path

    deepspeed \
        --master_port ${MASTER_PORT} \
        --module align_anything.trainers.text_image_to_text.dpo \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_template ${TRAIN_TEMPLATE} \
        --train_split ${TRAIN_SPLIT} \
        --output_dir ${OUTPUT_DIR}

We have configured a set of default parameters for the user, which may not necessarily be optimal. If you wish to adjust them, you can check which hyperparameters can be adjusted and their default values at `here <https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs>`__.