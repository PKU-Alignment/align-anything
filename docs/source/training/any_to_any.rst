Any to Any Scripts
==================

We provide examples of various scripts for finetuning based on the `Emu3 <https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f>`__ model as follows. You can execute these commands
under the ``./scripts`` to start the corresponding training.

Supervised Fine-Tuning
----------------------

.. warning::

    The format of training data is quite different from the other modalities, as it differs generation and understanding tasks. See `htlou/Any_to_any_example <https://huggingface.co/datasets/htlou/Any_to_any_example>` for a small example, which contains example training data of both generation (marked as ``TG``) and understanding (marked as ``TU``) tasks. You could download the dataset and use it to debug, or you could also build your own dataset in this format.


.. code:: bash

   # Initialize variables
    MODEL_NAME_OR_PATH="BAAI/Emu3-Chat"
    PROCESSOR_NAME_OR_PATH="BAAI/Emu3-VisionTokenizer"
    TRAIN_DATASETS=""
    TRAIN_DATA_FILE=""
    OUTPUT_DIR="../outputs/any2any"

    # Source the setup script
    source ./setup.sh

    # Execute deepspeed command
    deepspeed \
        --master_port ${MASTER_PORT} \
        --module align_anything.trainers.any_to_any.sft \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --processor_name_or_path ${PROCESSOR_NAME_OR_PATH} \
        --train_datasets ${TRAIN_DATASETS} \
        --train_data_file ${TRAIN_DATA_FILE} \
        --train_template Any2Any \
        --train_split train \
        --output_dir ${OUTPUT_DIR}