Welcome to align-anything
=========================

Welcome to `align-anything <https://github.com/PKU-Alignment/align-anything>`_ in AI alignment! align-anything aims to align any modality large models (any-to-any models), including LLMs, VLMs, and others, with human intentions and values. More details about the definition and milestones of alignment for Large Models can be found in `AI Alignment <https://alignmentsurvey.com>`_.

.. raw:: html

   <div style="text-align: center;">
     <img src="https://github.com/Gaiejj/align-anything-images/blob/main/docs/framework.png?raw=true" alt="Framework" style="width:100%; max-width:100%;">
   </div>


Align-Anything aims to align any modality large models (any-to-any
models), including LLMs, VLMs, and others, with human intentions and
values. More details about the definition and milestones of alignment
for Large Models can be found in `AI
Alignment <https://alignmentsurvey.com>`__. Overall, this framework has
the following characteristics:

-  **Highly Modular Framework.** Its versatility stems from the
   abstraction of different algorithm types and well-designed APIs,
   allowing users to easily modify and customize the code for different
   tasks.
-  **Support for Various Model Fine-Tuning.** This framework includes
   fine-tuning capabilities for models such as LLaMA3.1, LLaVA, Gemma,
   Qwen, Baichuan, and others (see `Model
   Zoo <https://github.com/PKU-Alignment/align-anything/blob/main/Model-Zoo.md>`__).
-  **Support Fine-Tuning across Any Modality.** It supports fine-tuning
   alignments for different modality model, including LLMs, VLMs, and
   other modalities (see `Development
   Roadmap <#development-roadmap>`__).
-  **Support Different Alignment Methods.** The framework supports
   different alignment algorithms, including SFT, DPO, PPO, and others.


Algorithms
----------

We support basic alignment algorithms for different modalities, each of
which may involve additional algorithms. For instance, in the text
modality, we have also implemented SimPO, KTO, and others.

==================================== === == === ===
Modality                             SFT RM DPO PPO
==================================== === == === ===
``Text -> Text (t2t)``               ✔️  ✔️ ✔️  ✔️
``Text+Image -> Text (ti2t)``        ✔️  ✔️ ✔️  ✔️
``Text+Image -> Text+Image (ti2ti)`` ✔️  ✔️ ✔️  ✔️
``Text -> Image (t2i)``              ✔️  ⚒️ ✔️  ⚒️
``Text -> Video (t2v)``              ✔️  ⚒️ ✔️  ⚒️
``Text -> Audio (t2a)``              ✔️  ⚒️ ✔️  ⚒️
==================================== === == === ===

Evaluation
----------

We support evaluation datasets for ``Text -> Text``,
``Text+Image -> Text`` and ``Text -> Image``.

+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Modality        | Supported Benchmarks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
+=================+==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| ``t2t``         | `ARC <https://huggingface.co/datasets/allenai/ai2_arc>`__, `BBH <https://huggingface.co/datasets/lukaemon/bbh>`__, `Belebele <https://huggingface.co/datasets/facebook/belebele>`__, `CMMLU <https://huggingface.co/datasets/haonan-li/cmmlu>`__, `GSM8K <https://huggingface.co/datasets/openai/gsm8k>`__, `HumanEval <https://huggingface.co/datasets/openai/openai_humaneval>`__, `MMLU <https://huggingface.co/datasets/cais/mmlu>`__, `MMLU-Pro <https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro>`__, `MT-Bench <https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts>`__, `PAWS-X <https://huggingface.co/datasets/google-research-datasets/paws-x>`__, `RACE <https://huggingface.co/datasets/ehovy/race>`__, `TruthfulQA <https://huggingface.co/datasets/truthfulqa/truthful_qa>`__                                                                                                                                                                                                                                        |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ti2t``        | `A-OKVQA <https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA>`__, `LLaVA-Bench(COCO) <https://huggingface.co/datasets/lmms-lab/llava-bench-coco>`__, `LLaVA-Bench(wild) <https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild>`__, `MathVista <https://huggingface.co/datasets/AI4Math/MathVista>`__, `MM-SafetyBench <https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench>`__, `MMBench <https://huggingface.co/datasets/lmms-lab/MMBench>`__, `MME <https://huggingface.co/datasets/lmms-lab/MME>`__, `MMMU <https://huggingface.co/datasets/MMMU/MMMU>`__, `MMStar <https://huggingface.co/datasets/Lin-Chen/MMStar>`__, `MMVet <https://huggingface.co/datasets/lmms-lab/MMVet>`__, `POPE <https://huggingface.co/datasets/lmms-lab/POPE>`__, `ScienceQA <https://huggingface.co/datasets/derek-thomas/ScienceQA>`__, `SPA-VL <https://huggingface.co/datasets/sqrti/SPA-VL>`__, `TextVQA <https://huggingface.co/datasets/lmms-lab/textvqa>`__, `VizWizVQA <https://huggingface.co/datasets/lmms-lab/VizWiz-VQA>`__ |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tv2t``        | `MVBench <https://huggingface.co/datasets/OpenGVLab/MVBench>`__, `Video-MME <https://huggingface.co/datasets/lmms-lab/Video-MME>`__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ta2t``        | `AIR-Bench <https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset>`__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``t2i``         | `ImageReward <https://huggingface.co/datasets/THUDM/ImageRewardDB>`__, `HPSv2 <https://huggingface.co/datasets/zhwang/HPDv2>`__, `COCO-30k(FID) <https://huggingface.co/datasets/sayakpaul/coco-30-val-2014>`__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``t2v``         | `ChronoMagic-Bench <https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench>`__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``t2a``         | `AudioCaps(FAD) <https://huggingface.co/datasets/AudioLLMs/audiocaps_test>`__                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
+-----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

-  ⚒️ : coming soon.

Installation
------------

.. code:: bash

   # clone the repository
   git clone git@github.com:PKU-Alignment/align-anything.git
   cd align-anything

   # create virtual env
   conda create -n align-anything python==3.11
   conda activate align-anything

-  ``[Optional]`` We recommend installing
   `CUDA <https://anaconda.org/nvidia/cuda>`__ in the conda environment
   and set the environment variable.

.. code:: bash

   # We tested on the H800 computing cluster, and this version of CUDA works well. 
   # You can adjust this version according to the actual situation of the computing cluster.

   conda install nvidia/label/cuda-12.2.0::cuda
   export CUDA_HOME=$CONDA_PREFIX

..

   If your CUDA installed in a different location, such as
   ``/usr/local/cuda/bin/nvcc``, you can set the environment variables
   as follows:

.. code:: bash

   export CUDA_HOME="/usr/local/cuda"

Fianlly, install ``align-anything`` by:

.. code:: bash

   pip install -e .

Wandb Logger
~~~~~~~~~~~~

We support ``wandb`` logging. By default, it is set to offline. If you
need to view wandb logs online, you can specify the environment
variables of ``WANDB_API_KEY`` before starting the training:

.. code:: bash

   export WANDB_API_KEY="..."  # your W&B API key here

Quick Start
-----------

Training Scripts
~~~~~~~~~~~~~~~~

To prepare for training, all scripts are located in the ``./scripts``
and parameters that require user input have been left empty. For
example, the DPO scripts for ``Text+Image -> Text`` modality is as
follow:

.. code:: bash

   MODEL_NAME_OR_PATH="" # model path
   TRAIN_DATASETS="" # dataset path
   TRAIN_TEMPLATE="" # dataset template
   TRAIN_SPLIT="" # split the dataset
   OUTPUT_DIR=""  # output dir

   source ./setup.sh # source the setup script

   export CUDA_HOME=$CONDA_PREFIX # replace it with your CUDA path

   deepspeed \
   	--master_port ${MASTER_PORT} \
   	--module align_anything.trainers.text_image_to_text.dpo \
   	--model_name_or_path ${MODEL_NAME_OR_PATH} \
   	--train_datasets ${TRAIN_DATASETS} \
   	--train_template SPA_VL \
   	--train_split train \
   	--output_dir ${OUTPUT_DIR}

We can run DPO with
`LLaVA-v1.5-7B <https://huggingface.co/llava-hf/llava-1.5-7b-hf>`__ (HF
format) and `SPA-VL <https://huggingface.co/datasets/sqrti/SPA-VL>`__
dataset using the follow script:

.. code:: bash

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

.. _evaluation-1:

Evaluation
~~~~~~~~~~

All evaluation scripts can be found in the ``./scripts``. The
``./scripts/evaluate.sh`` script runs model evaluation on the
benchmarks, and parameters that require user input have been left empty.
The corresponding script is as follow:

.. code:: bash

   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
   cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

   BENCHMARKS=("") # evaluation benchmarks
   OUTPUT_DIR="" # output dir
   GENERATION_BACKEND="" # generation backend
   MODEL_ID="" # model's unique id
   MODEL_NAME_OR_PATH="" # model path
   CHAT_TEMPLATE="" # model template

   for BENCHMARK in "${BENCHMARKS[@]}"; do
       python __main__.py \
           --benchmark ${BENCHMARK} \
           --output_dir ${OUTPUT_DIR} \
           --generation_backend ${GENERATION_BACKEND} \
           --model_id ${MODEL_ID} \
           --model_name_or_path ${MODEL_NAME_OR_PATH} \
           --chat_template ${CHAT_TEMPLATE}
   done

For example, you can evaluate
`LLaVA-v1.5-7B <https://huggingface.co/llava-hf/llava-1.5-7b-hf>`__ (HF
format) on `POPE <https://huggingface.co/datasets/lmms-lab/POPE>`__ and
`MM-SafetyBench <https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench>`__
benchmarks using the follow script:

.. code:: bash

   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
   cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

   BENCHMARKS=("POPE" "MM-SafetyBench") # evaluation benchmarks
   OUTPUT_DIR="../output/evaluation" # output dir
   GENERATION_BACKEND="vLLM" # generation backend
   MODEL_ID="llava-1.5-7b-hf" # model's unique id
   MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path
   CHAT_TEMPLATE="Llava" # model template

   for BENCHMARK in "${BENCHMARKS[@]}"; do
       python __main__.py \
           --benchmark ${BENCHMARK} \
           --output_dir ${OUTPUT_DIR} \
           --generation_backend ${GENERATION_BACKEND} \
           --model_id ${MODEL_ID} \
           --model_name_or_path ${MODEL_NAME_OR_PATH} \
           --chat_template ${CHAT_TEMPLATE}
   done

You can modify the configuration files for the benchmarks in `this
directory <https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/benchmarks>`__
to suit specific evaluation tasks and models, and adjust inference
parameters for
`vLLM <https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/vllm>`__
or
`DeepSpeed <https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/deepspeed>`__
based on your generation backend. For more details about the evaluation
pipeline, refer to the
`here <https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/evaluation/README.md>`__.

Inference
---------

Interactive Client
~~~~~~~~~~~~~~~~~~

.. code:: bash

   python3 -m align_anything.serve.cli --model_name_or_path your_model_name_or_path

Interactive Arena
~~~~~~~~~~~~~~~~~

.. code:: bash

   python3 -m align_anything.serve.arena \
       --red_corner_model_name_or_path your_red_model_name_or_path \
       --blue_corner_model_name_or_path your_blue_model_name_or_path

Why do we open source align-anything
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensuring that the behavior of AI system aligns with human intentions and values is crucial, and alignment techniques provide an effective solution. For large language models (LLMs), methods such as reinforcement learning with human feedback (RLHF) and direct preference optimization (DPO) have significantly improved performance and safety. As models evolve to handle any-modality inputs and outputs, effectively aligning them remains a current research challenge. align-anything framework integrates alignment tuning across modalities using well-designed interfaces and advanced abstractions, offering a comprehensive testbed for research.

Report Issues
~~~~~~~~~~~~~

If you have any questions in the process of using align-anything, don’t
hesitate to ask your questions on `the GitHub issue
page <https://github.com/PKU-Alignment/align-anything/issues/new/choose>`__,
we will reply to you in 2-3 working days.

Citation
--------

Please cite the repo if you use the data or code in this repo.

.. code:: bibtex

   @misc{align_anything,
     author = {PKU-Alignment Team},
     title = {Align Anything: training all modality models to follow instructions with unified language feedback},
     year = {2024},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/PKU-Alignment/align-anything}},
   }



.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Dataset

    data/text_to_text

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Training

    training/overview
    training/configs
    training/dataset_custom
    training/text_to_text
    training/text_to_image
    training/text_to_audio
    training/text_to_video
    training/text_image_to_text
    training/text_audio_to_text
    training/text_video_to_text
    training/any_to_any

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Tutorial

    tutorial/chameleon

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Evaluation

    evaluation/overview
    evaluation/example
    evaluation/evaluation_configurations