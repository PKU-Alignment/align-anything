<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="assets/logo.jpg" width="390"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">project website</font></b>
    <sup>
      <a href="https://space.bilibili.com/3493095748405551?spm_id_from=333.337.search-card.all.click">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">PKU-Alignment Team</font></b>
    <sup>
      <a href="https://space.bilibili.com/3493095748405551?spm_id_from=333.337.search-card.all.click">
        <i><font size="4">welcome</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/align-anything?logo=pypi)](https://pypi.org/project/align-anything)
[![License](https://img.shields.io/github/license/PKU-Alignment/align-anything?label=license)](#license)

[📘Documentation](https://pku-alignment.notion.site/Align-Anything-37a300fb5f774bb08e5b21fdeb476c64) |
[🆕Update News](#news) |
[🛠️Quick Start](#quick-start) |
[🚀Algorithms](#algorithms) |
[👀Evaluation](#evaluation) |
[🤔Reporting Issues](#report-issues)
</div>

<div align="center">

[Our 100K Instruction-Following Datasets](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K)

</div>

Align-Anything aims to align any modality large models (any-to-any models), including LLMs, VLMs, and others, with human intentions and values. More details about the definition and milestones of alignment for Large Models can be found in [AI Alignment](https://alignmentsurvey.com). Overall, this framework has the following characteristics:

- **Highly Modular Framework.** Its versatility stems from the abstraction of different algorithm types and well-designed APIs, allowing users to easily modify and customize the code for different tasks.
- **Support for Various Model Fine-Tuning.** This framework includes fine-tuning capabilities for models such as LLaMA3.1, LLaVA, Gemma, Qwen, Baichuan, and others (see [Model Zoo](https://github.com/PKU-Alignment/align-anything/blob/main/Model-Zoo.md)).
- **Support Fine-Tuning across Any Modality.** It supports fine-tuning alignments for different modality model, including LLMs, VLMs, and other modalities (see [Development Roadmap](#development-roadmap)).
- **Support Different Alignment Methods.** The framework supports different alignment algorithms, including SFT, DPO, PPO, and others.


|| <details><summary>prompt</summary>Small white toilet sitting in a small corner next to a wall.</details> | <details><summary>prompt</summary>A close up of a neatly made bed with two night stands</details>  | <details><summary>prompt</summary>A pizza is sitting on a plate at a restaurant.</details> |<details><summary>prompt</summary>A girl in a dress next to a piece of luggage and flowers.</details>|
|---| ---------------------------------- | --- | --- | --- |
|Before Alignment ([Chameleon-7B](https://huggingface.co/facebook/chameleon-7b))| <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/before/1.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;"> | <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/before/2.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;"> | <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/before/3.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;">  | <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/before/4.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;">|
|**After Alignment ([Chameleon 7B Plus](https://huggingface.co/PKU-Alignment/AA-chameleon-7b-plus))**| <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/after/1.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;"> | <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/after/2.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;"> | <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/after/3.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;">  | <img src="https://github.com/Gaiejj/align-anything-images/blob/main/chameleon/after/4.png?raw=true" alt="Image 8" style="max-width: 100%; height: auto;">|

> Alignment fine-tuning can significantly enhance the instruction-following capabilities of large multimodal models. After fine-tuning, Chameleon 7B Plus generates images that are more relevant to the prompt.

## Algorithms
We support basic alignment algorithms for different modalities, each of which may involve additional algorithms. For instance, in the text modality, we have also implemented SimPO, KTO, and others.

| Modality                           | SFT | RM  | DPO | PPO |
| ---------------------------------- | --- | --- | --- | --- |
| `Text -> Text (t2t)`               | ✔️   | ✔️   | ✔️   | ✔️   |
| `Text+Image -> Text (ti2t)`        | ✔️   | ✔️   | ✔️   | ✔️   |
| `Text+Image -> Text+Image (ti2ti)` | ✔️   | ✔️   | ✔️   | ✔️   |
| `Text+Audio -> Text (ta2t)`        | ✔️   | ✔️   | ✔️   | ✔️   |
| `Text+Video -> Text (tv2t)`        | ✔️   | ✔️   | ✔️   | ✔️   |
| `Text -> Image (t2i)`              | ✔️   | ⚒️   | ✔️   | ⚒️   |
| `Text -> Video (t2v)`              | ✔️   | ⚒️   | ✔️   | ⚒️   |
| `Text -> Audio (t2a)`              | ✔️   | ⚒️   | ✔️   | ⚒️   |

## Evaluation
We support evaluation datasets for `Text -> Text`, `Text+Image -> Text` and `Text -> Image`.

| Modality              | Supported Benchmarks                                                  |
| :-------------------- | :----------------------------------------------------------- |
| `t2t`       | [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBH](https://huggingface.co/datasets/lukaemon/bbh), [Belebele](https://huggingface.co/datasets/facebook/belebele), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x), [RACE](https://huggingface.co/datasets/ehovy/race), [TruthfulQA ](https://huggingface.co/datasets/truthfulqa/truthful_qa) |
| `ti2t` | [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA), [LLaVA-Bench(COCO)](https://huggingface.co/datasets/lmms-lab/llava-bench-coco), [LLaVA-Bench(wild)](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild), [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench), [MMBench](https://huggingface.co/datasets/lmms-lab/MMBench), [MME](https://huggingface.co/datasets/lmms-lab/MME), [MMMU](https://huggingface.co/datasets/MMMU/MMMU), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), [MMVet](https://huggingface.co/datasets/lmms-lab/MMVet), [POPE](https://huggingface.co/datasets/lmms-lab/POPE), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL), [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa), [VizWizVQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA) |
|`tv2t` |[MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) |
|`ta2t` |[AIR-Bench](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) |
| `t2i`      | [ImageReward](https://huggingface.co/datasets/THUDM/ImageRewardDB), [HPSv2](https://huggingface.co/datasets/zhwang/HPDv2), [COCO-30k(FID)](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014) |
| `t2v`      | [ChronoMagic-Bench](https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench) |
| `t2a`      | [AudioCaps(FAD)](https://huggingface.co/datasets/AudioLLMs/audiocaps_test) |

- ⚒️ : coming soon.

# News

- 2024-11-20: We release a bunch of scripts for all-modality models [here](./scripts). You can directly run the scripts to fine-tune your models, without any need to modify the code.
- 2024-10-10: We support SFT for `Any -> Any` modality models Emu3.
- 2024-09-24: We support SFT, DPO, RM and PPO for `Text + Video -> Text` modality models.
- 2024-09-13: We support SFT, DPO, RM and PPO for `Text + Audio -> Text` modality models.
- 2024-08-17: We support DPO and PPO for `Text+Image -> Text+Image` modality models.
- 2024-08-15: We support a new function in the evaluation module: the `models_pk` script in [here](./scripts/models_pk.sh), which enables comparing the performance of two models across different benchmarks.
- 2024-08-06: We restructure the framework to support any modality evaluation and the supported benchmark list is [here](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/evaluation/benchmarks).
- 2024-08-06: We support `Text+Image -> Text+Image` modality for the SFT trainer and Chameleon models.
<details><summary>More News</summary>

- 2024-07-23: We support `Text -> Image`, `Text -> Audio`, and `Text -> Video` modalities for the SFT trainer and DPO trainer.
- 2024-07-22: We support the **Chameleon** model for the SFT trainer and DPO trainer!
- 2024-07-17: We open-source the Align-Anything-Instruction-100K dataset for text modality. This dataset is available in both [English](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K) and [Chinese](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K-zh) versions, each sourced from different data sets and meticulously refined for quality by GPT-4.
- 2024-07-14: We open-source the align-anything framework.

</details>

# Installation


```bash
# clone the repository
git clone git@github.com:PKU-Alignment/align-anything.git
cd align-anything

# create virtual env
conda create -n align-anything python==3.11
conda activate align-anything
```

- **`[Optional]`** We recommend installing [CUDA](https://anaconda.org/nvidia/cuda) in the conda environment and set the environment variable.

```bash
# We tested on the H800 computing cluster, and this version of CUDA works well. 
# You can adjust this version according to the actual situation of the computing cluster.

conda install nvidia/label/cuda-12.2.0::cuda
export CUDA_HOME=$CONDA_PREFIX
```

> If your CUDA installed in a different location, such as `/usr/local/cuda/bin/nvcc`, you can set the environment variables as follows:

```bash
export CUDA_HOME="/usr/local/cuda"
```

Fianlly, install `align-anything` by:

```bash
pip install -e .
```

## Wandb Logger

We support `wandb` logging. By default, it is set to offline. If you need to view wandb logs online, you can specify the environment variables of `WANDB_API_KEY` before starting the training:

```bash
export WANDB_API_KEY="..."  # your W&B API key here
```

<!-- ## Install from Dockerfile

1. build docker image


```bash
FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN echo "export PS1='[\[\e[1;33m\]\u\[\e[0m\]:\[\e[1;35m\]\w\[\e[0m\]]\$ '" >> ~/.bashrc

WORKDIR /root/align-anything
COPY . .

RUN python -m pip install --upgrade pip \
    && pip install -e .
```

then,

```bash
docker build --tag align-anything .
```

2. run the container

```bash
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount type=bind,source=<host's mode path>,target=<docker's mode path> \
    align-anything
``` -->


# Quick Start

## Training Scripts

To prepare for training, all scripts are located in the `./scripts` and parameters that require user input have been left empty. For example, the DPO scripts for `Text + Image -> Text` modality is as follow:

```bash
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
```

We can run DPO with [LLaVA-v1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) (HF format) and [Align-Anything-400K](https://huggingface.co/datasets/PKU-Alignment/align-anything-400k) dataset using the follow script:

```bash
MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path
TRAIN_DATASETS="PKU-Alignment/align-anything-400k" # dataset path
TRAIN_TEMPLATE="AA_TI2T" # dataset template
TRAIN_NAME="text-image-to-text" # dataset name
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
	--train_name ${TRAIN_NAME} \
	--train_split ${TRAIN_SPLIT} \
	--output_dir ${OUTPUT_DIR}
```

## Evaluation

All evaluation scripts can be found in the `./scripts`. The `./scripts/evaluate.sh` script runs model evaluation on the benchmarks, and parameters that require user input have been left empty. The corresponding script is as follow:

```bash
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
```

For example, you can evaluate [LLaVA-v1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) (HF format) on [POPE](https://huggingface.co/datasets/lmms-lab/POPE) and [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench) benchmarks using the follow script:

```bash
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
```

You can modify the configuration files for the benchmarks in [this directory](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/benchmarks) to suit specific evaluation tasks and models, and adjust inference parameters for [vLLM](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/vllm) or [DeepSpeed](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/deepspeed) based on your generation backend. For more details about the evaluation pipeline, refer to the [here](https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/evaluation/README.md).

# Inference

## Interactive Client

```bash
python3 -m align_anything.serve.cli --model_name_or_path your_model_name_or_path
```

<img src="assets/cli_demo.gif" alt="cli_demo" style="width:600px;">


## Interactive Arena

```bash
python3 -m align_anything.serve.arena \
    --red_corner_model_name_or_path your_red_model_name_or_path \
    --blue_corner_model_name_or_path your_blue_model_name_or_path
```

<img src="assets/arena_demo.gif" alt="arena_demo" style="width:600px;">

## Report Issues

If you have any questions in the process of using align-anything, don't hesitate to ask your questions on [the GitHub issue page](https://github.com/PKU-Alignment/align-anything/issues/new/choose), we will reply to you in 2-3 working days.


# Citation

Please cite the repo if you use the data or code in this repo.

```bibtex
@misc{align_anything,
  author = {PKU-Alignment Team},
  title = {Align Anything: training all modality models to follow instructions with unified language feedback},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PKU-Alignment/align-anything}},
}
```

# License

align-anything is released under Apache License 2.0.