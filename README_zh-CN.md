<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="assets/logo.jpg" width="390"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">项目网站</font></b>
    <sup>
      <a href="https://space.bilibili.com/3493095748405551?spm_id_from=333.337.search-card.all.click">
        <i><font size="4">hot</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">北大对齐小组（PKU-Alignment Team）</font></b>
    <sup>
      <a href="https://space.bilibili.com/3493095748405551?spm_id_from=333.337.search-card.all.click">
        <i><font size="4">welcome</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/align-anything?logo=pypi)](https://pypi.org/project/align-anything)
[![License](https://img.shields.io/github/license/PKU-Alignment/align-anything?label=license)](#license)
<!-- TODO -->
<!-- [![CodeCov](https://img.shields.io/codecov/c/github/PKU-Alignment/omnisafe/main?logo=codecov)](https://app.codecov.io/gh/PKU-Alignment/omnisafe) -->

📘文档 |
[🚀功能](#功能) |
[🆕更新消息](#新闻) |
[🛠️安装](#安装) |
[👀训练](#训练) |
[🤔问题报告](#报告问题)
</div>

<div align="center">

[English](README.md) | 简体中文 ｜ [Our 100K Datasets](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K) | 👋 加入我们的[微信群](assets/wechat.jpg)

</div>

Align-Anything 是一个基于 DeepSpeed 或 NeMo （目前正在开发中）的开源对齐框架，旨在将各种模态的大模型（any to any模型），包括 LLM、VLM 等，与人类意图和价值观进行对齐。更多关于AI 系统（如LLMs\ MLLMs等）对齐的定义、关键技术以及其他相关信息，可在 [AI对齐综述网站](https://alignmentsurvey.com) 中找到。

### 功能

- 高度模块化的框架：我们的框架提供了一套全面的、多样的对齐算法集合，适用于不同模态模型的对齐。它的多功能性源于不同算法类型的抽象和精心设计的 API，使用户能够轻松修改和定制代码以适应不同任务。
- 支持各种模型微调：该框架包括了对如 LLaMA、LLaVA、Gemma、Qwen、Baichuan 等模型的微调功能（参见 [模型库](https://github.com/PKU-Alignment/align-anything/blob/main/Model_Zoo.md)）。
- 支持任何模态的对齐微调：它支持对不同模态模型，包括 LLM、VLM 和其他模态的微调对齐（参见 [开发路线图](#开发路线图)）。
- 支持多种对齐算法：该框架支持多种对齐算法，包括 SFT、DPO、PPO 等（参见 [示例](https://github.com/PKU-Alignment/align-anything/tree/main/examples)）。

#### 开发路线图

我们为未来的开发工作制定了 `Align-Anything` 的路线图：

- [ ] 支持在 `扩散模型`、`文本到任何模态的生成模型` 和其他 `视觉语言模型` 上的对齐算法。
- [ ] 支持包括 `LoRA`、`QLoRA` 在内的多种训练参数。
- [ ] 支持用于训练的 `NeMo` 框架，以及用于评估的 `vllm` 框架。

| 训练算法 | 文本 :arrow_right: 文本 | 文本+图像 :arrow_right: 文本 | 文本 :arrow_right: 图像 | 文本 :arrow_right: 视频 | 更多模态... |
|---|---|---|---|---|---|
| SFT Trainer | :white_check_mark: | :white_check_mark: | :airplane: | :car: | :car: |
| RM Trainer | :white_check_mark: | :white_check_mark: | :airplane: | :car: | :car: |
| DPO Trainer | :white_check_mark: | :white_check_mark: | :airplane: | :car: | :car: |
| PPO Trainer | :white_check_mark: | :white_check_mark: | :airplane: | :car: | :car: |
| KTO Trainer | :white_check_mark: | :car: | :car: | :car: | :car: |
| ORPO Trainer | :white_check_mark: | :car: | :car: | :car: | :car: |
| SimPO Trainer | :white_check_mark: | :car: | :car: | :car: | :car: |

- :white_check_mark: : 目前支持的功能。
- :airplane: : 正在内部测试的功能，将尽快被更新。
- :car: : 正在开发的功能。

# 新闻

- 2024-07-14 🎉我们开源了 `Align-Anything` 框架。

# 安装

所有模型权重、训练参数和分词器都存储在您事先指定的 `OUTPUT_DIR` 中。

```bash
conda create -n align-anything python==3.11
conda activate align-anything
git clone git@github.com:PKU-Alignment/align-anything.git
cd align-anything
pip install -e .
```

### Wandb 日志
我们支持 `wandb` 日志记录。默认情况下，设置为离线。如果您需要在线查看 wandb 日志，可以在开始训练前指定 `WANDB_API_KEY` 的环境变量：

```bash
export WANDB_API_KEY="..."  # your W&B API key here
```

### 从 Dockerfile 安装

<details>
<summary>如何从 Docker 构建？</summary>
1. 构建 docker 镜像

```bash
FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN echo "export PS1='[\[\e[1;33m\]\u\[\e[0m\]:\[\e[1;35m\]\w\[\e[0m\]]\$ '" >> ~/.bashrc

WORKDIR /root/align-anything
COPY . .

RUN python -m pip install --upgrade pip \
    && pip install -e .
```

然后,

```bash
docker build --tag align-anything .
```

2. 运行容器

```bash
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount type=bind,source=<host's mode path>,target=<docker's mode path> \
    test_docker
```

</details>

# 训练

## 快速开始

为了准备训练，所有脚本都位于 `./scripts` 目录下。需要用户输入的参数已被留空，必须在开始训练前填写。例如，对于 `ppo.sh`：

```bash
ACTOR_MODEL_NAME=""
REWARD_MODEL_NAME=""
CRITIC_MODEL_NAME=""
TRAIN_DATASETS=""
TRAIN_TEMPLATE=""
PTX_DATASET=""
PTX_TEMPLATE=""
OUTPUT_DIR=""

source ./setup.sh

deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.trainers.ppo \
  --actor_model_name_or_path ${ACTOR_MODEL_NAME} \
  --reward_model_name_or_path ${REWARD_MODEL_NAME} \
  --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME} \
  --train_datasets ${TRAIN_DATASETS} \
  --train_split train \
  --train_template ${TRAIN_TEMPLATE} \
  --ptx_datasets ${PTX_DATASET} \
  --ptx_split train \
  --ptx_template ${PTX_TEMPLATE} \
  --output_dir ${OUTPUT_DIR}
```

<!-- TODO -->
- `ACTOR_MODEL_NAME`: 要进行微调的模型，通常是已经经过初始监督微调的模型，如 `PKU-Alignment/alpaca-7b-reproduced`。
- `REWARD_MODEL_NAME`: 带有得分输出层的模型。运行 `rm.sh` 来训练奖励模型并获取其路径。
- `CRITIC_MODEL_NAME`: 用于 RLHF 值函数估计的模型，通常设置为与 `REWARD_MODEL_NAME` 相同。
- `TRAIN_DATASET`: RLHF 的训练数据集，如 `PKU-Alignment/PKU-SafeRLHF`。
- `TRAIN_TEMPLATE`: RLHF 的训练模板，如 `PKU-Alignment/PKU-SafeRLHF`。
- `PTX_DATASET`: 用于辅助 RLHF 微调的监督学习数据集，如 `tatsu-lab/alpaca`。
- `PTX_TEMPLATE`: 在 RLHF 中需要指定辅助监督学习数据集的模板，在这种情况下，它是 `Dialogue`。
- `OUTPUT_DIR`: 您希望保存训练模型、日志等的目录。

### 一些训练问题
1. 如果在训练过程中遇到错误：

为了包含 CUDA 安装路径并设置环境变量，请修改脚本如下：

```bash
export CUDA_HOME="/usr/local/cuda"
```
或者
```bash
export CUDA_HOME=$CONDA_PREFIX
```

具体取决于您的`cuda`安装路径。

## 自定义数据集

Align-Anything 提供了一个高度可扩展的数据集注册接口，允许用户通过设计和指定他们的 `template.py` 简单地嵌入自定义数据集。

以[PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)为例，我们在此展示如何设计template并将它整合进完整的RLHF工作流中。

PKU-Alignment/PKU-SafeRLHF数据的键值对如下：

```python
{
  'prompt': '...',
  'response_0': '...',
  'response_1': '...',
  'better_response_id': 0
}
```

### 模板创建

首先，我们需要创建一个名为PKUSafeRLHF的新模板，并指定所需参数，例如system_prompt。

```python
@register_template('PKUSafeRLHF')
class PKUSafeRLHF(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
```

### 奖励建模

奖励建模要求用户提供一个字典，数据键如下：

```python
{
  'better_text': '...',
  'worse_text': '...',
}
```

因此，用户需要在`align-anything/configs/template.py`中实现键值转换逻辑，例如，在这种情况下：

```python
@register_template('PKUSafeRLHF')
class PKUSafeRLHF(Dialogue):

    def format_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        metrics = raw_sample['better_response_id']
        better_response = raw_sample[f'response_{int(metrics)}']
        worse_response = raw_sample[f'response_{1-int(metrics)}']
        prompt = raw_sample['prompt']

        formatted_better_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'better_text': formatted_better_output,
            'worse_text': formatted_worse_output,
        }
```

在这里，`format_sample`解析PKU-Alignment/PKU-SafeRLHF数据集中的键，根据`better_response_id`确定哪个回应更好，并随后调用之前定义的参数，如`system_prompt`，来实现键值对的转换。

### 强化学习微调

在强化学习微调阶段，模型需要基于数据集中的提示生成输出。因此，用户需要在`template.py`中使用以下函数实现键值转换：

```python
@register_template('PKUSafeRLHF')
class PKUSafeRLHF(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'

    def format_prompt_only_sample(self, raw_sample: dict[str, Any]) -> dict[str, Any]:
        prompt = raw_sample['prompt']

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        return {'text': formatted_prompt}
```

## 为什么我们开源 Align-Anything？

确保 AI 系统的行为与人类意图和价值观一致至关重要，对齐技术提供了一个有效的解决方案。对于大语言模型（LLM），如RLHF和DPO等方法，已显著提高了性能和安全性。随着AI系统能力增强，模型将可以处理任何模态的输入和输出，如何有效地对齐多模态模型仍是当前的研究挑战。`Align-Anything` 框架通过精心设计的接口和高级抽象，整合了跨模态的对齐调整，为研究提供了一个全面的测试平台。

### 报告问题
如果在使用 Align-Anything 的过程中有任何问题，可以在 [GitHub 问题页面](https://github.com/PKU-Alignment/align-anything/issues/new/choose)上提出您的问题，我们将在 2-3 个工作日内回复您。

## 引用
如果您在研究中使用了 Align-Anything，请引用我们的工作：
```
@misc{align_anything,
  author = {PKU-Alignment Team},
  title = {Align Anything: Training Any Modality Model with Feedback},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PKU-Alignment/align-anything}},
}
```

## 证书

Align-Anything 在 Apache License 2.0 协议下发布.
