# 评测

## 快速开始

评测脚本为`./scripts/evaluation.sh`。通过修改脚本的`TASK`参数，选择评测集。

例如，如果要在评测集MMLU上进行评测：

```bash
TASK="mmlu"

source ./setup.sh

deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.evaluation.benchmarks \
  --task ${TASK}
```

- `TASK`支持的选项有`mmlu`, `gaokao`, `gsm8k`, `hellaswag`, `winogrande`, `mt_bench`, `mme`

如果使用OPENAI进行测评，需要设置环境变量`OPENAI_API_BASE`和`OPENAI_API_KEY`

如果您在评测过程中遭遇报错：
```bash
No such file or directory: ':/usr/local/cuda/bin/nvcc'
```

请结合设备的`cuda`安装地址，导入环境变量。例如：

```bash
export CUDA_HOME="/usr/local/cuda"
```

其中具体的路径取决于您的`cuda`安装地址

## 评测数据准备

我们当前支持的评测数据，部分可以通过Huggingface自动下载，部分需要在Github上手动下载保存到本地路径
- `mmlu`: [Huggingface自动加载](https://huggingface.co/datasets/cais/mmlu)
- `gaokao`: [github手动下载](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/Data/Objective_Questions)(需要手动划分训练集与测试集)
- `gsm8k`: [Huggingface自动加载](https://huggingface.co/datasets/openai/gsm8k)
- `hellaswag`: [Huggingface自动加载](https://huggingface.co/datasets/Rowan/hellaswag)
- `winogrande`: [github手动下载](https://github.com/allenai/winogrande)
- `mt_bench`: [github手动下载](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data)
- `mme`: [github手动下载](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

## 开发计划


| 特性                   |      实现情况     |
| ---------------------- | ------------------ |
| 语言模型本地测评        |✔️ |
| 语言模型在线测评        |✔️ |
| 图像-语言模型本地测评   |✔️ |
| 图像-语言模型在线测评   |✔️ |
| 视频-语言模型本地测评   |❌ |
| 视频-语言模型在线测评   |❌ |
| 音频-语言模型本地测评   |❌ |
| reward模型测评         |❌ |
| vllm加速推理           |❌ |
| 多种API调用测评        |❌ |
| 本地模型PK             |❌ |
