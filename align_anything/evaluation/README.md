# Evaluation

## Quick Start

The evaluation script is `./scripts/evaluation.sh`. By modifying the `TASK` parameter of the script, you can select the evaluation set.

For example, to evaluate on the MMLU dataset:

```bash
TASK="mmlu"

source ./setup.sh

deepspeed \
  --master_port ${MASTER_PORT} \
  --module align_anything.evaluation.benchmarks \
  --task ${TASK}
```

- Supported options for `TASK` are `mmlu`, `gaokao`, `gsm8k`, `hellaswag`, `winogrande`, `mt_bench`, and `mme`

If you are using OPENAI for evaluation, you need to set the environment variables `OPENAI_API_BASE` and `OPENAI_API_KEY`

If you encounter errors during evaluating:
```bash
No such file or directory: ':/usr/local/cuda/bin/nvcc'
```


Please set the environment variables according to the `cuda` installation path on your device. For example:

```bash
export CUDA_HOME="/usr/local/cuda"
```

The specific path depends on your `cuda` installation location.

## Evaluation Data Preparation

We currently support the following evaluation datasets, some of which can be automatically downloaded via Huggingface, while others need to be manually downloaded from Github and saved to a local path.
- `mmlu`: [Automatically load from Huggingface](https://huggingface.co/datasets/cais/mmlu)
- `gaokao`: [Manually download from GitHub](https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/Data/Objective_Questions)(requires manual split into training and test sets)
- `gsm8k`: [Automatically load from Huggingface](https://huggingface.co/datasets/openai/gsm8k)
- `hellaswag`: [Automatically load from Huggingface](https://huggingface.co/datasets/Rowan/hellaswag)
- `winogrande`: [Manually download from GitHub](https://github.com/allenai/winogrande)
- `mt_bench`: [Manually download from GitHub](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data)
- `mme`: [Manually download from GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

## Development Roadmap

| Feature                                     |Status  |
| ------------------------------------------- | ------ |
| Local evaluation of language models         |✔️     |
| Online evaluation of language models        |✔️     |
| Local evaluation of image-language models   |✔️     |
| Online evaluation of image-language models  |✔️     |
| Local evaluation of video-language models   |❌     |
| Online evaluation of video-language models  |❌     |
| Local evaluation of audio-language models   |❌     |
| Evaluation of reward models                 |❌     |
| vllm accelerated inference                  |❌     |
| Evaluation of multiple API calls            |❌     |
| Local model comparison                      |❌     |
