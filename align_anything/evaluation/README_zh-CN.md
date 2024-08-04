# 评估

## 开发路线图

我们目前支持两种主要的评估数据集类型：**"文本 ➡️ 文本"** 和 **"图像+文本 ➡️ 文本"**。在 **"文本 ➡️ 文本"** 模态下，我们已经支持了12个基准数据集；在 **"图像+文本 ➡️ 文本"** 模态下，目前支持了 MME 一个基准数据集。我们正在积极扩展对更多多模态基准测试的支持。

然后，我们为未来的开发工作制定了 `Align-Anything` 的路线图：

| 模态                 | 目前支持的数据集                                             |
| :------------------- | :----------------------------------------------------------- |
| **文本 ➡️ 文本**      | [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBH](https://huggingface.co/datasets/lukaemon/bbh), [Belebele](https://huggingface.co/datasets/facebook/belebele), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x), [RACE](https://huggingface.co/datasets/ehovy/race), [TruthfulQA ](https://huggingface.co/datasets/truthfulqa/truthful_qa), ➖ |
| **图像+文本 ➡️ 文本** | [MME](https://huggingface.co/datasets/lmms-lab/MME), ➖       |
| **文本 ➡️ 图像**      | ➖                                                            |
| **文本 ➡️ 视频**      | ➖                                                            |
| **文本 ➡️ 语音**      | ➖                                                            |

- ➖ : 正在内部测试的功能，将尽快被更新。

## 快速开始

评估启动脚本位于 `./scripts` 目录下。需要用户输入的参数已留空，必须在启动评估过程之前填写。例如，对于 `evaluate.sh`：

~~~bash
cd ../align_anything/evaluation

BENCHMARK=""
OUTPUT_DIR=""
GENERATION_BACKEND=""

python __main__.py \
    --benchmark ${BENCHMARK} \
    --output_dir ${OUTPUT_DIR} \
    --generation_backend ${GENERATION_BACKEND}
~~~

- `BENCHMARK`: 评估模型性能的基准或数据集。例如，使用 `ARC` 表示 ARC 数据集或其他相关基准。
- `OUTPUT_DIR`: 用于保存评估结果和输出文件的目录。
- `GENERATION_BACKEND`: 进行大语言模型推理的框架，包括`deepspeed` 和 `vLLM` 。

此外，你还应修改 `./align_anything/configs/evaluation/benchmarks` 下与基准测试对应的配置文件，以适应特定的评估任务，并指定测试模型。

例如，对于 `arc.yaml`：

~~~yaml
infer_cfgs:
  # The deepspeed configuration
  ds_cfgs: ds_z3_config.json
  vllm_cfgs: vllm_basic.json
  
default:
  # Evaluation configurations
  eval_cfgs:
    # Output directory name
    output_dir: null
    # Num shot
    n_shot: 5
    # Chain of Thought
    cot: false
  # Configuration for data
  data_cfgs:
    # Task name
    task: [ARC-Challenge, ARC-Easy]
    # Task directory
    task_dir: allenai/ai2_arc
    # Evaluation split
    split: test

  # Model configurations
  model_cfgs:
    model_id: Meta-Llama-3-8B-Instruct
    # Pretrained model name or path
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    # Chat template
    chat_template: Llama3
    # The max token length
    model_max_length: 2048
~~~

如果想修改更多推理参数，请查看 `./align_anything/configs/evaluation/vllm` 和 `./align_anything/configs/evaluation/deepspeed`，具体取决于你选择的推理框架。


### 整合自定义模板

要使用自定义模板，你需要在 `./align_anything/configs/template.py` 中实现并注册该模板。例如，对于 `PKUSafeRLHF` 模板：

~~~python
@register_template('PKUSafeRLHF')
class PKUSafeRLHF(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
~~~

在定义了自定义模板后，通过将 YAML 配置文件中的 `chat_template` 参数设置为模板名称（例如 `PKUSafeRLHF`）来指定它。