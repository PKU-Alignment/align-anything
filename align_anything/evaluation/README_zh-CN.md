# 评估

## 开发路线图

我们目前支持两种主要的评估数据集类型：**"文本 ➡️ 文本"** 和 **"图像+文本 ➡️ 文本"**。在 **"文本 ➡️ 文本"** 模态下，我们已经支持了12个基准数据集；在 **"图像+文本 ➡️ 文本"** 模态下，我们已经支持了15个基准数据集。我们正在积极扩展对更多多模态基准测试的支持。

然后，我们为未来的开发工作制定了 `Align-Anything` 的路线图：

| 模态                 | 目前支持的数据集                                             |
| :------------------- | :----------------------------------------------------------- |
| **文本 ➡️ 文本**      | [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBH](https://huggingface.co/datasets/lukaemon/bbh), [Belebele](https://huggingface.co/datasets/facebook/belebele), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x), [RACE](https://huggingface.co/datasets/ehovy/race), [TruthfulQA ](https://huggingface.co/datasets/truthfulqa/truthful_qa), ➖ |
| **图像+文本 ➡️ 文本** | [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA), [LLaVA-Bench(COCO)](https://huggingface.co/datasets/lmms-lab/llava-bench-coco), [LLaVA-Bench(wild)](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild), [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench), [MMBench](https://huggingface.co/datasets/lmms-lab/MMBench), [MME](https://huggingface.co/datasets/lmms-lab/MME), [MMMU](https://huggingface.co/datasets/MMMU/MMMU), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), [MMVet](https://huggingface.co/datasets/lmms-lab/MMVet), [POPE](https://huggingface.co/datasets/lmms-lab/POPE), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL), [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa), [VizWizVQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA), ➖ |
| **文本 ➡️ 图像**      | ➖                                                            |
| **文本 ➡️ 视频**      | ➖                                                            |
| **文本 ➡️ 语音**      | ➖                                                            |

- ➖ : 正在内部测试的功能，将尽快被更新。

## 快速开始

评估启动脚本位于 `./scripts` 目录下。需要用户输入的参数已留空，必须在启动评估过程之前填写。例如，对于 `evaluate.sh`：

~~~bash
cd ../align_anything/evaluation

BENCHMARKS=("")
OUTPUT_DIR=""
GENERATION_BACKEND=""
MODEL_ID=""
MODEL_NAME_OR_PATH=""
CHAT_TEMPLATE=""

for BENCHMARK in "${BENCHMARKS[@]}"; do
    python __main__.py \
        --benchmark ${BENCHMARK} \
        --output_dir ${OUTPUT_DIR} \
        --generation_backend ${GENERATION_BACKEND} \
        --model_id ${MODEL_ID} \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --chat_template ${CHAT_TEMPLATE}
done
~~~

- `BENCHMARKS`: 用于评估模型性能的一个或多个评估基准或数据集。例如，`("POPE" "MMBench")` 可用于在 POPE 和 MMBench 数据集上评估模型。列表中的每个基准将按顺序处理。
- `OUTPUT_DIR`: 用于保存评估结果和输出文件的目录。
- `GENERATION_BACKEND`: 进行大语言模型推理的框架，包括 `vLLM` 和 `deepspeed`。
- `MODEL_ID`: 模型的唯一标识符，用于跟踪和区分模型评估，如 `llava-1.5-7b-hf`。
- `MODEL_NAME_OR_PATH`: 模型的本地路径或 Hugging Face 链接，如 `llava-hf/llava-1.5-7b-hf` 。
- `CHAT_TEMPLATE`: 模型的聊天模板 id，如 `LLAVA`。更多细节可以参考 `./align_anything/configs/template.py`。

为了在一个或多个基准测试中比较多个模型的性能，目录 `./scripts` 中的 `models_pk.sh` 脚本允许您评估不同的模型，然后比较它们的结果。在运行脚本之前，请确保正确填写了所有参数。

~~~bash
cd ../align_anything/evaluation

BENCHMARKS=("")
OUTPUT_DIR=""
GENERATION_BACKEND=""
MODEL_IDS=("" "")
MODEL_NAME_OR_PATHS=("" "")
CHAT_TEMPLATES=("" "")

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "Processing benchmark: ${BENCHMARK}"
    
    for i in "${!MODEL_IDS[@]}"; do
        MODEL_ID=${MODEL_IDS[$i]}
        MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATHS[$i]}
        CHAT_TEMPLATE=${CHAT_TEMPLATES[$i]}
        
        echo "Running model ${MODEL_ID} for benchmark ${BENCHMARK}"
        python __main__.py \
            --benchmark ${BENCHMARK} \
            --output_dir ${OUTPUT_DIR} \
            --generation_backend ${GENERATION_BACKEND} \
            --model_id ${MODEL_ID} \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --chat_template ${CHAT_TEMPLATE}
    done

    python models_pk.py --benchmark ${BENCHMARK} \
                        --model_1 "${MODEL_IDS[0]}" \
                        --model_2 "${MODEL_IDS[1]}"
done
~~~

- `BENCHMARKS`: 用于评估模型性能的一个或多个评估基准或数据集。例如，`("POPE" "MMBench")` 可用于在 POPE 和 MMBench 数据集上评估模型。列表中的每个基准将按顺序处理。
- `OUTPUT_DIR`: 用于保存评估结果和输出文件的目录。
- `GENERATION_BACKEND`: 进行大语言模型推理的框架，包括 `vLLM` 和 `deepspeed`。
- `MODEL_IDS`: 正在评估的模型的两个唯一标识符的数组，例如 `("llava-1.5-7b-hf" "llava-1.5-13b-hf")`。这些 id 有助于跟踪和区分不同的模型评估。
- `MODEL_NAME_OR_PATHS`: 一个由两条路径组成的数组，这些路径指向模型的权重或它们发布在 Hugging Face 上的名称，例如 `("llava-hf/llava-1.5-7b-hf" "llava-hf/llava-1.5-13b-hf")`。
- `CHAT_TEMPLATES`: 由两个聊天模板 id 组成的数组，对应于每个模型，例如 `("LLAVA" "LLAVA")`。这定义了每个模型生成的响应的格式或样式。

模态无感的评估脚本位于 `./scripts` 目录下。需要用户输入的参数已留空，必须在启动评估过程之前填写。例如，对于 `evaluate_anything.sh`:

~~~bash
cd ../align_anything/evaluation

MODALITY=""
OUTPUT_DIR=""
GENERATION_BACKEND=""
MODEL_ID=""
MODEL_NAME_OR_PATH=""
CHAT_TEMPLATE=""

python eval_anything.py \
    --modality ${MODALITY} \
    --output_dir ${OUTPUT_DIR} \
    --generation_backend ${GENERATION_BACKEND} \
    --model_id ${MODEL_ID} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template ${CHAT_TEMPLATE}
~~~

- `MODALITY`: 输入——输出模态的指定，例如 `ti2t`。
- `OUTPUT_DIR`: 用于保存评估结果和输出文件的目录。
- `GENERATION_BACKEND`: 进行大语言模型推理的框架，包括 `vLLM` 和 `deepspeed`。
- `MODEL_ID`: 模型的唯一标识符，用于跟踪和区分模型评估，如 `llava-1.5-7b-hf`。
- `MODEL_NAME_OR_PATH`: 模型的本地路径或 Hugging Face 链接，如 `llava-hf/llava-1.5-7b-hf` 。
- `CHAT_TEMPLATE`: 模型的聊天模板 id，如 `LLAVA`。更多细节可以参考 `./align_anything/configs/template.py`。

此外，你还应修改 `./align_anything/configs/evaluation/benchmarks` 下与基准测试对应的配置文件，以适应特定的评估任务，并指定测试模型。

例如，对于 `pope.yaml`：

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
    # Unique identifier for cache folder
    uuid: null
    # Num shot
    n_shot: 0
  # Configuration for data
  data_cfgs:
    # Task name
    task: default
    # Task directory
    task_dir: lmms-lab/POPE
    # Evaluation split
    split: test
  # Model configurations
  model_cfgs:
    model_id: null
    # Pretrained model name or path
    model_name_or_path: null
    # Chat template
    chat_template: null
    # Whether to trust remote code
    trust_remote_code: True
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