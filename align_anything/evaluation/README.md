# Evaluate

## Development Roadmap

We currently support two types of evaluation datasets: **"Text ➡️ Text"** and **"Image+Text ➡️ Text"**. For **"Text ➡️ Text"**, we support 12 benchmark datasets, and for **"Image+Text ➡️ Text"**, we support 15 benchmark datasets. We are actively working on expanding support to additional multimodal benchmarks.

Then, We have a roadmap for future development work `align-anything`:

| Modality              | Supported                                                    |
| :-------------------- | :----------------------------------------------------------- |
| **Text ➡️ Text**       | [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBH](https://huggingface.co/datasets/lukaemon/bbh), [Belebele](https://huggingface.co/datasets/facebook/belebele), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x), [RACE](https://huggingface.co/datasets/ehovy/race), [TruthfulQA ](https://huggingface.co/datasets/truthfulqa/truthful_qa), ➖ |
| **Image+Text ➡️ Text** | [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA), [LLaVA-Bench(COCO)](https://huggingface.co/datasets/lmms-lab/llava-bench-coco), [LLaVA-Bench(wild)](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild), [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench), [MMBench](https://huggingface.co/datasets/lmms-lab/MMBench), [MME](https://huggingface.co/datasets/lmms-lab/MME), [MMMU](https://huggingface.co/datasets/MMMU/MMMU), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), [MMVet](https://huggingface.co/datasets/lmms-lab/MMVet), [POPE](https://huggingface.co/datasets/lmms-lab/POPE), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL), [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa), [VizWizVQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA), ➖ |
| **Text ➡️ Image**      | [ImageReward](https://huggingface.co/datasets/THUDM/ImageRewardDB), [HPSv2](https://huggingface.co/datasets/zhwang/HPDv2)➖                                                            |
| **Text ➡️ Video**      | ➖                                                            |
| **Text ➡️ Audio**      | ➖                                                            |

- ➖ : Features on going in our TODO list.

## Quick Start

To prepare for evaluation,  the script is located in the `./scripts`. Parameters that require user input have been left empty and must be filled in prior to initiating the evaluation process. For example, for `evaluate.sh`:

~~~bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

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

- `BENCHMARK`: One or more evaluation benchmarks or datasets for assessing the model's performance. For example, `("POPE" "MMBench")` can be used to evaluate the model on both the POPE and MMBench datasets. Each benchmark in the list will be processed sequentially.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generating predictions, `vLLM` or `deepspeed`.
- `MODEL_ID`: Unique identifier for the model, used to track and distinguish model evaluations, like `llava-1.5-7b-hf`.
- `MODEL_NAME_OR_PATH`: The local path or Hugging Face link of model, such as `llava-hf/llava-1.5-7b-hf`.
- `CHAT_TEMPLATE`: Chat template id of your model, like `LLAVA`. More details can be refered in `./align_anything/configs/template.py`.

To compare multiple models' performance across one or more benchmarks, located in the `./scripts`, the `models_pk.sh` script allows you to evaluate across different models and then compare their results. Ensure all parameters are correctly filled in before running the script.

~~~bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

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

- `BENCHMARKS`: One or more evaluation benchmarks or datasets for assessing the model's performance. For example, `("POPE" "MMBench")` can be used to evaluate the model on both the POPE and MMBench datasets. Each benchmark in the list will be processed sequentially.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generating predictions, `vLLM` or `deepspeed`.
- `MODEL_IDS`: An array of two unique identifiers for the models being evaluated, such as `("llava-1.5-7b-hf" "llava-1.5-13b-hf")`. These IDs help track and distinguish between different model evaluations.
- `MODEL_NAME_OR_PATHS`: An array of two paths to the models' weights or their names if hosted on Hugging Face, such as `("llava-hf/llava-1.5-7b-hf" "llava-hf/llava-1.5-13b-hf")`.
- `CHAT_TEMPLATES`: An array of two chat template IDs corresponding to each model, such as `("LLAVA" "LLAVA")`. This defines the format or style of responses generated by each model.

To realize modal insensitive evaluation, the script is located in the `./scripts`. Parameters that require user input have been left empty and must be filled in prior to initiating the evaluation process. 

In addition, you should modify the config file corresponding to the benchmark under `./align_anything/configs/evaluation/benchmarks` to adapt to specific evaluation tasks and specify test models.

For example, for `pope.yaml`:

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
    max_length: 1024
    # The max new tokens for generation
    max_new_tokens: 512
~~~

For more inference parameters, please see `./align_anything/configs/evaluation/vllm` and `./align_anything/configs/evaluation/deepspeed`, depends on our generation backend.


### Incorporating Custom Templates

To utilize a custom template, you need to implement and register the template in `./align_anything/configs/template.py`. For example, for `PKUSafeRLHF` template:

~~~python
@register_template('PKUSafeRLHF')
class PKUSafeRLHF(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
~~~

After defining our custom template, specify it in our YAML configuration file by setting the `chat_template` parameter to the name of our template, such as `PKUSafeRLHF`.