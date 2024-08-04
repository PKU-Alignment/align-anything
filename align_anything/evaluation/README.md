# Evaluation

## Development Roadmap

We currently support two types of evaluation datasets: **"Text ➡️ Text"** and **"Image+Text ➡️ Text"**. For **"Text ➡️ Text"**, we support 12 benchmark datasets, and for **"Image+Text ➡️ Text"**, we currently support one benchmark dataset, MME. We are actively working on expanding support to additional multimodal benchmarks.

Then, We have a roadmap for future development work `align-anything`:

| Modality              | Supported                                                    |
| :-------------------- | :----------------------------------------------------------- |
| **Text ➡️ Text**       | [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBH](https://huggingface.co/datasets/lukaemon/bbh), [Belebele](https://huggingface.co/datasets/facebook/belebele), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x), [RACE](https://huggingface.co/datasets/ehovy/race), [TruthfulQA ](https://huggingface.co/datasets/truthfulqa/truthful_qa), ➖ |
| **Image+Text ➡️ Text** | [MME](https://huggingface.co/datasets/lmms-lab/MME), ➖       |
| **Text ➡️ Image**      | ➖                                                            |
| **Text ➡️ Video**      | ➖                                                            |
| **Text ➡️ Audio**      | ➖                                                            |

- ➖ : Features on going in our TODO list.

## Quick Start

To prepare for evaluation,  the script is located in the `./scripts`. Parameters that require user input have been left empty and must be filled in prior to initiating the evaluation process. For example, for `evaluate.sh`:

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

- `BENCHMARK`: The evaluation benchmark or dataset for assessing the model's performance. For example, `ARC` for the ARC dataset or another relevant benchmark.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generating predictions, `deepspeed` or `vLLM`.

In addition, you should modify the config file corresponding to the benchmark under `./align_anything/configs/evaluation/benchmarks` to adapt to specific evaluation tasks and specify test models.

For example, for `arc.yaml`:

~~~yaml
infer_cfgs:
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