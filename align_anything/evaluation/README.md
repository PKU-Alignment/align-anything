# Evaluate

## Development Roadmap

Below is a list of benchmarks currently supported by our framework. For the full list, please refer to [this directory](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/evaluation/benchmarks). More benchmarks are in the process of being added.

| Modality | Supported Benchmarks                                         |
| :------- | :----------------------------------------------------------- |
| `t2t`    | [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBH](https://huggingface.co/datasets/lukaemon/bbh), [Belebele](https://huggingface.co/datasets/facebook/belebele), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MMLU](https://huggingface.co/datasets/cais/mmlu), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [MT-Bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x), [RACE](https://huggingface.co/datasets/ehovy/race), [TruthfulQA ](https://huggingface.co/datasets/truthfulqa/truthful_qa) |
| `ti2t`   | [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA), [LLaVA-Bench(COCO)](https://huggingface.co/datasets/lmms-lab/llava-bench-coco), [LLaVA-Bench(wild)](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild), [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench), [MMBench](https://huggingface.co/datasets/lmms-lab/MMBench), [MME](https://huggingface.co/datasets/lmms-lab/MME), [MMMU](https://huggingface.co/datasets/MMMU/MMMU), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), [MMVet](https://huggingface.co/datasets/lmms-lab/MMVet), [POPE](https://huggingface.co/datasets/lmms-lab/POPE), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL), [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa), [VizWizVQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA) |
|`tv2t` |[MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) |
|`ta2t` |[AIR-Bench](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) |
| `t2i`      | [ImageReward](https://huggingface.co/datasets/THUDM/ImageRewardDB), [HPSv2](https://huggingface.co/datasets/zhwang/HPDv2), [COCO-30k(FID)](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014) |
| `t2v`      | [ChronoMagic-Bench](https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench) |
| `t2a`      | [AudioCaps(FAD)](https://huggingface.co/datasets/AudioLLMs/audiocaps_test) |

## Quick Start

All evaluation scripts can be found in the `./scripts`. The `./scripts/evaluate.sh` script runs model evaluation on the benchmarks, and parameters that require user input have been left empty. The corresponding script is as follow:

~~~bash
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
~~~

- `BENCHMARK`: A list of evaluation benchmarks used to assess the model's performance.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generation, `vLLM` or `deepspeed`.
- `MODEL_ID`: Unique identifier for the model, used to track and differentiate between evaluations.
- `MODEL_NAME_OR_PATH`: Local path or Hugging Face link to the model weights.
- `CHAT_TEMPLATE`: The chat template corresponding to the model.

The `./scripts/models_pk.sh` evaluates and compares two models across benchmarks. Ensure all parameters are set before running. The corresponding script is as follow:

~~~bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

BENCHMARKS=("") # evaluation benchmarks
OUTPUT_DIR="" # output dir
GENERATION_BACKEND="" # generation backend
MODEL_IDS=("" "") # model's unique ids
MODEL_NAME_OR_PATHS=("" "") # model paths
CHAT_TEMPLATES=("" "") # model templates

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

- `BENCHMARKS`: A list of evaluation benchmarks used to assess the model's performance.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generation, `vLLM` or `deepspeed`.
- `MODEL_IDS`: A pair of unique identifiers for the models being evaluated, used to track and distinguish between different model evaluations.
- `MODEL_NAME_OR_PATHS`: A pair of local paths or Hugging Face links to the model weights.
- `CHAT_TEMPLATES`: A pair of chat templates, each corresponding to one of the models.

You can customize the configuration files for the benchmarks in [this directory](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/benchmarks) to fit specific evaluation tasks and models. For instance, for the `pope.yaml`:

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

For inference parameters, refer to [vLLM settings](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/vllm) or [DeepSpeed settings](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/deepspeed), based on your chosen generation backend.

### Benchmarks requiring OpenAI API access

For the following benchmarks, OpenAI API is required for evaluation:

- ChronoMagicBench
- MM-SafetyBench
- SPA-VL
- MMVet
- AGIEval
- mt_bench
- llava-bench-coco
- llava-bench-in-the-wild
- LongBench

You can add your OpenAI API credentials in one of two ways:

1. Using Environment Variables:

   Set the OPENAI_API_KEY and OPENAI_API_BASE_URL environment variables before running the evaluation:

   ~~~bash
   export OPENAI_API_KEY="YOUR_API_KEY"
   export OPENAI_API_BASE_URL="https://api.openai.com/v1/chat/completions"
   ~~~

2. Modifying the YAML Configuration Files:

   Alternatively, you can specify your API key and base URL directly in the corresponding YAML configuration files for the benchmark, located in `./align_anything/configs/evaluation/benchmarks`. Update the `openai_api_key` and `openai_api_base_url` properties accordingly.

### Attention

1. For `SPA-VL/HarmEval` benchmark, the base model used is `llava-hf/llava-1.5-7b-hf`, while for `SPA-VL/HelpEval` benchmark, the base model is `gpt-4-turbo`.
2. When evaluating `HPSv2`, you will need to manually configure the environment by downloading the file from https://github.com/tgxs002/HPSv2/blob/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz and placing it in the directory `./your_env_name/lib/python3.11/site-packages/hpsv2/src/open_clip/`.
3. When evaluating `AudioCaps`, the `background_dir` variable, located in the `./align_anything/evaluation/benchmarks/AudioCaps/eval.py` file, should point to the directory containing the standard audio files for the evaluation set. This directory must only contain audio files in MP3 format, and each audio file should be named according to its corresponding `audiocap_id`. For example, if the `audiocap_id` is 103541, the audio file should be named `103541.mp3`.


To utilize a custom template, you need to implement and register the template in [here](https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/configs/template.py). For example, here is the implementation of `Llava` template:

~~~python
@register_template('Llava')
class Llava:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'
~~~

After implementing your custom template, specify it in the configuration file found in [this directory](https://github.com/PKU-Alignment/align-anything/tree/main/align_anything/configs/evaluation/benchmarks) by setting the `chat_template` parameter to the name of your template, such as `Llava`.