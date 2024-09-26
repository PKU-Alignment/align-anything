# Evaluation Configurations

Align-Anything offers highly customizable settings for the benchmark evaluation process. You can adjus configurations file under `align_anything/configs/evaluation/benchmarks` to fit specific tasks and models. under these categories:

- `infer_cfgs`: Configuration file names related to the generation backend.
- `eval_cfgs`: The output path for evaluation results and log files can be specified in `output_dir`. For certain benchmarks, the options `n_shot` and `cot` are available, which can be configured based on your requirements.
- `data_cfgs`: The parameters file related to the test dataset can be modified here. For the specified `task_dir`, you can specify multiple tasks through `task`.
- `model_cfgs`:  Configuration settings related to the model to be tested.

Taking `mmlu.yaml` as an example:

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
    # Use Chain of Thought
    cot: false
  # Configuration for data
  data_cfgs:
    # Task name
    task: ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics']
    # Task directory
    task_dir: cais/mmlu
    # Evaluation split
    split: test
    # Candidate labels
    candidate_labels: ["A", "B", "C", "D"]

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

To accommodate various inference devices and environments, we support Deepspeed and vLLM as inference backends. You can view the supported generation backend of each benchmark [here](./overview.md). You are able to adjust the inference parameters in `align_anything/configs/deepspeed` and `align_anything/configs/vllm`. For instance, you can adjust the vLLM inference configurations to suit your needs:

~~~json
{
    "SamplingParams":
    {
        "n": 1,
        "top_k": 10,
        "top_p": 0.95,
        "temperature": 0.05,
        "max_tokens": 512,
        "frequency_penalty": 1.2,
        "prompt_logprobs": 0,
        "logprobs": 20
    },
    "LLM":
    {
        "tokenizer_mode": "auto",
        "trust_remote_code": true,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 16
    }
}
~~~

You can also modify the chat template according to your model in the way described in [here](../training/dataset_custom).