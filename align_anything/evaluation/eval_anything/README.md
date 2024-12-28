# Eval Anything

NOTE: The current version contains scripts organized from experimental content as usage examples. We will reorganize the relevant code in the future to make it more accessible to the community.

## All-Modality Understanding (AMU)

### Loading Evaluation Datasets

The default way to load the dataset is:

```python
from datasets import load_dataset

dataset = load_dataset(
    "PKU-Alignment/EvalAnything-AMU",
    name='image',
    trust_remote_code=True
)
```
This includes the complete set of 164 test cases, with visual inputs encompassing both images and videos.
Considering that future multi-modal models might only handle either images or videos, and that preprocessing interfaces may differ between image and video processing, we have divided the visual modalities into two subsets. Here's how to load them separately:

- For images:
```python
from datasets import load_dataset

dataset = load_dataset(
    "PKU-Alignment/EvalAnything-AMU",
    name='image',
    trust_remote_code=True
)
```

- For videos:
```python
from datasets import load_dataset

dataset = load_dataset(
    "PKU-Alignment/EvalAnything-AMU",
    name='video',
    trust_remote_code=True
)
```
For more details about the evaluation data, please refer to [PKU-Alignment/EvalAnything-AMU](https://huggingface.co/datasets/PKU-Alignment/EvalAnything-AMU)

### Model Evaluation
Model evaluation is initiated using the `eval_anything/amu/example.py` script. Note that you need to complete the model inference-related code before use. For evaluation prompts, refer to `eval_anything/amu/amu_eval_prompt.py`.

## All-Modality Generation (AMG)

### Instruction-Following Part

#### Loading Evaluation Datasets

The default loading method is:
```python
dataset = load_dataset(
    "PKU-Alignment/EvalAnything-InstructionFollowing",
    trust_remote_code=True
)
```
To load test data for a single modality (e.g., images), use:

```python
dataset = load_dataset(
    "PKU-Alignment/EvalAnything-InstructionFollowing",
    name="image",
    trust_remote_code=True
)
```

For more dataset details, refer to [PKU-Alignment/EvalAnything-InstructionFollowing](https://huggingface.co/datasets/PKU-Alignment/EvalAnything-InstructionFollowing)

#### Model Evaluation

The AMG Instruction-Following evaluation covers Text/Image/Audio/Video Generation across four dimensions. The evaluation code is located in the `eval_anything/amg/if` folder. To start the evaluation (e.g., images):

1. Use `eval_anything/amg/if/image_instruct/example.py` to generate multi-modal results
2. Use `eval_anything/amg/if/image_instruct/eval_instruct.py` to evaluate the generated results

Since there isn't a widely accepted video or audio large language model in the community to serve as a judge model for direct scoring, we use multiple-choice questions to check whether the generated modalities contain the instructed content for video and audio evaluations. For text and image modalities, we continue using GPT as a judge model to provide direct scores.

### Modality Selection Part

#### Loading Evaluation Datasets
Load the dataset using:
```python
dataset = load_dataset(
    "PKU-Alignment/EvalAnything-Selection_Synergy",
    trust_remote_code=True
)
```
For more details, refer to [PKU-Alignment/EvalAnything-Selection_Synergy](https://huggingface.co/datasets/PKU-Alignment/EvalAnything-EvalAnything-Selection_Synergy)

#### Model Evaluation

Use `eval_anything/amg/selection/example.py` for modality selection evaluation. Note that you need to implement the code related to generating responses.

### Modality Synergy Part

#### Loading Evaluation Datasets

Load the dataset using:

```python
dataset = load_dataset(
    "PKU-Alignment/EvalAnything-Selection_Synergy",
    trust_remote_code=True
)
```
For more details, refer to [PKU-Alignment/EvalAnything-Selection_Synergy](https://huggingface.co/datasets/PKU-Alignment/EvalAnything-EvalAnything-Selection_Synergy)

#### Model Evaluation

Since there isn't currently a complete multi-modal generation model, you can simulate the multi-modal generation process using Agent-related technologies. Reference the Agent code in `eval_anything/amg/agent`. For evaluation:

Use `eval_anything/amg/synergy/example.py` to generate relevant instructions
Use `eval_anything/amg/generate.sh` to call the agent to simulate the multi-modal generation process
Format the generated results as shown in `eval_anything/amg/synergy`
Use `eval_anything/amg/synergyreward_eval.py` to evaluate modality synergy

We've trained a multi-modal input model for Modality Synergy scoring. For model details, refer to [PKU-Alignment/AnyRewardModel](https://huggingface.co/PKU-Alignment/AnyRewardModel).


Load the model using:
```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("PKU-Alignment/AnyRewardModel", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("PKU-Alignment/AnyRewardModel", trust_remote_code=True)
```

For Image-Audio Modality Synergy scoring:
```python
user_prompt: str = 'USER: {input}'
assistant_prompt: str = '\nASSISTANT:\n{modality}{text_response}'

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def process_ia(prompt, image_path, audio_path):
    image_pixel_values = processor(data_paths = image_path, modality="image").pixel_values
    audio_pixel_values = processor(data_paths = audio_path, modality="audio").pixel_values

    text_input = processor(
        text = user_prompt.format(input = prompt) + \
                assistant_prompt.format(modality = "<image><audio>", text_response = ""),
        modality="text"
    )
    return {
        "input_ids": text_input.input_ids,
        "attention_mask": text_input.attention_mask,
        "pixel_values_1": image_pixel_values.unsqueeze(0),
        "pixel_values_2": audio_pixel_values.unsqueeze(0),
        "modality": [["image", "audio"]]
    }


score = sigmoid(model(**process_ia(prompt, image_path, audio_path)).end_scores.squeeze(dim=-1).item())
```

For Text-Image Modality Synergy scoring:
```python
user_prompt: str = 'USER: {input}'
assistant_prompt: str = '\nASSISTANT:\n{modality}{text_response}'

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def process_ti(prompt, response, image_path):
    image_pixel_values = processor(data_paths = image_path, modality="image").pixel_values
    text_input = processor(
        text = user_prompt.format(input = prompt) + \
                assistant_prompt.format(modality = "<image>", text_response = response),
        modality="text"
    )
    return {
        "input_ids": text_input.input_ids,
        "attention_mask": text_input.attention_mask,
        "pixel_values_1": image_pixel_values.unsqueeze(0),
        "modality": [["image", "text"]]
    }

score = sigmoid(model(**process_ti(prompt, response, image_path)).end_scores.squeeze(dim=-1).item())
```

For Text-Audio Modality Synergy scoring:
```python
user_prompt: str = 'USER: {input}'
assistant_prompt: str = '\nASSISTANT:\n{modality}{text_response}'

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def process_ta(prompt, response, audio_path):
    audio_pixel_values = processor(data_paths = audio_path, modality="audio").pixel_values
    text_input = processor(
        text = user_prompt.format(input = prompt) + \
                assistant_prompt.format(modality = "<audio>", text_response = response),
        modality="text"
    )
    return {
        "input_ids": text_input.input_ids,
        "attention_mask": text_input.attention_mask,
        "pixel_values_1": audio_pixel_values.unsqueeze(0),
        "modality": [["audio", "text"]]
    }

score = sigmoid(model(**process_ta(prompt, response, audio_path)).end_scores.squeeze(dim=-1).item())
```
