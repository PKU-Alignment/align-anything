# OpenAI API Evaluation

This section introduces a method for preference annotation of multimodal data using the OpenAI API. By calling advanced models such as GPT-4o, GPT-4, and others, you can accurately and efficiently compare preferences within your data. Whether dealing with images or text, our code allows you to quickly obtain precise preference comparisons with the assistance of OpenAI models, enabling you to swiftly build preference datasets or evaluate your models.

## Supported Tasks

This section supports GPT preference annotations for five types of tasks, including four pure text tasks and one image-text task. Currently, text tasks support preference comparisons for the safety, effectiveness, empathy, and reasoning ability of text datasets. For the image-text task, preference comparisons for answers given to questions related to a provided image are supported.

## Supported Models

For pure text tasks, this framework supports all OpenAI Chat models. For the image-text task, only the GPT-4o model is currently supported. \
If you need to change the model, please modify the `DEFAULT_OPENAI_CHAT_COMPLETION_MODELS` section in `./utils.py`. The default model is GPT-4o.

## Quick Start

Currently, the models support using JSONL files as input for OpenAI API preference evaluation, with strict formatting requirements for these JSON files.

For the safety, effectiveness, and empathy tasks, the JSONL file format should be as follows:
```json
{
  "question": <your question>,
  "response_a": <response_a>,
  "response_b": <response_b>
}
```
For the reasoning task, the JSONL file format should be as follows:
```json
{
  "question": <your question>,
  "response_a": <response_a>,
  "response_b": <response_b>,
  "target": <the ground-truth answer of your question>
}
```
For the image-text task, the JSONL file format should be as follows:
```json
{
  "question": <your question>,
  "response_a": <response_a>,
  "response_b": <response_b>,
  "image_url": <the image_url of your image>
}
```
Note that the current image-text preference evaluation only supports passing images in base 64 encoded format. For more details, see [OpenAI Vision: Uploading base 64 encoded images](https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images). Simply place the base 64 encoded image in the `image_url` section above.

Once you have checked your data format, place your OpenAI API key into the `./config/openai_api_keys.txt` file and run the `script.sh` script:
```bash
sh script.sh --input-file <input-file> --task <task>
```
Here, `input-file` represents the path to your dataset input, and `task` represents the task you are evaluating. After running, you will see your results file in the current folder.

## Prompt Modification

All prompts used for preference evaluation are placed in the `./config/system_prompt.py` file. You can personalize the prompts for evaluation by modifying them according to your needs.
