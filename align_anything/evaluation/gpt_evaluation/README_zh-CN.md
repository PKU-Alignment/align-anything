# OpenAI API Evaluation
本节介绍了一种使用OpenAI API对多模态数据进行偏好标注的方法。您可以通过调用GPT-4o、GPT-4等先进模型的API，准确、高效地为您的数据进行偏好对比。无论是图像还是文本，你都可以使用我们的代码在OpenAI模型的辅助下快速获取大量精确的偏好对比，从而快速构建偏好数据集，或对您的模型进行评测。
## 支持任务
本部分支持五种任务的GPT偏好标注，其中有四种纯文本任务以及一种图像-文本任务。目前，文本任务支持对文本数据集的安全性，有效性，同理心，推理能力进行偏好对比，而对于图像-文本任务当前支持对于给定图像与相关问题，对给出的回答进行偏好对比。
## 支持模型
对于纯文本任务，本框架支持所有OpenAI Chat模型，而对于图像-文本任务，当前仅支持GPT-4o模型。\
如果需要更改模型，请在`./utils.py`中的`DEFAULT_OPENAI_CHAT_COMPLETION_MODELS`部分进行修改，默认模型为GPT-4o。
## 快速开始
当前，模型仅支持以jsonl文件作为输入进行OpenAI API偏好评测，且对json文件的相关格式有严格要求。\
\
对于安全性，有效性，同理心任务，jsonl文件的格式要求如下
```bash
{
  "question": <your question>,
  "response_a": <response_a>,
  "response_b": <response_b>
}
```
对于推理任务，jsonl文件的格式要求如下
```bash
{
  "question": <your question>,
  "response_a": <response_a>,
  "response_b": <response_b>,
  "target": <the ground-truth answer of your question>
}
```
对于图像-文本任务，jsonl文件的格式要求如下
```bash
{
  "question": <your question>,
  "response_a": <response_a>,
  "response_b": <response_b>,
  "image_url": <the image_url of your image>
}
```
注意，当前图像-文本偏好评测仅支持将图像以base 64 编码格式传递，具体细节请见 [OpenAI Vision: Uploading base 64 encoded images](https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images) 。您仅需将图片的base 64编码放入上面的`image_url`部分。\
\
当您检查完您的数据格式后，请将您的OpenAI API key放入如下`./config/openai_api_keys.txt`文件，并运行`script.sh`脚本。
```bash
sh script.sh --input-file <input-file> --task <task>
```
其中，`input-file`代表您的数据集输入路径,`task`代表您将评测的任务。\
运行完成后，将您即可在当前文件夹下看到您的结果文件。
## prompt更改
所有用于进行偏好评测的prompt被放置于`./config/system_prompt.py`文件中，您可以根据您的需求，在此处更改您的prompt个性化进行评测。
