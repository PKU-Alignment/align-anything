# Janus Model Doc for Align-Anything

## Environment setup

Currently, we developed our code based on a variant of [Janus](https://github.com/deepseek-ai/Janus) repo. After installing Align-Anything and correctly set up the environment, you can install the forked stable version of the repo by running:

```bash
pip install git+https://github.com/htlou/Align_Anything_Janus.git
cd Align_Anything_Janus
pip install -e .
```

## Pre-tokenization

In the case of performing supervised finetuning, set the input, output, Janus model path and Janus repo path (the path to the cloned Janus repo) correctly in `supervised_tokenize.sh` and run:

```bash
bash supervised_tokenize.sh
```

In the case of performing preference learning (currently, only DPO is supported), set the input, output, Janus model path and Janus repo path (the path to the cloned Janus repo) correctly in `preference_tokenize.sh` and run:

```bash
bash preference_tokenize.sh
```
