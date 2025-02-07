# Janus Model Doc for Align-Anything

## Environment setup

Currently, we developed our code based on a variant of [Janus](https://github.com/deepseek-ai/Janus) repo. After installing Align-Anything and correctly set up the environment, you can install the forked stable version of the repo by running:

```bash
pip install git+https://github.com/htlou/Align_Anything_Janus.git
cd Align_Anything_Janus
pip install -e .
```

## Pre-tokenization

Set the input, output, Janus model path and Janus repo path (the path to the cloned Janus repo) correctly in `sft_tokenize.sh` and run:

```bash
bash sft_tokenize.sh
```

