# Chameleon Model Doc for Align-Anything

## Environment setup

Currently, the official Transformer repo does not support Chameleon model with image output (see [this PR](https://github.com/huggingface/transformers/pull/32013) for more details), so we rely on a certain fork of the repo.

After installing Align-Anything and correctly set up the envrionment, you can install the forked stable version of the repo by running:

```bash
pip install git+https://github.com/htlou/transformers.git@hantao_stable_cham
```

## Model Download

We will update this section after our model is uploaded to the Hugging Face model hub.

## Model Training

As chameleon expands the \<image\> in the input into image placeholders (\<image\> * 1024) in the processor, but add the real image tokens inside the forward function, we need to do pretokenize to the input and merge the real image tokens to the labels before doing the training.

### Pre Tokenization

Set the input, output and Chameleon model path correctly in `pre_tokenize_example.py` and run:

```bash
python pre_tokenize_example.py
```

### Chameleon Finetuning

Add a script named `sft_cham.sh` under the `scripts` file like this:

```bash
# You can replace it with a local model path
MODEL_NAME_OR_PATH=""
# You can replace it with a local dataset path, note that it should not include the name of the datat file
TRAIN_DATASETS="path/to/dataset"
# the file name should look like "dataset_file_name.pt"
FILE_NAME="dataset_file_name" 
# You can replace it with a new path
OUTPUT_DIR="../outputs/sft_chameleon"
# For wandb online logging
export WANDB_API_KEY=""
# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
	--hostfile ${HOSTFILE} \ 
	--master_port ${MASTER_PORT} \
	--module align_anything.trainers.ti_to_ti.sft \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_datasets ${TRAIN_DATASETS} \
	--train_data_files ${FILE_NAME} \ 
	--output_dir ${OUTPUT_DIR} \
	--train_template ANYTHING_TI2TI \
	--train_split 'train' \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--gradient_accumulation_steps 2 \
	--save_interval 500 \
	--learning_rate 1e-5 \
	--epochs 12 \
    --lr_scheduler_type constant
```

and set up the correct model path and dataset path, then run:
    
```bash
bash scripts/sft_cham.sh
```

## Model Evaluation

### Batch Inference

Currently the batch inference of Chameleon is not integrated into the Align-Anything repo, so we need to use [another repo](). Here's a forked (and revised to make it stable) version:

```bash
git clone https://github.com/htlou/mmsg_chameleon.git
cd mmsg_chameleon
```

Then set up the envrionment using 
```bash
pip install -e . 
```

After setting up the envrioment, set up the correct paths in `scripts/interleaved_gen.sh` and then run
```bash
bash scripts/interleaved_gen.sh
```

to do batch inference.

### GPT-based Evaluation

Currently the GPT-based evaluation of text-image interleaved messages is not integrated into the Align-Anything repo, so we need to use [another repo](https://github.com/htlou/gpt4_eval):

```bash
git clone https://github.com/htlou/gpt4_eval.git
cd gpt4_eval
```
You can set the `INPUT_TYPE` in the `script.sh` to `interleaved-compare` and run:
    
```bash
bash script.sh
```

to do the evaluation.