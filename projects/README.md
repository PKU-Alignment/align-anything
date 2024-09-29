# Align-Anything Projects

This folder contains the separate projects developed under the [Align-Anything](../README.md) framework.

The contents of each project are as follows:

## Text-Image-to-Text-Image

[This project](./text_image_to_text_image/README.md) contains the environment setup and some necessary scripts for the text-image-to-text-image alignment task, or to be specific, the SFT, DPO, RM and PPO of [Chameleon](https://huggingface.co/facebook/chameleon-7b) model. It provides the essential `pre_tokenize_example.py` and its variants for pre-processing the data before training.

## Any-to-Text

[This project](./any_to_text/README.md) contains the process for constructing an Any-to-Text multimodal language model, aimed at advancing alignment research by integrating various modalities such as text, images, and audio into a unified model. It provides step-by-step guides and scripts, specifically `build_llama_vision.py` and `build_llama_vision_audio.py`, to help researchers build and initialize different configurations of the model.

## Language Feedback

[This project](./lang_feedback/README.md) contains the process for generating language feedback data for Align-Anything algorithm. It utilizes VLLM to efficiently handle the language model inference. This project is currently under development and is only set for internal use.