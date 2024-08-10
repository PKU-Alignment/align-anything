import argparse
import os

from transformers import (
    LlavaConfig, 
    LlavaForConditionalGeneration, 
    LlavaProcessor, 
    CLIPImageProcessor, 
    AutoModel, 
    AutoTokenizer, 
    AutoModelForCausalLM,
)

def make_model_contiguous(module):
    for child in module.children():
        make_model_contiguous(child)
    
    for param in module.parameters(recurse=False):
        param.data = param.data.contiguous()

def build_llava(args):    
    language_model_path = args.language_model_path
    vision_tower_path = args.vision_tower_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    vision_model = AutoModel.from_pretrained(vision_tower_path)
    language_model = AutoModelForCausalLM.from_pretrained(language_model_path)
    tokenizer = AutoTokenizer.from_pretrained(language_model_path)
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_path)

    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<unk>", "<pad>"]})
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"

    vision_config = vision_model.vision_model.config
    language_config = language_model.config
    config = LlavaConfig(vision_config, language_config)
    config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    model = LlavaForConditionalGeneration(config)
    model.language_model = language_model
    model.vision_tower = vision_model
    model.resize_token_embeddings(len(tokenizer))
    
    make_model_contiguous(model)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Parser for model paths and save path.")

    parser.add_argument('--language_model_path', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Path to the language model.')
    parser.add_argument('--vision_tower_path', type=str, default='openai/clip-vit-large-patch14-336', help='Path to the vision tower.')
    parser.add_argument('--save_path', type=str, default='Any2Text/llama_vision', help='Path to save the results.')

    args = parser.parse_args()
    return args
    
if __name__=="__main__":
    args = parse_args()
    build_llava(args)