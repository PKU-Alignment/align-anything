# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the Yushi-Hu's tifa library.
# https://github.com/Yushi-Hu/tifa/blob/main/tifascore/vqa_models.py
# https://github.com/Yushi-Hu/tifa/blob/main/tifascore/unifiedqa.py
# https://github.com/Yushi-Hu/tifa/blob/main/tifascore/mc_sbert.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, AutoProcessor, BlipForQuestionAnswering

class SBERTModel:
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))
            
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.detach().cpu()
    
    def multiple_choice(self, answer, choices):
        answer_embedding = self.embed_sentences([answer])
        choices_embedding = self.embed_sentences(choices)
        top_choice_index = torch.argmax(torch.matmul(choices_embedding, answer_embedding.T)).item()
        return choices[top_choice_index]

class UnifiedQAModel:
    def __init__(self, model_name = "allenai/unifiedqa-v2-t5-large-1363200"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()
            self.model.eval()
            
    def run_model(self, input_string, **generator_args):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
            res = self.model.generate(input_ids.to(self.model.device), max_new_tokens=30, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)
        
    def qa(self, question, context):
        answer = self.run_model(f"{question} \n {context}")[0]
        return ''.join(c for c in answer if c.isalnum() or c.isspace()).strip().lower()
    
    def mcqa(self, question, context, choices=["yes", "no"]):
        choice_text = ""
        if len(choices) > 0:
            choice_text = ""
            headings = ["(A)", "(B)", "(C)", "(D)"]
            for i, choice in enumerate(choices):
                if i < len(headings):
                    choice_text += f"{headings[i]} {choice} "    
        
        return self.run_model(f"{question} \n {context} \n {choice_text}")[0]
    
class BLIP:
    def __init__(self, ckpt="Salesforce/blip-vqa-capfilt-large"):
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = BlipForQuestionAnswering.from_pretrained(ckpt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def vqa(self, image, question):
        image = Image.open(image).convert('RGB')
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=50)
        generated_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
       
        return generated_answer[0]

class VQAModel:
    def __init__(self, model_name='blip-large'):
        self.model_name = model_name
        self.model = eval("BLIP")("Salesforce/blip-vqa-capfilt-large")
        self.sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2")
        
    def vqa(self, image, question, choices=[]):
        with torch.no_grad():
            if (len(choices) != 0) and (self.model_name.startswith("blip2")):
                return self.model.vqa(image, question, choices)
            else:
                return self.model.vqa(image, question)
            
    def multiple_choice_vqa(self, image, question, choices):
        free_form_answer = self.vqa(image, question, choices)
        multiple_choice_answer = free_form_answer
        if free_form_answer not in choices:
            multiple_choice_answer = self.sbert_model.multiple_choice(free_form_answer, choices)
        return {"free_form_answer": free_form_answer, "multiple_choice_answer": multiple_choice_answer}