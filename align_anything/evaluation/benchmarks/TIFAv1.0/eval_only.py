# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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

import argparse
from align_anything.evaluation.inference.base_inference import *
from align_anything.utils.question_gen import get_QA_pair, QA_filter, get_tifa_score
from align_anything.models.vqa_model import UnifiedQAModel, VQAModel
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.evaluation.eval_logger import EvalLogger
import torch.multiprocessing as mp
import os

def load_dataset(gen_dir):
    processed_inputs = []
    with open(gen_dir, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    for data in datas:
        processed_inputs.append({
            'prompt': data['prompt'],
            'image_path': data['image'],
        })
    return processed_inputs

class TIFAGenerator(BaseInferencer):
    def evaluator(self, outputs, file_path, QA_pairs_path, api_key, base_url):
        qa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
        vqa_model = VQAModel("mplug-large")
        
        tot_score = 0.0
        num_sum = 0
        results = []
        for output in tqdm(outputs, desc="Evaluating"):
            prompt = output['prompt']
            img_path = output['image_path']
            
            caption_qas = get_QA_pair(prompt, api_key, base_url)
            filtered_questions = QA_filter(qa_model, caption_qas)
            
            num_sum += 1
            if not filtered_questions or not os.path.exists(img_path):
                score = 0.0
            else:
                result = get_tifa_score(vqa_model, filtered_questions, img_path)
                score = result['tifa_score']
                results.append(result)
            tot_score += score
            save_detail(prompt, '', '', img_path, score, file_path)
            
        with open(QA_pairs_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
                
        return tot_score, num_sum

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')
    
    dict_configs, infer_configs = read_eval_cfgs('tifav1.0', 'vLLM')
    
    try:
        assert dict_configs or infer_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        logger.log('error', "Config file is not exist or incomplete.")
        exit()

    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    logger.log_dir = eval_configs.output_dir
    test_data = load_dataset(eval_configs.generation_output)
    eval_module = TIFAGenerator(model_config.model_id, model_config.model_name_or_path, model_config.model_max_length, 42)

    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}/{model_config.model_id}"
    os.makedirs(uuid_path, exist_ok=True)

    file_path = f"{uuid_path}/result.json"
    QA_pairs_path = f"{uuid_path}/QA_pairs.jsonl"
    score, num_sum = eval_module.evaluator(test_data, file_path, QA_pairs_path, api_key, base_url)
            
    eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'num_sum': [num_sum],
            'avg_score': [score / num_sum],
            }
    logger.print_table(title=f'TIFA Benchmark', data=eval_results)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {eval_results['model_id'][0]},")
    logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
    logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
    logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
    logger.log('info', f"avg_score: {eval_results['avg_score'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()