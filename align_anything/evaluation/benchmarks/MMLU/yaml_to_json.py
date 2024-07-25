import os
import yaml
import json

def process_yaml_to_json(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹下的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.yaml') and filename.startswith('mmlu'):
            # 构造完整的文件路径
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename[5:].replace('.yaml', '.json'))

            # 读取并解析 YAML 文件
            with open(input_file_path, 'r', encoding='utf-8') as yaml_file:
                data = yaml.safe_load(yaml_file)
            
            # 提取 fewshot_config 中的 samples
            samples = data['fewshot_config']['samples']
            new_data = []
            for item in samples:
                new_data.append(
                    {
                        'question': item['question'],
                        'answer': item['target'],
                    }
                )
            # 将 samples 转换为 JSON 格式并写入文件
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(new_data, json_file, indent=4, ensure_ascii=False)

    print(f"Processed YAML files from '{input_folder}' and saved JSON files to '{output_folder}'")

# 指定输入和输出文件夹路径
input_folder = '/aifs4su/yaodong/donghai/align-anything/align_anything/evaluation/benchmarks/MMLU/flan_cot_fewshot'
output_folder = '/aifs4su/yaodong/donghai/align-anything/align_anything/evaluation/benchmarks/MMLU/cot_few_shot'

# 调用函数进行批量处理
process_yaml_to_json(input_folder, output_folder)