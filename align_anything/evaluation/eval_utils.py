import re
from collections import Counter
from typing import List, Dict

import statistics


def parse_responses(responses: List[str]) -> Dict[str, int]:
    pattern_full = re.compile(r'\[\[response_a: (-?\d+(\.\d+)?)\]\].*?\[\[response_b: (-?\d+(\.\d+)?)\]\]')
    pattern = re.compile(r'\[\[(.*?)\]\]')

    counter = Counter()

    for response in responses:
        match_full = pattern_full.search(response)
        match_pre = pattern.search(response)
        
        if match_full:
            a_score = float(match_full.group(1))
            b_score = float(match_full.group(3))
            if (a_score > b_score):
                counter['Win']+=1
            elif (a_score==b_score):
                counter['Draw']+=1
            elif (a_score < b_score):
                counter['Lose']+=1
        elif match_pre:
                content = match_pre.group(1).lower()
                if content == 'response_a':
                    counter['Win'] += 1
                elif content == 'response_b':
                    counter['Lose'] += 1
                elif content == 'draw':
                    counter['Draw'] +=1
            
            
            
    return counter


#标量（scalar）计算函数，用于计算标量分数的均值与方差；
def sca_calculator(responses: Dict[int, List[str]], pattern_name: str) -> Dict[str, Dict[int, List[float]]]:
    pattern_str = r'\[\[' + re.escape(pattern_name) + r'(-?\d+(?:\.\d+)?)\]\]'
    pattern = re.compile(pattern_str)

    results = {
        pattern_name: {}
    }
    
    for id_, texts in responses.items():
        scores = []
        for text in texts:
            found_matches = pattern.findall(text)
            scores.extend([float(score) for score in found_matches])
        if scores:
            results[pattern_name][id_] = scores
    
    return results


def extract_matches(responses: Dict[int, List[str]], pattern: str) -> Dict[int, List[str]]:
    compiled_pattern = re.compile(pattern)
    matches_per_response = {}

    for id_, texts in responses.items():
        matches = []
        for text in texts:
            found_matches = compiled_pattern.findall(text)
            matches.extend(found_matches)
        matches_per_response[id_] = matches

    return matches_per_response
    

#logprobs 输入输出
def pre_log_probs(log_probs: List[List[Dict[str,float]]], correct_answer: str) -> int :
    options = ['A', 'B', 'C', 'D']
    first_log_prob = log_probs[0]
    
    
    max_prob_option = max(options, key=lambda option: first_log_prob.get(option, float('-inf')))
    
    
    return 1 if max_prob_option == correct_answer else 0
    


