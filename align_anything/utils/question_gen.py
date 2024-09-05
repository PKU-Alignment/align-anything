# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the Yushi-Hu's tifa library.
# https://github.com/Yushi-Hu/tifa/blob/main/tifascore/question_gen.py
# https://github.com/Yushi-Hu/tifa/blob/main/tifascore/question_filter.py
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

from word2number import w2n
from statistics import mean
from tqdm import tqdm
import requests
import json

categories = ['object', 'human', 'animal', 'food', 'activity', 'attribute', 'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']

system_prompt = """
Given a image descriptions, generate one or two multiple-choice questions that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.

Description: A man posing for a selfie in a jacket and bow tie.
Entities: man, selfie, jacket, bow tie
Activities: posing
Colors:
Counting:
Other attributes:
Questions and answers are below:
About man (human):
Q: is this a man?
Choices: yes, no
A: yes
Q: who is posing for a selfie?
Choices: man, woman, boy, girl
A: man
About selfie (activity):
Q: is the man taking a selfie?
Choices: yes, no
A: yes
Q: what type of photo is the person taking?
Choices: selfie, landscape, sports, portrait
A: selfie
About jacket (object):
Q: is the man wearing a jacket?
Choices: yes, no
A: yes
Q: what is the man wearing?
Choices:jacket, t-shirt, tuxedo, swearter
A: jacket
About bow tie (object):
Q: is the man wearing a bow tie?
Choices: yes, no
A: yes
Q: is the man wearing a bow tie or a neck tie?
Choices: bow tie, neck tie, cravat, bolo tie
A: bow tie
About posing (activity):
Q: is the man posing for the selfie?
Choices: yes, no
A: yes
Q: what is the man doing besides taking the selfie?
Choices: posing, waving, nothing, shaking
A: posing

Description: A horse and several cows feed on hay.
Entities: horse, cows, hay
Activities: feed on
Colors:
Counting: several
Other attributes:
Questions and answers are below:
About horse (animal):
Q: is there a horse?
Choices: yes, no
A: yes
About cows (animal):
Q: are there cows?
Choices: yes, no
A: yes
About hay (object):
Q: is there hay?
Choices: yes, no
A: yes
Q: what is the horse and cows feeding on?
Choices: hay, grass, leaves, twigs
A: hay
About feed on (activity):
Q: are the horse and cows feeding on hay?
Choices: yes, no
A: yes
About several (counting):
Q: are there several cows?
Choices: yes, no
A: yes

Description: A red colored dog.
Entities: dog
Activities:
Colors: red
Counting:
Other attributes:
Questions and answers are below:
About dog (animal):
Q: is this a dog?
Choices: yes, no
A: yes
Q: what animal is in the picture?
Choices: dog, cat, bird, fish
A: dog
About red (color):
Q: is the dog red?
Choices: yes, no
A: yes
Q: what color is the dog?
Choices: red, black, white, yellow
A: red

Description: A busy intersection with an ice cream truck driving by.
Entities: intersection, ice cream truck
Activities: driving by
Colors:
Counting:
Other attributes: busy
Questions and answers are below:
About intersection (location):
Q: is this an intersection?
Choices: yes, no
A: yes
Q: is this a intersection or a straight road?
Choices: intersection, straight road, parking lot, school
A: intersection
About ice cream truck (object):
Q: is this an ice cream truck?
Choices: yes, no
A: yes
Q: what type of truck is driving by?
Choices: ice cream truck, car, pickup truck, dumper truck
A: ice cream truck
About driving by (activity):
Q: is the ice cream truck driving by?
Choices: yes, no
A: yes
About busy (attribute):
Q: is this a busy intersection?
Choices: yes, no
A: yes
Q: is this a busy or a quiet intersection?
Choices: busy, quiet, silent, noisy
A: busy

Description: Portrait of a gecko wearing a train conductor's hat and holding a flag that has a yin-yang symbol on it. Woodcut.
Entities: gecko, train conductor's hat, flag, yin-yang symbol
Activities: wearing, holding
Colors:
Counting:
Other attributes: portrait, woodcut
Questions and answers are below:
About gecko (animal):
Q: is this a gecko?
Choices: yes, no
A: yes
Q: what animal is in the photo?
Choices: gecko, human, snake, frog
A: gecko
About train conductor's hat (object):
Q: is the gecko wearing a train conductor's hat?
Choices: yes, no
A: yes
About flag (object):
Q: is the gecko holding a flag?
Choices: yes, no
A: yes
Q: what is the gecko holding?
Choices: flag, sign, banner, poster
A: flag
About yin-yang symbol (object):
Q: is there a yin-yang symbol on the flag?
Choices: yes, no
A: yes
Q: what is on the flag?
Choices: yin-yang symbol, star, heart, cross
A: yin-yang symbol
About wearing (activity):
Q: is the gecko wearing a train conductor's hat?
Choices: yes, no
A: yes
About holding (activity):
Q: is the gecko holding a flag?
Choices: yes, no
A: yes
About portrait (attribute):
Q: is this a portrait?
Choices: yes, no
A: yes
About woodcut (attribute):
Q: is this a woodcut?
Choices: yes, no
A: yes
Q: is this a woodcut, a painting, or a photograph?
Choices: woodcut, painting, drawing, photograph
A: woodcut

Description: A woman is showing a watermelon slice to a woman on a scooter.
Entities: woman, watermelon slice, woman, scooter
Activities: showing
Colors:
Counting:
Other attributes: on
Questions and answers are below:
About woman (human):
Q: is this a woman?
Choices: yes, no
A: yes
Q: who is showing a watermelon slice?
Choices: woman, man, boy, girl
A: woman
About watermelon slice (food):
Q: is there a watermelon slice?
Choices: yes, no
A: yes
Q: what is the woman showing to a woman on a scooter?
Choices: watermelon slice, apple, orange, banana
A: watermelon slice
About woman (human):
Q: is there a woman on a scooter?
Choices: yes, no
A: yes
Q: who is on a scooter?
Choices: woman, man, boy, girl
A: woman
About scooter (object):
Q: is there a scooter?
Choices: yes, no
A: yes
Q: what vehicle is one of the woman on?
Choices: scooter, bicycle, motorcycle, car
A: scooter
About showing (activity):
Q: is the woman showing a watermelon slice to another woman?
Choices: yes, no
A: yes
About on (spatial):
Q: is one of the woman on a scooter?
Choices: yes, no
A: yes
Q: is the woman on the scooter or standing next to it?
Choices: on, next to, behind, in front of
A: on

Description: A photo of three dogs.
Entities: dogs
Activities:
Colors:
Counting: three
Other attributes: photo
Questions and answers are below:
About dogs (animal):
Q: are there dogs?
Choices: yes, no
A: yes
Q: what animals are in the photo?
Choices: dogs, cats, birds, fish
A: dogs
About three (counting):
Q: are there three dogs?
Choices: yes, no
A: yes
Q: how many dogs are in the photo?
Choices: 1, 2, 3, 4
A: 3
About photo (attribute):
Q: is this a photo?
Choices: yes, no
A: yes
Q: is this a photo, a painting, or a comic?
Choices: photo, painting, comic, sculpture
A: photo

Description: A white milk truck with a license plate that reads \"pro milk.\"
Entities: milk truck, license plate
Activities:
Colors: white
Counting:
Other attributes: pro milk
Questions and answers are below:
About milk truck (object):
Q: is this a milk truck?
Choices: yes, no
A: yes
About license plate (object):
Q: is there a license plate on the vehicle?
Choices: yes, no
A: yes
About white (color):
Q: is the truck white?
Choices: yes, no
A: yes
Q: what color is the truck?
Choices: white, black, red, blue
A: white
About pro milk (other):
Q: does the license plate read \"pro milk\"?
Choices: yes, no
A: yes

Description: A person sitting on a horse in air over gate in grass with people and trees in background.
Entities: person, horse, gate, grass, people, trees
Activities: sitting
Colors:
Counting:
Other attributes: over, in air, background
Questions and answers are below:
About person (human):
Q: is there a person on a horse?
Choices: yes, no
A: yes
Q: who is sitting on a horse?
Choices: person, animal, robot, alien
A: person
About horse (animal):
Q: is this a horse?
Choices: yes, no
A: yes
Q: what animal is in the picture?
Choices: horse, cow, sheep, goat
A: horse
About gate (object):
Q: is there a gate?
Choices: yes, no
A: yes
About grass (object):
Q: is there grass?
Choices: yes, no
A: yes
About people (human):
Q: are there people in the background?
Choices: yes, no
A: yes
About trees (object):
Q: are there trees in the background?
Choices: yes, no
A: yes
About sitting (activity):
Q: is the person sitting on the horse?
Choices: yes, no
A: yes
About over (spatial):
Q: is the horse over the gate?
Choices: yes, no
A: yes
Q: is the horse over or next to the gate?
Choices: over, under, next to, behind
A: over
About in air (attribute):
Q: is the horse in air?
Choices: yes, no
A: yes
Q: is the horse in air or on the ground?
Choices: in air, on the ground, in the water, in the sky
A: in air
About background (attribute):
Q: are there people and trees in the background?
Choices: yes, no
A: yes
Q: are there people and trees in the background or in the foreground?
Choices: background, foreground, middle ground, sky
A: background

Description: a red blue and yellow train and some people on a platform
Entities: train, people, platform
Activities:
Colors: red blue and yellow
Counting: some
Other attributes:
Questions and answers are below:
About train (object):
Q: is this a train?
Choices: yes, no
A: yes
Q: what type of vehicle is this?
Choices: train, car, motorcycle, bus
A: train
About people (human):
Q: are there people on the platform?
Choices: yes, no
A: yes
About platform (location):
Q: is there a platform?
Choices: yes, no
A: yes
Q: what type of place is this?
Choices: platform, parking lot, airport, bus stop
A: platform
About red blue and yellow (color):
Q: is the train red blue and yellow?
Choices: yes, no
A: yes
Q: what color is the train?
Choices: red blue and yellow, pure blue, blue and yellow, red and blue
A: red blue and yellow
About some (counting):
Q: are there some people?
Choices: yes, no
A: yes

Description: square blue apples on a tree with circular yellow leaves
Entities: apples, tree, leaves
Activities:
Colors: blue, yellow
Counting:
Other attributes: square, circular
Questions and answers are below:
About apples (food):
Q: are there apples on the tree?
Choices: yes, no
A: yes
Q: what fruit is on the tree?
Choices: apples, oranges, bananas, pears
A: apples
About tree (object):
Q: is this a tree?
Choices: yes, no
A: yes
Q: what type of plant is this?
Choices: tree, flower, bush, grass
A: tree
About leaves (object):
Q: are there leaves on the tree?
Choices: yes, no
A: yes
Q: what is on the tree?
Choices: leaves, branches, roots, flowers
A: leaves
About blue (color):
Q: are the apples blue?
Choices: yes, no
A: yes
Q: what color are the apples?
Choices: blue, red, yellow, green
A: blue
About yellow (color):
Q: are the leaves yellow?
Choices: yes, no
A: yes
Q: what color are the leaves?
Choices: yellow, red, blue, green
A: yellow
About square (shape):
Q: are the apples square?
Choices: yes, no
A: yes
Q: what shape are the apples?
Choices: square, round, oval, triangle
A: square
About circular (shape):
Q: are the leaves circular?
Choices: yes, no
A: yes
Q: what shape are the leaves?
Choices: circular, square, oval, triangle
A: circular

Description: Overview of a pot of vegetable soup with wooden spoon on a stainless steel stove top.
Entities: pot, vegetable soup, spoon, stove top
Activities:
Colors:
Counting:
Other attributes: overview, wooden, stainless steel
Questions and answers are below:
About pot (object):
Q: is there a pot of vegetable soup?
Choices: yes, no
A: yes
Q: what type of container is this?
Choices: pot, bowl, cup, glass
A: pot
About vegetable soup (food):
Q: is there vegetable soup?
Choices: yes, no
A: yes
Q: what type of food is this?
Choices: vegetable soup, chicken soup, beef soup, fish soup
A: vegetable soup
About spoon (object):
Q: is there a spoon?
Choices: yes, no
A: yes
Q: what type of utensil is this?
Choices: spoon, fork, knife, chopsticks
A: spoon
About stove top (object):
Q: is there a stove top?
Choices: yes, no
A: yes
Q: what type of appliance is this?
Choices: stove top, oven, microwave, refrigerator
A: stove top
About overview (attribute):
Q: is this an overview?
Choices: yes, no
A: yes
About wooden (material):
Q: is the spoon wooden?
Choices: yes, no
A: yes
Q: what is the material of the spoon?
Choices: wooden, metal, plastic, glass
A: wooden
About stainless steel (material):
Q: is the stove top stainless steel?
Choices: yes, no
A: yes
Q: what is the material of the stove top?
Choices: stainless steel, iron, aluminum, copper
A: stainless steel

Description: """

def get_tifa_score(vqa_model, question_answer_pairs, img_path):
    tifa_scores = []
    question_logs = {}
    
    for question_answer_pair in tqdm(question_answer_pairs):
        if question_answer_pair['question'] not in question_logs:
            question_logs[question_answer_pair['question']] = question_answer_pair
        choices=question_answer_pair['choices']
        vqa_answer = vqa_model.multiple_choice_vqa(img_path, question_answer_pair['question'], choices=choices)

        free_form_answer, mc_answer = vqa_answer["free_form_answer"], vqa_answer["multiple_choice_answer"]
        question_logs[question_answer_pair['question']]['free_form_vqa'] = free_form_answer
        question_logs[question_answer_pair['question']]['multiple_choice_vqa'] = mc_answer
        score = int(mc_answer == question_answer_pair['answer'])
        question_logs[question_answer_pair['question']]['scores'] = score
        
        tifa_scores.append(score)
        
    question_logs = dict(question_logs)
    result_dict = {}
    averaged_scores = mean(tifa_scores)
    result_dict = {"tifa_score": averaged_scores} 
    result_dict["question_details"] = question_logs
    
    return result_dict

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1 = 1, 1, 1
        else:
            precision, recall, F1 = 0, 0, 0
    return float(F1)

def QA_filter(qa_model, caption_qas):
    filtered_question_instances = []
    
    question_set = set()
    
    for question_instance in caption_qas:
        question = question_instance['question']
        caption = question_instance['caption']
        choices = question_instance['choices']
        
        if question in question_set:
            continue
        else:
            question_set.add(question)
            
        qa_answer = qa_model.mcqa(question, caption, choices=choices)
        if qa_answer != question_instance['answer']:
            continue
        
        if question_instance['answer'] not in ['yes', 'no']:
            free_form_answer = qa_model.qa(question, caption).strip()
            
            gpt_answer = question_instance['answer']
            if gpt_answer.isnumeric():
                try:
                    free_form_answer = str(w2n.word_to_num(free_form_answer))
                except:
                    pass
            
            if compute_prf(gpt_answer.split(), free_form_answer.split()) <= 0.6:
                continue
            
        filtered_question_instances.append(question_instance)
        
    return filtered_question_instances

def parse_resp(resp):
    resp = resp.split('\n')
    question_instances = []
    
    this_entity = None
    this_type = None
    this_question = None
    this_choices = None
    this_answer = None
    
    for line_number in range(6, len(resp)):
        line = resp[line_number]
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]
            
        elif line.startswith('Q: '):
            this_question = line[3:]
        elif line.startswith('Choices: '):
            this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            this_answer = line[3:]
            
            if this_entity and this_question and this_choices:
                question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
            this_question = None
            this_choices = None
            this_answer = None
            
    return question_instances

def openai_completion(prompt, api_key, base_url, gpt_model, max_tokens=700, temperature=0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": gpt_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["\n\n", "<|endoftext|>"]
    }
    response = requests.post(base_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        response.raise_for_status()
        
def get_QA_pair(caption, api_key, base_url, gpt_model="gpt-4-turbo"):
    prompt = system_prompt + caption+ "\nEntities"
    resp = openai_completion(prompt, api_key, base_url, gpt_model)
    question_instances = parse_resp(resp)
    
    caption_qas = []
    for question_instance in question_instances:
        caption_qa = {}
        caption_qa['caption'] = caption
        caption_qa['element'] = question_instance[0]
        caption_qa['question'] = question_instance[1]
        caption_qa['choices'] = question_instance[2]
        caption_qa['answer'] = question_instance[3]
        caption_qa['element_type'] = question_instance[4]
        
        if question_instance[4] not in categories:
            continue
        if caption_qa['element_type'] in ['animal', 'human']:
            caption_qa['element_type'] = 'animal/human'
            
        caption_qas.append(caption_qa)
        
    return caption_qas