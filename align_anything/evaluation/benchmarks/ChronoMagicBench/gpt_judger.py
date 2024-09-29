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

import os
import requests
from tqdm import tqdm
from align_anything.utils.tools import image_b64

system_prompt = '''
Suppose you are a data rater, specialized in generating scores for time-lapse videos. You will be supplied with eight key frames extracted from a video, each with a filename labeled with its position in the video sequence. Your task is to rate the variation of this time-lapse video on a scale of 5. The scoring criteria are as follows:
1: Minimal change. The scene appears almost like a still image, with static elements remaining motionless and only minor changes in lighting or subtle movements of elements. No significant activity is noticeable.
2: Slight change. There is a small amount of movement or change in the elements of the scene, such as a few people or vehicles moving and minor changes in light or shadows. The overall variation is still minimal, with changes mostly being quantitative.
3: Moderate change. Multiple elements in the scene undergo changes, but the overall pace is slow. This includes gradual changes in daylight, moving clouds, growing plants, or occasional vehicle and pedestrian movements. The scene begins to show a transition from quantitative to qualitative change.
4: Significant change. The elements in the scene show obvious dynamic changes with a higher speed and frequency of variation. This includes noticeable changes in city traffic, crowd activities, or significant weather transitions. The scene displays a mix of quantitative and qualitative changes.
5: Dramatic change. Elements in the scene undergo continuous and rapid significant changes, creating a very rich visual effect. This includes events like sunrise and sunset, construction of buildings, and seasonal changes, making the variation process vivid and impactful. The scene exhibits clear qualitative change.

For guidance on the expected format, refer to the provided examples: 

Brief Reasoning Statement: The changes in the frames are quite significant as we see the progression of the 3D print from start to finish. The variation in the video includes the build-up of the print layer by layer, showing dynamic changes and detailed progress. The scene exhibits both quantitative changes (incremental addition of the print layers) and qualitative changes (completed print and human interaction).
"Score":{4}

Brief Reasoning Statement: The changes in the frames show a significant dynamic process of a lunar eclipse. The video captures the progression from a full moon to a full lunar eclipse and back to a more visible moon, showcasing both quantitative (shadow coverage) and qualitative (color change) changes. The visual effect is rich, with continuous and rapid significant changes in the moon's appearance.
"Score":{5}

Brief Reasoning Statement: The changes in the frames show moderate to significant dynamic activity. The light trails from boats and water traffic, as well as cloud movement, create visual changes that is quantitative. The visual effect includes ongoing changes, but they are not as rapid or continuous as some more dramatic scenes.
"Score":{3}

Attention: Do not reply outside the example template! Below are the video title and input frames:
'''

def build_gpt_prompt(datas, image_dir):
    prompts = {}
    for video_id, data in tqdm(datas.items(), desc="Creating prompts"):
        prompt = [{"type": "text", "text": system_prompt}]
        for image_name in data['frames']:
            prompt.append({"type": "text", "text": image_name.strip()})
            prompt.append({"type": "image_url", "image_url":{"url": f"data:image/png;base64,{image_b64(os.path.join(image_dir, image_name.strip()))}"}})
        
        prompts[video_id] = {
            'prompt': data['prompt'],
            'video_path': data['video_path'],
            'gpt_prompt': prompt
        }

    return prompts

def gpt_judger(answer, api_key, base_url, judge_model="gpt-4o-2024-05-13"):
    def get_response(prompt):
        data = {
            "model": judge_model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    result = get_response(answer)
    return result