# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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

import importlib
import os
import sys


benchmark_modules = [
    'benchmarks.text_to_text.gsm8k.eval',
    'benchmarks.text_to_text.mmlu.eval',
    'benchmarks.text_to_text.HumanEval.eval',
    'benchmarks.text_to_text.AGIEval.eval',
    'benchmarks.text_to_text.TruthfulQA.eval',
    'benchmarks.text_to_text.ARC.eval',
    'benchmarks.text_to_text.CMMLU.eval',
    'benchmarks.text_to_text.MMLUPRO.eval',
    'benchmarks.text_to_text.CEval.eval',
    'benchmarks.text_to_text.DoAnythingNow.eval',
    # 来自 my-dev 的模块
    'benchmarks.text_to_text.AdvBench.eval',
    'benchmarks.text_to_text.Anthropics.eval',
    'benchmarks.text_to_text.BBQ.eval',
    'benchmarks.text_to_text.BeaverTails.eval',
    'benchmarks.text_to_text.CDialBias.eval',
    'benchmarks.text_to_text.Cona.eval',
    'benchmarks.text_to_text.CyberAttackAssistance.eval',
    'benchmarks.text_to_text.Dice.eval',
    'benchmarks.text_to_text.Confaide.eval',
    'benchmarks.text_to_text.DecodingTrust.eval',
    'benchmarks.text_to_text.Bad.eval',
    # 来自 main 的模块
    'benchmarks.text_to_text.DoNotAnswer.eval',
    'benchmarks.text_to_text.HarmBench.eval',
    'benchmarks.text_to_text.RedEval.eval',
    'benchmarks.text_to_text.HExPHI.eval',
    'benchmarks.text_to_text.LatentJailbreak.eval',
    'benchmarks.text_to_text.MaliciousInstruct.eval',
    'benchmarks.text_to_text.MaliciousInstructions.eval',
    'benchmarks.text_to_text.HarmfulQ.eval',
    'benchmarks.text_to_text.gptfuzzer.eval',
    'benchmarks.text_to_text.llm_jailbreak_study.eval',
    'benchmarks.text_to_text.jbb_behaviors.eval',
    'benchmarks.text_to_text.salad_bench.eval',
    'benchmarks.text_to_text.air_bench_2024.eval',
    'benchmarks.text_to_text.aegis_aicontent_safety_dataset.eval',
    'benchmarks.text_to_text.s_eval.eval',
    'benchmarks.text_to_text.FakeAlignment.eval',
    'benchmarks.text_to_text.DeceptionBench.eval',
    'benchmarks.text_to_text.Flames.eval',
    'benchmarks.text_to_text.XSafety.eval',
    'benchmarks.text_to_text.jade_db.eval',
    'benchmarks.text_to_text.MoralBench.eval',
    # 共同模块
    'benchmarks.text_image_to_text.mmmu.eval',
    'benchmarks.text_image_to_text.mathvision.eval',
    'benchmarks.text_audio_to_text.mmau.eval',
    'benchmarks.text_video_to_text.mmvu.eval',
    # 'benchmarks.text_vision_to_action.chores.eval',
]
script_name = os.path.basename(sys.argv[0])
if script_name == 'eval_vla.sh':
    benchmark_modules = [
        'benchmarks.text_vision_to_action.chores.eval',
    ]

for module in benchmark_modules:
    # print(f"Importing: {module}")
    importlib.import_module(module)
