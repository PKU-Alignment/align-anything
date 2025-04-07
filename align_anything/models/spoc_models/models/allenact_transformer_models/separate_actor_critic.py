# Copyright 2024 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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
from allenact.base_abstractions.misc import SafeActorCriticOutput

from align_anything.models.spoc_models.models.allenact_transformer_models.allenact_dino_transformer import (
    DinoLLAMATxNavActorCritic,
)


class DinoLLAMATxNavActorCriticSeparate(DinoLLAMATxNavActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_tsfm = DinoLLAMATxNavActorCritic(*args, **kwargs)

    def forward(self, *args, **kwargs):
        actor_output, memory = super().forward(*args, **kwargs)
        critic_output, critic_memory = self.critic_tsfm(*args, **kwargs)

        critic_output.distributions = actor_output.distributions

        return critic_output, critic_memory


class SafeDinoLLAMATxNavActorCriticSeparate(DinoLLAMATxNavActorCriticSeparate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_critic_tsfm = DinoLLAMATxNavActorCritic(*args, **kwargs)

    def forward(self, *args, **kwargs):
        actor_output, memory = super().forward(*args, **kwargs)
        c_critic_output, c_critic_memory = self.c_critic_tsfm(*args, **kwargs)

        actor_critic_output = SafeActorCriticOutput(
            distributions=actor_output.distributions,
            values=actor_output.values,
            c_values=c_critic_output.values,
            extras=c_critic_output.extras,
        )
        return actor_critic_output, c_critic_memory
