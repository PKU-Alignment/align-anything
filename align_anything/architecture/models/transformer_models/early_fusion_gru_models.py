
# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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


import numpy as np
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from open_clip.tokenizer import HFTokenizer
from open_clip.transformer import TextTransformer
from dataclasses import field
from align_anything.architecture.agent import AbstractAgent
from align_anything.architecture.models.transformer_models.image_encoders import *
from align_anything.architecture.models.transformer_models.preprocessors import (
    Preprocessor,
    PreprocessorConfig,
    tensor_image_preprocessor,
    SigLipPreprocessorConfig,
    SigLipPreprocessor,
)
from align_anything.architecture.models.transformer_models.text_cond_visual_encoder import (
    TextCondMultiCameraVisualEncoder,
    TextCondVisualEncoderConfig,
    NonTxMultiCameraVisualEncoder,
    NonTxVisualEncoderConfig,
    TransformerConfig,
)
from align_anything.trainers.text_video_to_action.training.offline.train_utils import load_pl_ckpt
from align_anything.utils.utils.constants.stretch_initialization_utils import ALL_STRETCH_ACTIONS
from align_anything.utils.utils.nn_utils import sample_action_index_from_logits
from align_anything.utils.utils.sensor_constant_utils import is_a_visual_sensor, is_a_non_visual_sensor

EarlyFusionCnnRNNPreprocessorConfig = PreprocessorConfig
EarlyFusionCnnRNNPreprocessor = Preprocessor


@dataclass
class RNNConfig:
    num_layers: int = 3
    input_size: int = 512
    hidden_size: int = 512
    rnn_type: str = "GRU"
    trainable_masked_hidden_state: bool = True


@dataclass
class EarlyFusionCnnRNNConfig:
    visual_encoder: TextCondVisualEncoderConfig = field(default_factory=TextCondVisualEncoderConfig)
    visual_text_encoder_class: str = "TextCondMultiCameraVisualEncoder"
    decoder: RNNConfig = RNNConfig(3, 512, 512, "GRU", True)
    num_actions: int = len(ALL_STRETCH_ACTIONS)
    max_length: int = 1000
    action_loss: bool = True


class EarlyFusionCnnRNN(nn.Module):
    def __init__(
        self,
        cfg: EarlyFusionCnnRNNConfig,
    ):
        super().__init__()
        self.cfg = cfg

        _ENCODER_NAME_TO_CLASS = {
            c.__name__: c
            for c in [
                TextCondMultiCameraVisualEncoder,
                NonTxMultiCameraVisualEncoder,
            ]
        }

        assert self.cfg.visual_text_encoder_class in _ENCODER_NAME_TO_CLASS, (
            f"{self.cfg.visual_text_encoder_class} is not yet implemented,"
            f" only {list(_ENCODER_NAME_TO_CLASS.keys())} are supported."
        )

        self.visual_encoder = _ENCODER_NAME_TO_CLASS[self.cfg.visual_text_encoder_class](
            self.cfg.visual_encoder
        )

        self.decoder = RNNStateEncoder(
            input_size=self.cfg.decoder.input_size,
            hidden_size=self.cfg.decoder.hidden_size,
            num_layers=self.cfg.decoder.num_layers,
            rnn_type=self.cfg.decoder.rnn_type,
            trainable_masked_hidden_state=self.cfg.decoder.trainable_masked_hidden_state,
        )
        self.action_classifier = nn.Linear(self.cfg.decoder.hidden_size, self.cfg.num_actions)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.input_sensors = self.cfg.visual_encoder.input_sensors
        if "last_actions" in self.input_sensors:
            # if num_actions=20; then 0-19 are actions, 20 is for "" (start token), and 21 is for padding
            self.last_actions_embed = nn.Embedding(
                self.cfg.num_actions + 2,
                self.cfg.decoder.hidden_size,
                padding_idx=self.cfg.num_actions + 1,
            )
            self.last_actions_embed.weight.data.uniform_(-0.01, 0.01)

        if "an_object_is_in_hand" in self.input_sensors:
            self.object_in_hand_embed = nn.Embedding(3, self.cfg.decoder.hidden_size)
            self.object_in_hand_embed.weight.data.uniform_(-0.01, 0.01)

    def mock_batch(self):
        B, T, C, H, W = 2, 10, 3, 224, 384
        L = 15
        frames = torch.rand((B, T, C, H, W), dtype=torch.float32)
        goals = dict(
            input_ids=torch.randint(0, 10, size=[B, L]),
            attention_mask=torch.ones([B, L], dtype=torch.bool),
        )
        actions = torch.randint(0, self.cfg.num_actions, size=[B, T])
        padding_mask = torch.zeros([B, T], dtype=torch.bool)
        time_ids = torch.arange(T).unsqueeze(0).tile(B, 1)
        return goals, frames, actions, time_ids, padding_mask

    def compute_loss(self, logits, actions):
        B, T, C = logits.shape
        return self.ce_loss(logits.reshape(-1, C), actions.reshape(-1))

    def get_input_embedding_per_timestep(
        self,
        visual_sensors,
        non_visual_sensors,
        goals,
        time_ids,
        text_features=None,
    ):
        visual_feats, text_feats = self.visual_encoder(
            visual_sensors, goals, text_features, non_visual_sensors
        )

        if "last_actions" in non_visual_sensors:
            last_actions_enc = self.last_actions_embed(non_visual_sensors["last_actions"])
            visual_feats = visual_feats + last_actions_enc

        if "an_object_is_in_hand" in non_visual_sensors:
            object_in_hand_enc = self.object_in_hand_embed(
                non_visual_sensors["an_object_is_in_hand"]
            )
            visual_feats = visual_feats + object_in_hand_enc

        return visual_feats, text_feats

    def decode_and_get_logits(self, embedded_features, mask, memory):
        decoder_output, memory = self.decoder(
            x=embedded_features,
            hidden_states=memory,
            masks=mask,
        )
        decoder_output = decoder_output.permute(1, 0, 2)
        logits = dict(actions_logits=self.action_classifier(decoder_output))
        return logits, memory

    def forward(self, batch):
        goals = batch["goals"]
        time_ids = batch["time_ids"]
        padding_mask = batch["padding_mask"]

        visual_sensors = {key: obs for (key, obs) in batch.items() if is_a_visual_sensor(key)}
        non_visual_sensors = {
            key: obs for (key, obs) in batch.items() if is_a_non_visual_sensor(key)
        }

        embedded_features, text_feats = self.get_input_embedding_per_timestep(
            visual_sensors,
            non_visual_sensors,
            goals,
            time_ids,
        )

        embedded_features = embedded_features.permute(1, 0, 2)

        mask = ~padding_mask
        mask[:, 0] = False
        mask = mask.permute(1, 0).unsqueeze(2).float()

        memory = torch.zeros(
            (self.cfg.decoder.num_layers, mask.shape[1], self.cfg.decoder.hidden_size)
        ).to(embedded_features.device)

        logits, _ = self.decode_and_get_logits(embedded_features, mask, memory)

        outputs = dict(**logits)
        if self.cfg.action_loss:
            action_loss = self.compute_loss(logits["actions_logits"], batch["actions"])
            outputs["actions_loss"] = action_loss
            outputs["loss"] = action_loss

        return outputs

    @classmethod
    def build_model(
        cls,
        model_version,
        input_sensors,
        loss,
        ckpt_pth=None,
    ):
        model_cfg = EarlyFusionCnnRNNConfig()
        model_cfg.action_loss = "action" in loss
        model_cfg.visual_encoder.input_sensors = input_sensors
        if model_version == "small_3" or model_version == "small":
            model_cfg.visual_encoder.image_encoder = "Dinov2Small"
            model_cfg.visual_encoder.text_encoder = "t5-small"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = RNNConfig(3, 512, 512, "GRU", True)
        elif model_version == "small_6":
            model_cfg.visual_encoder.image_encoder = "Dinov2Small"
            model_cfg.visual_encoder.text_encoder = "t5-small"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(6, 512, 8)
            model_cfg.decoder = RNNConfig(6, 512, 512, "GRU", True)
        elif model_version == "base_3":
            model_cfg.visual_encoder.image_encoder = "Dinov2Base"
            model_cfg.visual_encoder.text_encoder = "t5-small"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = RNNConfig(3, 512, 512, "GRU", True)
        elif model_version == "base_6":
            model_cfg.visual_encoder.image_encoder = "Dinov2Base"
            model_cfg.visual_encoder.text_encoder = "t5-small"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(6, 768, 8)
            model_cfg.decoder = RNNConfig(6, 768, 768, "GRU", True)
        elif model_version == "siglip_base_3" or model_version == "siglip_3":
            model_cfg.visual_encoder.image_encoder = "SigLIPBase"
            model_cfg.visual_encoder.text_encoder = "SigLIPBase"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = RNNConfig(3, 512, 512, "GRU", True)
        elif model_version == "siglip_large_3":
            model_cfg.visual_encoder.image_encoder = "SigLIPLarge"
            model_cfg.visual_encoder.text_encoder = "SigLIPLarge"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = RNNConfig(3, 512, 512, "GRU", True)
        elif model_version == "siglip_base_3_nonTxEnc":
            model_cfg.visual_text_encoder_class = "NonTxMultiCameraVisualEncoder"
            model_cfg.visual_encoder = NonTxVisualEncoderConfig()
            model_cfg.visual_encoder.image_encoder = "SigLIPBase"
            model_cfg.visual_encoder.text_encoder = "SigLIPBase"
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.visual_encoder.input_sensors = input_sensors
            model_cfg.decoder = RNNConfig(3, 512, 512, "GRU", True)
        else:
            raise NotImplementedError

        model = EarlyFusionCnnRNN(model_cfg)
        if ckpt_pth is not None:
            load_pl_ckpt(model, ckpt_pth)

        if "siglip" in model_version.lower():
            preproc_cfg = SigLipPreprocessorConfig(
                model_version=model.visual_encoder.image_encoder.cfg.model,
                text_encoder_context_length=model.visual_encoder.image_encoder.context_length,
            )
            preprocessor_type = SigLipPreprocessor
        else:
            preproc_cfg = EarlyFusionCnnRNNPreprocessorConfig()
            preprocessor_type = EarlyFusionCnnRNNPreprocessor

        preproc_cfg.data_augmentation = True
        preproc_cfg.augmentation_version = "v2"
        preproc = preprocessor_type(cfg=preproc_cfg, device="cpu")
        return model, preproc

    @classmethod
    def build_agent(
        cls,
        model_version,
        input_sensors,
        loss,
        device,
        sampling,
        ckpt_pth=None,
    ):
        model, preproc = cls.build_model(model_version, input_sensors, loss, ckpt_pth)
        return EarlyFusionCnnRNNAgent(model, preproc, device, sampling)


class EarlyFusionCnnRNNAgent(AbstractAgent):
    def __init__(self, model, preprocessor, device, sampling="greedy", max_seq_len=1000):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.max_seq_len = max_seq_len
        self.sampling = sampling
        self.reset()
        self.model = self.model.to(self.device)
        self.preprocessor.device = self.device

    def reset(self):
        self.curr_t = 0
        self.preprocessor.image_preprocessor = tensor_image_preprocessor(
            size=(
                (256, 256)
                if isinstance(self.model.visual_encoder.image_encoder, SigLIP)
                else (224, 384)
            ),
            data_augmentation=self.preprocessor.cfg.data_augmentation,
            specific=False,
            augmentation_version=self.preprocessor.cfg.augmentation_version,
            mean=(
                (0.5, 0.5, 0.5)
                if isinstance(self.model.visual_encoder.image_encoder, SigLIP)
                else (0.48145466, 0.4578275, 0.40821073)
            ),
            std=(
                (0.5, 0.5, 0.5)
                if isinstance(self.model.visual_encoder.image_encoder, SigLIP)
                else (0.26862954, 0.26130258, 0.27577711)
            ),
        )
        self.cache = dict()

    def get_action_list(self):
        return self.preprocessor.cfg.action_list

    def process_sensors_for_model_eval(self, observations):
        observations = {
            k: torch.tensor(np.array([v])).to(self.device) for (k, v) in observations.items()
        }

        frames_dict = {
            sensor: self.preprocessor.process_frames([observations], sensor)
            for (sensor, frame) in observations.items()
            if is_a_visual_sensor(sensor)
        }

        preprocessed_nonvisual_sensors = {}
        if "last_actions" in self.model.input_sensors:
            start_token = self.preprocessor.action2idx[""]
            preprocessed_nonvisual_sensors["last_actions"] = (
                torch.tensor(np.array([[start_token]])).to(self.device)
                if self.curr_t == 0
                else self.cache["last_actions"]
            )

        if "an_object_is_in_hand" in self.model.input_sensors:
            observations["an_object_is_in_hand"] = observations["an_object_is_in_hand"][:, 0]
            preprocessed_nonvisual_sensors["an_object_is_in_hand"] = (
                self.preprocessor.process_objinhand([observations])
            )

        return dict(
            visual_sensors=frames_dict,
            non_visual_sensors=preprocessed_nonvisual_sensors,
        )

    def get_action(self, observations, goal_spec):
        processed_observations = self.process_sensors_for_model_eval(observations)

        if self.curr_t == 0:
            if isinstance(self.preprocessor.text_preprocessor, (TextTransformer, HFTokenizer)):
                goal = self.preprocessor.text_preprocessor([goal_spec], context_length=64).to(
                    self.preprocessor.device
                )
                self.cache["goal"] = goal
            else:
                goal = self.preprocessor.text_preprocessor([goal_spec], return_tensors="pt")
                self.cache["goal"] = {k: v.to(self.device) for k, v in goal.items()}

            text_feats = self.model.visual_encoder.encode_text(self.cache["goal"])
            self.cache["text_feats"] = text_feats
        else:
            goal = self.cache["goal"]
            text_feats = self.cache["text_feats"]

        embedded_features, _ = self.model.get_input_embedding_per_timestep(
            processed_observations["visual_sensors"],
            processed_observations["non_visual_sensors"],
            None,
            time_ids=torch.tensor([[self.curr_t]]).to(self.device),
            text_features=text_feats,
        )

        if self.curr_t == 0:
            memory = torch.zeros(
                (self.model.cfg.decoder.num_layers, 1, self.model.cfg.decoder.hidden_size)
            ).to(embedded_features.device)
            self.cache["memory"] = memory
        memory = self.cache["memory"]

        embedded_features = embedded_features.permute(1, 0, 2)

        if self.curr_t == 0:
            mask = torch.zeros((1, 1, 1)).to(embedded_features)
        else:
            mask = torch.ones((1, 1, 1)).to(embedded_features.device)

        logits, memory = self.model.decode_and_get_logits(
            embedded_features,
            mask,
            memory,
        )
        self.cache["memory"] = memory

        curr_logits = logits["actions_logits"][0, -1]
        action_idx = sample_action_index_from_logits(
            curr_logits,
            self.sampling,
            self.preprocessor.cfg.action_list,
        )
        action = self.preprocessor.cfg.action_list[action_idx]

        if "last_actions" in self.model.input_sensors:
            self.cache["last_actions"] = action_idx.reshape(1, 1)

        self.curr_t += 1

        return action, torch.softmax(curr_logits, -1)
