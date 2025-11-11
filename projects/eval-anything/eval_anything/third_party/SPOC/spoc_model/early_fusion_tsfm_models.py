# Copyright 2024 Allen Institute for AI
# ==============================================================================

import json
import os
from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn
from open_clip.tokenizer import HFTokenizer
from open_clip.transformer import TextTransformer
from transformers import PretrainedConfig, PreTrainedModel

from eval_anything.third_party.SPOC.spoc_model.agent import AbstractAgent
from eval_anything.third_party.SPOC.spoc_model.preprocessors import (
    Preprocessor,
    PreprocessorConfig,
    tensor_image_preprocessor,
)
from eval_anything.third_party.SPOC.spoc_model.text_cond_visual_encoder import (
    PositionalEncoder,
    TextCondMultiCameraVisualEncoder,
    TextCondVisualEncoderConfig,
    TransformerConfig,
)
from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import (
    ALL_STRETCH_ACTIONS,
)
from eval_anything.third_party.SPOC.utils.type_utils import THORActions


EarlyFusionCnnTransformerPreprocessorConfig = PreprocessorConfig
EarlyFusionCnnTransformerPreprocessor = Preprocessor


def is_a_visual_sensor(sensor):
    return sensor in [
        'raw_manipulation_camera',
        'raw_navigation_camera',
        'raw_navigation_camera_2',
        'raw_manipulation_camera_2',
    ]  # more can be added later


def create_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.full([T, T], float('-inf'), device=device), diagonal=1)


def sample_action_index_from_logits(
    logits: torch.Tensor,
    sampling: Literal[
        'greedy', 'sample', 'sample_done_only_if_argmax', 'sample_done_only_if_prob_gt_thresh'
    ],
    action_list: List[str] = None,
) -> torch.Tensor:
    assert len(logits.shape) == 1, f'expected logits to be 1D, got {logits.shape}'
    if sampling == 'greedy':
        action_idx = torch.argmax(logits, dim=-1)
    elif sampling == 'sample':
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    elif sampling == 'sample_done_only_if_argmax':
        assert action_list is not None, f'action_list must be provided for {sampling}'
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        # THORActions.done action is really "end"; but checking "done" too if we ever decide to make it "done"
        sampled_done = action_list[action_idx] in [THORActions.done, THORActions.sub_done]
        is_argmax = action_idx == torch.argmax(logits)
        if sampled_done and not is_argmax:
            while action_list[action_idx] in [THORActions.done, THORActions.sub_done]:
                action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    elif sampling == 'sample_done_only_if_prob_gt_thresh':
        assert action_list is not None, f'action_list must be provided for {sampling}'
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        sampled_done = action_list[action_idx] in [THORActions.done, THORActions.sub_done]
        probs = torch.softmax(logits, dim=-1)
        is_gt_thresh = probs[action_idx] > 0.3
        if sampled_done and not is_gt_thresh:
            while action_list[action_idx] in [THORActions.done, THORActions.sub_done]:
                action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    else:
        raise NotImplementedError(f'unknown sampling method {sampling}')

    return action_idx


class EarlyFusionCnnTransformerConfig(PretrainedConfig):
    model_type = 'MM'

    def __init__(
        self,
        visual_encoder: dict = None,
        visual_text_encoder_class: str = 'TextCondMultiCameraVisualEncoder',
        decoder: dict = None,
        num_actions: int = len(ALL_STRETCH_ACTIONS),
        max_length: int = 1000,
        action_loss: bool = True,
        use_llama_decoder: bool = True,
        **kwargs,
    ):
        # 将字典转换为对应的配置对象
        self.visual_encoder = (
            TextCondVisualEncoderConfig(**visual_encoder)
            if visual_encoder
            else TextCondVisualEncoderConfig()
        )
        self.decoder = TransformerConfig(**decoder) if decoder else TransformerConfig(3, 512, 8)

        self.visual_text_encoder_class = visual_text_encoder_class
        self.num_actions = num_actions
        self.max_length = max_length
        self.action_loss = action_loss
        self.use_llama_decoder = use_llama_decoder

        super().__init__(**kwargs)

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict['visual_encoder'] = self.visual_encoder.to_dict()
        config_dict['decoder'] = self.decoder.to_dict()
        return config_dict

    def save_pretrained(self, save_directory):
        """
        Override save_pretrained to ensure proper saving of configuration.
        """
        config_dict = self.to_dict()
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class EarlyFusionCnnTransformer(PreTrainedModel):
    config_class = EarlyFusionCnnTransformerConfig

    def __init__(
        self,
        cfg: EarlyFusionCnnTransformerConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg

        self.visual_encoder = TextCondMultiCameraVisualEncoder(self.cfg.visual_encoder)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.cfg.decoder.d_model, nhead=self.cfg.decoder.nhead, batch_first=True
            ),
            num_layers=self.cfg.decoder.num_layers,
        )
        self.action_classifier = nn.Linear(self.cfg.decoder.d_model, self.cfg.num_actions)
        self.time_encoder = PositionalEncoder(self.cfg.decoder.d_model, self.cfg.max_length)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.input_sensors = self.cfg.visual_encoder.input_sensors
        if 'last_actions' in self.input_sensors:
            # if num_actions=20; then 0-19 are actions, 20 is for "" (start token), and 21 is for padding
            self.last_actions_embed = nn.Embedding(
                self.cfg.num_actions + 2,
                self.cfg.decoder.d_model,
                padding_idx=self.cfg.num_actions + 1,
            )
            self.last_actions_embed.weight.data.uniform_(-0.01, 0.01)

        if 'an_object_is_in_hand' in self.input_sensors:
            self.object_in_hand_embed = nn.Embedding(3, self.cfg.decoder.d_model)
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

        if 'last_actions' in non_visual_sensors:
            last_actions_enc = self.last_actions_embed(non_visual_sensors['last_actions'])
            visual_feats = visual_feats + last_actions_enc

        if 'an_object_is_in_hand' in non_visual_sensors:
            object_in_hand_enc = self.object_in_hand_embed(
                non_visual_sensors['an_object_is_in_hand']
            )
            visual_feats = visual_feats + object_in_hand_enc

        time_enc = self.time_encoder(time_ids)
        visual_feats = visual_feats + time_enc

        return visual_feats, text_feats

    def decode_and_get_logits(self, embedded_features, text_feats, padding_mask=None):
        causal_mask = create_causal_mask(embedded_features.shape[1], embedded_features.device)
        decoder_output = self.decoder(
            tgt=embedded_features,
            memory=text_feats,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        logits = dict(actions_logits=self.action_classifier(decoder_output))
        return logits

    @classmethod
    def build_model(
        cls,
        model_version,
        input_sensors,
        loss,
    ):
        model_cfg = EarlyFusionCnnTransformerConfig()
        model_cfg.action_loss = 'action' in loss
        model_cfg.visual_encoder.input_sensors = input_sensors
        if model_version == 'small_3' or model_version == 'small':
            model_cfg.visual_encoder.image_encoder = 'Dinov2Small'
            model_cfg.visual_encoder.text_encoder = 't5-small'
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = TransformerConfig(3, 512, 8)
        elif model_version == 'small_6':
            model_cfg.visual_encoder.image_encoder = 'Dinov2Small'
            model_cfg.visual_encoder.text_encoder = 't5-small'
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(6, 512, 8)
            model_cfg.decoder = TransformerConfig(6, 512, 8)
        elif model_version == 'base_3':
            model_cfg.visual_encoder.image_encoder = 'Dinov2Base'
            model_cfg.visual_encoder.text_encoder = 't5-small'
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = TransformerConfig(3, 512, 8)
        elif model_version == 'base_6':
            model_cfg.visual_encoder.image_encoder = 'Dinov2Base'
            model_cfg.visual_encoder.text_encoder = 't5-small'
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(6, 768, 8)
            model_cfg.decoder = TransformerConfig(6, 768, 8)
        elif model_version == 'clip_resnet_50_3':
            model_cfg.visual_encoder.image_encoder = 'ClipResNet50'
            model_cfg.visual_encoder.text_encoder = 't5-small'
            model_cfg.visual_encoder.fusion_xformer = TransformerConfig(3, 512, 8)
            model_cfg.decoder = TransformerConfig(3, 512, 8)
        else:
            raise NotImplementedError

        model = EarlyFusionCnnTransformer(model_cfg)

        preproc_cfg = EarlyFusionCnnTransformerPreprocessorConfig()
        preprocessor_type = EarlyFusionCnnTransformerPreprocessor

        preproc_cfg.data_augmentation = True
        preproc_cfg.augmentation_version = 'v2'
        preproc = preprocessor_type(cfg=preproc_cfg, device='cpu')
        return model, preproc

    @classmethod
    def build_agent(
        cls,
        model_version,
        input_sensors,
        loss,
        device,
        sampling,
    ):
        model, preproc = cls.build_model(model_version, input_sensors, loss)
        return EarlyFusionCnnTransformerAgent(model, preproc, device, sampling)


class EarlyFusionCnnTransformerAgent(AbstractAgent):
    def __init__(self, model, device, sampling='greedy', max_seq_len=1000):
        self.model = model
        # self.preprocessor = preprocessor
        preproc_cfg = EarlyFusionCnnTransformerPreprocessorConfig()
        self.preprocessor = EarlyFusionCnnTransformerPreprocessor(cfg=preproc_cfg, device='cpu')
        self.device = device
        self.max_seq_len = max_seq_len
        self.sampling = sampling
        self.reset()
        self.model = self.model.to(self.device)
        self.preprocessor.device = self.device

    def reset(self):
        self.curr_t = 0
        self.preprocessor.image_preprocessor = tensor_image_preprocessor(
            size=((224, 384)),
            data_augmentation=self.preprocessor.cfg.data_augmentation,
            specific=False,
            augmentation_version=self.preprocessor.cfg.augmentation_version,
            mean=((0.48145466, 0.4578275, 0.40821073)),
            std=((0.26862954, 0.26130258, 0.27577711)),
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
        if 'last_actions' in self.model.input_sensors:
            start_token = self.preprocessor.action2idx['']
            preprocessed_nonvisual_sensors['last_actions'] = (
                torch.tensor(np.array([[start_token]])).to(self.device)
                if self.curr_t == 0
                else self.cache['last_actions']
            )

        for sensor_name in [
            'nav_task_relevant_object_bbox',
            'manip_task_relevant_object_bbox',
            'nav_accurate_object_bbox',
            'manip_accurate_object_bbox',
        ]:
            if sensor_name in self.model.input_sensors:
                preprocessed_nonvisual_sensors[sensor_name] = (
                    self.preprocessor.process_task_relevant_bbox([observations], sensor_name)
                )

        if 'an_object_is_in_hand' in self.model.input_sensors:
            observations['an_object_is_in_hand'] = observations['an_object_is_in_hand'][:, 0]
            preprocessed_nonvisual_sensors['an_object_is_in_hand'] = (
                self.preprocessor.process_objinhand([observations])
            )

        return dict(
            visual_sensors=frames_dict,
            non_visual_sensors=preprocessed_nonvisual_sensors,
        )

    def generate(self, observations, goal_spec):
        processed_observations = self.process_sensors_for_model_eval(observations)

        if self.curr_t == 0:
            if isinstance(self.preprocessor.text_preprocessor, (TextTransformer, HFTokenizer)):
                goal = self.preprocessor.text_preprocessor(
                    [goal_spec], context_length=self.preprocessor.cfg.text_encoder_context_length
                ).to(self.preprocessor.device)
                self.cache['goal'] = goal
            else:
                goal = self.preprocessor.text_preprocessor([goal_spec], return_tensors='pt')
                self.cache['goal'] = {k: v.to(self.device) for k, v in goal.items()}

            text_feats = self.model.visual_encoder.encode_text(self.cache['goal'])
            self.cache['text_feats'] = text_feats
        else:
            goal = self.cache['goal']
            text_feats = self.cache['text_feats']

        embedded_features, _ = self.model.get_input_embedding_per_timestep(
            processed_observations['visual_sensors'],
            processed_observations['non_visual_sensors'],
            None,
            time_ids=torch.tensor([[self.curr_t]]).to(self.device),
            text_features=text_feats,
        )

        if self.curr_t == 0:
            self.cache['embedded_features'] = embedded_features
        else:
            self.cache['embedded_features'] = torch.cat(
                (self.cache['embedded_features'], embedded_features), dim=1
            )

        decoder_input = self.cache['embedded_features']
        if self.curr_t >= self.max_seq_len:
            decoder_input = decoder_input[:, -self.max_seq_len :]

        logits = self.model.decode_and_get_logits(decoder_input, text_feats)

        curr_logits = logits['actions_logits'][0, -1]
        action_idx = sample_action_index_from_logits(
            curr_logits,
            self.sampling,
            self.preprocessor.cfg.action_list,
        )
        action = self.preprocessor.cfg.action_list[action_idx]

        if 'last_actions' in self.model.input_sensors:
            self.cache['last_actions'] = action_idx.reshape(1, 1)

        self.curr_t += 1

        return action, torch.softmax(curr_logits, -1)
