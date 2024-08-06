# Copyright 2024 PKU-Alignment Team and Meta Platforms. All Rights Reserved.
#
# This code is inspired by the ImageBind.
# https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py
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
"""The audio processing utility."""

from typing import Union

import numpy as np
import torch
import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision.transforms import Normalize


DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds


def get_audio_processor(
    clip_duration: float = 2,
    clips_per_audio: int = 1,
    num_mel_bins: int = 128,
    max_frames: int = 200,
    mel_first: bool = True,
):
    class AudioProcessor:
        def __init__(
            self,
            clip_duration: float = 2,
            clips_per_audio: int = 1,
            num_mel_bins: int = 128,
            max_frames: int = 204,
            mel_first: bool = True,
            mean=-4.268,
            std=9.138,
        ):
            self.clip_sampler = ConstantClipsPerVideoSampler(
                clip_duration=clip_duration, clips_per_video=clips_per_audio
            )
            self.normalize = Normalize(mean=mean, std=std)
            self.num_mel_bins = num_mel_bins
            self.max_frames = max_frames
            self.mel_first = mel_first

        def waveform2melspec(self, waveform, sample_rate, num_mel_bins, max_frames):
            # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
            waveform -= waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=sample_rate,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=num_mel_bins,
                dither=0.0,
                frame_length=25,
                frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
            )
            # Convert to [mel_bins, num_frames] shape
            fbank = fbank.transpose(0, 1)
            # Pad to target_length
            n_frames = fbank.size(1)
            p = max_frames - n_frames
            # cut and pad
            if p > 0:
                fbank = torch.nn.functional.pad(fbank, (0, p), mode='constant', value=0)
            elif p < 0:
                fbank = fbank[:, 0:max_frames]
            # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
            # channel image
            fbank = fbank.unsqueeze(0)
            return fbank

        def get_clip_timepoints(self, duration):
            # Read out all clips
            all_clips_timepoints = []
            is_last_clip = False
            end = 0.0
            while not is_last_clip:
                start, end, _, _, is_last_clip = self.clip_sampler(end, duration, annotation=None)
                all_clips_timepoints.append((start, end))
            return all_clips_timepoints

        def __call__(
            self,
            waveform: Union[np.array, torch.Tensor],
            sample_rate: int = 16000,
        ) -> torch.Tensor:
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            if waveform.dim() == 1:
                waveform = torch.stack([waveform, waveform], dim=0)

            all_clips_timepoints = self.get_clip_timepoints(waveform.size(1) / sample_rate)
            all_clips = []
            for clip_timepoints in all_clips_timepoints:
                waveform_clip = waveform[
                    :,
                    int(clip_timepoints[0] * sample_rate) : int(clip_timepoints[1] * sample_rate),
                ]
                waveform_melspec = self.waveform2melspec(
                    waveform_clip, sample_rate, self.num_mel_bins, self.max_frames
                )
                all_clips.append(waveform_melspec)

            all_clips = [self.normalize(ac) for ac in all_clips]
            mel_spec = torch.stack(all_clips, dim=0)
            if not self.mel_first:
                mel_spec = mel_spec.permute(0, 1, 3, 2)
            if mel_spec.size(0) == 1:
                mel_spec = mel_spec.squeeze(0)
            return mel_spec

    return AudioProcessor(
        clip_duration=clip_duration,
        clips_per_audio=clips_per_audio,
        num_mel_bins=num_mel_bins,
        max_frames=max_frames,
        mel_first=mel_first,
    )
