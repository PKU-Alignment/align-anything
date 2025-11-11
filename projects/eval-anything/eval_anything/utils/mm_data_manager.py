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

"""
Utility functions for multi-modal data.
- Handle the conversion between multi-modal data and the required format.
- TODO: Display multi-modal data.
"""

import base64
import math
import os
import re
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
import PIL
import requests
import torch
from PIL import Image
from scipy.io import wavfile
from torchvision.io.video import read_video

from eval_anything.evaluate_tools.base_tools import BaseMMDataManager
from eval_anything.utils.register import MMDataManagerRegistry


@MMDataManagerRegistry.register('image')
class ImageManager(BaseMMDataManager):
    """
    Convert between image and base64 string.
    """

    @classmethod
    def decode_base64_to_image(cls, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image object and ensure consistent format"""
        try:
            # Remove data URI prefix if present
            if base64_string.startswith('data:image/'):
                base64_data = base64_string.split(',', 1)[1]
            else:
                base64_data = base64_string

            # Decode base64 to bytes and convert to Image
            image = Image.open(BytesIO(base64.b64decode(base64_data)))

            # Convert to RGB/RGBA based on whether transparency is needed
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                return image.convert('RGBA')
            return image.convert('RGB')

        except Exception as e:
            raise ValueError(f'Failed to decode base64 image: {str(e)}')

    @classmethod
    def encode_image_to_base64(cls, image: Union[str, 'PIL.Image']) -> str:
        """Convert image to base64 string with consistent format"""
        try:
            if isinstance(image, str):
                image_input = Image.open(image)
            else:
                image_input = image

            # Determine if image needs transparency
            if image_input.mode == 'RGBA' and cls._has_transparency(image_input):
                buffer = BytesIO()
                image_input.save(buffer, format='PNG')  # Use PNG for images with transparency
            else:
                image_input = image_input.convert('RGB')
                buffer = BytesIO()
                image_input.save(buffer, format='JPEG', quality=95)

            img_bytes = buffer.getvalue()
            base64_data = base64.b64encode(img_bytes).decode('utf-8')
            return base64_data

        except Exception as e:
            raise ValueError(f'Failed to encode image to base64: {str(e)}')

    @classmethod
    def _has_transparency(cls, image: Image.Image) -> bool:
        """Check if image has any transparent pixels"""
        if image.mode == 'RGBA':
            extrema = image.getextrema()
            if len(extrema) >= 4:  # Make sure we have alpha channel
                alpha_min, alpha_max = extrema[3]
                return alpha_min < 255
        return False

    @classmethod
    def extract_images_from_conversation(cls, conversation: List[Dict]) -> List[Image.Image]:
        """Extract all images from the conversation with consistent format"""
        images = []

        for message in conversation:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    try:
                        if 'image' in item:
                            image = cls.decode_base64_to_image(item['image'])
                            images.append(image)
                        elif 'image_url' in item:
                            image = cls.decode_base64_to_image(item['image_url'])
                            images.append(image)
                    except Exception as e:
                        print(f'Warning: Failed to process image in conversation: {str(e)}')
                        continue
        return images, {}

    @classmethod
    def prompt_to_conversation(
        cls,
        user_prompt: str,
        system_prompt: Union[str, None] = None,
        images: Union[List[PIL.Image], List[str]] = [],
    ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format

        Args:
            user_prompt (str): Input user_prompt with image placeholders such as <image>, <imagen>, or <image n>. If no placeholder is present, images will be automatically prepended to the beginning of the text
            system_prompt (str): Input system_prompt (if exists)
            images (list): List of PIL.Image objects to be encoded and inserted into the conversation

        Returns:
            list: Conversation object in the specified format
        """
        content_parts = []
        matches = list(re.finditer(r'<image\s*(\d*)>', user_prompt))

        if matches:
            assert len(images) == len(
                matches
            ), f'Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}'

            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append(
                        {'type': 'text', 'text': user_prompt[last_end : match.start()]}
                    )
                content_parts.append(
                    {
                        'type': 'image',
                        'image': f'data:image/jpeg;base64,{cls.encode_image_to_base64(images[i])}',
                    }
                )
                last_end = match.end()

            if last_end < len(user_prompt):
                content_parts.append({'type': 'text', 'text': user_prompt[last_end:]})
        else:
            content_parts.extend(
                [
                    {
                        'type': 'image',
                        'image': f'data:image/jpeg;base64,{cls.encode_image_to_base64(img)}',
                    }
                    for img in images
                ]
            )
            if user_prompt:
                content_parts.append({'type': 'text', 'text': user_prompt})

        conversation = [{'role': 'user', 'content': content_parts}]
        if system_prompt is not None:
            conversation.insert(
                0, {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}
            )

        return conversation

    @classmethod
    def encode_to_base64(cls, data: Union[str, 'PIL.Image'], **kwargs) -> str:
        """Implement abstract method to encode image to base64."""
        return cls.encode_image_to_base64(data)

    @classmethod
    def decode_from_base64(cls, base64_string: str) -> Image.Image:
        """Implement abstract method to decode base64 to image."""
        return cls.decode_base64_to_image(base64_string)

    @classmethod
    def extract_from_conversation(cls, conversation: List[Dict]) -> List[Image.Image]:
        """Implement abstract method to extract images from conversation."""
        return cls.extract_images_from_conversation(conversation)


@MMDataManagerRegistry.register('audio')
class AudioManager(BaseMMDataManager):
    """
    Convert between audio and base64 string.
    """

    @classmethod
    def decode_base64_to_audio(cls, base64_string: str) -> tuple[int, np.ndarray]:
        """
        Decode base64 string to audio.

        Args:
            base64_string: Base64 encoded audio string

        Returns:
            tuple: (sample_rate, audio_array)
        """
        try:
            # Remove data URI prefix if present
            if base64_string.startswith('data:audio/'):
                base64_data = base64_string.split(',', 1)[1]
            else:
                base64_data = base64_string

            binary_data = base64.b64decode(base64_data)
            buffer = BytesIO(binary_data)

            sample_rate, audio_array = wavfile.read(buffer)

            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32767.0

            return sample_rate, audio_array

        except Exception as e:
            raise ValueError(f'Failed to decode base64 audio: {str(e)}')

    @classmethod
    def encode_audio_to_base64(cls, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Encode NumPy audio data to base64 format.

        Args:
            audio_array: NumPy audio data array
            sample_rate: Audio sample rate, default is 16000 Hz

        Returns:
            str: Base64 encoded audio data
        """
        buffer = BytesIO()

        if audio_array.dtype != np.int16:
            # Normalize float audio to [-1, 1] range if needed
            if audio_array.dtype == np.float32 and np.max(np.abs(audio_array)) <= 1.0:
                # Convert to int16 format for WAV
                audio_array = (audio_array * 32767).astype(np.int16)
            else:
                audio_array = audio_array.astype(np.int16)

        wavfile.write(buffer, sample_rate, audio_array)
        buffer.seek(0)

        wav_data = buffer.getvalue()
        return base64.b64encode(wav_data).decode('utf-8')

    @classmethod
    def prompt_to_conversation(
        cls,
        user_prompt: str,
        system_prompt: Union[str, None] = None,
        audios: Union[np.ndarray, str, List[np.ndarray], List[str]] = [],
        sample_rates: Union[int, List[int]] = [],
    ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format

        Args:
            user_prompt (str): Input user_prompt containing audio placeholders such as <audio>, <audio1>, or <audio n>. If no placeholder is present, audio will be automatically prepended to the beginning of the text
            system_prompt (str, optional): Input system_prompt (if exists). Defaults to None.
            audios (Union[np.ndarray, str, List[np.ndarray], List[str]], optional): Single audio array/path or list of audio arrays/paths to be encoded and inserted into the conversation. Defaults to [].
            sample_rates (Union[int, List[int]], optional): Single sample rate or list of sample rates corresponding to each audio. Defaults to [].

        Returns:
            List[Dict]: Conversation object in the specified format
        """
        if isinstance(audios, (np.ndarray, str)):
            audios = [audios]
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates] * len(audios)

        # Load audio from file paths
        if len(audios) > 0 and not isinstance(audios[0], np.ndarray):
            loaded = [
                librosa.load(path, sr=sr if sr else None) for path, sr in zip(audios, sample_rates)
            ]
            audios, sample_rates = zip(*loaded) if loaded else ([], [])

        content_parts = []
        matches = list(re.finditer(r'<audio\s*(\d*)>', user_prompt))
        assert len(audios) == len(sample_rates), 'Number of audios and sample rates must match'

        if matches:
            assert len(audios) == len(
                matches
            ), f'Number of audios ({len(audios)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}'

            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append(
                        {'type': 'text', 'text': user_prompt[last_end : match.start()]}
                    )
                content_parts.append(
                    {
                        'type': 'audio_url',
                        'audio_url': {
                            'url': f'data:audio/wav;base64,{cls.encode_audio_to_base64(audios[i], sample_rates[i])}'
                        },
                    }
                )
                last_end = match.end()

            if last_end < len(user_prompt):
                content_parts.append({'type': 'text', 'text': user_prompt[last_end:]})
        else:
            content_parts.extend(
                [
                    {
                        'type': 'audio_url',
                        'audio_url': {
                            'url': f'data:audio/wav;base64,{cls.encode_audio_to_base64(audio, sample_rate)}'
                        },
                    }
                    for audio, sample_rate in zip(audios, sample_rates)
                ]
            )
            if user_prompt:
                content_parts.append({'type': 'text', 'text': user_prompt})

        conversation = [{'role': 'user', 'content': content_parts}]
        if system_prompt is not None:
            conversation.insert(
                0, {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}
            )

        return conversation

    @classmethod
    def extract_audios_from_conversation(
        cls, conversation: List[Dict]
    ) -> List[Tuple[np.ndarray, int]]:
        """Extract all audios from the conversation with consistent format"""
        audios = []

        for message in conversation:
            if 'content' in message and isinstance(message['content'], list):
                for part in message['content']:
                    if part.get('type') == 'audio_url' and 'audio_url' in part:
                        audio_url = part['audio_url'].get('url', '')
                        if audio_url.startswith('data:audio/'):
                            try:
                                sample_rate, audio_array = cls.decode_base64_to_audio(audio_url)
                                audios.append((audio_array, sample_rate))
                            except ValueError as e:
                                print(f'Warning: Failed to decode audio: {str(e)}')
        # TODO: Currently vllm only supports one audio, so return the first audio. When vllm supports multiple audios, return all audios.
        return audios[0], {}

    @classmethod
    def encode_to_base64(cls, data: Union[np.ndarray, str], **kwargs) -> str:
        """Implement abstract method to encode audio to base64."""
        sample_rate = kwargs.get('sample_rate', 16000)
        if isinstance(data, str):
            # Load audio from file
            audio_array, sr = librosa.load(data, sr=sample_rate)
            return cls.encode_audio_to_base64(audio_array, sr)
        return cls.encode_audio_to_base64(data, sample_rate)

    @classmethod
    def decode_from_base64(cls, base64_string: str) -> Tuple[int, np.ndarray]:
        """Implement abstract method to decode base64 to audio."""
        return cls.decode_base64_to_audio(base64_string)

    @classmethod
    def extract_from_conversation(cls, conversation: List[Dict]) -> Tuple[np.ndarray, int]:
        """Implement abstract method to extract audio from conversation."""
        return cls.extract_audios_from_conversation(conversation)


@MMDataManagerRegistry.register('video')
class VideoManager(BaseMMDataManager):
    """
    Convert between video and base64 string.
    Supports video from URLs, file paths, and lists of PIL Images.
    """

    @classmethod
    def decode_base64_to_video(cls, base64_string: str) -> List[Image.Image]:
        """
        Decode base64 string to video frames.

        Args:
            base64_string: Base64 encoded video string

        Returns:
            List[Image.Image]: List of video frames as PIL Images
        """
        try:
            # Remove data URI prefix if present
            if base64_string.startswith('data:video/'):
                base64_data = base64_string.split(',', 1)[1]
            else:
                base64_data = base64_string

            binary_data = base64.b64decode(base64_data)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(binary_data)
                temp_file_path = temp_file.name

            # Open the video file
            cap = cv2.VideoCapture(temp_file_path)
            frames = []

            # Read all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

            # Release resources
            cap.release()

            # Remove temporary file
            os.unlink(temp_file_path)

            return frames

        except Exception as e:
            raise ValueError(f'Failed to decode base64 video: {str(e)}')

    @classmethod
    def encode_video_to_base64(
        cls,
        video: Union[str, List[Image.Image], np.ndarray],
        fps: float = 1.0,
        fps_min_frames: int = 4,
        fps_max_frames: int = 768,
        frame_factor: int = 2,
        image_factor: int = 28,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
    ) -> str:
        """
        Encode video to base64 string by extracting and processing frames.

        This function takes a video input (URL, file path, list of images, or numpy array),
        extracts frames at a specified rate, resizes them if necessary to meet size constraints,
        and encodes them as base64 strings separated by commas.

        Args:
            video: Input video as a URL/path string, list of PIL Images, or numpy array with shape (frames, height, width, channels)
            fps: Target frames per second for extraction (default: 1.0)
            fps_min_frames: Minimum number of frames to extract (default: 4)
            fps_max_frames: Maximum number of frames to extract (default: 768)
            frame_factor: Factor to ensure frame count is divisible by this value (default: 2)
            image_factor: Factor for dimension rounding in resizing (default: 28)
            min_pixels: Minimum total pixels allowed in resized frames (default: 3,136)
            max_pixels: Maximum total pixels allowed in resized frames (default: 12,845,056)

        Returns:
            str: Comma-separated base64-encoded JPEG frames

        Raises:
            ValueError: If video format is unsupported or encoding fails
        """
        try:
            # Extract frames
            if isinstance(video, str):
                # Handle URL or path
                temp_path = None
                if video.startswith(('http://', 'https://')):
                    response = requests.get(video)
                    temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                    with open(temp_path, 'wb') as f:
                        f.write(response.content)
                    path = temp_path
                else:
                    path = video

                # Read video and sample frames
                tensor, _, info = read_video(path, pts_unit='sec', output_format='TCHW')
                total = tensor.shape[0]
                count = min(
                    max(int(total / info['video_fps'] * fps), fps_min_frames), fps_max_frames, total
                )
                count = (count // frame_factor) * frame_factor

                idx = torch.linspace(0, total - 1, count).round().long()
                frames = [
                    Image.fromarray(tensor[i].permute(1, 2, 0).numpy().astype(np.uint8))
                    for i in idx
                ]

                if temp_path:
                    os.unlink(temp_path)

            elif isinstance(video, list) and all(isinstance(img, Image.Image) for img in video):
                frames = video
            elif isinstance(video, np.ndarray) and len(video.shape) == 4:
                frames = [Image.fromarray(frame) for frame in video]
            else:
                raise ValueError('Unsupported video format')

            # Resize frames if needed
            if frames:
                h, w = frames[0].height, frames[0].width

                # Smart resize calculation
                h_out = max(image_factor, round(h / image_factor) * image_factor)
                w_out = max(image_factor, round(w / image_factor) * image_factor)

                if h_out * w_out > max_pixels:
                    scale = math.sqrt((h * w) / max_pixels)
                    h_out = math.floor(h / scale / image_factor) * image_factor
                    w_out = math.floor(w / scale / image_factor) * image_factor
                elif h_out * w_out < min_pixels:
                    scale = math.sqrt(min_pixels / (h * w))
                    h_out = math.ceil(h * scale / image_factor) * image_factor
                    w_out = math.ceil(w * scale / image_factor) * image_factor

                frames = [img.resize((w_out, h_out)) for img in frames]

            # Encode frames to base64
            base64_frames = []
            for frame in frames:
                buffer = BytesIO()
                frame.save(buffer, format='JPEG')
                base64_frames.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))

            return ','.join(base64_frames)

        except Exception as e:
            raise ValueError(f'Failed to encode video to base64: {str(e)}')

    @classmethod
    def extract_videos_from_conversation(
        cls, conversation: List[Dict]
    ) -> Tuple[List[List[Image.Image]], Dict]:
        """
        Extract all videos and their metadata from the conversation.

        Args:
            conversation: Conversation object

        Returns:
            Tuple[List[List[Image.Image]], Dict]:
                - List of videos, each represented as a list of PIL Image frames
                - Dictionary of video keyword arguments, including fps values
        """
        videos = []
        video_sample_fps_list = []

        for message in conversation:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    try:
                        if (
                            isinstance(item, dict)
                            and 'type' in item
                            and item['type'] == 'video_url'
                            and 'video_url' in item
                            and 'url' in item['video_url']
                        ):
                            video_url = item['video_url']['url']
                            video_frames = cls.decode_base64_to_video(video_url)
                            videos.append(video_frames)

                            # Extract fps if available, otherwise use default of 1.0
                            fps = item.get('fps', 1.0)
                            video_sample_fps_list.append(fps)
                    except Exception as e:
                        print(f'Warning: Failed to process video in conversation: {str(e)}')
                        continue

        video_kwargs = {'fps': video_sample_fps_list}
        return videos, video_kwargs

    @classmethod
    def prompt_to_conversation(
        cls,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        videos: Union[
            str, List[Image.Image], np.ndarray, List[Union[str, List[Image.Image], np.ndarray]]
        ] = [],
        fps: Union[float, List[float]] = 1.0,
    ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format.

        Args:
            user_prompt (str): Input user_prompt with video placeholders such as <video>, <videon>, or <video n>.
                              If no placeholder is present, videos will be automatically prepended to the beginning of the text.
            system_prompt (str, optional): Input system_prompt. Defaults to None.
            videos: Video file path(s), URL(s), list(s) of PIL Images, or numpy array(s)
            fps: Frames per second for video encoding (if videos are PIL Images or numpy arrays)

        Returns:
            List[Dict]: Conversation object in the specified format
        """
        # Normalize inputs to lists
        if not isinstance(videos, list) or (
            len(videos) > 0
            and (
                isinstance(videos[0], Image.Image)
                or (isinstance(videos[0], np.ndarray) and len(videos[0].shape) > 2)
            )
        ):
            videos = [videos]

        if isinstance(fps, float):
            fps_list = [fps] * len(videos)
        elif isinstance(fps, int):
            fps_list = [float(fps)] * len(videos)
        else:
            fps_list = fps
            assert len(fps_list) == len(videos), 'Number of FPS values must match number of videos'

        content_parts = []
        matches = list(re.finditer(r'<video\s*(\d*)>', user_prompt))

        if matches:
            assert len(videos) == len(
                matches
            ), f'Number of videos ({len(videos)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}'

            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append(
                        {'type': 'text', 'text': user_prompt[last_end : match.start()]}
                    )

                video_base64 = cls.encode_video_to_base64(videos[i], fps_list[i])
                content_parts.append(
                    {
                        'type': 'video_url',
                        'video_url': {'url': f'data:video/jpeg;base64,{video_base64}'},
                        'fps': fps_list[i],
                    }
                )

                # Placeholder for video. Currently, the QWenVL series of models can only recognize placeholders through the key 'video', unlike in the image domain where they support recognizing 'image_url'.
                # TODO: Remove this placeholder when the framework support inference backend serving
                content_parts.append({'type': 'video', 'video': ''})

                last_end = match.end()

            if last_end < len(user_prompt):
                content_parts.append({'type': 'text', 'text': user_prompt[last_end:]})
        else:
            # No video placeholders, prepend videos to the beginning
            for i, video in enumerate(videos):
                video_base64 = cls.encode_video_to_base64(video, fps_list[i])
                content_parts.append(
                    {
                        'type': 'video_url',
                        'video_url': {'url': f'data:video/jpeg;base64,{video_base64}'},
                        'fps': fps_list[i],
                    }
                )

                # Placeholder for video. Currently, the QWenVL series of models can only recognize placeholders through the key 'video', unlike in the image domain where they support recognizing 'image_url'.
                # TODO: Remove this placeholder when the framework support inference backend serving
                content_parts.append({'type': 'video', 'video': ''})

            if user_prompt:
                content_parts.append({'type': 'text', 'text': user_prompt})

        # Create conversation structure
        conversation = [{'role': 'user', 'content': content_parts}]
        if system_prompt is not None:
            conversation.insert(
                0, {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}
            )

        return conversation

    @classmethod
    def encode_to_base64(cls, data: Union[str, List[Image.Image], np.ndarray], **kwargs) -> str:
        """Implement abstract method to encode video to base64."""
        fps = kwargs.get('fps', 1.0)
        return cls.encode_video_to_base64(
            data,
            fps=fps,
            fps_min_frames=kwargs.get('fps_min_frames', 4),
            fps_max_frames=kwargs.get('fps_max_frames', 768),
            frame_factor=kwargs.get('frame_factor', 2),
            image_factor=kwargs.get('image_factor', 28),
            min_pixels=kwargs.get('min_pixels', 4 * 28 * 28),
            max_pixels=kwargs.get('max_pixels', 16384 * 28 * 28),
        )

    @classmethod
    def decode_from_base64(cls, base64_string: str) -> List[Image.Image]:
        """Implement abstract method to decode base64 to video frames."""
        return cls.decode_base64_to_video(base64_string)

    @classmethod
    def extract_from_conversation(
        cls, conversation: List[Dict]
    ) -> Tuple[List[List[Image.Image]], Dict]:
        """Implement abstract method to extract videos from conversation."""
        return cls.extract_videos_from_conversation(conversation)
