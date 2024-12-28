import os
from typing import Any, Dict, Optional, Type

import scipy
import torch
from agent.actions.base_action import BaseAction, tool_api
from agent.actions.parser import BaseParser, JsonParser
from agent.llms.base_llm import BaseDiffusionModelPipeline
from agent.utils.util import generate_hash_uid
from audiocraft.data.audio import audio_write
from audiocraft.models import AudioGen
from diffusers import (
    AudioLDM2Pipeline,
    AudioLDMPipeline,
    CogVideoXPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)

# from tango import Tango
from diffusers.utils import export_to_video


class ImageFLUXPipeline(BaseDiffusionModelPipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)
        self.n_inference_steps = 4

    def _load_pipe(self):
        pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate image according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for image generation

        Returns:
            :class:`str`: The saved image path.
        """
        to_hash = {'image_prompt': prompts, 'timestamp': self.get_timestamp()}
        image_name = f'{generate_hash_uid(to_hash)}.png'
        save_path = os.path.join(self.output_path, instruction_uid, image_name)

        image = self.pipe(
            prompts,
            guidance_scale=0.0,
            num_inference_steps=self.n_inference_steps,
            max_sequence_length=256,
            generator=torch.Generator('cuda').manual_seed(0),
        ).images[0]
        image.save(save_path)

        return save_path


class ImageSD3ModelPipeline(BaseDiffusionModelPipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)
        # self.n_inference_steps = 50

    def _load_pipe(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate image according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for image generation

        Returns:
            :class:`str`: The saved image path.
        """
        to_hash = {'image_prompt': prompts, 'timestamp': self.get_timestamp()}
        image_name = f'{generate_hash_uid(to_hash)}.png'
        save_path = os.path.join(self.output_path, instruction_uid, image_name)

        image = self.pipe(
            prompts,
            negative_prompt='',
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        image.save(save_path)

        return save_path


class ImageSDModelPipeline(BaseDiffusionModelPipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)
        # self.n_inference_steps = 50

    def _load_pipe(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate image according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for image generation

        Returns:
            :class:`str`: The saved image path.
        """
        to_hash = {'image_prompt': prompts, 'timestamp': self.get_timestamp()}
        image_name = f'{generate_hash_uid(to_hash)}.png'
        save_path = os.path.join(self.output_path, instruction_uid, image_name)
        print('-----------------------')
        image = self.pipe(prompts).images[0]
        print(image)
        print('-----------------------')
        image.save(save_path)
        print(save_path)
        return save_path


class CogVideoXModelPipeline(BaseDiffusionModelPipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)

    def _load_pipe(self):
        pipe = CogVideoXPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )

        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate video according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for video generation

        Returns:
            :class:`str`: The saved video path.
        """
        to_hash = {'video_prompt': prompts, 'timestamp': self.get_timestamp()}
        video_name = f'{generate_hash_uid(to_hash)}.mp4'
        save_path = os.path.join(self.output_path, instruction_uid, video_name)

        video = self.pipe(
            prompt=prompts,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device='cuda').manual_seed(42),
        ).frames[0]
        export_to_video(video, save_path, fps=8)

        return save_path


class AudioAudioLDM2Pipeline(BaseDiffusionModelPipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)

    def _load_pipe(self):
        pipe = AudioLDM2Pipeline.from_pretrained(self.model_path, torch_dtype=self.dtype)
        pipe = pipe.to('cuda')
        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate audio according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for audio generation

        Returns:
            :class:`str`: The saved audio path.
        """

        to_hash = {'audio_prompt': prompts, 'timestamp': self.get_timestamp()}
        audio_name = f'{generate_hash_uid(to_hash)}.wav'
        save_path = os.path.join(self.output_path, instruction_uid, audio_name)

        audio = self.pipe(prompts, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
        scipy.io.wavfile.write(save_path, rate=16000, data=audio)

        return save_path


class AudioAudioLDMPipeline(AudioAudioLDM2Pipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)

    def _load_pipe(self):
        pipe = AudioLDMPipeline.from_pretrained(self.model_path, torch_dtype=self.dtype)
        pipe = pipe.to('cuda')
        return pipe


class AudioAudioGenPipeline(BaseDiffusionModelPipeline):
    def __init__(
        self,
        model_path: str,
        output_path: str,
        dtype: torch.dtype = torch.bfloat16,
        gen_params: Optional[Dict] = None,
    ) -> None:
        self.model_path = model_path
        self.output_path = output_path
        self.dtype = dtype
        self.gen_params = gen_params or dict()

        self.pipe = self._load_pipe()

    def _load_pipe(self):
        pipe = AudioGen.get_pretrained(self.model_path)
        pipe.set_generation_params(duration=5)
        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate audio according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for audio generation

        Returns:
            :class:`str`: The saved audio path.
        """

        to_hash = {'audio_prompt': prompts, 'timestamp': self.get_timestamp()}
        audio_name = f'{generate_hash_uid(to_hash)}'
        save_path = os.path.join(self.output_path, instruction_uid, audio_name)
        audio = self.pipe.generate([prompts])
        # scipy.io.wavfile.write(save_path, rate=16000, data=audio)
        print(type(prompts))
        print(audio.shape)

        audio_write(
            save_path,
            audio[0].cpu(),
            self.pipe.sample_rate,
            strategy='loudness',
            loudness_compressor=True,
        )
        # audio_write(save_path, audio.cpu(), 16000, strategy="loudness", loudness_compressor=True)

        return f'{save_path}.wav'


class AudioTangoPipeline(AudioAudioLDM2Pipeline):
    def __init__(
        self, model_path: str, output_path: str, dtype: torch.dtype, gen_params: Dict[str, Any]
    ):
        super().__init__(model_path, output_path, dtype, gen_params)

    def _load_pipe(self):
        pipe = Tango(self.model_path)
        return pipe

    def generate(self, prompts: str, instruction_uid) -> str:
        """Generate audio according to the prompt.

        Args:
            prompt (:class:`str`): Prompt for audio generation

        Returns:
            :class:`str`: The saved audio path.
        """

        to_hash = {'audio_prompt': prompts, 'timestamp': self.get_timestamp()}
        audio_name = f'{generate_hash_uid(to_hash)}.wav'
        save_path = os.path.join(self.output_path, instruction_uid, audio_name)

        audio = self.pipe.generate(prompts, steps=200)
        scipy.io.wavfile.write(save_path, rate=16000, data=audio)

        return save_path


class ModalityGenerator(BaseAction):
    """A generator using diffusion model to generate modalities.

    Args:
        image_model_path (str, optional): The path to the model. Defaults to
            'black-forest-labs/FLUX.1-schnell'.
        video_model_path (str, optional): The path to the model. Defaults to
            'THUDM/CogVideoX-5b'.
        audio_model_path (str, optional): The path to the model. Defaults to
            'cvssp/audioldm2'.
        output_path (str, optional): The path to save the generated modalities. Defaults to
            'output'.
        detype (torch.dtype, optional): The data type of the model. Defaults to
            torch.bfloat16.
        gen_params (Optional[Dict], optional): The input params for generation.
        description (dict, Optional): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool, optional): Whether the action is enabled. Defaults to
            ``True``.
    """

    def __init__(
        self,
        output_dir: str,
        image_model_path: str,
        video_model_path: str,
        audio_model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        gen_params: Optional[Dict] = None,
        description: Optional[Dict] = None,
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ) -> None:
        super().__init__(description, parser, enable)
        self.image_model_path = image_model_path
        self.video_model_path = video_model_path
        self.audio_model_path = audio_model_path
        self.output_dir = output_dir
        self.dtype = dtype
        self.gen_params = gen_params or {}

        os.makedirs(output_dir, exist_ok=True)

        ImagePipeline = ImageFLUXPipeline

        if image_model_path is None:
            pass
        else:
            if 'FLUX.1-schnell' in image_model_path:
                ImagePipeline = ImageFLUXPipeline
            elif 'stable-diffusion-3-medium' in image_model_path:
                ImagePipeline = ImageSD3ModelPipeline
            elif 'stable-diffusion' in image_model_path:
                ImagePipeline = ImageSDModelPipeline
            else:
                raise ValueError('Invalid image model.')

        if 'audioldm2' in audio_model_path:
            AudioPipeline = AudioAudioLDM2Pipeline
            self.audio_pipeline = AudioPipeline(
                model_path=audio_model_path,
                output_path=output_dir,
                dtype=dtype,
                gen_params=gen_params,
            )
        elif 'audioldm' in audio_model_path:
            AudioPipeline = AudioAudioLDMPipeline
            self.audio_pipeline = AudioPipeline(
                model_path=audio_model_path,
                output_path=output_dir,
                dtype=dtype,
                gen_params=gen_params,
            )
        elif 'audiogen' in audio_model_path:
            AudioPipeline = AudioAudioGenPipeline
            self.audio_pipeline = AudioPipeline(
                model_path=audio_model_path,
                output_path=output_dir,
                # dtype=dtype,
                gen_params=gen_params,
            )
        elif audio_model_path:
            raise ValueError('Invalid audio model.')

        if video_model_path is None:
            pass
        else:
            if 'CogVideoX' in video_model_path:
                VideoPipeline = CogVideoXModelPipeline
                self.video_pipeline = VideoPipeline(
                    model_path=video_model_path,
                    output_path=output_dir,
                    dtype=dtype,
                    gen_params=gen_params,
                )
            else:
                raise ValueError('Invalid video model.')

        if image_model_path is not None:
            self.image_pipeline = ImagePipeline(
                model_path=image_model_path,
                output_path=output_dir,
                dtype=dtype,
                gen_params=gen_params,
            )

    @tool_api(explode_return=True)
    def generate(
        self,
        instruction_uid: str,
        image_prompt: Optional[str] = None,
        video_prompt: Optional[str] = None,
        audio_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate modalities based on the given text prompts.

        This method uses the diffusion model to create modalities that matches the description provided in the given prompts.

        Args:
            image_prompt (Optional[str]): A detailed text description of the image to be generated.
                This should be a clear and specific description of the desired image content.
            audio_prompt (Optional[str]): A detailed text description of the audio to be generated.
                This should be a clear and specific description of the desired audio content.

        Returns:
            Dict[str, str]: A dictionary containing the saved paths of the generated modalities.

        Example:
            >>> ModalityGenerator.generate(
            ...     image_prompt="A beautiful landscape with a sunset",
            ...     audio_prompt="A soothing melody with piano and violin"
            ... )

        Note:
            The quality and accuracy of the generated modalities depend on the clarity
            and specificity of the provided prompt. More detailed and vivid descriptions
            often lead to better results.
        """
        if image_prompt is None and audio_prompt is None and video_prompt is None:
            raise ValueError('At least one of the prompts should be provided.')

        if not os.path.exists(os.path.join(self.output_dir, instruction_uid)):
            os.makedirs(os.path.join(self.output_dir, instruction_uid))

        image_saved_path = (
            self.image_pipeline.generate(image_prompt, instruction_uid) if image_prompt else None
        )
        video_saved_path = (
            self.video_pipeline.generate(video_prompt, instruction_uid) if video_prompt else None
        )
        audio_saved_path = (
            self.audio_pipeline.generate(audio_prompt, instruction_uid) if audio_prompt else None
        )

        return {
            'image_path': image_saved_path,
            'video_path': video_saved_path,
            'audio_path': audio_saved_path,
        }
