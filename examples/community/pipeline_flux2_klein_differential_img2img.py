# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKLFlux2
from diffusers.models.transformers import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers.utils import load_image
        >>> from pipeline import Flux2KleinDifferentialImg2ImgPipeline

        >>> image = load_image(
        >>>     "https://github.com/exx8/differential-diffusion/blob/main/assets/input.jpg?raw=true",
        >>> )

        >>> mask = load_image(
        >>>     "https://github.com/exx8/differential-diffusion/blob/main/assets/map.jpg?raw=true",
        >>> )

        >>> pipe = Flux2KleinDifferentialImg2ImgPipeline.from_pretrained(
        >>>     "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16
        >>> )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "painting of a mountain landscape with a meadow and a forest, meadow background, anime countryside landscape, anime nature wallpap, anime landscape wallpaper, studio ghibli landscape, anime landscape, mountain behind meadow, anime background art, studio ghibli environment, background of flowery hill, anime beautiful peace scene, forrest background, anime scenery, landscape background, background art, anime scenery concept art"
        >>> out = pipe(
        >>>     prompt=prompt,
        >>>     num_inference_steps=4,
        >>>     guidance_scale=4.0,
        >>>     image=image,
        >>>     mask_image=mask,
        >>>     strength=1.0,
        >>> ).images[0]

        >>> out.save("image.png")
        ```
        """

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class Flux2KleinDifferentialImg2ImgPipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    r"""
    Differential Image to Image pipeline for Flux2 Klein.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: Flux2Transformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        
        # Mask processor should match the resolution of the latents before packing
        # Flux2 latents are (B, C, H, W) where H, W are height/width // (vae_scale_factor * 2)
        # Wait, Flux2 scale factor logic in pipeline_flux2_klein:
        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        # This means the effective downsampling is vae_scale_factor * 2.
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.vae.config.latent_channels if getattr(self, "vae", None) else 16,
            do_normalize=False,
            do_binarize=False,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

    def _get_qwen3_prompt_embeds(
        self,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: str | list[str],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        hidden_states_layers: list[int] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    def _prepare_text_ids(self, x: torch.Tensor):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    def _prepare_latent_ids(self, latents: torch.Tensor):
        batch_size, _, height, width = latents.shape
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    def _patchify_latents(self, latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    def _pack_latents(self, latents):
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    def _unpack_latents_with_ids(self, x: torch.Tensor, x_ids: torch.Tensor, height: int, width: int):
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)
            flat_ids = h_ids * width + w_ids
            out = torch.zeros((height * width, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
            out = out.view(height, width, ch).permute(2, 0, 1)
            x_list.append(out)
        return torch.stack(x_list, dim=0)

    def _unpatchify_latents(self, latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
    ):
        device = device or self._execution_device
        if prompt is None:
            prompt = ""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )
        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")
        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
        return image_latents

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)
        latents = self._pack_latents(latents)
        return latents, latent_ids

    def prepare_mask_latents(
        self,
        mask: torch.Tensor,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
    ):
        # Resize mask to latent space (H, W) before packing
        # In Flux2, latents are (B, C, H, W) then packed to (B, H*W, C)
        # H = height // (vae_scale_factor * 2), W = width // (vae_scale_factor * 2)
        h = height // (self.vae_scale_factor * 2)
        w = width // (self.vae_scale_factor * 2)
        mask = torch.nn.functional.interpolate(mask, size=(h, w))
        mask = mask.to(device=device, dtype=dtype)
        
        if mask.shape[0] < batch_size:
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        
        # Pack mask to match the (B, H*W, C) format
        # We repeat mask over channels to match latent channel dimension
        mask = mask.repeat(1, num_channels_latents, 1, 1)
        mask = self._pack_latents(mask)
        
        return mask

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, num_inference_steps - t_start

    def check_inputs(self, prompt, image, mask_image, strength, height, width, output_type):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2}")

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 4,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(prompt, image, mask_image, strength, height, width, output_type)
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (height // (self.vae_scale_factor * 2)) * (width // (self.vae_scale_factor * 2))
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(f"Adjusted num_inference_steps {num_inference_steps} is < 1.")

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Differential diffusion setup
        # 1. Preprocess mask and image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(device=device, dtype=prompt_embeds.dtype)
        
        # Original image latents for blending
        original_image_latents = self._encode_vae_image(init_image, generator=generator)
        original_image_latents = self._pack_latents(original_image_latents)

        # Process mask for differential diffusion
        mask_raw = self.mask_processor.preprocess(mask_image, height=height, width=width)
        mask_raw = torch.nn.functional.interpolate(mask_raw, size=(height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2)))
        mask_raw = mask_raw.to(device=device, dtype=prompt_embeds.dtype)
        if mask_raw.shape[0] < (batch_size * num_images_per_prompt):
            mask_raw = mask_raw.repeat((batch_size * num_images_per_prompt) // mask_raw.shape[0], 1, 1, 1)
        
        # Mask thresholds for differential diffusion: mask = original_mask > (step / total_steps)
        thresholds = torch.arange(num_inference_steps, dtype=prompt_embeds.dtype, device=device) / num_inference_steps
        
        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                # Flux2 Transformer call
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=None, # Flux2 Klein often uses guidance internally or via a different mechanism
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    # Generate noise for scaling original image latents
                    noise = randn_tensor(original_image_latents.shape, generator=generator, device=device, dtype=original_image_latents.dtype)
                    # Scale original image latents with noise
                    image_latent = self.scheduler.scale_noise(
                        original_image_latents, torch.tensor([noise_timestep], device=device), noise
                    )

                    # Apply differential mask
                    # mask_raw: (B, 1, H, W), thresholds[i]: scalar
                    current_mask_raw = (mask_raw > thresholds[i]).to(latents_dtype)
                    current_mask_packed = self._pack_latents(current_mask_raw.repeat(1, num_channels_latents, 1, 1))
                    
                    latents = image_latent * current_mask_packed + latents * (1 - current_mask_packed)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                
                progress_bar.update()

        # Finalize and decode
        latent_height = height // (self.vae_scale_factor * 2)
        latent_width = width // (self.vae_scale_factor * 2)
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height, latent_width)

        # VAE Normalization
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)