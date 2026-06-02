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

"""Flux2 Klein Differential Diffusion Pipeline

This pipeline builds upon :class:`Flux2KleinInpaintPipeline` but adds support for
*Differential Diffusion* – a workflow where a *source* image and a *map* are used
to control where changes should be applied.  The implementation follows the
behaviour of the example ``pipeline_hunyuandit_differential_img2img.py`` while
keeping the API compatible with the existing inpainting pipeline.

The pipeline uses:

* ``Qwen3ForCausalLM`` as the text encoder (same as the inpainting pipeline).
* ``Qwen2TokenizerFast`` for tokenisation.
* ``FlowMatchEulerDiscreteScheduler`` as the scheduler.

The key addition is the ``map`` argument which is an image (or batch of images)
that defines per‑pixel weights for differential updates.  The map is encoded
with the VAE, normalised to ``[0, 1]`` and used to blend the source latents with
the generated latents during the diffusion process.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKLFlux2
from diffusers.models.transformers import Flux2Transformer2DModel
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
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
        >>> from diffusers import FlowMatchEulerDiscreteScheduler
        >>> from diffusers.utils import load_image
        >>> from PIL import Image
        >>> from torchvision import transforms
        >>> from pipeline_flux2_klein_differential_diff import Flux2KleinDifferentialDiffPipeline

        >>> source_image = load_image("https://example.com/source.png")
        >>> diff_map = load_image("https://example.com/map.png")
        >>> pipe = Flux2KleinDifferentialDiffPipeline.from_pretrained(
        >>>     "BlackForestLabs/flux2-klein", torch_dtype=torch.float16
        >>> ).to("cuda")

        >>> image = pipe(
        >>>     prompt="A futuristic cityscape",
        >>>     image=source_image,
        >>>     map=diff_map,
        >>>     strength=0.7,
        >>> ).images[0]
        ```
"""


class Flux2KleinDifferentialDiffPipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    """Flux2 Klein pipeline with differential diffusion support.

    The API mirrors :class:`Flux2KleinInpaintPipeline` with an extra ``map``
    argument.  When ``map`` is supplied, the latent representation of the map is
    used to weight the contribution of the source image latents during the
    diffusion process.
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
        is_distilled: bool = False,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.is_distilled = is_distilled
        # Scale factor and latent channels (matches inpaint pipeline)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 32

        # Use Flux2ImageProcessor (same as inpaint) for image handling
        self.image_processor = Flux2ImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=self.latent_channels
        )

        # Processor for the differential map (no special mask handling needed)
        self.map_processor = Flux2ImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=self.latent_channels
        )

        self.tokenizer_max_length = 512

        # Default sample size – keep consistent with inpaint pipeline
        self.default_sample_size = 128

    # ---------------------------------------------------------------------
    # Helper methods (copied from the original inpainting pipeline)
    # ---------------------------------------------------------------------
    def _encode_image(self, image: PipelineImageInput) -> torch.Tensor:
        """Encode an image (or batch) to latent space using the VAE.

        The returned tensor has shape ``[B, C, H, W]`` where ``C`` equals the
        number of VAE latent channels.
        """
        image = self.image_processor.preprocess(image)
        return self.vae.encode(image).latent_dist.sample()

    def _process_map(self, map_image: PipelineImageInput) -> torch.Tensor:
        """Encode ``map_image`` and normalise to the range ``[0, 1]``.

        The map is expected to be a single‑channel image (grayscale).  If it has
        three channels we convert it to luminance using the standard ITU‑BT.601
        coefficients.
        """
        map_tensor = self.image_processor.preprocess(map_image, do_normalize=False)
        if map_tensor.shape[1] == 3:
            r, g, b = map_tensor[:, 0:1], map_tensor[:, 1:2], map_tensor[:, 2:3]
            map_tensor = 0.299 * r + 0.587 * g + 0.114 * b
        with torch.no_grad():
            map_latents = self.vae.encode(map_tensor).latent_dist.sample()
        map_min = map_latents.amin(dim=[1, 2, 3], keepdim=True)
        map_max = map_latents.amax(dim=[1, 2, 3], keepdim=True)
        eps = 1e-7
        map_norm = (map_latents - map_min) / (map_max - map_min + eps)
        return map_norm

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = (1024, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        use_resolution_binning: bool = True,
        map: PipelineImageInput = None,
        denoising_start: Optional[float] = None,
    ) -> Union[Flux2PipelineOutput, Tuple]:
        """Generate images using differential diffusion.

        Args:
            prompt (Union[str, List[str]]): Text prompt to guide image generation.
            image (PipelineImageInput): Source image to be edited.
            strength (float): How much noise to add; 0.0 retains the original image, 1.0 ignores it.
            height (int, optional): Desired output height. Defaults to ``original_size`` height.
            width (int, optional): Desired output width. Defaults to ``original_size`` width.
            num_inference_steps (int, optional): Number of diffusion steps.
            timesteps (List[int], optional): Custom timesteps for the scheduler.
            sigmas (List[float], optional): Custom sigmas for the scheduler.
            guidance_scale (float, optional): Classifier‑free guidance scale.
            negative_prompt (Union[str, List[str]], optional): Prompt for negative guidance.
            num_images_per_prompt (int, optional): Number of images to generate per prompt.
            eta (float, optional): Parameter for DDIM scheduler; ignored for others.
            generator (torch.Generator or List[torch.Generator], optional): Random generator.
            latents (torch.Tensor, optional): Pre‑computed latents.
            prompt_embeds (torch.Tensor, optional): Pre‑computed prompt embeddings.
            negative_prompt_embeds (torch.Tensor, optional): Pre‑computed negative prompt embeddings.
            prompt_attention_mask (torch.Tensor, optional): Attention mask for prompt embeddings.
            negative_prompt_attention_mask (torch.Tensor, optional): Attention mask for negative prompt embeddings.
            output_type (str, optional): ``"pil"`` or ``"np"``.
            return_dict (bool, optional): Whether to return a ``Flux2PipelineOutput``.
            callback_on_step_end (callable, optional): Callback after each denoising step.
            callback_on_step_end_tensor_inputs (List[str], optional): Tensors passed to the callback.
            guidance_rescale (float, optional): Guidance rescaling factor.
            original_size (Tuple[int, int], optional): Original image size for resolution binning.
            target_size (Tuple[int, int], optional): Desired target size after binning.
            crops_coords_top_left (Tuple[int, int], optional): Top‑left coordinates for cropping.
            use_resolution_binning (bool, optional): Whether to use resolution binning.
            map (PipelineImageInput, optional): Differential map image that controls where changes are applied.
            denoising_start (float, optional): Start point for denoising.

        Returns:
            Flux2PipelineOutput or tuple: Generated image(s).

        Example:
            >>> pipe = Flux2KleinDifferentialDiffPipeline.from_pretrained(
            >>>     "BlackForestLabs/flux2-klein", torch_dtype=torch.float16
            >>> ).to("cuda")
            >>> source_image = load_image("https://example.com/source.png")
            >>> diff_map = load_image("https://example.com/map.png")
            >>> image = pipe(
            >>>     prompt="A futuristic cityscape",
            >>>     image=source_image,
            >>>     map=diff_map,
            >>>     strength=0.7,
            >>> ).images[0]
        """
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        batch_size = 1 if isinstance(prompt, str) else len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
        if height is None:
            height = original_size[0]
        if width is None:
            width = original_size[1]

        if image is None:
            raise ValueError("`image` must be provided for differential diffusion.")
        init_latents = self._encode_image(image)

        map_norm = self._process_map(map) if map is not None else None

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps=num_inference_steps, device=self.device, timesteps=timesteps, sigmas=sigmas
        )
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self._num_timesteps = len(timesteps)

        prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = self.encode_prompt(
            prompt,
            device=self.device,
            dtype=self.text_encoder.dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_attention_mask is not None:
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])

        init_timestep = int(num_inference_steps * strength)
        init_timestep = max(init_timestep, 1)
        sigma = self.scheduler.sigmas[init_timestep]
        noise = randn_tensor(init_latents.shape, generator=generator, device=self.device, dtype=init_latents.dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, sigma)

        if map_norm is not None:
            if map_norm.shape != init_latents.shape:
                map_norm = torch.nn.functional.interpolate(
                    map_norm,
                    size=init_latents.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            clean_source = self._encode_image(image)
            init_latents = clean_source * map_norm + init_latents * (1 - map_norm)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        latents = init_latents
        for i, t in enumerate(timesteps):
            latent_input = latents
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
            noise_pred = self.transformer(
                latent_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timesteps=t,
                return_dict=False,
            )[0]
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if guidance_rescale > 0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if callback_on_step_end is not None:
                callback_on_step_end(i, t, {"step": i, "t": t, "latents": latents})

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        if output_type == "pil":
            image = self.image_processor.postprocess(image, output_type="pil")
        elif output_type == "np":
            image = self.image_processor.postprocess(image, output_type="np")
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """Retrieve timesteps from the scheduler.

    Mirrors the implementation from ``pipeline_flux2_klein_inpaint.py``.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """Rescale the guidance‑adjusted noise as described in the paper.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

"""NOTE:
- All functionality resides in this single file as requested.
- Tests and additional utilities have been omitted.
"""
