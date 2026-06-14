import torch


def flux_klein_color_anchor_callback(pipeline, step, timestep, callback_kwargs):
    """Callback that corrects color drift during Flux Klein denoising by anchoring
    the per-channel statistics (mean and standard deviation) of the latent toward
    a reference image's latent statistics.

    This performs a **per-channel affine color transfer** — matching both the mean
    (brightness/color cast) and standard deviation (contrast/hue balance) of each
    latent channel to the anchor image. A mean-only shift cannot correct hue
    rotation (e.g. purple → red) because hue is encoded in the *relative scale*
    between channels, not just their offsets.

    The structural detail (high-frequency content) is preserved because the
    correction is purely per-channel affine: each token is shifted and scaled by
    the same per-channel factors, so spatial patterns within each channel are
    unchanged.

    **Last-step-only correction** (by default): The correction is applied only at
    the final denoising step(s) to avoid two problems that occur with per-step
    correction:
      1. **Blurriness** — modifying latents at every step pushes them off the
         model's expected denoising trajectory, reducing sharpness.
      2. **Color distortion** — at early/intermediate steps the latent is a noisy
         mixture, and correcting its statistics interferes with the model's
         predictions, causing the model to "fight" the corrections.

    At the last step the latent is nearly clean, so its per-channel spatial
    statistics directly represent the image's color characteristics.

    Setup (attach these to the pipeline instance before calling):
        pipeline.anchor_latents  — packed reference latent [B, seq_len, C]
                                   (produced by the helper ``encode_anchor_image``)
        pipeline.color_anchor_strength     — float 0..1, default 0.25
        pipeline.color_anchor_last_n_steps — int, default 1 (apply only at last step;
                                             increase to 2-3 for stronger correction)

    Inside the Flux Klein denoising loop, latents are *packed* as:
        [batch, height*width, channels]   (i.e.  [B, seq_len, C])
    The per-channel spatial statistics are computed over dim=1 (the spatial-token
    axis).
    """
    latents = callback_kwargs.get("latents")

    # ── guard: anchor must be attached to the pipeline ──────────────────
    if not hasattr(pipeline, "anchor_latents") or pipeline.anchor_latents is None:
        return callback_kwargs

    anchor = pipeline.anchor_latents  # [B, seq_len_ref, C]  (may differ in seq_len)

    # ── configurable hyperparameters (with sensible defaults) ───────────
    strength = getattr(pipeline, "color_anchor_strength", 0.25)

    if strength <= 0.0:
        return callback_kwargs

    # ── only apply at the last N denoising steps ────────────────────────
    num_steps = getattr(pipeline, "_num_timesteps", 1)
    last_n = getattr(pipeline, "color_anchor_last_n_steps", 1)
    start_step = max(0, num_steps - last_n)

    if step < start_step:
        return callback_kwargs

    # ── Affine color transfer (mean + std matching) ─────────────────────
    # latents shape : [B, seq_len,     C]
    # anchor  shape : [B, seq_len_ref, C]   (possibly different spatial size)
    #
    # Per-channel spatial statistics (over the token / spatial dimension):
    eps = 1e-6  # prevent division by zero for near-constant channels

    with torch.no_grad():
        latent_mean = latents.mean(dim=1, keepdim=True)   # [B, 1, C]
        latent_std = latents.std(dim=1, keepdim=True) + eps  # [B, 1, C]

        anchor_mean = anchor.mean(dim=1, keepdim=True)    # [B, 1, C]
        anchor_std = anchor.std(dim=1, keepdim=True) + eps  # [B, 1, C]

        # Normalize latent to zero-mean / unit-variance, then rescale to
        # match anchor statistics.  This corrects both color cast (mean)
        # and hue / contrast balance (std ratio between channels).
        corrected = (latents - latent_mean) / latent_std * anchor_std + anchor_mean

        # Blend between original and fully-corrected at the given strength
        latents = torch.lerp(latents, corrected, strength)

    callback_kwargs["latents"] = latents
    return callback_kwargs


def encode_anchor_image(image, pipeline, device=None, dtype=None):
    """Encode a reference PIL image into the packed latent format used during
    Flux Klein denoising, suitable for use as ``pipeline.anchor_latents``.

    This applies the same patchify → batch-norm → pack pipeline that
    ``Flux2KleinPipeline.prepare_latents`` uses, so the resulting tensor lives
    in the same coordinate space as the denoising latents.

    Args:
        image:  A PIL Image (any size; will be preprocessed by the pipeline).
        pipeline:  A ``Flux2KleinPipeline`` instance.
        device:  Target device (defaults to pipeline's execution device).
        dtype:  Target dtype (defaults to ``pipeline.vae.dtype``).

    Returns:
        Packed latent tensor of shape ``[1, H'*W', C]``.
    """
    device = device or pipeline._execution_device
    dtype = dtype or pipeline.vae.dtype

    with torch.no_grad():
        # Preprocess to [-1, 1] tensor of shape [1, 3, H, W]
        img_tensor = pipeline.image_processor.preprocess(image).to(device=device, dtype=dtype)

        # VAE encode → patchify → batch-norm  (mirrors _encode_vae_image)
        image_latents = pipeline._encode_vae_image(img_tensor, generator=None)
        # image_latents shape: [1, C_patch, H_patch, W_patch]

        # Pack to sequence format: [1, H_patch * W_patch, C_patch]
        packed = pipeline._pack_latents(image_latents)

    return packed


# ═══════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.utils import load_image

    # 1. Load the pipeline
    dtype = torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-9B",
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()

    # 2. Encode a reference image as the color anchor
    ref_image = load_image("https://example.com/reference.jpg")
    ref_image = ref_image.resize((1024, 1024))

    pipe.anchor_latents = encode_anchor_image(ref_image, pipe)
    # Optional: tweak parameters
    # pipe.color_anchor_strength = 0.25       # default; increase for stronger correction
    # pipe.color_anchor_last_n_steps = 1      # default; increase to 2-3 for more effect

    # 3. Generate with the color anchor callback
    prompt = (
        "A high-end product photography shot of a perfume bottle, dramatic lighting, "
        "detailed color grade, crisp white balance, uniform tones"
    )

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=4,
        callback_on_step_end=flux_klein_color_anchor_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]

    image.save("flux_klein_anchored_output.png")