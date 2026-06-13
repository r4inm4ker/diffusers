import torch


def flux_klein_color_anchor_callback(pipeline, step, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")

    # Verify the anchor exists on the pipeline
    if not hasattr(pipeline, "anchor_latents") or pipeline.anchor_latents is None:
        return callback_kwargs

    anchor = pipeline.anchor_latents  # Expected shape: [batch, 16, height, width]

    # Hyperparameters matching ComfyUI node logic
    strength = 0.80  # 0.0 = no effect, 1.0 = hard color match
    epsilon = 1e-5  # Prevents division by zero

    # Loop through the 16 latent channels of FLUX
    with torch.no_grad():
        for b in range(latents.shape[0]):
            for c in range(latents.shape[1]):
                # 1. Calculate current latent statistics
                curr_mean = latents[b, c].mean()
                curr_std = latents[b, c].std()

                # 2. Calculate anchor reference statistics
                anch_mean = anchor[b, c].mean()
                anch_std = anchor[b, c].std()

                # 3. Apply AdaIN / Latent Color Matching formula
                normalized = (latents[b, c] - curr_mean) / (curr_std + epsilon)
                matched_channel = (normalized * anch_std) + anch_mean

                # 4. Blend back with the generated latent based on strength
                latents[b, c] = (1.0 - strength) * latents[b, c] + strength * matched_channel

    callback_kwargs["latents"] = latents
    return callback_kwargs





import torch
from diffusers import Flux2KleinPipeline # Requires diffusers>=0.26.0 for Klein family
from diffusers.utils import load_image

# 1. Load the official 9B pipeline
dtype = torch.bfloat16
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B", # Or the 4-step distilled repo
    torch_dtype=dtype
)
pipe.enable_model_cpu_offload() # Saves VRAM for local execution

# 2. Prepare the Anchor Reference Image
ref_image = load_image("https://example.com")
ref_image = ref_image.resize((1024, 1024))

# 3. Encode reference image to FLUX's 16-channel latent space
def image_to_flux_latents(image, pipeline):
    with torch.no_grad():
        # Convert image to tensor [-1, 1]
        img_tensor = pipeline.image_processor.preprocess(image).to(device="cuda", dtype=dtype)
        # Pass through VAE encoder
        vae_output = pipeline.vae.encode(img_tensor).latent_dist.sample()
        # Scale according to FLUX scaling factors
        latents = (vae_output - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    return latents

# Attach the processed anchor directly to the pipeline instance
pipe.anchor_latents = image_to_flux_latents(ref_image, pipe)

# 4. Execute the Generation
prompt = "A high-end product photography shot of a perfume bottle, dramatic lighting"
# Include explicit color grade terms in prompt to combat Klein's red-bias
prompt += ", detailed color grade, crisp white balance, uniform tones"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=4, # 4 steps for the distilled Klein 9B model
    callback_on_step_end=flux_klein_color_anchor_callback,
    callback_on_step_end_tensor_inputs=["latents"]
).images[0]

image.save("flux_klein_anchored_output.png")