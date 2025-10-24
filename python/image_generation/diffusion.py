"""
    Credit: Anthony Assi
"""
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import datetime

# --- Configuration ---
# Define the model ID for Realistic Vision V6.0 B1.
model_id = "stablediffusionapi/realistic-vision-v60-b1"

# Define the prompt for the image you want to generate.
# prompt = "paw patrol on a mission are fighting against spiderman"
prompt = "Wild Kratts brothers cartoon in the african savannah in the cheetah power costume"

# Define things you want to avoid in the image.
negative_prompt = """
dark, unhappy
"""

# --- Setup ---
# Check if CUDA (GPU) is available and set the device accordingly.
if torch.cuda.is_available():
    device = "cuda"
    print("--- Using GPU for image generation ---")
else:
    device = "cpu"
    print("--- CUDA not available, using CPU. This will be very slow. ---")

# --- Pipeline Creation ---
print(f"--- Loading Model: {model_id} ---")
# Create the Stable Diffusion pipeline.
# We use torch.float16 for better performance on modern GPUs like in your DGX Spark.
pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
)

# Configure a high-quality scheduler for the diffusion process.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move the entire pipeline to the GPU.
pipe.to(device)

# --- Image Generation ---
print(f"--- Generating image with prompt: '{prompt}' ---")

# Generate the image. This is the main step that uses the GPU.
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5
).images[0]

# --- Save the Output ---
# Create a unique filename using the current timestamp.
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

IMG_DIR = "./img"
os.makedirs(IMG_DIR, exist_ok=True)
output_filename = f"{IMG_DIR}/image_{timestamp}.png"
# Save the generated image to a file.
image.save(output_filename)

print(f"--- Image successfully saved as '{output_filename}' ---")