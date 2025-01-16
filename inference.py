import torch
import os
from pipeline_bria import BriaPipeline

# Set your Hugging Face token (replace with your own token)
os.environ["HF_TOKEN"] = "your_hf_token_here"

# Create the pipeline
pipe = BriaPipeline.from_pretrained(
    "briaai/BRIA-4B-Adapt", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load the LoRA weights (replace with your own path to weights)
pipe.load_lora_weights(
    "path_to_your_lora_weights/checkpoint-1500",
    weight_name="pytorch_lora_weights.safetensors"
)

# Move the pipeline to GPU
pipe.to(device="cuda")

# Define the prompt and negative prompt
prompt = "your prompt here."
negative_prompt = "your negative prompt here."

# Log before image generation
print("Starting image generation...")

# Generate the image
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    height=1024, 
    width=1024
).images[0]

# Save the generated image
image.save("generated_image.png")
print("Image saved successfully!")
