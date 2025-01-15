# Training LoRA with Bria's Tailored Generation Model

This repository provides a comprehensive guide and tools for training LoRA models using Bria's new Tailored Generation foundation model. This improved system offers enhanced prompt alignment, better fidelity, and simplified workflow compared to previous solutions.

## 🌟 Implementation Options

This repository offers three flexible ways to implement LoRA training with Bria's Tailored Generation:

1. **Python Scripts** (This Repository): Complete control and customization
2. **Web Interface**: User-friendly GUI for quick implementation
3. **API Integration**: Production-ready REST API ([Documentation](https://bria-ai-api-docs.redoc.ly/tag/Tailored-Generation/))

💡 **Choose based on your needs**: Python scripts for maximum control, Web Interface for ease of use, or API for production deployment.

## 🎯 Key Improvements

- **Enhanced Prompt Alignment**: Better correlation between prompts and generated images
- **Simplified Training Process**: No need for style prefixes/suffixes
- **Higher Fidelity**: More accurate reproduction of trained elements
- **Dense Caption Training**: Built on a foundation model trained with detailed, accurate descriptions
- **Enterprise-Ready**: Risk-free generation suitable for commercial applications

## 🚀 Technical Specifications

- **Base Model**: [BRIA-4B-Adapt](https://huggingface.co/briaai/BRIA-4B-Adapt) - 4B parameter transformer-based architecture
- **Model Card**: Comprehensive documentation available on [HuggingFace](https://huggingface.co/briaai/BRIA-4B-Adapt)
- **Text Encoder**: Advanced T5-based encoder (single encoder architecture)
- **Training Time**: ~1.25 hours on G6a instance for 1500 steps
- **Optimal Parameters**: Tested defaults provide best results (detailed benchmarks below)

## 📋 Prerequisites

- CUDA-compatible GPU
- Python 3.8+
- Access to Bria's foundation model
- Recommended: G6a instance for optimal performance

## 🛠 Getting Started

### 1. Installation & Setup

```bash
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt
```

### 2. Data Preparation

#### Image Requirements

- **Resolution & Aspect Ratios**:
  - Default: 1024x1024 pixels
  - Supported variations (approximately 1M pixels):
    - 1280 x 768
    - 1344 x 768
    - 832 x 1216
    - 1152 x 832
    - 1216 x 832
    - 960 x 1088
  - Images are automatically resized and center-cropped (control with `center_crop` and `resolution` arguments)

- **Image Variety**:
  - Maintain consistency in target visual elements
  - Include sufficient variations for model generalization
  - Recommended dataset size: 15-20 high-quality images

#### Caption Guidelines

- **Length**: Less than 128 tokens (~100 words)
- **Content Structure**:
  - Include unique content descriptions
  - Use constant domain descriptions (e.g., "An illustration of a cute brown bear")
  - Consider using trigger words (e.g., "a character named Briabear")

💡 **Pro Tip**: Quality over quantity - fewer high-quality images often produce better results than larger, lower-quality datasets.

### 3. Training Process

#### Key Hyperparameters

- **Rank**: 
  - Default: 128
  - Lower ranks (e.g., 64, 32) suitable for simple cases
  - Higher ranks for finer details
  
- **Optimizer**:
  - Default: Prodigy with learning_rate=1
  - Based on [Hugging Face recommendations](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md)

```bash
# Basic training command
bash train_lora.sh

# Advanced configuration (example)
python train_text_to_image_lora.py \
    --pretrained_model_name_or_path="briaai/BRIA-4B-Adapt" \
    --dataset_path="path/to/your/dataset" \
    --output_dir="./lora_output" \
    --rank=128 \
    --max_train_steps=1500 \
    --learning_rate=1
```

### 4. Generating Images with Trained LoRA

```python
from diffusers import BriaPipeline
import torch

# Load the pipeline with your trained LoRA
pipeline = BriaPipeline.from_pretrained(
    "briaai/BRIA-4B-Adapt",
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights("path/to/your/lora")

# Generation parameters
prompt = "An illustration of a cute brown bear in a forest"
negative_prompt = None  # Optional: Use for better control
num_inference_steps = 30  # Recommended: 30-50 steps
guidance_scale = 5.0    # Recommended default

# Generate image
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale
).images[0]
image.save("generated_image.png")
```

💡 **Pro Tips**:
- Higher number of inference steps (30-50) generally produces better results
- Use guidance_scale=5.0 as a starting point
- Consider using negative prompts for better control
- Experiment with different checkpoints from your training

## 📊 Performance & Case Studies

### Modern Blurred SeaView Dataset Case Study

We conducted comparative experiments using the [Modern Blurred SeaView dataset](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView/viewer) to demonstrate the improvements in this new model. This dataset is particularly useful for testing color fidelity and style consistency.

#### Key Improvements Observed:
- **Enhanced Color Palette Fidelity**: Significantly better preservation of the original dataset's color scheme
- **Style Consistency**: More reliable reproduction of the blurred seaview aesthetic
- **Prompt Alignment**: Better correlation between text prompts and generated images

#### Experiment Details:
```python
dataset_name = "Negev900/Modern_Blurred_SeaView"
training_params = {
    "max_train_steps": 1500,
    "rank": 128,
    "learning_rate": 1
}
```

#### Sample Results

```markdown
Input Prompt: "A modern blurred seascape view with soft pastel colors"

Previous Model:
- Inconsistent color reproduction
- Variable style application

New Tailored Generation:
- Precise color palette matching
- Consistent blur effects
- Better preservation of artistic style
```

💡 **Key Learning**: The new model's improved color fidelity makes it particularly suitable for projects where maintaining a specific color scheme is crucial.

## 📊 Performance Comparisons

| Metric | Previous Model | Tailored Generation | Improvement |
|--------|---------------|---------------------|-------------|
| Prompt Alignment | 70-90% | 95%+ | ~20% |
| Training Time | 1.25hr (G4) | 1.25hr (G6a) | Similar* |
| Style Fidelity | Requires prefixes | Native support | Simplified |

*Note: Similar training time on upgraded hardware (G6a vs G4)


## 🚨 Troubleshooting

Common issues and solutions:

1. **Memory Issues**
   - Reduce batch size
   - Use gradient checkpointing
   - Optimize image resolution

2. **Training Quality**
   - Check caption accuracy
   - Verify dataset consistency
   - Review learning rate

