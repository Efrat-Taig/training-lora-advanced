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

## 🛠 Installations

```bash
pip install -qr https://huggingface.co/briaai/BRIA-4B-Adapt/resolve/main/requirements.txt

```


```python
from huggingface_hub import hf_hub_download
import os

try:
    local_dir = os.path.dirname(__file__)
except:
    local_dir = '.'
    
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='pipeline_bria.py', local_dir=local_dir)
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='transformer_bria.py', local_dir=local_dir)
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='bria_utils.py', local_dir=local_dir)
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='train_lora.py', local_dir=local_dir)



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

