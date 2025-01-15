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

Your training dataset should meet these specifications:

- **Resolution**: 1024x1024 pixels (required)
- **Format**: Images paired with detailed captions
- **Storage**: Compatible with local directories, S3, or Hugging Face datasets
- **Minimum Dataset Size**: Recommended 15-20 high-quality images

💡 **Pro Tip**: Quality over quantity - fewer high-quality images often produce better results than larger, lower-quality datasets.

### 3. Training Process

```bash
# Basic training command
bash train_lora.sh

# Advanced configuration (example)
python train_text_to_image_lora.py \
    --pretrained_model_name_or_path="bria/tailored-gen-v1" \
    --dataset_path="path/to/your/dataset" \
    --output_dir="./lora_output" \
    --num_train_epochs=100 \
    --learning_rate=1e-4
```

## 📊 Performance Comparisons

| Metric | Previous Model | Tailored Generation | Improvement |
|--------|---------------|---------------------|-------------|
| Prompt Alignment | 70-90% | 95%+ | ~20% |
| Training Time | 1.25hr (G4) | 1.25hr (G6a) | Similar* |
| Style Fidelity | Requires prefixes | Native support | Simplified |

*Note: Similar training time on upgraded hardware (G6a vs G4)

## 🔍 Best Practices

1. **Caption Quality**: Focus on detailed, accurate descriptions
2. **Dataset Preparation**: Ensure consistent style and quality
3. **Parameter Selection**: Default parameters are optimized
4. **Hardware Selection**: G6a instances recommended for best performance

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

## 💻 API Integration

For production environments, you can access the same functionality through our REST API:

```python
import requests

url = "https://api.bria.ai/v1/tailored-generation/train"
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer YOUR_API_KEY"
}

payload = {
    "dataset_url": "your_dataset_url",
    "training_parameters": {
        "num_epochs": 100,
        "learning_rate": 1e-4
    }
}

response = requests.post(url, json=payload, headers=headers)
```

Complete API documentation is available [here](https://bria-ai-api-docs.redoc.ly/tag/Tailored-Generation/).

💡 **Pro Tip**: The API is ideal for automated workflows and production environments where you need reliable, scalable training capabilities.

## 🔄 Differences from Model Card

While the [official model card](https://huggingface.co/briaai/BRIA-4B-Adapt) provides basic information, this repository offers:

- Detailed step-by-step training procedures
- Comprehensive troubleshooting guide
- Optimized training parameters based on extensive testing
- Real-world usage examples and best practices
- Performance comparisons and benchmarks
- Integration examples for all three implementation methods

## 📚 Additional Resources

- [Detailed Documentation](link-to-docs)
- [API Reference](link-to-api)
- [Example Notebooks](link-to-notebooks)
- [Community Discord](link-to-discord)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📫 Contact & Support

- Email: [Your Email]
- Discord: [Discord Link]
- Issues: Please use the GitHub issues tab

## 📄 License

[Your License Type] - See LICENSE file for details
