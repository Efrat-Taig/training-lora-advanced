# Training LoRA with Bria

This repository provides a comprehensive guide and tools for training LoRA models using Bria's new Generation foundation model. This improved system offers enhanced prompt alignment, better fidelity, and simplified workflow compared to previous solutions.

## 🌟 Implementation Options

Bria offers three flexible ways to implement LoRA training with Bria's Tailored Generation:

1. **Python Scripts** (This Repository): Allwing complete control and customization, based on [this](https://platform.bria.ai/console/tailored-generation/projects) model card
2. **Web Interface**: User-friendly [GUI](https://platform.bria.ai/console/tailored-generation/projects) for quick implementation 
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

- CUDA-compatible GPU (Recommended: G6a instance for optimal performance)
- Access to Bria's foundation model
     >💡 Note that this one is important :) If you are an academic researcher, you can get free access by filling out [this form](https://docs.google.com/forms/u/1/d/e/1FAIpQLSe-E1r-QoBmsAZbJ5MJKB76wGnk6bUn2kBq5imPQVVJviv1Kg/viewform) .Otherwise, feel free to reach out to us on our [Discord channel](https://discord.gg/Pkbp2BmZbq)



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

We need to perform these because the training scripts differ slightly from those provided by Diffusers. This is due to the unique architecture of the Bria models.



# Modern Blurred SeaView Dataset Case Study

We conducted comparative experiments using the [Modern Blurred SeaView dataset](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView/viewer)
an open-source datasets available on Hugging Face to demonstrate the improvements in this new model.

# Training LoRA with BRIA
To train the BRIA model using LoRA fine-tuning, use the run_fintune script.

```bash
run_fintune.sh
```

### Parameters:
- **`--pretrained_model_name_or_path`**: Path to the pre-trained BRIA model.
- **`--dataset_name`**: Hugging Face dataset containing images and prompts.
- **`--output_dir`**: Directory to save fine-tuned weights.


Note that for training, we are using the script `train_lora.py`, which we just downloaded using the command:  

```python
hf_hub_download(repo_id="briaai/BRIA-4B-Adapt", filename='train_lora.py', local_dir=local_dir)

```
If you have modified any paths, update them here as well, or provide the full path to the 'train_lora.py' script.




# Running Inference

The `inference.py` script allows you to generate images using the BRIA pipeline with pre-trained and optionally fine-tuned LoRA weights.

#### Steps for Running Inference:

1. **Set Up Your Environment**:
   - Ensure all dependencies are installed as described in the setup instructions.
   - Verify that your Hugging Face token is correctly set in the script (`your_hf_token_here`).

2. **Customize Your Inputs**:
- **Prompts**: Adjust `prompt` and `negative_prompt` for your use case.
- **Image Size**: Modify the `height` and `width` parameters in the `pipe()` function to specify the resolution.
- **LoRA Weights**: Specify the path to your fine-tuned weights in the `pipe.load_lora_weights()` function.

3. **Run the Script**:
   Execute the script using:
   ```bash
   python inference.py
   ```

4. **Output**:
   - The script will generate an image based on your prompt.
   - The generated image will be saved in the current directory as `generated_image.png`.

The script will output an image and save it as `generated_image.png`. 



#  Sample Results

Sample from my [Modern Blurred SeaView](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView) Dataset:

<img src="https://github.com/Efrat-Taig/training-lora/blob/main/Data_set_sample.png" width="600">>

And here are results: 

<img src="https://github.com/Efrat-Taig/training-lora-advanced/blob/main/Lora_res.png">

What is shown here are images generated by the foundation model (aligned with the prompt but not specific to the dataset color scheme) and images generated by the fine-tuned model (aligned with both the prompt and the dataset color scheme).


# Key Improvements 
### Key Improvements Observed from https://github.com/Efrat-Taig/training-lora-basics
- **Enhanced Color Palette Fidelity**: Significantly better preservation of the original dataset's color scheme
- **Style Consistency**: More reliable reproduction of the blurred seaview aesthetic
- **Prompt Alignment**: Better correlation between text prompts and generated images



💡 **Key Learning**: The new model's improved color fidelity makes it particularly suitable for projects where maintaining a specific color scheme is crucial.

     
## Final Notes
For further assistance or collaboration opportunities, feel free to reach out:
- Email: efrat@bria.ai
- [LinkedIn](https://www.linkedin.com/in/efrattaig/)
- Join my [Discord community](https://discord.gg/Nxe9YW9zHS) for more information, tutorials, tools, and to connect with other users!
