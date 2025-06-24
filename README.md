# Flux Replicate ComfyUI Node

A custom ComfyUI node that enables image generation using Flux models through the Replicate API, with support for multiple image inputs and advanced processing modes.

## Features

- 🎨 **Multiple Flux Models**: Support for Flux Schnell, Flux Dev, and Flux Pro
- 🖼️ **Dual Image Support**: Two image inputs for various processing modes
- ⚙️ **Advanced Controls**: Configurable dimensions, steps, guidance scale, and more
- 🔧 **ComfyUI Integration**: Seamless integration with your existing ComfyUI workflows
- 🚀 **High Quality Output**: Support for multiple output formats (WebP, JPG, PNG)
- 🎭 **Multiple Modes**: img2img, style reference, and image blending capabilities

## Quick Start

### Method 1: One-Liner Installation (Recommended)

**For macOS/Linux:**
```bash
cd ComfyUI/custom_nodes/ && git clone https://github.com/leonardomiramondi/flux-replicate-comfyui.git && cd flux-replicate-comfyui && ./install.sh
```

**For Windows:**
```cmd
cd ComfyUI\custom_nodes && git clone https://github.com/leonardomiramondi/flux-replicate-comfyui.git && cd flux-replicate-comfyui && pip install -r requirements.txt
```

### Method 2: Step-by-Step Git Clone

1. **Navigate to your ComfyUI custom nodes directory**:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/leonardomiramondi/flux-replicate-comfyui.git
   ```

3. **Install dependencies**:
   ```bash
   cd flux-replicate-comfyui
   pip install -r requirements.txt
   # OR run the automated installer
   ./install.sh
   ```

4. **Restart ComfyUI**

### Method 3: Manual Download

1. **Download the repository**:
   - Click the green "Code" button on GitHub
   - Select "Download ZIP"
   - Extract the ZIP file

2. **Copy to ComfyUI**:
   ```bash
   # Move the extracted folder to your ComfyUI custom nodes directory
   mv flux-replicate-comfyui-main ComfyUI/custom_nodes/flux-replicate-comfyui
   ```

3. **Install dependencies**:
   ```bash
   cd ComfyUI/custom_nodes/flux-replicate-comfyui
   pip install -r requirements.txt
   ```

4. **Restart ComfyUI**

## Verification

After installation, the node should appear in ComfyUI under:
**Add Node → image → generation → Flux Replicate Context**

## Usage

### Basic Text-to-Image Generation

1. Add the "Flux Replicate Context" node to your workflow
2. **Enter your Replicate API token** in the `api_token` field (get one from [replicate.com](https://replicate.com/account/api-tokens))
3. Connect a text input to the `prompt` parameter
4. Configure your desired settings:
   - **Model**: Choose between Flux Schnell (fastest), Flux Dev (balanced), or Flux Pro (highest quality)
   - **Dimensions**: Set width and height (multiples of 64, max 2048x2048)
   - **Steps**: Number of inference steps (4-50, lower is faster)
   - **Guidance Scale**: How closely to follow the prompt (Dev/Pro only)
   - **Seed**: For reproducible results (-1 for random)

### Image Generation Modes

1. **Text-to-Image**: Use only the text prompt (no image inputs)
2. **Image-to-Image**: Connect an image to `image_input_1` and set mode to "img2img"
3. **Style Reference**: Connect a reference image to `image_input_1` and set mode to "reference"
4. **Image Blending**: Connect two images to `image_input_1` and `image_input_2`, set mode to "blend"

### Advanced Parameters

- **Negative Prompt**: Specify what you don't want in the image
- **Output Format**: Choose between WebP (smallest), JPG (compatible), or PNG (lossless)
- **Output Quality**: JPEG/WebP compression quality (1-100)

## Node Parameters

### Required Inputs
- `api_token` (STRING): Your Replicate API token (get from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens))
- `prompt` (STRING): Text description of the image to generate
- `model` (CHOICE): Flux model variant to use
- `width` (INT): Image width in pixels (256-2048, step 64)
- `height` (INT): Image height in pixels (256-2048, step 64)
- `num_inference_steps` (INT): Number of denoising steps (1-50)
- `guidance_scale` (FLOAT): Prompt adherence strength (0.0-20.0)
- `seed` (INT): Random seed for reproducibility (-1 for random)

### Optional Inputs
- `image_input_1` (IMAGE): Primary image input for img2img, reference, or blending
- `image_input_2` (IMAGE): Secondary image input for blending mode
- `negative_prompt` (STRING): Text describing what to avoid
- `output_format` (CHOICE): Image format (webp/jpg/png)
- `output_quality` (INT): Compression quality (1-100)
- `image_mode` (CHOICE): How to use input images (img2img/reference/blend)

### Output
- `IMAGE`: Generated image tensor compatible with other ComfyUI nodes

## Model Comparison

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| **Flux Schnell** | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | Quick iterations, drafts |
| **Flux Dev** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Balanced quality/speed |
| **Flux Pro** | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Final renders, high quality |

## Troubleshooting

### Common Issues

1. **"API token is required" error**:
   - Enter your Replicate API token in the `api_token` field
   - Get your token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)

2. **"Failed to create prediction" error**:
   - Check your API token is correct and valid
   - Ensure you have sufficient Replicate credits
   - Verify your internet connection

3. **"Prediction timed out" error**:
   - Try reducing image dimensions
   - Lower the number of inference steps
   - Check Replicate's service status

4. **Poor image quality**:
   - Increase inference steps (try 10-20 for Dev/Pro)
   - Use more descriptive prompts
   - Try Flux Dev or Pro models
   - Adjust guidance scale (7.5 is often good)

5. **Node not appearing in ComfyUI**:
   - Restart ComfyUI completely
   - Check the console for error messages
   - Ensure all dependencies are installed

### Performance Tips

- **For fastest generation**: Use Flux Schnell with 4 steps
- **For best quality**: Use Flux Pro with 20+ steps
- **For balanced workflow**: Use Flux Dev with 10-15 steps
- **Memory optimization**: Keep dimensions at 1024x1024 or lower
- **Batch processing**: Generate multiple variations by changing seed values

## API Rate Limits

Replicate has usage limits based on your plan:
- **Free tier**: Limited requests per month
- **Pro tier**: Higher rate limits and priority processing
- **Enterprise**: Custom limits

Monitor your usage at [replicate.com/account](https://replicate.com/account)

## Examples

### Basic Landscape
```
Prompt: "A serene mountain landscape at sunset with a crystal clear lake reflecting the orange and pink sky"
Model: Flux Dev
Steps: 15
Guidance: 7.5
```

### Portrait Style Transfer
```
Prompt: "Transform this photo into a Renaissance painting style portrait"
Model: Flux Pro
Image Input 1: [uploaded portrait]
Image Mode: img2img
Steps: 25
Guidance: 10.0
```

### Image Blending
```
Prompt: "Merge these two landscapes into a surreal dreamscape"
Model: Flux Dev
Image Input 1: [mountain landscape]
Image Input 2: [ocean scene]
Image Mode: blend
Steps: 20
Guidance: 8.0
```

### Abstract Art
```
Prompt: "Abstract geometric composition with vibrant colors and flowing shapes"
Model: Flux Schnell
Steps: 4
Guidance: 0.0 (ignored for Schnell)
```

## License

This project is open source. Please check the individual licenses of dependencies.

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## Repository Structure

```
flux-replicate-comfyui/
├── __init__.py              # ComfyUI node registration
├── flux_replicate_node.py   # Main node implementation
├── requirements.txt         # Python dependencies
├── install.sh              # Automated installation script
├── README.md               # This documentation
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore rules
└── .github/workflows/      # GitHub Actions for CI
    └── test.yml           # Automated testing
```

## GitHub Repository Setup

When creating your GitHub repository:

1. **Repository name**: `flux-replicate-comfyui`
2. **Description**: "ComfyUI custom node for Flux image generation via Replicate API with dual image support"
3. **Topics/Tags**: `comfyui`, `flux`, `replicate`, `image-generation`, `ai`, `custom-node`
4. **Make it public** for easy sharing and community access

## Support

- 🐛 **Issues**: [Create an issue](https://github.com/YOUR_USERNAME/flux-replicate-comfyui/issues) for bugs or feature requests
- 💬 **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/flux-replicate-comfyui/discussions) for questions and community chat
- 📖 **ComfyUI**: Check the main [ComfyUI repository](https://github.com/comfyanonymous/ComfyUI)
- 🔗 **Replicate**: [Replicate API documentation](https://replicate.com/docs) 