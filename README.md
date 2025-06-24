# Flux Context ComfyUI Node

A custom ComfyUI node for Flux image generation with context support via Replicate API.

## Installation

```bash
cd ComfyUI/custom_nodes/ && git clone https://github.com/leonardomiramondi/flux-context-comfyui.git && cd flux-context-comfyui && pip install -r requirements.txt
```

Restart ComfyUI. The node will appear under: **Add Node → image → generation → Flux Context**

## Usage

1. Add the "Flux Context" node to your workflow
2. Enter your Replicate API token (get from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens))
3. Enter your text prompt
4. Optionally connect 1-2 images for context/reference
5. Choose your Flux model (Schnell/Dev/Pro) and settings

Get your Replicate API token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)

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