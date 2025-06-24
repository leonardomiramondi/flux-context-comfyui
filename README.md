# Flux Context ComfyUI Node

A ComfyUI custom node for **Flux Context** (Kontext) - advanced AI-powered image editing using text prompts. This node integrates with Replicate's Flux Kontext models to perform sophisticated image transformations.

## What is Flux Context?

Flux Context (Kontext) is a specialized AI model from Black Forest Labs designed for **image editing**, not generation. It excels at:

- **Style Transfer** - Convert photos to watercolor, oil painting, sketches, etc.
- **Object Editing** - Modify hairstyles, clothing, colors, accessories
- **Text Editing** - Replace text in signs, posters, labels
- **Background Changes** - Swap environments while preserving subjects
- **Artistic Transformations** - Apply specific art styles with precision

## Features

- **Direct Replicate API Integration** - No external dependencies
- **Built-in API Token Input** - No environment variables needed
- **Two Image Inputs** - Primary image to edit + optional reference image
- **Real-time Progress** - Monitor editing progress in ComfyUI console
- **Multiple Models** - Choose between Kontext Pro and Kontext Max

## Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Go to "Install Custom Nodes"
3. Search for "flux context"
4. Click Install
5. Restart ComfyUI

### Method 2: Manual Installation
```bash
# Navigate to your ComfyUI custom_nodes directory
# Common paths:
# Windows: ComfyUI\custom_nodes\
# Mac: ComfyUI/custom_nodes/
# Linux: ComfyUI/custom_nodes/

cd ComfyUI/custom_nodes/
git clone https://github.com/leonardomiramondi/flux-context-comfyui.git
cd flux-context-comfyui
pip install -r requirements.txt
```

Then restart ComfyUI.

## Usage

1. **Add the node** - Search for "Flux Context" in ComfyUI node browser
2. **Connect an image** to the `image_1` input (the image you want to edit)
3. **Enter your Replicate API token** - Get yours from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
4. **Write an editing prompt** - e.g., "Make this a watercolor painting"
5. **Optional**: Connect a second image to `image_2` for style reference
6. **Run the workflow**

## Node Inputs

### Required:
- `api_token` - Your Replicate API token
- `image_1` - The image you want to edit
- `editing_prompt` - Text description of how to edit the image
- `model` - Choose between flux-kontext-pro or flux-kontext-max

### Optional:
- `image_2` - Reference image for style/composition guidance

## Example Prompts

- `"Transform this portrait into an oil painting"`
- `"Change the red car to blue while keeping everything else the same"`
- `"Make this photo look like a vintage 1950s advertisement"`
- `"Convert this landscape into a watercolor painting with soft brushstrokes"`
- `"Replace the text 'SALE' with 'NEW' in this sign"`

## Models Available

- **flux-kontext-pro** - High-quality, balanced performance
- **flux-kontext-max** - Maximum quality, best for professional work

## Troubleshooting

### "Invalid version or not permitted" Error
- Make sure you're using a valid Replicate API token
- Check that your token has permissions for the selected model

### Node Not Appearing
- Restart ComfyUI completely after installation
- Check the console for any error messages during startup

## Updates

This node automatically updates when you use ComfyUI Manager's "Update All" feature. No manual intervention needed!

## Requirements

- ComfyUI
- Active internet connection
- Replicate API account and token

## License

MIT License - see LICENSE file for details. 