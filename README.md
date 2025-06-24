# Flux Context ComfyUI Node

Edit and transform images using AI with text prompts. Powered by FLUX.1 Kontext - the best-in-class text-guided image editing model.

## What is Flux Context?

Flux Context (Kontext) is an advanced image editing AI that lets you transform existing images using simple text descriptions. Unlike general image generators, it specializes in:

- **Style Transfer**: Turn photos into paintings, sketches, or any artistic style
- **Object Editing**: Change hairstyles, clothing, colors, accessories
- **Background Replacement**: Swap environments while preserving subjects
- **Text Editing**: Replace text in signs, posters, logos
- **Character Consistency**: Edit people while maintaining their identity

## Installation

**Find your ComfyUI custom nodes folder:**

```bash
# Common locations - use the one that matches your setup:
cd /path/to/your/ComfyUI/custom_nodes/           # Replace with your actual path
cd ~/ComfyUI/custom_nodes/                       # Home folder installation
cd /Applications/ComfyUI/custom_nodes/           # macOS app bundle
cd C:\ComfyUI\custom_nodes\                      # Windows installation
```

💡 **Can't find your ComfyUI folder?** Look for where you installed ComfyUI or check inside the ComfyUI interface under `Settings → Paths` or look for a `custom_nodes` folder next to your `main.py` file.

**Then install the node:**

```bash
git clone https://github.com/leonardomiramondi/flux-context-comfyui.git
cd flux-context-comfyui
pip install -r requirements.txt
```

**Restart ComfyUI.** The node will appear under: **Add Node → image → editing → Flux Context**

## How to Use

1. **Add the "Flux Context" node** to your workflow
2. **Connect an image** to the `input_image` input (the image you want to edit)
3. **Enter your Replicate API token** (get from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens))
4. **Write your editing prompt** describing how you want to transform the image
5. **Optionally add a context image** for style reference
6. **Run the workflow** to get your edited image

## Example Editing Prompts

- `"Make this a watercolor painting"`
- `"Change the hair to blonde with a pixie cut"`
- `"Replace the background with a forest setting"`
- `"Turn this into a 1990s cartoon style"`
- `"Add a red leather jacket"`
- `"Convert to black and white photography"`

## Node Parameters

**Required:**
- `input_image` - The image you want to edit
- `editing_prompt` - Text describing the desired changes
- `api_token` - Your Replicate API token
- `model` - Choose between Kontext Pro (fast) or Max (highest quality)

**Optional:**
- `context_image` - Reference image for style/composition guidance
- `context_strength` - How strongly to follow the context image (0.1-1.0)
- `preserve_identity` - Keep facial features unchanged when editing people
- `output_format` - WebP, JPG, or PNG
- `output_quality` - Compression quality (1-100)

## Tips for Better Results

- **Be specific**: "Change hair to short blonde bob" vs "change hair"
- **Preserve what matters**: The node automatically helps maintain identity for people
- **Use context images**: Upload a reference image for complex style transfers
- **Start simple**: Make one change at a time for best results

Get your Replicate API token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) 