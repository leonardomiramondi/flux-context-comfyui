#!/bin/bash

# Flux Kontext ComfyUI Node Installation Script

set -e

echo "🚀 Installing Flux Kontext ComfyUI Node..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    echo "   Make sure you're in the flux-kontext-comfyui directory inside your ComfyUI/custom_nodes/ folder."
    echo "   Your ComfyUI path may be different - check where you installed ComfyUI."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo "❌ Error: pip not found. Please install Python and pip first."
    exit 1
fi

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Installation completed successfully!"
    echo ""
    echo "🎉 Flux Kontext ComfyUI Node is now ready to use!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Restart ComfyUI"
    echo "   2. Get your Replicate API token from: https://replicate.com/account/api-tokens"
    echo "   3. In ComfyUI workflows, you can now use the Flux Kontext node"
    echo "   The node will appear under: Add Node → image → generation → Flux Kontext"
    echo ""
    echo "💡 Need help? Check the README.md or visit the GitHub repository!"
else
    echo "❌ Installation failed!"
    echo "   Please check the error messages above and try again."
    exit 1
fi 