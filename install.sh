#!/bin/bash

# Flux Context ComfyUI Node Installation Script

set -e

echo "🚀 Installing Flux Context ComfyUI Node..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found."
    echo "   Make sure you're in the flux-context-comfyui directory inside your ComfyUI/custom_nodes/ folder."
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

echo ""
echo "✅ Installation complete!"
echo "   Please restart ComfyUI to load the new node."
echo "   The node will appear under: Add Node → image → generation → Flux Context"
echo ""
echo "ℹ️  Don't forget to get your Replicate API token from:"
echo "   https://replicate.com/account/api-tokens" 