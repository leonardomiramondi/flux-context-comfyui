#!/bin/bash

# Flux Replicate ComfyUI Node Installation Script

set -e

echo "🚀 Installing Flux Replicate ComfyUI Node..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the node directory."
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

# Check for API token
if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo "⚠️  Warning: REPLICATE_API_TOKEN environment variable not set."
    echo "   You can set it by running:"
    echo "   export REPLICATE_API_TOKEN=your_token_here"
    echo ""
    echo "   Or add it to your shell profile:"
    echo "   echo 'export REPLICATE_API_TOKEN=your_token_here' >> ~/.zshrc"
    echo "   source ~/.zshrc"
else
    echo "✅ REPLICATE_API_TOKEN is set"
fi

echo ""
echo "✅ Installation complete!"
echo "   Please restart ComfyUI to load the new node."
echo "   The node will appear under: Add Node → image → generation → Flux Replicate Context" 