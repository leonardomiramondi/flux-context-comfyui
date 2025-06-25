import requests
import time
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import json

class FluxKontextNode:
    """
    ComfyUI node for Flux Kontext - image editing and transformation using text prompts
    Built using official Replicate API documentation
    """
    
    def __init__(self):
        self.base_url = "https://api.replicate.com/v1"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "r8_",
                    "placeholder": "Enter your Replicate API token"
                }),
                "image_1": ("IMAGE",),
                "editing_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Put the person into a white t-shirt",
                    "placeholder": "Describe how you want to edit the image"
                }),
                "model": ([
                    "flux-kontext-apps/multi-image-kontext-max",
                    "flux-kontext-apps/multi-image-kontext-pro",
                    "black-forest-labs/flux-kontext-pro",
                    "black-forest-labs/flux-kontext-max"
                ], {
                    "default": "flux-kontext-apps/multi-image-kontext-max"
                }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], {
                    "default": "1:1"
                }),
            }
        }
    

    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/editing"
    DESCRIPTION = "Edit images using Flux Kontext models via Replicate API"
    
    def pil_to_tensor(self, image):
        """Convert PIL image to tensor format expected by ComfyUI"""
        image_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None,]
    
    def tensor_to_base64(self, tensor, max_size=1024):
        """Convert ComfyUI tensor to base64 data URL"""
        # Handle tensor dimensions
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert tensor to PIL Image
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Resize if needed to prevent payload issues
        if max(img.width, img.height) > max_size:
            ratio = max_size / max(img.width, img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image to {new_width}x{new_height}")
        
        # Convert to base64 data URL
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    def upload_image_to_replicate(self, image_data_url):
        """
        Upload image to Replicate and get a public URL
        This is needed because some models expect URLs instead of data URLs
        """
        try:
            # For now, return the data URL directly
            # Replicate should handle data URLs in most cases
            return image_data_url
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")
    
    def edit_image(self, api_token, **kwargs):
        """Main image editing function using Flux Kontext"""
        
        # Extract parameters
        image_1 = kwargs.get("image_1")
        image_2 = kwargs.get("image_2") 
        editing_prompt = kwargs.get("editing_prompt", "Put the person into a white t-shirt")
        model = kwargs.get("model", "flux-kontext-apps/multi-image-kontext-max")
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        
        # Validate API token
        if not api_token or not api_token.strip() or api_token.strip() == "r8_":
            raise ValueError("Please enter your actual Replicate API token from replicate.com/account/api-tokens")
        
        print(f"🚀 Starting Flux Kontext editing with model: {model}")
        print(f"📝 Prompt: {editing_prompt}")
        
        try:
            # Convert images to the format expected by the API
            print("📸 Processing input images...")
            image_1_url = self.tensor_to_base64(image_1, max_size=1024)
            
            # Build input parameters based on official API documentation
            if "flux-kontext-apps" in model and "multi-image" in model:
                # Official flux-kontext-apps multi-image API format
                api_input = {
                    "prompt": editing_prompt,
                    "input_image_1": image_1_url,
                    "aspect_ratio": aspect_ratio
                }
                
                if image_2 is not None:
                    image_2_url = self.tensor_to_base64(image_2, max_size=1024)
                    api_input["input_image_2"] = image_2_url
                    print("✅ Using both images for multi-image editing")
                else:
                    # Use the same image for both inputs if only one provided
                    api_input["input_image_2"] = image_1_url
                    print("ℹ️  Using same image for both inputs (only one image provided)")
                    
            elif "black-forest-labs" in model:
                # Black Forest Labs API format (based on research)
                api_input = {
                    "prompt": editing_prompt,
                    "image": image_1_url,
                }
                print("✅ Using Black Forest Labs format")
                
            else:
                # Fallback format
                api_input = {
                    "prompt": editing_prompt,
                    "image": image_1_url,
                }
                print("ℹ️  Using fallback format")
            
            print(f"🔧 API Input keys: {list(api_input.keys())}")
            
            # Make the API request using Replicate's direct format
            headers = {
                "Authorization": f"Token {api_token.strip()}",
                "Content-Type": "application/json"
            }
            
            # Use Replicate's run API format (simpler than predictions)
            payload = {
                "input": api_input
            }
            
            print("🌐 Making API request to Replicate...")
            response = requests.post(
                f"https://api.replicate.com/v1/models/{model}/predictions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 201:
                error_details = response.text
                print(f"❌ API Error ({response.status_code}): {error_details}")
                raise Exception(f"API request failed ({response.status_code}): {error_details}")
            
            prediction = response.json()
            prediction_id = prediction["id"]
            print(f"✅ Prediction created successfully! ID: {prediction_id}")
            
            # Wait for the prediction to complete
            print("⏳ Waiting for image editing to complete...")
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Check prediction status
                status_response = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {api_token.strip()}"}
                )
                
                if status_response.status_code != 200:
                    raise Exception(f"Failed to check status: {status_response.text}")
                
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "succeeded":
                    output = status_data.get("output")
                    if not output:
                        raise Exception("No output received from API")
                    
                    # Handle output (can be URL or list of URLs)
                    if isinstance(output, list) and len(output) > 0:
                        output_url = output[0]
                    else:
                        output_url = output
                    
                    print(f"✅ Image editing completed! Downloading result...")
                    
                    # Download and convert result
                    img_response = requests.get(output_url)
                    if img_response.status_code != 200:
                        raise Exception(f"Failed to download result: {img_response.status_code}")
                    
                    result_image = Image.open(BytesIO(img_response.content))
                    if result_image.mode != 'RGB':
                        result_image = result_image.convert('RGB')
                    
                    print("🎉 Success! Flux Kontext editing completed.")
                    return (self.pil_to_tensor(result_image),)
                
                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    raise Exception(f"Prediction failed: {error}")
                
                elif status in ["starting", "processing"]:
                    print(f"⏳ Status: {status}")
                    time.sleep(3)
                else:
                    print(f"🔄 Unknown status: {status}, continuing to wait...")
                    time.sleep(3)
            
            raise Exception("Request timed out after 5 minutes")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            
            # Provide helpful error messages
            if "401" in error_msg or "authentication" in error_msg.lower():
                raise ValueError("Invalid API token. Please check your Replicate API token from replicate.com/account/api-tokens")
            elif "404" in error_msg:
                raise ValueError(f"Model not found: {model}. Please check the model name.")
            elif "422" in error_msg or "validation" in error_msg.lower():
                raise ValueError(f"Input validation failed. Check your inputs and try again. Error: {error_msg}")
            else:
                raise ValueError(f"Flux Kontext editing failed: {error_msg}")

# ComfyUI Node Registration
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "FluxKontextNode": FluxKontextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextNode": "Flux Kontext"
} 