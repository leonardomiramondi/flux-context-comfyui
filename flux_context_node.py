import requests
import time
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import json

class FluxContextNode:
    """
    ComfyUI node for Flux Context (Kontext) - image editing and transformation using text prompts
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
                "image 1": ("IMAGE",),
                "editing_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Make this a watercolor painting",
                    "placeholder": "Describe how you want to edit the image"
                }),
                "model": ([
                    "black-forest-labs/flux-kontext-pro",
                    "black-forest-labs/flux-kontext-max"
                ], {
                    "default": "black-forest-labs/flux-kontext-pro"
                }),
                "output_format": (["jpg", "png"], {
                    "default": "jpg"
                }),
            },
            "optional": {
                "image 2": ("IMAGE",),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs before processing"""
        return True
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "image/editing"
    
    def pil_to_tensor(self, image):
        """Convert PIL image to tensor format expected by ComfyUI"""
        image_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None,]
    
    def tensor_to_base64(self, tensor, max_size=1280):
        """Convert ComfyUI tensor to base64 string for API upload with high quality preservation"""
        # Handle tensor dimensions
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert tensor to PIL Image
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        original_size = (img.width, img.height)
        print(f"Original image size: {original_size[0]}x{original_size[1]}")
        
        # Only resize if the image is significantly larger than max_size
        if img.width > max_size or img.height > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_size / img.width, max_size / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            
            # Use highest quality resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image to {new_width}x{new_height} for API compatibility")
        
        # Save as PNG for maximum quality first, then check size
        buffered = BytesIO()
        img.save(buffered, format="PNG", optimize=True, compress_level=1)  # Low compression for quality
        png_size = len(buffered.getvalue())
        
        # Estimate base64 size (roughly 4/3 of original)
        estimated_b64_size_mb = png_size * 4/3 / (1024 * 1024)
        
        # If PNG is too large (>20MB), use high-quality JPEG
        if estimated_b64_size_mb > 20:
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=98, optimize=True, progressive=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"Using JPEG at 98% quality, estimated size: {estimated_b64_size_mb:.1f}MB")
            return f"data:image/jpeg;base64,{img_str}"
        else:
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"Using PNG for maximum quality, estimated size: {estimated_b64_size_mb:.1f}MB")
            return f"data:image/png;base64,{img_str}"
    
    def create_prediction(self, input_data):
        """Create a prediction using Replicate API with better error handling"""
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }
        
        try:
            # Check JSON size before sending
            json_str = json.dumps(input_data)
            json_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            print(f"Request payload size: {json_size_mb:.2f} MB")
            
            if json_size_mb > 50:  # If larger than 50MB, try with smaller images
                print("Payload too large, retrying with smaller images...")
                raise Exception("Payload too large - will retry with compressed images")
            
            response = requests.post(
                f"{self.base_url}/predictions",
                headers=headers,
                json=input_data,
                timeout=30  # Add timeout
            )
            
            if response.status_code != 201:
                raise Exception(f"Failed to create prediction: {response.text}")
            
            return response.json()
            
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing error: {str(e)}. Try with smaller images.")
        except requests.exceptions.Timeout:
            raise Exception("Request timeout. Try with smaller images or check your connection.")
        except Exception as e:
            if "Payload too large" in str(e):
                raise e  # Re-raise for retry logic
            raise Exception(f"API request failed: {str(e)}")
    
    def get_prediction(self, prediction_id):
        """Get prediction status and result"""
        headers = {
            "Authorization": f"Token {self.api_token}",
        }
        
        response = requests.get(
            f"{self.base_url}/predictions/{prediction_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get prediction: {response.text}")
        
        return response.json()
    
    def wait_for_prediction(self, prediction_id, timeout=300):
        """Wait for prediction to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            prediction = self.get_prediction(prediction_id)
            status = prediction.get("status")
            
            if status == "succeeded":
                return prediction
            elif status == "failed":
                raise Exception(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
            elif status in ["starting", "processing"]:
                print(f"Flux Context editing in progress: {status}")
                time.sleep(3)
            else:
                print(f"Unknown status: {status}")
                time.sleep(3)
        
        raise Exception("Prediction timed out")
    
    def download_image(self, url):
        """Download image from URL and convert to tensor"""
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.status_code}")
        
        image = Image.open(BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return self.pil_to_tensor(image)
    
    def get_model_version(self, model_name):
        """Get the latest version ID for a Replicate model"""
        headers = {
            "Authorization": f"Token {self.api_token}",
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/models/{model_name}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                model_info = response.json()
                latest_version = model_info.get("latest_version", {}).get("id")
                if latest_version:
                    print(f"Found latest version for {model_name}: {latest_version}")
                    return latest_version
                else:
                    raise Exception(f"No latest version found for {model_name}")
            else:
                raise Exception(f"Failed to get model info for {model_name}: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Error getting latest version for {model_name}: {str(e)}")

    def edit_image(self, api_token, **kwargs):
        """Main image editing function using Flux Context"""
        
        # Extract parameters from kwargs
        image_1 = kwargs.get("image 1")
        image_2 = kwargs.get("image 2")
        editing_prompt = kwargs.get("editing_prompt")
        model = kwargs.get("model", "black-forest-labs/flux-kontext-pro")
        output_format = kwargs.get("output_format", "jpg")
        
        # Validate API token
        if not api_token or not api_token.strip():
            raise ValueError("API token is required. Get yours from replicate.com/account/api-tokens")
        
        # Store the API token for this generation
        self.api_token = api_token.strip()

        # Try with high quality settings first
        max_sizes = [1280, 1024, 768]  # Start with higher quality
        
        for max_size in max_sizes:
            try:
                print(f"Attempting with max image size: {max_size}px")
                
                # Get the model version ID
                version_id = self.get_model_version(model)
                
                # Convert primary image to base64 with high quality
                image_1_b64 = self.tensor_to_base64(image_1, max_size)
                
                # Prepare input data using version instead of model
                input_data = {
                    "version": version_id,
                    "input": {
                        "prompt": editing_prompt,
                        "output_format": output_format,
                    }
                }
                
                # Add images - single image or array of images
                if image_2 is not None:
                    # Two images - pass as array
                    image_2_b64 = self.tensor_to_base64(image_2, max_size)
                    input_data["input"]["image_url"] = [image_1_b64, image_2_b64]
                    print("Added both images as array to 'image_url'")
                else:
                    # Single image
                    input_data["input"]["image_url"] = image_1_b64
                    print("Added single image as 'image_url'")
                
                print(f"Starting Flux Context editing with model: {model}")
                print(f"Using version: {version_id}")
                print(f"Editing prompt: {editing_prompt}")
                print(f"Output format: {output_format}")
                
                # Create prediction
                prediction = self.create_prediction(input_data)
                prediction_id = prediction["id"]
                
                print(f"Prediction created with ID: {prediction_id}")
                
                # Wait for completion
                result = self.wait_for_prediction(prediction_id)
                
                # Get output URL
                output_url = result.get("output")
                if not output_url:
                    raise Exception("No output generated")
                
                # Handle both single URL and list of URLs
                if isinstance(output_url, list):
                    output_url = output_url[0]
                
                print(f"Downloading edited image from: {output_url}")
                
                return (self.download_image(output_url),)
                
            except Exception as e:
                error_msg = str(e).lower()
                if ("payload too large" in error_msg or "json parsing" in error_msg or "request entity too large" in error_msg) and max_size > 768:
                    print(f"Retrying with smaller image size due to: {str(e)}")
                    continue  # Try next smaller size
                else:
                    # If it's not a size issue or we're already at minimum size, raise the error
                    raise e
        
        # If all sizes failed
        raise Exception("Failed to process images even with smallest size. Please try with lower resolution images or contact support.") 