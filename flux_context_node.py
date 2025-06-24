import requests
import time
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

class FluxContextNode:
    """
    ComfyUI node for Flux Context (Kontext) - image editing and transformation using text prompts with context images
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
                "input_image": ("IMAGE",),
                "editing_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Make this a watercolor painting",
                    "placeholder": "Describe how you want to edit the image"
                }),
                "model": (["black-forest-labs/flux-kontext-pro", "black-forest-labs/flux-kontext-max"], {
                    "default": "black-forest-labs/flux-kontext-pro"
                }),
            },
            "optional": {
                "context_image": ("IMAGE",),
                "context_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "preserve_identity": ("BOOLEAN", {
                    "default": True
                }),
                "output_format": (["webp", "jpg", "png"], {
                    "default": "webp"
                }),
                "output_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "image/editing"
    
    def pil_to_tensor(self, image):
        """Convert PIL image to tensor format expected by ComfyUI"""
        image_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None,]
    
    def tensor_to_base64(self, tensor):
        """Convert ComfyUI tensor to base64 string for API upload"""
        # Handle tensor dimensions
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert tensor to PIL Image
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_prediction(self, input_data):
        """Create a prediction using Replicate API"""
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/predictions",
            headers=headers,
            json=input_data
        )
        
        if response.status_code != 201:
            raise Exception(f"Failed to create prediction: {response.text}")
        
        return response.json()
    
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
    
    def enhance_prompt(self, prompt, preserve_identity, context_image):
        """Enhance the editing prompt with context-specific instructions"""
        enhanced_prompt = prompt
        
        if preserve_identity and "person" in prompt.lower() or "face" in prompt.lower():
            enhanced_prompt += " while keeping the same facial features and identity"
        
        if context_image is not None:
            enhanced_prompt += " using the second image as visual reference for style and composition"
        
        return enhanced_prompt
    
    def edit_image(self, api_token, input_image, editing_prompt, model, context_image=None, 
                   context_strength=0.8, preserve_identity=True, output_format="webp", output_quality=90):
        """Main image editing function using Flux Context"""
        
        # Validate API token
        if not api_token or not api_token.strip():
            raise ValueError("API token is required. Get yours from replicate.com/account/api-tokens")
        
        # Store the API token for this generation
        self.api_token = api_token.strip()
        
        # Convert input image to base64
        input_image_b64 = self.tensor_to_base64(input_image)
        
        # Enhance prompt for better context understanding
        enhanced_prompt = self.enhance_prompt(editing_prompt, preserve_identity, context_image)
        
        # Get model version
        version = self.get_model_version(model)
        
        # Prepare input data for Flux Context
        input_data = {
            "version": version,
            "input": {
                "prompt": enhanced_prompt,
                "input_image": input_image_b64,
                "output_format": output_format,
                "output_quality": output_quality,
            }
        }
        
        # Add context image if provided
        if context_image is not None:
            context_image_b64 = self.tensor_to_base64(context_image)
            input_data["input"]["context_image"] = context_image_b64
            input_data["input"]["context_strength"] = context_strength
        
        print(f"Starting Flux Context editing with model: {model}")
        print(f"Editing prompt: {enhanced_prompt}")
        
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
    
    def get_model_version(self, model):
        """Get the latest version hash for the specified Flux Context model"""
        # These version hashes may need updates - check Replicate model pages
        model_versions = {
            "black-forest-labs/flux-kontext-pro": "latest",
            "black-forest-labs/flux-kontext-max": "latest",
        }
        
        return model_versions.get(model, "latest") 