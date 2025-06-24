import requests
import time
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

class FluxReplicateNode:
    """
    ComfyUI node for generating images using Flux models via Replicate API
    """
    
    def __init__(self):
        self.base_url = "https://api.replicate.com/v1"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "r8_"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape with mountains and a lake"
                }),
                "model": (["black-forest-labs/flux-schnell", "black-forest-labs/flux-dev", "black-forest-labs/flux-pro"], {
                    "default": "black-forest-labs/flux-schnell"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
                "num_inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
            },
            "optional": {
                "image_input_1": ("IMAGE",),
                "image_input_2": ("IMAGE",),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "output_format": (["webp", "jpg", "png"], {
                    "default": "webp"
                }),
                "output_quality": ("INT", {
                    "default": 80,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "image_mode": (["img2img", "reference", "blend"], {
                    "default": "img2img"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    
    def pil_to_tensor(self, image):
        """Convert PIL image to tensor format expected by ComfyUI"""
        image_array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None,]
    
    def tensor_to_base64(self, tensor):
        """Convert ComfyUI tensor to base64 string for API upload"""
        # Convert tensor to PIL Image
        i = 255. * tensor.cpu().numpy().squeeze()
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
                print(f"Prediction status: {status}")
                time.sleep(2)
            else:
                print(f"Unknown status: {status}")
                time.sleep(2)
        
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
    
    def generate(self, api_token, prompt, model, width, height, num_inference_steps, guidance_scale, seed, 
                image_input_1=None, image_input_2=None, negative_prompt="", output_format="webp", 
                output_quality=80, image_mode="img2img"):
        """Main generation function"""
        
        # Validate API token
        if not api_token or not api_token.strip():
            raise ValueError("API token is required. Please enter your Replicate API token.")
        
        # Store the API token for this generation
        self.api_token = api_token.strip()
        
        # Prepare input data
        input_data = {
            "version": self.get_model_version(model),
            "input": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "output_format": output_format,
                "output_quality": output_quality,
            }
        }
        
        # Add guidance scale for models that support it
        if model != "black-forest-labs/flux-schnell":
            input_data["input"]["guidance_scale"] = guidance_scale
        
        # Add seed if specified
        if seed != -1:
            input_data["input"]["seed"] = seed
        
        # Add negative prompt if provided
        if negative_prompt.strip():
            input_data["input"]["negative_prompt"] = negative_prompt
        
        # Handle image inputs based on mode
        if image_input_1 is not None:
            image1_b64 = self.tensor_to_base64(image_input_1)
            
            if image_mode == "img2img":
                # Use first image as base for img2img
                input_data["input"]["image"] = image1_b64
                input_data["input"]["prompt_strength"] = 0.8
                
            elif image_mode == "reference":
                # Use first image as style reference
                input_data["input"]["style_reference"] = image1_b64
                
            elif image_mode == "blend" and image_input_2 is not None:
                # Blend two images (use first as base, second as overlay)
                input_data["input"]["image"] = image1_b64
                image2_b64 = self.tensor_to_base64(image_input_2)
                input_data["input"]["overlay_image"] = image2_b64
                input_data["input"]["blend_strength"] = 0.5
            else:
                # Default to img2img mode
                input_data["input"]["image"] = image1_b64
                input_data["input"]["prompt_strength"] = 0.8
        
        print(f"Creating prediction with model: {model}")
        print(f"Prompt: {prompt}")
        
        # Create prediction
        prediction = self.create_prediction(input_data)
        prediction_id = prediction["id"]
        
        print(f"Prediction created with ID: {prediction_id}")
        
        # Wait for completion
        result = self.wait_for_prediction(prediction_id)
        
        # Get output URL
        output_urls = result.get("output", [])
        if not output_urls:
            raise Exception("No output generated")
        
        # Download and return the generated image
        image_url = output_urls[0] if isinstance(output_urls, list) else output_urls
        print(f"Downloading image from: {image_url}")
        
        return (self.download_image(image_url),)
    
    def get_model_version(self, model):
        """Get the latest version hash for the specified model"""
        # These are the current version hashes as of the latest update
        # You may need to update these periodically
        model_versions = {
            "black-forest-labs/flux-schnell": "f2ab8a5569bb018f84ed58c6f64e7c17ff52d7ffe90a8b95f67a266bd54c987f",
            "black-forest-labs/flux-dev": "d7e6d19786dafebe13ff7ba6ad17bdbb8644c96a3e28b4d1fe5eef649b8b7899",
            "black-forest-labs/flux-pro": "fdf5b5f32094f18d55388c3c0d2c01de456073b2c9b1a5b7f6af5b3e12e3b97e"
        }
        
        return model_versions.get(model, model_versions["black-forest-labs/flux-schnell"]) 