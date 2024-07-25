import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

class DoubutsuDescriber:
    def __init__(self):
        self.model_id = "qresearch/doubutsu-2b-pt-756"
        self.model = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": "Describe the image"}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 512}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "precision": (["float16", "bfloat16"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "image/text"

    def describe_image(self, image, question, max_new_tokens, temperature, precision):
        if self.model is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", self.model_id)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model files not found. Please download the model to {model_path}")
            
            dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
            )
            adapter_path = os.path.join(os.path.dirname(__file__), "..", "models", "qresearch/doubutsu-2b-lora-756-docci")
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter files not found. Please download the adapter to {adapter_path}")
            self.model.load_adapter(adapter_path)

        # Convert the image tensor to a numpy array
        image_np = image.squeeze().cpu().numpy()
        
        # Check if the image is in the format (H, W, C) or (C, H, W)
        if image_np.shape[0] == 3 and len(image_np.shape) == 3:
            # If it's in (C, H, W) format, transpose to (H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Ensure the image is in RGB format (3 channels)
        if image_np.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {image_np.shape}. Expected 3 channels (RGB).")
        
        # Convert to uint8 if it's not already
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if precision == "bfloat16" else torch.float16):
            result = self.model.answer_question(
                pil_image, question, self.tokenizer, max_new_tokens=max_new_tokens, temperature=temperature
            )
        
        return (result,)