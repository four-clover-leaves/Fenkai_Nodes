import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import re
from typing import Optional, Tuple, List
import concurrent.futures

# --- FIX 1: Safer Imports ---
# If api_config.py is missing/renamed, this prevents the entire node from crashing on load.
try:
    from .api_config import load_gemini_api_key
except ImportError:
    try:
        from api_config import load_gemini_api_key
    except ImportError:
        # Graceful fallback: allows ComfyUI to load the node, but it will error at runtime nicely
        load_gemini_api_key = lambda: None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class FenkaiGeminiNode:
    def __init__(self):
        self.api_key = load_gemini_api_key()
        self.client = None
        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)

    @classmethod
    def INPUT_TYPES(cls):
        models = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
        # Added "Auto" option
        ratios = ["Auto", "1:1", "2:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "3:2", "2:3", "21:9"]
        resolutions = ["1K (Standard)", "2K (High)", "4K (Ultra)"]

        return {
            "required": {
                "prompt": ("STRING", {"default": "Enter prompt here...", "multiline": True}),
                "model": (models, {"default": "gemini-3-pro-image-preview"}),
                "ratio": (ratios, {"default": "Auto"}),
                "native_resolution": (resolutions, {"default": "1K (Standard)"}), 
                "batch_mode": ("BOOLEAN", {"default": False}),
                "single_batch_count": ("INT", {"default": 1, "min": 1, "max": 16}),
                "max_parallel": ("INT", {"default": 3, "min": 1, "max": 10}), 
            },
            "optional": {f"image_{i}": ("IMAGE", {}) for i in range(1, 15)}
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompts_log")
    FUNCTION = "generate_image"
    CATEGORY = "Fenkai Nodes Image Generation"

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # --- FIX 2: Batch Dimension Crash Fix ---
        # ComfyUI sends [Batch, Height, Width, Channels].
        # If Batch > 1, squeeze(0) fails or leaves it as 4D, crashing PIL.
        # This fix explicitly grabs the first image [0] from the batch.
        if tensor.ndim == 4:
            tensor = tensor[0] 
        return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        if pil_image.mode != "RGB": pil_image = pil_image.convert("RGB")
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def _create_error_image(self, text: str, size: tuple) -> torch.Tensor:
        # Ensure tuple contains integers (safety cast)
        w, h = int(size[0]), int(size[1])
        img = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((w//3, h//2), text, fill=(255, 0, 0))
        return self._pil_to_tensor(img)

    def _process_single_prompt(self, index, prompt, model_name, ref_contents, ratio, res_str, target_size):
        try:
            local_client = genai.Client(api_key=self.api_key)
            contents = ref_contents + [prompt] if ref_contents else [prompt]
            
            clean_res = res_str.split(" ")[0]
            img_config_args = {"aspect_ratio": ratio}
            if clean_res in ["2K", "4K"]:
                img_config_args["image_size"] = clean_res

            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**img_config_args),
                safety_settings=[types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH")]
            )
            
            response = local_client.models.generate_content(model=model_name, contents=contents, config=config)
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img = Image.open(BytesIO(part.inline_data.data)).resize(target_size, Image.Resampling.LANCZOS)
                        return (index, self._pil_to_tensor(img), "Success")
            return (index, self._create_error_image("NO DATA", target_size), "No Data")
        except Exception as e:
            return (index, self._create_error_image("API ERROR", target_size), str(e))

    def generate_image(self, prompt, model, ratio, native_resolution, batch_mode, single_batch_count, max_parallel, **kwargs):
        if not self.api_key:
            return (self._create_error_image("MISSING KEY", (1024, 1024)), "Setup API_GOOGLE")

        # Capture the raw input for logging later
        original_ratio_input = ratio

        # --- FIX 3: Robust Auto Logic ---
        if ratio == "Auto":
            # Default to 1:1 if no reference image is found
            ratio = "1:1"
            
            if kwargs.get("image_1") is not None:
                img_tensor = kwargs["image_1"]
                # Handle inconsistent tensor shapes safely
                if img_tensor.ndim == 4:
                    _, h, w, _ = img_tensor.shape
                else:
                    h, w, _ = img_tensor.shape
                
                # Zero division protection
                if h > 0:
                    input_ratio = w / h
                    
                    supported_ratios = {
                        "1:1": 1.0, "2:1": 2.0, "16:9": 16/9, "9:16": 9/16,
                        "4:3": 4/3, "3:4": 3/4, "4:5": 4/5, "5:4": 5/4,
                        "3:2": 3/2, "2:3": 2/3, "21:9": 21/9
                    }
                    
                    # Logic to find the closest supporting ratio
                    closest_ratio = "1:1"
                    min_diff = float('inf')
                    
                    for r_str, r_val in supported_ratios.items():
                        diff = abs(input_ratio - r_val)
                        if diff < min_diff:
                            min_diff = diff
                            closest_ratio = r_str
                    
                    ratio = closest_ratio

        # Resolution Math
        mp_multiplier = 1.0
        if "2K" in native_resolution: mp_multiplier = 4.0
        if "4K" in native_resolution: mp_multiplier = 8.0

        # Safety Regex: This now works even if 'Auto' was selected because we overwrote 'ratio' above
        ratio_match = re.match(r"^(\d+):(\d+)$", ratio)
        rw, rh = map(int, ratio_match.groups()) if ratio_match else (1, 1)
        aspect = rw / rh
        
        target_w = max(64, int(round(((mp_multiplier * 1024 * 1024) * aspect)**0.5 / 64) * 64))
        target_h = int(target_w / aspect)
        target_size = (int(target_w), int(target_h))

        # Batch Processing
        prompts = [p.strip() for p in prompt.split('\n') if p.strip()] if batch_mode else [prompt] * single_batch_count
        
        ref_contents = []
        for i in range(1, 15):
            img_key = f"image_{i}"
            if kwargs.get(img_key) is not None:
                ref_contents.append(self._tensor_to_pil(kwargs[img_key]))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(self._process_single_prompt, i, p, model, ref_contents, ratio, native_resolution, target_size) for i, p in enumerate(prompts)]
            for f in concurrent.futures.as_completed(futures): results.append(f.result())

        results.sort(key=lambda x: x[0])
        batch_tensor = torch.cat([r[1] for r in results], dim=0)
        
        # --- FIX 4: Correct Logging ---
        # Checks if the User originally selected "Auto" before we changed it
        log_suffix = f"\n[Auto-Ratio Logic: Input detected, snapped to {ratio}]" if original_ratio_input == "Auto" else ""
        
        return (batch_tensor, "\n".join(prompts) + log_suffix)

NODE_CLASS_MAPPINGS = {"FenkaiGeminiNode": FenkaiGeminiNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FenkaiGeminiNode": "FENKAI Nano Banana Pro"}