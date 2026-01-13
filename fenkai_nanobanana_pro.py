import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import configparser
import re
from typing import Optional, Tuple, List
import concurrent.futures

# Attempt to load dotenv for local .env support
try:
    from dotenv import load_dotenv
    load_dotenv() # This looks for a .env file in your ComfyUI root
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class FenkaiGeminiNode:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.client = None
        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)
        
    @staticmethod
    def _load_api_key() -> Optional[str]:
        """
        Priority:
        1. Environment Variable 'API_GOOGLE' (Local .env or ComfyDeploy Cloud)
        2. Local 'googleapi.ini' (Legacy support)
        """
        # --- 1. Check Environment (Best Practice) ---
        # This works for ComfyDeploy and your local .env file
        env_key = os.environ.get("API_GOOGLE")
        if env_key:
            print("--- Nano Banana Pro: Using Cloud/Env Key (API_GOOGLE) ---")
            return env_key

        # --- 2. Check Legacy .ini Files (Fallback) ---
        node_ini = os.path.join(os.path.dirname(__file__), "googleapi.ini")
        if os.path.exists(node_ini):
            cfg = configparser.ConfigParser()
            try:
                cfg.read(node_ini)
                local_key = cfg.get("API_KEY", "key", fallback=None)
                if local_key:
                    print("--- Nano Banana Pro: Using Local .ini Key ---")
                    return local_key
            except Exception:
                pass

        print("--- Nano Banana Pro: WARNING - No API Key Found ---")
        return None

    @classmethod
    def INPUT_TYPES(cls):
        models = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
        ratios = ["1:1", "2:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "3:2", "2:3", "21:9"]
        resolutions = ["1K (Standard)", "2K (High)", "4K (Ultra)"]

        return {
            "required": {
                "prompt": ("STRING", {"default": "Enter your prompt here...", "multiline": True}),
                "model": (models, {"default": "gemini-3-pro-image-preview"}),
                "ratio": (ratios, {"default": "16:9"}),
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
        if tensor.ndim == 4: tensor = tensor.squeeze(0)
        return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        if pil_image.mode != "RGB": pil_image = pil_image.convert("RGB")
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def _create_error_image(self, text: str, detail: str = None) -> torch.Tensor:
        img = Image.new('RGB', (1024, 1024), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((450, 512), text, fill=(255, 0, 0))
        return self._pil_to_tensor(img)

    def _process_single_prompt(self, index, prompt, model_name, ref_contents, ratio, res_str, target_size):
        try:
            local_client = genai.Client(api_key=self.api_key)
            contents = ref_contents + [prompt] if ref_contents else [prompt]
            
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=ratio),
                safety_settings=[types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH")]
            )
            
            response = local_client.models.generate_content(model=model_name, contents=contents, config=config)
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img = Image.open(BytesIO(part.inline_data.data)).resize(target_size, Image.Resampling.LANCZOS)
                        return (index, self._pil_to_tensor(img), "Success")
            return (index, self._create_error_image("NO DATA"), "No Data")
        except Exception as e:
            return (index, self._create_error_image("ERROR", str(e)), str(e))

    def generate_image(self, prompt, model, ratio, native_resolution, batch_mode, single_batch_count, max_parallel, **kwargs):
        if not self.api_key:
            return (self._create_error_image("MISSING KEY"), "Setup API_GOOGLE environment variable")

        prompts = [p.strip() for p in prompt.split('\n') if p.strip()] if batch_mode else [prompt] * single_batch_count
        
        # Collect images
        ref_contents = []
        for i in range(1, 15):
            img = kwargs.get(f"image_{i}")
            if img is not None: ref_contents.append(self._tensor_to_pil(img))

        # Setup sizing
        ratio_match = re.match(r"^(\d+):(\d+)$", ratio)
        rw, rh = map(int, ratio_match.groups()) if ratio_match else (1, 1)
        target_w = max(64, int(round(((1024*1024) * (rw/rh))**0.5 / 64) * 64))
        target_h = int(target_w / (rw/rh))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(self._process_single_prompt, i, p, model, ref_contents, ratio, native_resolution, (target_w, target_h)) for i, p in enumerate(prompts)]
            for f in concurrent.futures.as_completed(futures): results.append(f.result())

        results.sort(key=lambda x: x[0])
        batch_tensor = torch.cat([r[1] for r in results], dim=0)
        return (batch_tensor, "\n".join(prompts))

NODE_CLASS_MAPPINGS = {"FenkaiGeminiNode": FenkaiGeminiNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FenkaiGeminiNode": "FENKAI Nano Banana Pro"}