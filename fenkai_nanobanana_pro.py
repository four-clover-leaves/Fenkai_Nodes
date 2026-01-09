import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import configparser
import re
from typing import Optional, Tuple, List
import concurrent.futures

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
        node_ini = os.path.join(os.path.dirname(__file__), "googleapi.ini")
        external_default = r"D:\\AI\\ComfyUINEW\\googleapi.ini"

        def _read_key_from_ini(ini_path: str) -> Optional[str]:
            cfg = configparser.ConfigParser()
            try:
                cfg.read(ini_path)
                return cfg.get("API_KEY", "key", fallback=None)
            except Exception:
                return None

        if os.path.exists(node_ini):
            cfg = configparser.ConfigParser()
            try:
                cfg.read(node_ini)
                if cfg.has_section("API_PATH"):
                    pointer_path = cfg.get("API_PATH", "path", fallback=None)
                    if pointer_path:
                        pointer_path = os.path.expandvars(pointer_path)
                        if os.path.exists(pointer_path):
                            key_val = _read_key_from_ini(pointer_path)
                            if key_val: return key_val
                key_val = cfg.get("API_KEY", "key", fallback=None)
                if key_val: return key_val
            except Exception:
                pass

        if os.path.exists(external_default):
            key_val = _read_key_from_ini(external_default)
            if key_val: return key_val

        return None

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "gemini-2.5-flash-image",
            "gemini-3-pro-image-preview",
        ]

        ratios = ["1:1", "2:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "3:2", "2:3", "21:9"]
        resolutions = ["1K (Standard)", "2K (High)", "4K (Ultra)"]

        inputs = {
            "required": {
                "prompt": ("STRING", {"default": "Enter your prompt here...", "multiline": True}),
                "model": (models, {"default": "gemini-3-pro-image-preview"}),
                "ratio": (ratios, {"default": "16:9"}),
                "native_resolution": (resolutions, {"default": "1K (Standard)"}), 
                "batch_mode": ("BOOLEAN", {"default": False, "label_on": "Enabled (Split Lines)", "label_off": "Disabled (Single Prompt)"}),
                "single_batch_count": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "display": "number"}),
                "max_parallel": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1, "display": "slider"}), 
            },
            "optional": {
                "image_batch": ("IMAGE", {}), 
            }
        }
        
        for i in range(1, 15):
            inputs["optional"][f"image_{i}"] = ("IMAGE", {})

        return inputs

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompts_log")
    FUNCTION = "generate_image"
    CATEGORY = "Fenkai Nodes Image Generation"

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def _create_error_image(self, text: str = "PROMPT DECLINED", detail: str = None, width=1024, height=1024) -> torch.Tensor:
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((width//2 - 50, height//2), text, fill=(255, 0, 0))
        if detail:
            draw.text((width//2 - 100, height//2 + 20), detail[:50], fill=(255, 255, 255))
        return self._pil_to_tensor(img)

    def _try_generate(self, client, model_name, contents, config):
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

    def _process_single_prompt(self, index, prompt, model_name, ref_image_contents, ratio_string, resolution_str, target_size):
        try:
            local_client = genai.Client(api_key=self.api_key)
            
            contents = []
            if ref_image_contents:
                contents.extend(ref_image_contents)
            
            contents.append(prompt)
            clean_res = resolution_str.split(" ")[0] 

            safety_settings = [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH")
            ]

            def build_config(size_str=None):
                img_config_args = {"aspect_ratio": ratio_string}
                if size_str:
                    img_config_args["image_size"] = size_str

                return types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(**img_config_args),
                    safety_settings=safety_settings
                )
            
            config_high = build_config(clean_res if clean_res in ["2K", "4K"] else None)
            response = None
            used_fallback = False

            try:
                response = self._try_generate(local_client, model_name, contents, config_high)
            except Exception as e_high:
                print(f"⚠️ Prompt {index}: Native {clean_res} request failed ({str(e_high)}). Falling back to Standard.")
                used_fallback = True

            if used_fallback or not response:
                config_std = build_config(None)
                response = self._try_generate(local_client, model_name, contents, config_std)

            generated_image_bytes = None
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        generated_image_bytes = part.inline_data.data
                        break
            
            if generated_image_bytes:
                generated_pil = Image.open(BytesIO(generated_image_bytes))
                
                # REVISED: High-quality resize to target_size (which is now mathematically ratio-locked)
                # This ensures batch tensor consistency without stretching or cropping.
                if generated_pil.size != target_size:
                    generated_pil = generated_pil.resize(target_size, Image.Resampling.LANCZOS)
                
                return (index, self._pil_to_tensor(generated_pil), "Success")
            
            return (index, self._create_error_image("NO IMAGE", "API returned no data"), "No Data")

        except Exception as e:
            print(f"Error on prompt {index}: {str(e)}")
            return (index, self._create_error_image("ERROR", str(e)), str(e))

    def generate_image(
        self,
        prompt: str,
        model: str,
        ratio: str,
        native_resolution: str, 
        batch_mode: bool = False,
        single_batch_count: int = 1,
        max_parallel: int = 3,
        **kwargs 
    ) -> Tuple[torch.Tensor, str]:

        if not self.api_key or not genai:
            return (self._create_error_image("SETUP ERROR", "Run: pip install google-genai"), "Missing Key or Lib")

        # 1. Prepare Prompts based on Mode
        if batch_mode:
            prompts_list = [p.strip() for p in prompt.split('\n') if p.strip()]
            if not prompts_list:
                prompts_list = ["Empty Prompt"]
        else:
            prompts_list = [prompt] * max(1, single_batch_count)

        # 2. Collect All Images (Batch + 14 Slots)
        ref_contents = []
        if "image_batch" in kwargs and kwargs["image_batch"] is not None:
            batch_tensor = kwargs["image_batch"]
            for i in range(batch_tensor.shape[0]):
                pil_img = self._tensor_to_pil(batch_tensor[i].unsqueeze(0))
                ref_contents.append(pil_img)

        for i in range(1, 15):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                pil_img = self._tensor_to_pil(kwargs[key])
                ref_contents.append(pil_img)
        
        if len(ref_contents) > 14:
            ref_contents = ref_contents[:14]

        # 3. Determine Target Dimensions (FIXED RATIO LOGIC)
        clean_res = native_resolution.split(" ")[0]
        mp_val = 1
        if "2K" in clean_res: mp_val = 2
        if "4K" in clean_res: mp_val = 8 

        ratio_match = re.match(r"^(\d+):(\d+)$", ratio.strip())
        rw, rh = map(int, ratio_match.groups()) if ratio_match else (1, 1)
        
        base_area = 1024 * 1024
        target_area = mp_val * base_area
        aspect = rw / rh
        width_f = (target_area * aspect) ** 0.5
        
        # ROUNDING FIX: Only round the width to a multiple of 64.
        # Derive height directly from aspect to lock the ratio 100%.
        def _round_to_64(x): return max(64, int(round(x / 64) * 64))
        target_w = _round_to_64(width_f)
        target_h = int(target_w / aspect)
        target_size = (target_w, target_h)

        # 4. PARALLEL EXECUTION
        results = []
        print(f"--- Starting: {len(prompts_list)} jobs, Workers: {max_parallel} ---")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_index = {
                executor.submit(
                    self._process_single_prompt, 
                    i, p, model, ref_contents, ratio, native_resolution, target_size
                ): i 
                for i, p in enumerate(prompts_list)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                res = future.result()
                results.append(res)

        results.sort(key=lambda x: x[0])
        final_tensors = [r[1] for r in results]
        
        if not final_tensors:
            return (self._create_error_image("FATAL", "No results"), "")

        # Normalize batch sizes (all should already be target_size)
        max_h = max(t.shape[1] for t in final_tensors)
        max_w = max(t.shape[2] for t in final_tensors)
        
        normalized_tensors = []
        for t in final_tensors:
            if t.shape[1] != max_h or t.shape[2] != max_w:
                t = torch.nn.functional.interpolate(t.permute(0,3,1,2), size=(max_h, max_w), mode='bilinear').permute(0,2,3,1)
            normalized_tensors.append(t)

        batch_tensor = torch.cat(normalized_tensors, dim=0)
        
        return (batch_tensor, "\n".join(prompts_list))

NODE_CLASS_MAPPINGS = {
    "FenkaiGeminiNode": FenkaiGeminiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FenkaiGeminiNode": "FENKAI Nano Banana Pro",
}