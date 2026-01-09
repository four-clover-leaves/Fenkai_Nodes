import os
import time
import torch
import numpy as np
from PIL import Image

# --- Soft Import Strategy ---
try:
    from google import genai
    from google.genai import types
    GOOGLE_LIB_INSTALLED = True
except ImportError:
    GOOGLE_LIB_INSTALLED = False

class FenkaiGeminiTextNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Analyze this image and reasoning process.",
                    "dynamicPrompts": True
                }),
                "model_name": (
                    [
                        # --- GEMINI 3 (LATEST) ---
                        "gemini-3-pro-preview",       # Flagship Reasoning (Uses thinking_level)
                        "gemini-3-pro-image-preview", # Specialized for Image Inputs
                        
                        # --- GEMINI EXPERIMENTAL ---
                        "gemini-exp-1206",            # Dec 2024 High-Perf Experimental
                        "gemini-2.0-flash-exp",       # Experimental Flash
                        
                        # --- GEMINI 2.5 (STABLE) ---
                        "gemini-2.5-pro",             # High Intelligence (Uses thinking_budget)
                        "gemini-2.5-flash",           # Fast & Versatile
                        "gemini-2.5-flash-lite",      # Cheapest / Fastest
                        "gemini-2.5-flash-image",     # Optimized for Vision Tasks
                        
                        # --- LEGACY ---
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ],
                    {"default": "gemini-3-pro-preview"} 
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                
                # --- DELAY TIMER ---
                "delay_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "Wait this many seconds before calling API (prevents rate limits)"}),
                
                "custom_model_id": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "Override (e.g., gemini-4-ultra)"
                }),
                "system_instruction": ("STRING", {
                    "multiline": True, 
                    "default": "You are a helpful AI assistant."
                }),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "Leave empty if using ENV: GOOGLE_API_KEY"
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                
                # --- NEW: SEED PARAMETER ---
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Sets a static seed for more deterministic outputs (Gemini 2.5+)"}),
                
                # --- REASONING SETTINGS ---
                "thinking_mode": ("BOOLEAN", {"default": False, "label_on": "Enable Reasoning", "label_off": "Standard Speed"}),
                "thinking_level": (["high", "low"], {"default": "high", "tooltip": "For Gemini 3 Only"}),
                "thinking_budget": ("INT", {"default": 1024, "min": 1024, "max": 32000, "step": 1024, "tooltip": "For Gemini 2.5 Only"}),
                
                "max_retries": ("INT", {"default": 3, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "generate_content"
    CATEGORY = "FENKAI/Text Generation"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # We return the seed value to ensure ComfyUI re-runs when the seed changes
        return kwargs.get("seed", float("nan"))

    def generate_content(self, prompt, model_name, api_key, image=None, custom_model_id="", system_instruction=None, temperature=0.7, seed=0, thinking_mode=False, thinking_level="high", thinking_budget=1024, max_retries=3, delay_seconds=0.0):
        
        if not GOOGLE_LIB_INSTALLED:
            return ("Error: 'google-genai' library not found. Run install script.",)

        # 1. API Key Check
        final_api_key = api_key.strip()
        if not final_api_key:
            final_api_key = os.getenv("GOOGLE_API_KEY")
        if not final_api_key:
            return ("Error: No API Key provided.",)

        # 2. Model Selection
        final_model = custom_model_id.strip() if custom_model_id.strip() else model_name

        # 3. Client Init
        try:
            client = genai.Client(api_key=final_api_key)
        except Exception as e:
            return (f"Client Error: {str(e)}",)

        # 4. Image Processing
        contents = [prompt]
        if image is not None:
            try:
                tensor = image[0]
                i = 255. * tensor.cpu().numpy()
                img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                contents.append(img_pil)
            except Exception as e:
                return (f"Image Conversion Error: {str(e)}",)

        # 5. Config (Thinking Logic: Gemini 3 vs 2.5)
        thinking_config = None
        if thinking_mode:
            if "gemini-3" in final_model:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=thinking_level
                )
            else:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=thinking_budget
                )

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
        ]

        # Apply Configuration including the new Seed
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,  # Static seed passed to API
            thinking_config=thinking_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction
        )

        # 6. Safety Delay (Pre-Execution)
        if delay_seconds > 0:
            print(f"Waiting {delay_seconds}s before API call...")
            time.sleep(delay_seconds)

        # 7. Generation Loop (with Retries)
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                print(f"Generating with {final_model} (Seed: {seed}) (Attempt {attempt+1})...")
                
                response = client.models.generate_content(
                    model=final_model,
                    contents=contents,
                    config=config
                )
                
                if response.text:
                    return (response.text,)
                else:
                    return ("Empty response (Model might have refused).",)

            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                if "503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg:
                    if attempt < max_retries:
                        wait_time = 2 * (attempt + 1)
                        print(f"API Busy. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                return (f"API Failed after {attempt+1} attempts. Last Error: {last_error}",)
        
        return (f"Failed after {max_retries} retries. Error: {last_error}",)

NODE_CLASS_MAPPINGS = {
    "FenkaiGeminiTextNode": FenkaiGeminiTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FenkaiGeminiTextNode": "FENKAI Gemini 3 Pro (Text/Reasoning)",
}