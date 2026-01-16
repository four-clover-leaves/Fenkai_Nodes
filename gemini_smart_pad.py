import torch
import math

class GeminiSmartPad:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    # Outputs: The padded image, the ratio string for Gemini, and the new dimensions
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("padded_image", "aspect_ratio_str", "target_width", "target_height", "debug_info")
    FUNCTION = "smart_pad"
    CATEGORY = "Gemini/Utils"

    def smart_pad(self, image):
        # Image shape is [Batch, Height, Width, Channels]
        # We take the first image in the batch to calculate dims
        batch_size, orig_h, orig_w, channels = image.shape
        
        current_ratio = orig_w / orig_h
        
        # 1. Define Gemini Supported Ratios
        # Maps "String Enum" -> Float Value
        ratios = {
            "1:1": 1.0,
            "16:9": 16/9,
            "9:16": 9/16,
            "4:3": 4/3,
            "3:4": 3/4,
            "3:2": 3/2,
            "2:3": 2/3,
            "5:4": 5/4,
            "4:5": 4/5,
            "21:9": 21/9
        }

        # 2. Find the closest supported ratio
        best_ratio_str = "1:1"
        best_ratio_val = 1.0
        min_diff = float('inf')

        for r_str, r_val in ratios.items():
            diff = abs(current_ratio - r_val)
            if diff < min_diff:
                min_diff = diff
                best_ratio_str = r_str
                best_ratio_val = r_val

        # 3. Calculate New Dimensions (Padding Logic)
        # We want to encase the original image fully inside the new ratio
        
        if current_ratio > best_ratio_val:
            # Image is "wider" than target -> Pad Height (Top/Bottom bars)
            # Width stays same, Height increases
            target_w = orig_w
            target_h = int(orig_w / best_ratio_val)
        else:
            # Image is "taller" than target -> Pad Width (Left/Right bars)
            # Height stays same, Width increases
            target_h = orig_h
            target_w = int(orig_h * best_ratio_val)

        # 4. Create New Blank Canvas (Black)
        # Shape: [Batch, New_H, New_W, Channels]
        new_canvas = torch.zeros((batch_size, target_h, target_w, channels), dtype=image.dtype, device=image.device)

        # 5. Paste Original Image into Center
        # Calculate offsets
        y_offset = (target_h - orig_h) // 2
        x_offset = (target_w - orig_w) // 2

        # Insert image (handling slicing for center paste)
        new_canvas[:, y_offset:y_offset+orig_h, x_offset:x_offset+orig_w, :] = image

        # Debug Info
        info = (f"Input: {orig_w}x{orig_h} (Ratio {current_ratio:.2f}) | "
                f"Matched: {best_ratio_str} | "
                f"Output: {target_w}x{target_h}")

        return (new_canvas, best_ratio_str, target_w, target_h, info)

NODE_CLASS_MAPPINGS = {
    "GeminiSmartPad": GeminiSmartPad
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiSmartPad": "Gemini 3 Smart Padder"
}