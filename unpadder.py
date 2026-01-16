import torch

class GeminiUnpadder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "orig_width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 1}),
                "orig_height": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("cropped_image", "debug_info")
    FUNCTION = "unpad_image"
    CATEGORY = "Fenkai"

    def unpad_image(self, image, orig_width, orig_height):
        # 1. Get dimensions of the CURRENT (possibly upscaled) image
        # Shape is [Batch, Height, Width, Channels]
        batch, curr_h, curr_w, channels = image.shape
        
        # 2. Calculate Ratios
        target_ratio = orig_width / orig_height
        current_ratio = curr_w / curr_h
        
        # 3. Calculate Crop Coordinates based on RATIO (not fixed pixels)
        # We want to find the largest box that fits in the current image 
        # while matching the 'target_ratio'.
        
        if current_ratio > target_ratio:
            # Current image is "Wider" than target -> Black bars are on Left/Right
            # Strategy: Keep full Height, calculate new Width to match target ratio
            new_h = curr_h
            new_w = int(curr_h * target_ratio)
            
            # Calculate margins to slice off
            margin = (curr_w - new_w) // 2
            
            # Crop Width (Left/Right), Keep Height
            # Syntax: [:, Top:Bottom, Left:Right, :]
            cropped_image = image[:, :, margin : margin + new_w, :]
            
            action = "Cropped Left/Right (Width)"

        else:
            # Current image is "Taller" than target -> Black bars are on Top/Bottom
            # Strategy: Keep full Width, calculate new Height to match target ratio
            new_w = curr_w
            new_h = int(curr_w / target_ratio)
            
            # Calculate margins
            margin = (curr_h - new_h) // 2
            
            # Crop Height (Top/Bottom), Keep Width
            cropped_image = image[:, margin : margin + new_h, :, :]
            
            action = "Cropped Top/Bottom (Height)"

        # 4. Debug Info
        info = (f"Input: {curr_w}x{curr_h} | Target Ratio: {target_ratio:.2f} | "
                f"Action: {action} | Result: {new_w}x{new_h}")

        return (cropped_image, info)

# Mappings
NODE_CLASS_MAPPINGS = {
    "GeminiUnpadder": GeminiUnpadder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiUnpadder": "Gemini 3 Unpadder (Restore Original)"
}