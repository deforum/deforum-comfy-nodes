import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo
from .tools import VariantSupport
from .base_node import ImageNode, NODE_POSTFIX
import os
import json
import time
import random
import folder_paths
import comfy.utils
from comfy.cli_args import args
from nodes import SaveImage

def tensor_to_pil(tensor):
    """Convert ComfyUI image tensor to PIL Image."""
    # ComfyUI images are (batch, height, width, channels) in range [0,1]
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Convert to numpy and scale to 0-255
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI image tensor."""
    # Convert to numpy and normalize to 0-1
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor

def measure_saturation(img_array):
    """Measure the saturation of an RGB image array."""
    _, _, saturation = rgb_to_hls(img_array)
    return float(saturation.mean())

def rgb_to_hls(img_array):
    rgb = img_array.astype(np.float32)
    if rgb.max(initial=0) > 1.0:
        rgb = rgb / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    maxc = np.max(rgb[:, :, :3], axis=2)
    minc = np.min(rgb[:, :, :3], axis=2)
    lightness = (minc + maxc) * 0.5
    saturation = np.zeros_like(lightness)
    hue = np.zeros_like(lightness)

    non_gray = maxc != minc
    delta = maxc - minc
    low_light = lightness <= 0.5
    saturation[non_gray & low_light] = (
        delta[non_gray & low_light] / (maxc[non_gray & low_light] + minc[non_gray & low_light])
    )
    saturation[non_gray & ~low_light] = (
        delta[non_gray & ~low_light] / (2.0 - maxc[non_gray & ~low_light] - minc[non_gray & ~low_light])
    )

    safe_delta = np.where(non_gray, delta, 1.0)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    red_max = non_gray & (r == maxc)
    green_max = non_gray & (g == maxc)
    blue_max = non_gray & (b == maxc)
    hue[red_max] = bc[red_max] - gc[red_max]
    hue[green_max] = 2.0 + rc[green_max] - bc[green_max]
    hue[blue_max] = 4.0 + gc[blue_max] - rc[blue_max]
    hue = (hue / 6.0) % 1.0

    return hue, lightness, saturation

def _hue_to_rgb(p, q, hue):
    hue = hue % 1.0
    result = np.empty_like(hue)

    first = hue < (1.0 / 6.0)
    second = (hue >= (1.0 / 6.0)) & (hue < 0.5)
    third = (hue >= 0.5) & (hue < (2.0 / 3.0))
    fourth = hue >= (2.0 / 3.0)

    result[first] = p[first] + (q[first] - p[first]) * 6.0 * hue[first]
    result[second] = q[second]
    result[third] = p[third] + (q[third] - p[third]) * ((2.0 / 3.0) - hue[third]) * 6.0
    result[fourth] = p[fourth]
    return result

def hls_to_rgb(hue, lightness, saturation):
    q = np.where(
        lightness < 0.5,
        lightness * (1.0 + saturation),
        lightness + saturation - lightness * saturation,
    )
    p = 2.0 * lightness - q

    r = _hue_to_rgb(p, q, hue + (1.0 / 3.0))
    g = _hue_to_rgb(p, q, hue)
    b = _hue_to_rgb(p, q, hue - (1.0 / 3.0))

    gray = saturation == 0
    r[gray] = lightness[gray]
    g[gray] = lightness[gray]
    b[gray] = lightness[gray]

    return np.stack([r, g, b], axis=2)

def adjust_saturation(img_array, factor):
    hue, lightness, saturation = rgb_to_hls(img_array)
    saturation = np.clip(saturation * factor, 0.0, 1.0)
    rgb = hls_to_rgb(hue, lightness, saturation)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

@VariantSupport()
class ColorCorrection(ImageNode):
    """
    Apply adaptive color correction to images.
    Corrects saturation and optionally applies white balance.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation_threshold": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "max_saturation_factor": ("FLOAT", {
                    "default": 1.3, 
                    "min": 1.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "enable_white_balance": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "white_balance_scale_min": ("FLOAT", {
                    "default": 0.97, 
                    "min": 0.8, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "white_balance_scale_max": ("FLOAT", {
                    "default": 1.03, 
                    "min": 1.0, 
                    "max": 1.2, 
                    "step": 0.01
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_correction"
    
    def apply_correction(self, image, saturation_threshold, max_saturation_factor, 
                        enable_white_balance, white_balance_scale_min=0.97, 
                        white_balance_scale_max=1.03):
        
        # Process each image in the batch
        batch_size = image.shape[0]
        corrected_images = []
        
        for i in range(batch_size):
            # Convert tensor to PIL for processing
            pil_img = tensor_to_pil(image[i:i+1])
            img_array = np.array(pil_img.convert("RGB"))
            
            # Measure saturation
            saturation = measure_saturation(img_array)
            
            if saturation >= saturation_threshold:
                # No correction needed
                corrected_img = pil_img
            else:
                corrected_img_array = img_array.copy()
                
                # Apply white balance correction if enabled
                if enable_white_balance:
                    avg_per_channel = corrected_img_array.reshape(-1, 3).mean(axis=0)
                    avg_intensity = avg_per_channel.mean()
                    scale = avg_intensity / (avg_per_channel + 1e-6)
                    scale = np.clip(scale, white_balance_scale_min, white_balance_scale_max)
                    corrected_img_array = np.clip(corrected_img_array * scale, 0, 255).astype(np.uint8)
                
                # Apply saturation correction
                sat_factor = min(saturation_threshold / (saturation + 1e-6), max_saturation_factor)
                corrected_img_array = adjust_saturation(corrected_img_array, sat_factor)
                
                corrected_img = Image.fromarray(corrected_img_array)
            
            # Convert back to tensor
            corrected_tensor = pil_to_tensor(corrected_img)
            corrected_images.append(corrected_tensor)
        
        # Combine batch
        result_tensor = torch.cat(corrected_images, dim=0)
        return (result_tensor,)

@VariantSupport()
class AdaptiveSharpening(ImageNode):
    """
    Apply adaptive sharpening to improve image details and clarity.
    Uses unsharp masking technique with configurable intensity and radius.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "radius": ("FLOAT", {
                    "default": 1.1, 
                    "min": 0.1, 
                    "max": 3.0, 
                    "step": 0.1
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sharpening"
    
    def apply_sharpening(self, image, intensity, radius):
        # Process each image in the batch
        batch_size = image.shape[0]
        sharpened_images = []
        
        for i in range(batch_size):
            # Convert tensor to PIL for processing
            pil_img = tensor_to_pil(image[i:i+1])
            
            # Apply sharpening using unsharp masking
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            sharpened = Image.blend(pil_img, blurred, alpha=-intensity)
            
            # Convert back to tensor
            sharpened_tensor = pil_to_tensor(sharpened)
            sharpened_images.append(sharpened_tensor)
        
        # Combine batch
        result_tensor = torch.cat(sharpened_images, dim=0)
        return (result_tensor,)

@VariantSupport()
class ColorCorrectionAndSharpening(ImageNode):
    """
    Combined node that applies both color correction and sharpening in one step.
    Useful for batch processing with consistent settings.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_sharpening": ("BOOLEAN", {"default": True}),
                "saturation_threshold": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "max_saturation_factor": ("FLOAT", {
                    "default": 1.3, 
                    "min": 1.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "sharpening_intensity": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "sharpening_radius": ("FLOAT", {
                    "default": 1.1, 
                    "min": 0.1, 
                    "max": 3.0, 
                    "step": 0.1
                }),
            },
            "optional": {
                "enable_white_balance": ("BOOLEAN", {"default": False}),
                "white_balance_scale_min": ("FLOAT", {
                    "default": 0.97, 
                    "min": 0.8, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "white_balance_scale_max": ("FLOAT", {
                    "default": 1.03, 
                    "min": 1.0, 
                    "max": 1.2, 
                    "step": 0.01
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_combined_processing"
    
    def apply_combined_processing(self, image, enable_color_correction, enable_sharpening,
                                 saturation_threshold, max_saturation_factor,
                                 sharpening_intensity, sharpening_radius,
                                 enable_white_balance=False, white_balance_scale_min=0.97,
                                 white_balance_scale_max=1.03):
        
        processed_image = image
        
        # Apply color correction if enabled
        if enable_color_correction:
            color_corrector = ColorCorrection()
            processed_image = color_corrector.apply_correction(
                processed_image, saturation_threshold, max_saturation_factor,
                enable_white_balance, white_balance_scale_min, white_balance_scale_max
            )[0]
        
        # Apply sharpening if enabled
        if enable_sharpening:
            sharpener = AdaptiveSharpening()
            processed_image = sharpener.apply_sharpening(
                processed_image, sharpening_intensity, sharpening_radius
            )[0]
        
        return (processed_image,)

def _match_mean_std(target, reference, eps=1e-6):
    channels = min(target.shape[-1], reference.shape[-1], 3)
    matched = target.clone()
    target_rgb = target[..., :channels]
    reference_rgb = reference[..., :channels]

    target_mean = target_rgb.reshape(-1, channels).mean(dim=0)
    target_std = target_rgb.reshape(-1, channels).std(dim=0, unbiased=False)
    reference_mean = reference_rgb.reshape(-1, channels).mean(dim=0)
    reference_std = reference_rgb.reshape(-1, channels).std(dim=0, unbiased=False)

    matched[..., :channels] = (
        (target_rgb - target_mean)
        * (reference_std / (target_std + eps))
        + reference_mean
    )
    return matched

def _histogram_match_channel(source, reference, bins=256):
    source_hist, source_edges = np.histogram(source, bins=bins, range=(0.0, 1.0))
    reference_hist, reference_edges = np.histogram(reference, bins=bins, range=(0.0, 1.0))

    source_cdf = source_hist.cumsum().astype(np.float64)
    reference_cdf = reference_hist.cumsum().astype(np.float64)
    if source_cdf[-1] == 0 or reference_cdf[-1] == 0:
        return source

    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]
    source_centers = (source_edges[:-1] + source_edges[1:]) * 0.5
    reference_centers = (reference_edges[:-1] + reference_edges[1:]) * 0.5

    source_quantiles = np.interp(source, source_centers, source_cdf)
    return np.interp(source_quantiles, reference_cdf, reference_centers)

def _match_histogram(target, reference):
    channels = min(target.shape[-1], reference.shape[-1], 3)
    target_np = target.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()
    matched_np = target_np.copy()

    for channel in range(channels):
        matched_np[..., channel] = _histogram_match_channel(
            target_np[..., channel].reshape(-1),
            reference_np[..., channel].reshape(-1),
        ).reshape(target_np.shape[:-1])

    return torch.from_numpy(matched_np).to(device=target.device, dtype=target.dtype)

def _luminance(image):
    channels = image.shape[-1]
    if channels < 3:
        return image.mean(dim=-1, keepdim=True)
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype)
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)

def _restore_luminance(target, matched):
    channels = min(target.shape[-1], matched.shape[-1], 3)
    if channels == 0:
        return matched

    restored = matched.clone()
    luminance_delta = _luminance(target) - _luminance(matched)
    restored[..., :channels] = restored[..., :channels] + luminance_delta
    return restored

@VariantSupport()
class ImageColorMatch(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["mean_std", "histogram"],),
                "preserve_luminance": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"

    def match_color(self, images, reference_image, strength, method, preserve_luminance):
        if images.ndim != 4 or reference_image.ndim != 4:
            raise ValueError("Expected IMAGE tensors with shape BHWC")

        strength = max(0.0, min(float(strength), 1.0))
        if strength == 0.0:
            return (images,)

        matched_frames = []
        reference_count = reference_image.shape[0]
        for frame_index in range(images.shape[0]):
            target = images[frame_index]
            reference = reference_image[min(frame_index, reference_count - 1)].to(
                device=target.device,
                dtype=target.dtype,
            )

            if method == "mean_std":
                matched = _match_mean_std(target, reference)
            elif method == "histogram":
                matched = _match_histogram(target, reference)
            else:
                raise ValueError(f"Unknown color match method: {method}")

            if preserve_luminance:
                matched = _restore_luminance(target, matched)

            matched = torch.lerp(target, matched, strength).clamp(0.0, 1.0)
            matched_frames.append(matched)

        return (torch.stack(matched_frames, dim=0),)

def _sequence_length(value):
    if torch.is_tensor(value):
        return int(value.numel()) if value.ndim > 0 else 1
    if isinstance(value, np.ndarray):
        return int(value.size) if value.ndim > 0 else 1
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1

def _value_at_frame(value, frame_index, default=0.0):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.item())
        flat = value.detach().flatten()
        return float(flat[min(frame_index, flat.numel() - 1)].item())
    if isinstance(value, np.ndarray):
        flat = value.flatten()
        return float(flat[min(frame_index, flat.size - 1)])
    if isinstance(value, (list, tuple)):
        return float(value[min(frame_index, len(value) - 1)])
    return float(value)

def _transform_matrix(transform_type, x_value, y_value, width, height, device, dtype):
    cx = width / 2.0
    cy = height / 2.0
    matrix = torch.eye(3, device=device, dtype=dtype)

    if transform_type == "translate":
        matrix[0, 2] = x_value
        matrix[1, 2] = y_value
    elif transform_type == "rotate":
        angle = torch.tensor(np.deg2rad(x_value), device=device, dtype=dtype)
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        matrix[0, 0] = alpha
        matrix[0, 1] = beta
        matrix[0, 2] = (1.0 - alpha) * cx - beta * cy
        matrix[1, 0] = -beta
        matrix[1, 1] = alpha
        matrix[1, 2] = beta * cx + (1.0 - alpha) * cy
    elif transform_type == "scale":
        scale_x = 1.0 + x_value
        scale_y = 1.0 + y_value
        matrix[0, 0] = scale_x
        matrix[0, 2] = cx * (1.0 - scale_x)
        matrix[1, 1] = scale_y
        matrix[1, 2] = cy * (1.0 - scale_y)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    return matrix

def _normalize_pixel_coords(coords, size):
    if size <= 1:
        return torch.zeros_like(coords)
    return (coords / (size - 1)) * 2.0 - 1.0

@VariantSupport()
class ImageTransform(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transform_type": (["translate", "rotate", "scale"],),
                "x_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "y_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "edge_mode": (["extend", "wrap", "reflect", "none"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_transform"

    def apply_transform(
        self,
        images,
        transform_type,
        x_value,
        y_value,
        edge_mode,
    ):
        if images.ndim != 4:
            raise ValueError("Expected IMAGE tensor with shape BHWC")

        batch_size, height, width, _ = images.shape
        num_frames = max(
            batch_size,
            _sequence_length(x_value),
            _sequence_length(y_value),
        )

        device = images.device
        dtype = images.dtype
        frame_indices = torch.arange(num_frames, device=device) % batch_size
        source_images = images[frame_indices].permute(0, 3, 1, 2)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        ones = torch.ones_like(x_coords)
        output_coords = torch.stack([x_coords, y_coords, ones], dim=0).reshape(3, -1)

        inverse_matrices = []
        for frame_index in range(num_frames):
            frame_x = _value_at_frame(x_value, frame_index, 0.0)
            frame_y = _value_at_frame(y_value, frame_index, 0.0)
            matrix = _transform_matrix(transform_type, frame_x, frame_y, width, height, device, dtype)
            inverse_matrices.append(torch.linalg.inv(matrix))

        inverse = torch.stack(inverse_matrices, dim=0)
        coords = torch.matmul(inverse, output_coords.unsqueeze(0).expand(num_frames, -1, -1))
        x_source = coords[:, 0, :]
        y_source = coords[:, 1, :]

        if edge_mode == "wrap":
            x_source = torch.remainder(x_source, width)
            y_source = torch.remainder(y_source, height)

        grid = torch.stack(
            [
                _normalize_pixel_coords(x_source, width),
                _normalize_pixel_coords(y_source, height),
            ],
            dim=-1,
        ).reshape(num_frames, height, width, 2)

        padding_modes = {
            "extend": "border",
            "wrap": "zeros",
            "reflect": "reflection",
            "none": "zeros",
        }
        transformed = F.grid_sample(
            source_images,
            grid,
            mode="bilinear",
            padding_mode=padding_modes[edge_mode],
            align_corners=True,
        )

        return (transformed.permute(0, 2, 3, 1).clamp(0.0, 1.0),)

    
class SmartLoopPreviewImage(ImageNode, SaveImage):
    """
    Advanced version that maintains a sliding window of recent iterations
    to prevent unlimited accumulation while still showing iteration progress
    """
    
    # Class variable to track iteration counts per node
    _iteration_counters = {}
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.compress_level = 1
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "smart_preview_images"
    OUTPUT_NODE = True

    def smart_preview_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, unique_id=None):
        
        # Track iteration count for this node instance
        if unique_id not in self._iteration_counters:
            self._iteration_counters[unique_id] = 0
        
        iteration_num = self._iteration_counters[unique_id]
        self._iteration_counters[unique_id] += 1
        
        # Generate filename with iteration number
        timestamp = int(time.time() * 1000)
        prefix_append = f"_iter{iteration_num:03d}_{timestamp}"
        filename_prefix += prefix_append

        # Save images for this iteration
        results = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Add iteration info to metadata if enabled
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                        
                metadata.add_text("loop_iteration", str(iteration_num))

            filename_with_batch_num = filename_prefix.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{batch_number:05}_.png"
            img.save(os.path.join(self.output_dir, file), pnginfo=metadata, compress_level=self.compress_level)
            
            results.append({
                "filename": file,
                "subfolder": "",
                "type": self.type,
                "iteration": iteration_num  # Custom field for frontend use
            })

        # Clean up old files if we exceed 1
        self._cleanup_old_previews(unique_id, 1)

        # Determine merge strategy
        merge_results = iteration_num > 0  # Merge after first iteration
        
        return {
            "ui": {
                "images": results
            },
            "result": (images,),
            "merge": merge_results,
        }
    
    def _cleanup_old_previews(self, unique_id, max_previews):
        """Clean up old preview files to prevent disk space issues"""
        current_iteration = self._iteration_counters[unique_id] - 1
        if current_iteration >= max_previews:
            # Remove files from iterations that are too old
            iterations_to_remove = range(current_iteration - max_previews, current_iteration - max_previews + 1)
            for old_iter in iterations_to_remove:
                if old_iter >= 0:
                    # Find and remove files from this iteration
                    pattern = f"*_iter{old_iter:03d}_*"
                    import glob
                    old_files = glob.glob(os.path.join(self.output_dir, pattern))
                    for old_file in old_files:
                        try:
                            os.remove(old_file)
                        except OSError:
                            pass  # File already removed or doesn't exist

# -----------------------------------------------------------------------------
#  Registration tables
# -----------------------------------------------------------------------------
IMAGE_NODE_CLASS_MAPPINGS = {
    f"ColorCorrection{NODE_POSTFIX}": ColorCorrection,
    f"AdaptiveSharpening{NODE_POSTFIX}": AdaptiveSharpening,
    f"ColorCorrectionAndSharpening{NODE_POSTFIX}": ColorCorrectionAndSharpening,
    f"ImageColorMatch{NODE_POSTFIX}": ImageColorMatch,
    f"ImageTransform{NODE_POSTFIX}": ImageTransform,
    f"SmartLoopPreviewImage{NODE_POSTFIX}": SmartLoopPreviewImage
}

IMAGE_NODE_DISPLAY_NAME_MAPPINGS = {
    f"ColorCorrection{NODE_POSTFIX}": f"Color Correction {NODE_POSTFIX}",
    f"AdaptiveSharpening{NODE_POSTFIX}": f"Adaptive Sharpening {NODE_POSTFIX}",
    f"ColorCorrectionAndSharpening{NODE_POSTFIX}": f"Color Correction + Sharpening {NODE_POSTFIX}",
    f"ImageColorMatch{NODE_POSTFIX}": f"Image Color Match {NODE_POSTFIX}",
    f"ImageTransform{NODE_POSTFIX}": f"Image Transform {NODE_POSTFIX}",
    f"SmartLoopPreviewImage{NODE_POSTFIX}": f"Smart Loop Preview Image {NODE_POSTFIX}"
} 
