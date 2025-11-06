#!/usr/bin/env python3
"""
Color Utilities Module

Provides color space conversion (RGB ↔ Lab) and Delta E (ΔE) calculations
for precise color matching and transfer operations.
"""

import numpy as np
import cv2
from typing import Tuple, Union
from skimage import color as skcolor


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color space to CIE Lab color space.

    Lab color space is perceptually uniform and ideal for color difference
    calculations (Delta E).

    Args:
        rgb: RGB image or color values (uint8 or float32)
             Shape: (H, W, 3) for images or (3,) for single colors

    Returns:
        Lab color values (float32)
        L: Lightness [0, 100]
        a: Green-Red axis [-128, 127]
        b: Blue-Yellow axis [-128, 127]
    """
    # Ensure input is numpy array
    rgb = np.asarray(rgb, dtype=np.float32)

    # If single color (1D array), add dimensions
    is_single_color = len(rgb.shape) == 1
    if is_single_color:
        rgb = rgb.reshape(1, 1, 3)

    # Normalize to [0, 1] if in [0, 255]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    # Convert RGB to Lab using scikit-image (more accurate than OpenCV)
    lab = skcolor.rgb2lab(rgb)

    # Return to original shape if single color
    if is_single_color:
        lab = lab.reshape(3)

    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE Lab color space to RGB color space.

    Args:
        lab: Lab color values (float32)
             Shape: (H, W, 3) for images or (3,) for single colors

    Returns:
        RGB values (uint8) in range [0, 255]
    """
    # Ensure input is numpy array
    lab = np.asarray(lab, dtype=np.float32)

    # If single color (1D array), add dimensions
    is_single_color = len(lab.shape) == 1
    if is_single_color:
        lab = lab.reshape(1, 1, 3)

    # Convert Lab to RGB using scikit-image
    rgb = skcolor.lab2rgb(lab)

    # Convert to uint8 [0, 255]
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # Return to original shape if single color
    if is_single_color:
        rgb = rgb.reshape(3)

    return rgb


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate Delta E (CIE 1976) color difference.

    ΔE = √((L₂-L₁)² + (a₂-a₁)² + (b₂-b₁)²)

    Interpretation:
    - ΔE < 1: Imperceptible difference
    - 1 < ΔE < 2: Perceptible through close observation
    - 2 < ΔE < 10: Perceptible at a glance
    - 11 < ΔE < 49: Colors are more similar than opposite
    - ΔE >= 50: Colors are completely different

    Args:
        lab1: First Lab color(s)
        lab2: Second Lab color(s)

    Returns:
        Delta E value(s) as float or array
    """
    lab1 = np.asarray(lab1, dtype=np.float32)
    lab2 = np.asarray(lab2, dtype=np.float32)

    # Calculate Euclidean distance in Lab color space
    delta = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))

    return delta


def delta_e_cie94(lab1: np.ndarray, lab2: np.ndarray,
                  kL: float = 1.0, K1: float = 0.045, K2: float = 0.015) -> Union[float, np.ndarray]:
    """
    Calculate Delta E (CIE 1994) color difference.

    More sophisticated than CIE76, accounts for perceptual non-uniformities.

    Args:
        lab1: First Lab color(s)
        lab2: Second Lab color(s)
        kL: Lightness weight factor (default 1.0)
        K1: Chroma weight factor (default 0.045)
        K2: Hue weight factor (default 0.015)

    Returns:
        Delta E value(s)
    """
    lab1 = np.asarray(lab1, dtype=np.float32)
    lab2 = np.asarray(lab2, dtype=np.float32)

    # Extract channels
    if len(lab1.shape) == 1:
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2
    else:
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Calculate differences
    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2

    # Calculate Chroma
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2

    # Calculate dH (hue difference)
    dH_squared = da**2 + db**2 - dC**2
    dH_squared = np.maximum(dH_squared, 0)  # Avoid negative due to rounding
    dH = np.sqrt(dH_squared)

    # Weight factors
    SL = 1.0
    SC = 1.0 + K1 * C1
    SH = 1.0 + K2 * C1

    # Calculate Delta E 94
    delta_e = np.sqrt(
        (dL / (kL * SL))**2 +
        (dC / SC)**2 +
        (dH / SH)**2
    )

    return delta_e


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate Delta E (CIEDE2000) color difference.

    Most accurate perceptual color difference formula.
    Industry standard for critical color matching applications.

    Args:
        lab1: First Lab color(s)
        lab2: Second Lab color(s)

    Returns:
        Delta E value(s)
    """
    from skimage.color import deltaE_ciede2000

    lab1 = np.asarray(lab1, dtype=np.float32)
    lab2 = np.asarray(lab2, dtype=np.float32)

    # Ensure proper shape for scikit-image
    if len(lab1.shape) == 1:
        lab1 = lab1.reshape(1, 1, 3)
        lab2 = lab2.reshape(1, 1, 3)
        result = deltaE_ciede2000(lab1, lab2)[0, 0]
    else:
        result = deltaE_ciede2000(lab1, lab2)

    return result


def find_closest_color(target_lab: np.ndarray,
                       palette_lab: np.ndarray,
                       method: str = 'cie2000') -> Tuple[int, float]:
    """
    Find the closest color in a palette to a target color.

    Args:
        target_lab: Target color in Lab space (shape: (3,))
        palette_lab: Palette colors in Lab space (shape: (N, 3))
        method: Delta E calculation method ('cie76', 'cie94', 'cie2000')

    Returns:
        Tuple of (index, delta_e) for the closest match
    """
    target_lab = np.asarray(target_lab, dtype=np.float32)
    palette_lab = np.asarray(palette_lab, dtype=np.float32)

    # Calculate Delta E for all palette colors
    if method == 'cie76':
        delta_es = np.array([delta_e_cie76(target_lab, pal) for pal in palette_lab])
    elif method == 'cie94':
        delta_es = np.array([delta_e_cie94(target_lab, pal) for pal in palette_lab])
    elif method == 'cie2000':
        delta_es = np.array([delta_e_ciede2000(target_lab, pal) for pal in palette_lab])
    else:
        raise ValueError(f"Unknown Delta E method: {method}")

    # Find minimum
    closest_idx = np.argmin(delta_es)
    min_delta_e = delta_es[closest_idx]

    return int(closest_idx), float(min_delta_e)


def get_mean_and_std_lab(image_lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and standard deviation for each Lab channel.

    Args:
        image_lab: Image in Lab color space (H, W, 3)

    Returns:
        Tuple of (mean, std) arrays with 3 values each
    """
    mean = np.mean(image_lab, axis=(0, 1))
    std = np.std(image_lab, axis=(0, 1))

    return mean.astype(np.float32), std.astype(np.float32)


def create_delta_e_heatmap(image: np.ndarray,
                           reference_color: np.ndarray,
                           method: str = 'cie2000') -> np.ndarray:
    """
    Create a Delta E heatmap showing color difference from reference.

    Args:
        image: Input image (RGB or Lab)
        reference_color: Reference color (RGB or Lab, shape: (3,))
        method: Delta E calculation method

    Returns:
        Heatmap array (H, W) with Delta E values
    """
    # Convert to Lab if needed
    if image.max() > 100:  # Assume RGB if values > 100
        image_lab = rgb_to_lab(image)
    else:
        image_lab = image

    if reference_color.max() > 100:
        ref_lab = rgb_to_lab(reference_color)
    else:
        ref_lab = reference_color

    # Calculate Delta E for each pixel
    height, width = image_lab.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    if method == 'cie76':
        heatmap = delta_e_cie76(image_lab, ref_lab)
    elif method == 'cie94':
        heatmap = delta_e_cie94(image_lab, ref_lab)
    elif method == 'cie2000':
        # For CIEDE2000, calculate per-pixel
        for i in range(height):
            for j in range(width):
                heatmap[i, j] = delta_e_ciede2000(image_lab[i, j], ref_lab)

    return heatmap


def downsample_image(image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
    """
    Downsample image for faster processing while maintaining aspect ratio.

    Args:
        image: Input image
        max_dimension: Maximum width or height

    Returns:
        Downsampled image
    """
    height, width = image.shape[:2]

    if height <= max_dimension and width <= max_dimension:
        return image

    # Calculate new dimensions
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))

    # Resize using high-quality interpolation
    downsampled = cv2.resize(image, (new_width, new_height),
                            interpolation=cv2.INTER_AREA)

    return downsampled


def interpret_delta_e(delta_e: float) -> str:
    """
    Provide human-readable interpretation of Delta E value.

    Args:
        delta_e: Delta E value

    Returns:
        Interpretation string
    """
    if delta_e < 1.0:
        return "Imperceptible difference"
    elif delta_e < 2.0:
        return "Perceptible through close observation"
    elif delta_e < 3.5:
        return "Noticeable difference"
    elif delta_e < 5.0:
        return "Clear difference"
    elif delta_e < 10.0:
        return "Significant difference"
    elif delta_e < 50.0:
        return "Very different colors"
    else:
        return "Completely different colors"
