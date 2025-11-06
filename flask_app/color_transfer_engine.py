#!/usr/bin/env python3
"""
Color Transfer Engine

Advanced color transfer using Lab color space and RAL palette matching
with quality control and Delta E reporting.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from color_utils import (
    rgb_to_lab, lab_to_rgb, delta_e_ciede2000,
    get_mean_and_std_lab, downsample_image
)
from palette_manager import get_palette


class ColorTransferEngine:
    """
    Sophisticated color transfer engine with RAL palette integration.

    Supports:
    - Statistical color transfer (Reinhard method)
    - RAL palette-based color mapping
    - Single or multiple color matching
    - Delta E quality reporting
    """

    def __init__(self, downsample_max: int = 2048):
        """
        Initialize color transfer engine.

        Args:
            downsample_max: Maximum dimension for downsampling (performance optimization)
        """
        self.downsample_max = downsample_max
        self.palette = get_palette()

    def transfer_reinhard(self,
                         source_rgb: np.ndarray,
                         target_rgb: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Reinhard color transfer: transfer statistical color properties.

        Uses mean and variance in Lab color space to adjust source image
        to match target image's color characteristics.

        Args:
            source_rgb: Source image (RGB, uint8)
            target_rgb: Target image or color (RGB, uint8)

        Returns:
            Tuple of (result_rgb, statistics)
        """
        # Convert to Lab
        source_lab = rgb_to_lab(source_rgb)
        target_lab = rgb_to_lab(target_rgb)

        # Calculate statistics
        source_mean, source_std = get_mean_and_std_lab(source_lab)
        target_mean, target_std = get_mean_and_std_lab(target_lab)

        # Perform transfer: ((x - μₛ) * (σₜ / σₛ)) + μₜ
        result_lab = source_lab.copy().astype(np.float32)

        for channel in range(3):
            if source_std[channel] > 1e-6:  # Avoid division by zero
                result_lab[:, :, channel] = (
                    (source_lab[:, :, channel] - source_mean[channel]) *
                    (target_std[channel] / source_std[channel]) +
                    target_mean[channel]
                )

        # Convert back to RGB
        result_rgb = lab_to_rgb(result_lab)

        # Statistics for reporting
        stats = {
            'source_mean_lab': source_mean.tolist(),
            'source_std_lab': source_std.tolist(),
            'target_mean_lab': target_mean.tolist(),
            'target_std_lab': target_std.tolist(),
            'result_mean_lab': get_mean_and_std_lab(result_lab)[0].tolist()
        }

        return result_rgb, stats

    def transfer_to_ral_color(self,
                              source_rgb: np.ndarray,
                              ral_code: str,
                              method: str = 'reinhard') -> Tuple[np.ndarray, Dict]:
        """
        Transfer source image colors to match a specific RAL color.

        Args:
            source_rgb: Source image (RGB, uint8)
            ral_code: RAL color code (e.g., "RAL 3000")
            method: Transfer method ('reinhard' or 'direct')

        Returns:
            Tuple of (result_rgb, info)
        """
        # Get RAL color
        ral_color = self.palette.get_color_by_code(ral_code)
        if not ral_color:
            raise ValueError(f"RAL color not found: {ral_code}")

        target_rgb = np.array(ral_color['rgb'], dtype=np.uint8)

        if method == 'reinhard':
            # Statistical transfer
            result_rgb, stats = self.transfer_reinhard(source_rgb, target_rgb)

            info = {
                'ral_code': ral_code,
                'ral_name': ral_color['name'],
                'ral_rgb': ral_color['rgb'],
                'method': 'reinhard',
                'statistics': stats
            }

        elif method == 'direct':
            # Direct color replacement (maintain lightness, replace chrominance)
            source_lab = rgb_to_lab(source_rgb)
            target_lab = rgb_to_lab(target_rgb)

            result_lab = source_lab.copy()
            # Keep L channel, replace a and b
            result_lab[:, :, 1] = target_lab[1]  # a
            result_lab[:, :, 2] = target_lab[2]  # b

            result_rgb = lab_to_rgb(result_lab)

            info = {
                'ral_code': ral_code,
                'ral_name': ral_color['name'],
                'ral_rgb': ral_color['rgb'],
                'method': 'direct'
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        return result_rgb, info

    def transfer_palette_mapping(self,
                                 source_rgb: np.ndarray,
                                 color_mapping: Dict[str, str],
                                 delta_e_method: str = 'cie2000') -> Tuple[np.ndarray, Dict]:
        """
        Transfer colors using palette-to-palette mapping.

        Maps source colors to specific RAL colors based on user-defined mapping.

        Args:
            source_rgb: Source image (RGB, uint8)
            color_mapping: Dict mapping source RAL codes to target RAL codes
                          e.g., {"RAL 3000": "RAL 5000"}
            delta_e_method: Delta E calculation method

        Returns:
            Tuple of (result_rgb, mapping_info)
        """
        # Convert source to Lab
        source_lab = rgb_to_lab(source_rgb)
        result_rgb = source_rgb.copy()

        mapping_stats = []

        for source_code, target_code in color_mapping.items():
            source_color = self.palette.get_color_by_code(source_code)
            target_color = self.palette.get_color_by_code(target_code)

            if not source_color or not target_color:
                continue

            source_ral_rgb = np.array(source_color['rgb'], dtype=np.uint8)
            target_ral_rgb = np.array(target_color['rgb'], dtype=np.uint8)

            source_ral_lab = rgb_to_lab(source_ral_rgb)
            target_ral_lab = rgb_to_lab(target_ral_rgb)

            # Find pixels close to source color
            pixel_delta_e = np.zeros((source_lab.shape[0], source_lab.shape[1]))
            for i in range(source_lab.shape[0]):
                for j in range(source_lab.shape[1]):
                    pixel_delta_e[i, j] = delta_e_ciede2000(
                        source_lab[i, j], source_ral_lab
                    )

            # Threshold: pixels within Delta E < 10 are considered matching
            mask = pixel_delta_e < 10.0
            pixel_count = np.sum(mask)

            if pixel_count > 0:
                # Replace matched pixels with target color
                result_rgb[mask] = target_ral_rgb

                mapping_stats.append({
                    'source_code': source_code,
                    'target_code': target_code,
                    'pixels_affected': int(pixel_count),
                    'percentage': float(pixel_count / (source_rgb.shape[0] * source_rgb.shape[1]) * 100)
                })

        info = {
            'method': 'palette_mapping',
            'mappings': mapping_stats,
            'total_mappings': len(color_mapping)
        }

        return result_rgb, info

    def auto_match_to_palette(self,
                              source_rgb: np.ndarray,
                              num_colors: int = 5,
                              method: str = 'kmeans') -> Tuple[np.ndarray, Dict]:
        """
        Automatically match source image colors to closest RAL colors.

        Args:
            source_rgb: Source image (RGB, uint8)
            num_colors: Number of dominant colors to extract
            method: Color quantization method ('kmeans')

        Returns:
            Tuple of (result_rgb, match_info)
        """
        # Extract dominant colors using K-means
        pixels = source_rgb.reshape(-1, 3).astype(np.float32)

        # Use K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        centers = centers.astype(np.uint8)

        # Find closest RAL matches for each cluster center
        color_matches = []
        for i, center_rgb in enumerate(centers):
            matches = self.palette.find_closest_match(center_rgb, top_n=1)
            if matches:
                ral_color, delta_e = matches[0]
                color_matches.append({
                    'cluster_id': i,
                    'source_rgb': center_rgb.tolist(),
                    'ral_match': ral_color,
                    'delta_e': delta_e
                })

        # Create result image with RAL colors
        result_pixels = pixels.copy()
        for i, match in enumerate(color_matches):
            mask = (labels.flatten() == i)
            result_pixels[mask] = match['ral_match']['rgb']

        result_rgb = result_pixels.reshape(source_rgb.shape).astype(np.uint8)

        info = {
            'method': 'auto_match',
            'num_colors': num_colors,
            'matches': color_matches
        }

        return result_rgb, info


class QualityControl:
    """
    Quality Control system for color transfer validation.

    Provides Delta E statistics, thresholds, and acceptance criteria.
    """

    def __init__(self,
                 delta_e_threshold: float = 5.0,
                 acceptance_percentage: float = 95.0):
        """
        Initialize QC system.

        Args:
            delta_e_threshold: Maximum acceptable Delta E value
            acceptance_percentage: Minimum percentage of pixels that must pass
        """
        self.delta_e_threshold = delta_e_threshold
        self.acceptance_percentage = acceptance_percentage

    def evaluate(self,
                 source_rgb: np.ndarray,
                 result_rgb: np.ndarray,
                 target_rgb: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate color transfer quality.

        Args:
            source_rgb: Source image
            result_rgb: Result image after transfer
            target_rgb: Target image (optional, for comparison)

        Returns:
            Quality report dictionary
        """
        # Convert to Lab
        source_lab = rgb_to_lab(source_rgb)
        result_lab = rgb_to_lab(result_rgb)

        # Calculate pixel-wise Delta E between source and result
        height, width = source_rgb.shape[:2]
        delta_e_map = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                delta_e_map[i, j] = delta_e_ciede2000(source_lab[i, j], result_lab[i, j])

        # Statistics
        delta_e_stats = {
            'mean': float(np.mean(delta_e_map)),
            'std': float(np.std(delta_e_map)),
            'min': float(np.min(delta_e_map)),
            'max': float(np.max(delta_e_map)),
            'median': float(np.median(delta_e_map)),
            'percentile_95': float(np.percentile(delta_e_map, 95))
        }

        # Threshold evaluation
        pixels_within_threshold = np.sum(delta_e_map < self.delta_e_threshold)
        total_pixels = height * width
        acceptance_rate = (pixels_within_threshold / total_pixels) * 100

        passed = acceptance_rate >= self.acceptance_percentage

        report = {
            'delta_e_statistics': delta_e_stats,
            'threshold': self.delta_e_threshold,
            'pixels_within_threshold': int(pixels_within_threshold),
            'total_pixels': int(total_pixels),
            'acceptance_rate': float(acceptance_rate),
            'required_acceptance': self.acceptance_percentage,
            'passed': passed,
            'delta_e_map': delta_e_map  # For heatmap visualization
        }

        # If target provided, also compare result to target
        if target_rgb is not None:
            target_lab = rgb_to_lab(target_rgb)
            target_mean, _ = get_mean_and_std_lab(target_lab)
            result_mean, _ = get_mean_and_std_lab(result_lab)

            target_delta_e = delta_e_ciede2000(result_mean, target_mean)

            report['target_comparison'] = {
                'mean_delta_e': float(target_delta_e),
                'result_mean_lab': result_mean.tolist(),
                'target_mean_lab': target_mean.tolist()
            }

        return report

    def generate_csv_report(self, report: Dict, output_path: str) -> None:
        """
        Generate CSV report from QC evaluation.

        Args:
            report: QC report dictionary
            output_path: Path to save CSV file
        """
        import pandas as pd

        data = {
            'Metric': [],
            'Value': []
        }

        # Add statistics
        for key, value in report['delta_e_statistics'].items():
            data['Metric'].append(f'Delta E {key}')
            data['Value'].append(value)

        # Add threshold info
        data['Metric'].extend([
            'Threshold',
            'Pixels Within Threshold',
            'Total Pixels',
            'Acceptance Rate (%)',
            'Required Acceptance (%)',
            'Passed QC'
        ])
        data['Value'].extend([
            report['threshold'],
            report['pixels_within_threshold'],
            report['total_pixels'],
            report['acceptance_rate'],
            report['required_acceptance'],
            'Yes' if report['passed'] else 'No'
        ])

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
