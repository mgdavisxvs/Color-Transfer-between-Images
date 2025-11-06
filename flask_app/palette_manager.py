#!/usr/bin/env python3
"""
RAL Palette Manager

Handles loading, caching, and searching the RAL color palette with
Lab color space conversion and Delta E matching.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from color_utils import rgb_to_lab, find_closest_color, delta_e_ciede2000


class RALPalette:
    """
    RAL Color Palette Manager

    Loads RAL colors from JSON, maintains RGB and Lab representations,
    and provides efficient color matching capabilities.
    """

    def __init__(self, palette_file: str = 'data/ral.json'):
        """
        Initialize RAL palette.

        Args:
            palette_file: Path to ral.json file
        """
        self.palette_file = Path(palette_file)
        self.colors = []  # List of color dictionaries
        self.rgb_array = None  # NumPy array of RGB values
        self.lab_array = None  # NumPy array of Lab values
        self.code_to_index = {}  # Map RAL code to index
        self.loaded = False

    def load(self) -> None:
        """
        Load RAL palette from JSON file and convert to Lab.

        Raises:
            FileNotFoundError: If palette file doesn't exist
            ValueError: If palette file is invalid
        """
        if not self.palette_file.exists():
            raise FileNotFoundError(f"Palette file not found: {self.palette_file}")

        try:
            with open(self.palette_file, 'r') as f:
                data = json.load(f)

            self.colors = data.get('colors', [])

            if not self.colors:
                raise ValueError("Palette file contains no colors")

            # Extract RGB values
            rgb_list = [c['rgb'] for c in self.colors]
            self.rgb_array = np.array(rgb_list, dtype=np.uint8)

            # Convert all colors to Lab for efficient matching
            self.lab_array = np.array([
                rgb_to_lab(np.array(rgb, dtype=np.uint8))
                for rgb in rgb_list
            ], dtype=np.float32)

            # Create code lookup
            self.code_to_index = {
                c['code']: idx for idx, c in enumerate(self.colors)
            }

            self.loaded = True

            print(f"✓ Loaded {len(self.colors)} RAL colors")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in palette file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load palette: {e}")

    def get_color_by_code(self, code: str) -> Optional[Dict]:
        """
        Get color information by RAL code.

        Args:
            code: RAL code (e.g., "RAL 1000")

        Returns:
            Color dictionary or None if not found
        """
        if not self.loaded:
            self.load()

        idx = self.code_to_index.get(code)
        if idx is not None:
            return self.colors[idx]
        return None

    def get_color_by_index(self, index: int) -> Optional[Dict]:
        """
        Get color information by index.

        Args:
            index: Color index

        Returns:
            Color dictionary or None if out of bounds
        """
        if not self.loaded:
            self.load()

        if 0 <= index < len(self.colors):
            return self.colors[index]
        return None

    def find_closest_match(self, rgb: np.ndarray,
                          method: str = 'cie2000',
                          top_n: int = 1) -> List[Tuple[Dict, float]]:
        """
        Find closest matching RAL color(s) for given RGB color.

        Args:
            rgb: RGB color values (3,) or (H, W, 3)
            method: Delta E calculation method
            top_n: Number of top matches to return

        Returns:
            List of (color_dict, delta_e) tuples sorted by Delta E
        """
        if not self.loaded:
            self.load()

        # Convert input to Lab
        target_lab = rgb_to_lab(np.array(rgb, dtype=np.uint8))

        # If image, use mean color
        if len(target_lab.shape) > 1:
            target_lab = np.mean(target_lab, axis=(0, 1))

        # Calculate Delta E for all palette colors
        delta_es = []
        for i, pal_lab in enumerate(self.lab_array):
            if method == 'cie2000':
                de = delta_e_ciede2000(target_lab, pal_lab)
            else:
                from color_utils import delta_e_cie76, delta_e_cie94
                if method == 'cie76':
                    de = delta_e_cie76(target_lab, pal_lab)
                elif method == 'cie94':
                    de = delta_e_cie94(target_lab, pal_lab)
                else:
                    de = delta_e_ciede2000(target_lab, pal_lab)

            delta_es.append((i, float(de)))

        # Sort by Delta E
        delta_es.sort(key=lambda x: x[1])

        # Return top N matches
        results = []
        for idx, de in delta_es[:top_n]:
            color_info = self.colors[idx].copy()
            color_info['lab'] = self.lab_array[idx].tolist()
            results.append((color_info, de))

        return results

    def search_by_name(self, query: str) -> List[Dict]:
        """
        Search RAL colors by name (case-insensitive).

        Args:
            query: Search query string

        Returns:
            List of matching color dictionaries
        """
        if not self.loaded:
            self.load()

        query_lower = query.lower()
        results = []

        for color in self.colors:
            if query_lower in color['name'].lower():
                results.append(color)

        return results

    def get_palette_rgb(self) -> np.ndarray:
        """
        Get all palette colors as RGB array.

        Returns:
            NumPy array of shape (N, 3) with RGB values
        """
        if not self.loaded:
            self.load()

        return self.rgb_array.copy()

    def get_palette_lab(self) -> np.ndarray:
        """
        Get all palette colors as Lab array.

        Returns:
            NumPy array of shape (N, 3) with Lab values
        """
        if not self.loaded:
            self.load()

        return self.lab_array.copy()

    def get_all_colors(self) -> List[Dict]:
        """
        Get all color information.

        Returns:
            List of all color dictionaries
        """
        if not self.loaded:
            self.load()

        return [c.copy() for c in self.colors]

    def get_color_statistics(self) -> Dict:
        """
        Get statistical information about the palette.

        Returns:
            Dictionary with palette statistics
        """
        if not self.loaded:
            self.load()

        return {
            'total_colors': len(self.colors),
            'rgb_range': {
                'min': self.rgb_array.min(axis=0).tolist(),
                'max': self.rgb_array.max(axis=0).tolist(),
                'mean': self.rgb_array.mean(axis=0).tolist()
            },
            'lab_range': {
                'L': [float(self.lab_array[:, 0].min()), float(self.lab_array[:, 0].max())],
                'a': [float(self.lab_array[:, 1].min()), float(self.lab_array[:, 1].max())],
                'b': [float(self.lab_array[:, 2].min()), float(self.lab_array[:, 2].max())]
            }
        }


# Global palette instance (singleton)
_palette_instance = None


def get_palette() -> RALPalette:
    """
    Get global RAL palette instance (singleton pattern).

    Returns:
        RALPalette instance
    """
    global _palette_instance

    if _palette_instance is None:
        _palette_instance = RALPalette()
        _palette_instance.load()

    return _palette_instance
