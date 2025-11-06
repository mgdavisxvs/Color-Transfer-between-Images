#!/usr/bin/env python3
"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2


@pytest.fixture(scope='session')
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def create_test_image():
    """Factory fixture to create test images."""
    def _create_image(width=100, height=100, color_type='gradient'):
        """
        Create a test image.

        Args:
            width: Image width
            height: Image height
            color_type: 'gradient', 'solid', 'random'

        Returns:
            NumPy array (H, W, 3) uint8
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)

        if color_type == 'gradient':
            for i in range(height):
                img[i, :, 0] = int(50 + i * (200 / height))   # B
                img[i, :, 1] = int(100 + i * (150 / height))  # G
                img[i, :, 2] = int(150 + i * (100 / height))  # R

        elif color_type == 'solid':
            img[:, :] = [128, 128, 128]

        elif color_type == 'random':
            img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        return img

    return _create_image


@pytest.fixture
def save_test_image(temp_dir):
    """Factory fixture to save test images."""
    def _save_image(image, filename='test.png'):
        """
        Save image to temp directory.

        Args:
            image: NumPy array
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = temp_dir / filename
        cv2.imwrite(str(filepath), image)
        return filepath

    return _save_image
