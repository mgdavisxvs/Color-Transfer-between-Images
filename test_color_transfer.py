#!/usr/bin/env python3
"""
Unit tests for color_transfer.py

Run with: python -m pytest test_color_transfer.py -v
Or simply: python test_color_transfer.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

# Import functions to test
from color_transfer import (
    validate_filename,
    validate_image,
    read_image,
    get_mean_and_std,
    transfer_color_vectorized,
    ImageValidationError,
    ColorTransferError,
    MIN_STD_THRESHOLD,
    MAX_IMAGE_SIZE
)


class TestValidateFilename(unittest.TestCase):
    """Test filename validation security checks."""

    def test_valid_filename(self):
        """Valid filenames should not raise exceptions."""
        validate_filename('image.bmp')
        validate_filename('s1.bmp')
        validate_filename('test_image_123.jpg')

    def test_path_traversal_dotdot(self):
        """Should reject filenames with '..' (path traversal)."""
        with self.assertRaises(ImageValidationError):
            validate_filename('../../../etc/passwd')

    def test_path_traversal_slash(self):
        """Should reject filenames with '/' (path traversal)."""
        with self.assertRaises(ImageValidationError):
            validate_filename('folder/image.bmp')

    def test_path_traversal_backslash(self):
        """Should reject filenames with '\\' (path traversal)."""
        with self.assertRaises(ImageValidationError):
            validate_filename('folder\\image.bmp')


class TestValidateImage(unittest.TestCase):
    """Test image validation."""

    def test_valid_image(self):
        """Valid 3-channel image should pass validation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_image(image, Path('test.bmp'))

    def test_none_image(self):
        """None image should raise ImageValidationError."""
        with self.assertRaises(ImageValidationError):
            validate_image(None, Path('test.bmp'))

    def test_empty_image(self):
        """Empty image should raise ImageValidationError."""
        image = np.array([])
        with self.assertRaises(ImageValidationError):
            validate_image(image, Path('test.bmp'))

    def test_wrong_channels_grayscale(self):
        """Grayscale (1-channel) image should raise ImageValidationError."""
        image = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ImageValidationError):
            validate_image(image, Path('test.bmp'))

    def test_wrong_channels_rgba(self):
        """RGBA (4-channel) image should raise ImageValidationError."""
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        with self.assertRaises(ImageValidationError):
            validate_image(image, Path('test.bmp'))

    def test_image_too_large(self):
        """Image exceeding MAX_IMAGE_SIZE should raise ImageValidationError."""
        # Create metadata for an image that would be too large
        # Don't actually allocate the memory
        large_height = MAX_IMAGE_SIZE + 1
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        # Manually set shape to simulate large image
        image = np.lib.stride_tricks.as_strided(
            image,
            shape=(large_height, 1, 3),
            strides=(0, 0, 1)
        )
        with self.assertRaises(ImageValidationError):
            validate_image(image, Path('huge.bmp'))


class TestGetMeanAndStd(unittest.TestCase):
    """Test mean and standard deviation calculation."""

    def test_uniform_image(self):
        """Uniform color image should have zero std dev."""
        image = np.full((100, 100, 3), 128, dtype=np.float32)
        mean, std = get_mean_and_std(image)

        self.assertEqual(len(mean), 3)
        self.assertEqual(len(std), 3)
        np.testing.assert_array_almost_equal(mean, [128, 128, 128], decimal=1)
        np.testing.assert_array_almost_equal(std, [0, 0, 0], decimal=1)

    def test_gradient_image(self):
        """Gradient image should have non-zero std dev."""
        image = np.zeros((100, 100, 3), dtype=np.float32)
        for i in range(100):
            image[i, :, :] = i * 2.55  # 0-255 gradient

        mean, std = get_mean_and_std(image)

        # Mean should be around middle value
        self.assertTrue(all(100 < m < 150 for m in mean))
        # Std should be non-zero
        self.assertTrue(all(s > 0 for s in std))


class TestTransferColorVectorized(unittest.TestCase):
    """Test vectorized color transfer implementation."""

    def test_identity_transfer(self):
        """Transfer with same statistics should return similar image."""
        source = np.random.randint(0, 256, (50, 50, 3)).astype(np.float32)
        mean = np.array([100.0, 128.0, 150.0])
        std = np.array([20.0, 30.0, 25.0])

        result = transfer_color_vectorized(source, mean, std, mean, std)

        # Result should be very close to source
        np.testing.assert_array_almost_equal(result, source, decimal=0)

    def test_zero_std_handling(self):
        """Should handle zero std dev without division by zero."""
        source = np.full((50, 50, 3), 128, dtype=np.float32)
        source_mean = np.array([128.0, 128.0, 128.0])
        source_std = np.array([0.0, 0.0, 0.0])  # Zero std
        target_mean = np.array([100.0, 150.0, 120.0])
        target_std = np.array([20.0, 30.0, 25.0])

        # Should not raise exception
        result = transfer_color_vectorized(
            source, source_mean, source_std, target_mean, target_std
        )

        # Result should be valid
        self.assertEqual(result.shape, source.shape)
        self.assertTrue(np.all((result >= 0) & (result <= 255)))

    def test_output_range(self):
        """Output should be clipped to [0, 255] range."""
        source = np.random.randint(0, 256, (50, 50, 3)).astype(np.float32)
        source_mean = np.array([50.0, 60.0, 70.0])
        source_std = np.array([10.0, 15.0, 12.0])
        target_mean = np.array([200.0, 220.0, 210.0])
        target_std = np.array([40.0, 50.0, 45.0])

        result = transfer_color_vectorized(
            source, source_mean, source_std, target_mean, target_std
        )

        # All values should be in valid range
        self.assertTrue(np.all((result >= 0) & (result <= 255)))
        self.assertEqual(result.dtype, np.uint8)

    def test_vectorization_performance(self):
        """Vectorized version should process large images efficiently."""
        import time

        # Create a reasonably large image
        large_image = np.random.randint(0, 256, (1000, 1000, 3)).astype(np.float32)
        source_mean = np.array([100.0, 120.0, 110.0])
        source_std = np.array([20.0, 25.0, 22.0])
        target_mean = np.array([150.0, 140.0, 160.0])
        target_std = np.array([30.0, 28.0, 32.0])

        start_time = time.time()
        result = transfer_color_vectorized(
            large_image, source_mean, source_std, target_mean, target_std
        )
        elapsed_time = time.time() - start_time

        # Should complete in under 1 second for 1Mpixel image
        self.assertLess(elapsed_time, 1.0,
                       f"Vectorized operation took {elapsed_time:.3f}s, should be < 1s")
        self.assertEqual(result.shape, large_image.shape)


class TestReadImage(unittest.TestCase):
    """Test image reading and conversion."""

    def setUp(self):
        """Create temporary directory for test images."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_read_valid_image(self):
        """Should successfully read and convert valid image."""
        # Create a test image
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        test_path = self.temp_path / 'test.bmp'
        cv2.imwrite(str(test_path), test_image)

        # Read and convert
        result = read_image(test_path)

        # Check result
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_read_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        nonexistent_path = self.temp_path / 'nonexistent.bmp'

        with self.assertRaises(FileNotFoundError):
            read_image(nonexistent_path)

    def test_read_jpg_image(self):
        """Should successfully read JPG images."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        test_path = self.temp_path / 'test.jpg'
        cv2.imwrite(str(test_path), test_image)

        result = read_image(test_path)

        self.assertEqual(result.shape, (100, 100, 3))

    def test_read_png_image(self):
        """Should successfully read PNG images."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        test_path = self.temp_path / 'test.png'
        cv2.imwrite(str(test_path), test_image)

        result = read_image(test_path)

        self.assertEqual(result.shape, (100, 100, 3))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""

    def setUp(self):
        """Create temporary directory and test images."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create source image with gradient (not uniform) to have non-zero std dev
        self.source = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            # Create vertical gradient for each channel
            self.source[i, :, 0] = int(50 + i * 0.5)   # B: 50-100
            self.source[i, :, 1] = int(100 + i * 0.5)  # G: 100-150
            self.source[i, :, 2] = int(150 + i * 0.5)  # R: 150-200

        # Create target image with different gradient (reddish)
        self.target = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            self.target[i, :, 0] = int(80 + i * 0.8)   # B: 80-160
            self.target[i, :, 1] = int(100 + i * 0.6)  # G: 100-160
            self.target[i, :, 2] = int(180 + i * 0.7)  # R: 180-250

        self.source_path = self.temp_path / 'source.bmp'
        self.target_path = self.temp_path / 'target.bmp'

        cv2.imwrite(str(self.source_path), self.source)
        cv2.imwrite(str(self.target_path), self.target)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_complete_workflow(self):
        """Test complete color transfer workflow."""
        from color_transfer import color_transfer

        output_path = self.temp_path / 'result.bmp'

        # Perform color transfer
        color_transfer(self.source_path, self.target_path, output_path)

        # Verify output was created
        self.assertTrue(output_path.exists())

        # Read and verify result
        result = cv2.imread(str(output_path))
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.source.shape)

        # Result should have different color characteristics than source
        # (this is a basic sanity check, not exact color matching)
        self.assertFalse(np.array_equal(result, self.source))


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
