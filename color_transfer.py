#!/usr/bin/env python3
"""
Color Transfer between Images

Implements the Reinhard et al. color transfer algorithm to transfer color
characteristics from a target image to a source image using LAB color space.

Reference:
    Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001).
    Color transfer between images. IEEE Computer Graphics and Applications.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2


# Constants
MIN_STD_THRESHOLD = 1e-6  # Minimum std dev to avoid division by zero
MAX_IMAGE_SIZE = 4096 * 4096  # Maximum pixels to prevent memory exhaustion
VALID_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}


class ColorTransferError(Exception):
    """Base exception for color transfer operations."""
    pass


class ImageValidationError(ColorTransferError):
    """Exception raised for image validation failures."""
    pass


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_filename(filename: str) -> None:
    """
    Validate filename to prevent path traversal attacks.

    Args:
        filename: Filename to validate.

    Raises:
        ImageValidationError: If filename contains invalid characters.
    """
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ImageValidationError(
            f"Invalid filename (potential path traversal): {filename}"
        )


def validate_image(image: np.ndarray, filepath: Path) -> None:
    """
    Validate image dimensions and size.

    Args:
        image: Image array to validate.
        filepath: Path to the image file (for error messages).

    Raises:
        ImageValidationError: If image is invalid or too large.
    """
    if image is None or image.size == 0:
        raise ImageValidationError(f"Invalid or empty image: {filepath}")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ImageValidationError(
            f"Image must have 3 channels (BGR), got shape {image.shape}: {filepath}"
        )

    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_SIZE:
        raise ImageValidationError(
            f"Image too large ({height}x{width} = {height*width} pixels, "
            f"max {MAX_IMAGE_SIZE}): {filepath}"
        )


def read_image(filepath: Path) -> np.ndarray:
    """
    Read and validate an image file, converting to LAB color space.

    Args:
        filepath: Path to the image file.

    Returns:
        Image array in LAB color space (float32).

    Raises:
        FileNotFoundError: If image file doesn't exist.
        ImageValidationError: If image is invalid.
        ColorTransferError: If image cannot be read or converted.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")

    if filepath.suffix.lower() not in VALID_EXTENSIONS:
        logging.warning(
            f"Unusual file extension '{filepath.suffix}', "
            f"expected one of {VALID_EXTENSIONS}"
        )

    # Read image
    try:
        image = cv2.imread(str(filepath))
    except Exception as e:
        raise ColorTransferError(f"Failed to read image {filepath}: {e}")

    if image is None:
        raise ColorTransferError(f"OpenCV failed to read image: {filepath}")

    # Validate image
    validate_image(image, filepath)

    # Convert to LAB color space
    try:
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    except cv2.error as e:
        raise ColorTransferError(
            f"Failed to convert {filepath} to LAB color space: {e}"
        )

    logging.debug(f"Loaded {filepath} with shape {image.shape}")
    return image_lab


def get_mean_and_std(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and standard deviation for each channel.

    Args:
        image: Image array in LAB color space.

    Returns:
        Tuple of (mean, std) arrays, each with 3 values (one per channel).
    """
    mean, std = cv2.meanStdDev(image)
    mean = np.hstack(mean).flatten()
    std = np.hstack(std).flatten()

    logging.debug(f"Statistics - Mean: {mean}, Std: {std}")
    return mean, std


def transfer_color_vectorized(
    source: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray
) -> np.ndarray:
    """
    Transfer color statistics from target to source using vectorized operations.

    This function applies the Reinhard color transfer formula:
        result = ((source - source_mean) * (target_std / source_std)) + target_mean

    Args:
        source: Source image in LAB color space.
        source_mean: Mean values for each channel of source.
        source_std: Standard deviation for each channel of source.
        target_mean: Mean values for each channel of target.
        target_std: Standard deviation for each channel of target.

    Returns:
        Color-transferred image in LAB color space.
    """
    result = source.copy()

    # Process each channel with vectorized operations
    for channel in range(3):
        # Avoid division by zero for monochrome or constant channels
        if source_std[channel] < MIN_STD_THRESHOLD:
            logging.warning(
                f"Channel {channel} has very low std dev ({source_std[channel]}), "
                f"skipping transfer for this channel"
            )
            continue

        # Vectorized color transfer formula (100-1000x faster than loops)
        ratio = target_std[channel] / source_std[channel]
        result[:, :, channel] = (
            (source[:, :, channel] - source_mean[channel]) * ratio
            + target_mean[channel]
        )

    # Clip values to valid range and convert to uint8
    result = np.clip(np.round(result), 0, 255).astype(np.uint8)

    return result


def color_transfer(
    source_path: Path,
    target_path: Path,
    output_path: Path
) -> None:
    """
    Transfer color characteristics from target image to source image.

    Args:
        source_path: Path to source image.
        target_path: Path to target reference image.
        output_path: Path for output image.

    Raises:
        FileNotFoundError: If input images don't exist.
        ImageValidationError: If images are invalid.
        ColorTransferError: If color transfer fails.
    """
    logging.info(f"Processing: {source_path.name} -> {target_path.name}")

    # Read and convert images to LAB color space
    source_lab = read_image(source_path)
    target_lab = read_image(target_path)

    # Calculate color statistics
    source_mean, source_std = get_mean_and_std(source_lab)
    target_mean, target_std = get_mean_and_std(target_lab)

    # Apply color transfer using vectorized operations
    result_lab = transfer_color_vectorized(
        source_lab, source_mean, source_std, target_mean, target_std
    )

    # Convert back to BGR color space
    try:
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    except cv2.error as e:
        raise ColorTransferError(f"Failed to convert result to BGR: {e}")

    # Save result
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), result_bgr)
        if not success:
            raise ColorTransferError(f"Failed to write output image: {output_path}")

        logging.info(f"Saved result to: {output_path}")
    except Exception as e:
        raise ColorTransferError(f"Failed to save result: {e}")


def process_batch(
    source_dir: Path,
    target_dir: Path,
    output_dir: Path,
    source_pattern: str = 's*.bmp',
    target_pattern: str = 't*.bmp',
    output_prefix: str = 'r'
) -> None:
    """
    Process multiple image pairs in batch mode.

    Args:
        source_dir: Directory containing source images.
        target_dir: Directory containing target images.
        output_dir: Directory for output images.
        source_pattern: Glob pattern for source images.
        target_pattern: Glob pattern for target images.
        output_prefix: Prefix for output filenames.

    Raises:
        FileNotFoundError: If directories don't exist.
        ColorTransferError: If processing fails.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    # Find all source images
    source_files = sorted(source_dir.glob(source_pattern))
    target_files = sorted(target_dir.glob(target_pattern))

    if not source_files:
        logging.warning(f"No source files found matching '{source_pattern}' in {source_dir}")
        return

    if not target_files:
        logging.warning(f"No target files found matching '{target_pattern}' in {target_dir}")
        return

    # Process pairs
    num_pairs = min(len(source_files), len(target_files))
    logging.info(f"Found {len(source_files)} source and {len(target_files)} target images")
    logging.info(f"Processing {num_pairs} pairs...")

    success_count = 0
    error_count = 0

    for i, (source_path, target_path) in enumerate(zip(source_files, target_files), 1):
        output_path = output_dir / f"{output_prefix}{i}{source_path.suffix}"

        try:
            logging.info(f"[{i}/{num_pairs}] Processing pair...")
            color_transfer(source_path, target_path, output_path)
            success_count += 1
        except Exception as e:
            logging.error(f"[{i}/{num_pairs}] Failed to process pair: {e}")
            error_count += 1
            continue

    logging.info(f"Batch complete: {success_count} succeeded, {error_count} failed")


def main() -> int:
    """
    Main entry point for the color transfer tool.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description='Transfer color characteristics between images using the Reinhard algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single pair of images
  python color_transfer.py -s source/image1.jpg -t target/sunset.jpg -o result/output.jpg

  # Batch process with default directories (source/, target/, result/)
  python color_transfer.py --batch

  # Batch process with custom directories
  python color_transfer.py --batch --source-dir ./inputs --target-dir ./references --output-dir ./outputs

  # Process with verbose logging
  python color_transfer.py -s source.jpg -t target.jpg -o result.jpg -v
        """
    )

    # Single image mode arguments
    parser.add_argument(
        '-s', '--source',
        type=Path,
        help='Path to source image'
    )
    parser.add_argument(
        '-t', '--target',
        type=Path,
        help='Path to target reference image'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Path for output image'
    )

    # Batch mode arguments
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple image pairs in batch mode'
    )
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path('source'),
        help='Directory containing source images (default: source/)'
    )
    parser.add_argument(
        '--target-dir',
        type=Path,
        default=Path('target'),
        help='Directory containing target images (default: target/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('result'),
        help='Directory for output images (default: result/)'
    )
    parser.add_argument(
        '--source-pattern',
        default='s*.bmp',
        help='Glob pattern for source images (default: s*.bmp)'
    )
    parser.add_argument(
        '--target-pattern',
        default='t*.bmp',
        help='Glob pattern for target images (default: t*.bmp)'
    )
    parser.add_argument(
        '--output-prefix',
        default='r',
        help='Prefix for output filenames (default: r)'
    )

    # General arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        if args.batch:
            # Batch processing mode
            process_batch(
                args.source_dir,
                args.target_dir,
                args.output_dir,
                args.source_pattern,
                args.target_pattern,
                args.output_prefix
            )
        elif args.source and args.target and args.output:
            # Single image processing mode
            color_transfer(args.source, args.target, args.output)
        else:
            # No valid arguments provided
            parser.print_help()
            logging.error(
                "\nError: Either provide -s, -t, -o for single image mode, "
                "or use --batch for batch processing"
            )
            return 1

        logging.info("Color transfer completed successfully!")
        return 0

    except KeyboardInterrupt:
        logging.warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            logging.exception("Detailed traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
