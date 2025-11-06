"""
Multiple Color Transfer Algorithm Implementations for TSM
==========================================================

This module implements multiple "workers" (algorithms) for color transfer,
each with a unique approach. The Tom Sawyer Method orchestrates these workers
to produce optimal results through ensemble learning.

Workers:
1. Reinhard Statistical Transfer (Lab space)
2. Linear Color Mapping
3. Histogram Matching
4. Channel-Specific LAB Transfer
5. Region-Aware Segmented Transfer

Each worker implements the same interface for consistency.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from color_utils import rgb_to_lab, lab_to_rgb, delta_e_ciede2000
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result from a single transfer algorithm worker"""
    algorithm_name: str
    result_rgb: np.ndarray
    processing_time: float
    metadata: Dict
    worker_id: str


class BaseTransferWorker:
    """Base class for all transfer algorithm workers"""

    def __init__(self, worker_id: str, name: str):
        self.worker_id = worker_id
        self.name = name
        self.specialties = []  # RAL codes or image types this worker excels at

    def transfer(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> TransferResult:
        """Execute transfer algorithm. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement transfer()")

    def estimate_complexity(self, image: np.ndarray) -> float:
        """Estimate image complexity (0-1 scale). Higher = more complex."""
        # Calculate variance in color channels as complexity indicator
        std_per_channel = np.std(image, axis=(0, 1))
        complexity = np.mean(std_per_channel) / 255.0
        return float(np.clip(complexity, 0, 1))


class ReinhardStatisticalWorker(BaseTransferWorker):
    """
    Worker 1: Reinhard Statistical Color Transfer

    Transfers color statistics (mean, std) in LAB color space.
    Based on Reinhard et al. "Color Transfer between Images" (2001)

    Specialties: General-purpose, works well on most images
    """

    def __init__(self, worker_id: str = "worker_reinhard"):
        super().__init__(worker_id, "Reinhard Statistical Transfer")
        self.specialties = ["general", "natural_images", "portraits"]

    def transfer(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> TransferResult:
        import time
        start_time = time.time()

        # Convert to LAB
        source_lab = rgb_to_lab(source_rgb)
        target_lab = rgb_to_lab(target_rgb)

        # Calculate statistics
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
        target_std = np.std(target_lab.reshape(-1, 3), axis=0)

        # Transfer statistics
        result_lab = np.copy(source_lab)
        MIN_STD_THRESHOLD = 1e-6

        for channel in range(3):
            if source_std[channel] < MIN_STD_THRESHOLD:
                result_lab[:, :, channel] = target_mean[channel]
            else:
                result_lab[:, :, channel] = (
                    (source_lab[:, :, channel] - source_mean[channel]) *
                    (target_std[channel] / source_std[channel]) +
                    target_mean[channel]
                )

        # Convert back to RGB
        result_rgb = lab_to_rgb(result_lab)

        processing_time = time.time() - start_time

        return TransferResult(
            algorithm_name=self.name,
            result_rgb=result_rgb,
            processing_time=processing_time,
            metadata={
                "source_mean": source_mean.tolist(),
                "source_std": source_std.tolist(),
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist()
            },
            worker_id=self.worker_id
        )


class LinearMappingWorker(BaseTransferWorker):
    """
    Worker 2: Linear Color Mapping

    Creates linear transformation matrices to map source colors to target.
    Works well for images with distinct color regions.

    Specialties: Flat colors, graphics, RAL grays (7000-7999)
    """

    def __init__(self, worker_id: str = "worker_linear"):
        super().__init__(worker_id, "Linear Color Mapping")
        self.specialties = ["flat_colors", "graphics", "RAL_7000-7999"]

    def transfer(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> TransferResult:
        import time
        start_time = time.time()

        # Reshape images for linear algebra
        h, w, c = source_rgb.shape
        source_flat = source_rgb.reshape(-1, 3).astype(np.float32)
        target_flat = target_rgb.reshape(-1, 3).astype(np.float32)

        # Calculate color centroids
        source_centroid = np.mean(source_flat, axis=0)
        target_centroid = np.mean(target_flat, axis=0)

        # Center the data
        source_centered = source_flat - source_centroid
        target_centered = target_flat - target_centroid

        # Calculate scaling factors per channel
        source_scale = np.std(source_centered, axis=0) + 1e-6
        target_scale = np.std(target_centered, axis=0) + 1e-6
        scale_factors = target_scale / source_scale

        # Apply linear transformation
        result_flat = (source_centered * scale_factors) + target_centroid
        result_rgb = np.clip(result_flat, 0, 255).reshape(h, w, c).astype(np.uint8)

        processing_time = time.time() - start_time

        return TransferResult(
            algorithm_name=self.name,
            result_rgb=result_rgb,
            processing_time=processing_time,
            metadata={
                "scale_factors": scale_factors.tolist(),
                "source_centroid": source_centroid.tolist(),
                "target_centroid": target_centroid.tolist()
            },
            worker_id=self.worker_id
        )


class HistogramMatchingWorker(BaseTransferWorker):
    """
    Worker 3: Histogram Matching

    Matches the histogram distribution of source to target.
    Excellent for preserving local details while transferring color tone.

    Specialties: Complex textures, natural scenes, RAL reds (3000-3999)
    """

    def __init__(self, worker_id: str = "worker_histogram"):
        super().__init__(worker_id, "Histogram Matching")
        self.specialties = ["complex_textures", "natural_scenes", "RAL_3000-3999"]

    def _match_histogram_channel(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match histogram of one channel"""
        # Calculate histograms
        source_hist, source_bins = np.histogram(source.flatten(), 256, [0, 256])
        target_hist, target_bins = np.histogram(target.flatten(), 256, [0, 256])

        # Calculate CDFs
        source_cdf = source_hist.cumsum()
        source_cdf = source_cdf / source_cdf[-1]  # Normalize

        target_cdf = target_hist.cumsum()
        target_cdf = target_cdf / target_cdf[-1]  # Normalize

        # Create mapping
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest CDF value in target
            diff = np.abs(target_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)

        # Apply mapping
        return mapping[source.astype(np.uint8)]

    def transfer(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> TransferResult:
        import time
        start_time = time.time()

        result_rgb = np.zeros_like(source_rgb)

        # Match histogram for each channel independently
        for channel in range(3):
            result_rgb[:, :, channel] = self._match_histogram_channel(
                source_rgb[:, :, channel],
                target_rgb[:, :, channel] if target_rgb.ndim == 3 else target_rgb.flatten()
            )

        processing_time = time.time() - start_time

        return TransferResult(
            algorithm_name=self.name,
            result_rgb=result_rgb,
            processing_time=processing_time,
            metadata={
                "method": "per_channel_histogram_matching"
            },
            worker_id=self.worker_id
        )


class LABChannelSpecificWorker(BaseTransferWorker):
    """
    Worker 4: LAB Channel-Specific Transfer

    Applies different strategies to L, A, and B channels independently.
    L channel: histogram matching for luminance
    A/B channels: statistical transfer for chrominance

    Specialties: Preserving brightness, RAL blues (5000-5999)
    """

    def __init__(self, worker_id: str = "worker_lab_specific"):
        super().__init__(worker_id, "LAB Channel-Specific Transfer")
        self.specialties = ["brightness_preservation", "RAL_5000-5999"]

    def transfer(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> TransferResult:
        import time
        start_time = time.time()

        # Convert to LAB
        source_lab = rgb_to_lab(source_rgb)
        target_lab = rgb_to_lab(target_rgb)

        result_lab = np.copy(source_lab)

        # L channel: preserve relative luminance structure
        l_mean_source = np.mean(source_lab[:, :, 0])
        l_mean_target = np.mean(target_lab.reshape(-1, 3)[:, 0])
        l_offset = l_mean_target - l_mean_source
        result_lab[:, :, 0] = np.clip(source_lab[:, :, 0] + l_offset, 0, 100)

        # A and B channels: statistical transfer
        for channel in [1, 2]:
            source_mean = np.mean(source_lab[:, :, channel])
            source_std = np.std(source_lab[:, :, channel])
            target_mean = np.mean(target_lab.reshape(-1, 3)[:, channel])
            target_std = np.std(target_lab.reshape(-1, 3)[:, channel])

            if source_std > 1e-6:
                result_lab[:, :, channel] = (
                    (source_lab[:, :, channel] - source_mean) *
                    (target_std / source_std) + target_mean
                )

        # Convert back to RGB
        result_rgb = lab_to_rgb(result_lab)

        processing_time = time.time() - start_time

        return TransferResult(
            algorithm_name=self.name,
            result_rgb=result_rgb,
            processing_time=processing_time,
            metadata={
                "l_offset": float(l_offset),
                "method": "luminance_offset_chrominance_statistical"
            },
            worker_id=self.worker_id
        )


class RegionAwareWorker(BaseTransferWorker):
    """
    Worker 5: Region-Aware Segmented Transfer

    Segments image into regions using K-means clustering, then applies
    transfer to each region independently. Excellent for complex images
    with multiple distinct color areas.

    Specialties: Complex images, multiple color regions, high resolution
    """

    def __init__(self, worker_id: str = "worker_region", n_clusters: int = 5):
        super().__init__(worker_id, "Region-Aware Segmented Transfer")
        self.specialties = ["complex_images", "multi_region", "high_resolution"]
        self.n_clusters = n_clusters

    def transfer(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> TransferResult:
        import time
        start_time = time.time()

        h, w, c = source_rgb.shape

        # Downsample for K-means if image is large
        max_pixels = 10000
        if h * w > max_pixels:
            scale = np.sqrt(max_pixels / (h * w))
            small_h, small_w = int(h * scale), int(w * scale)
            source_small = cv2.resize(source_rgb, (small_w, small_h))
        else:
            source_small = source_rgb
            small_h, small_w = h, w

        # K-means clustering to segment image
        pixels = source_small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, self.n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Resize labels back to original size if needed
        if h * w > max_pixels:
            labels_img = labels.reshape(small_h, small_w)
            labels = cv2.resize(labels_img, (w, h), interpolation=cv2.INTER_NEAREST).flatten()

        labels = labels.reshape(h, w)

        # Apply transfer to each region
        result_rgb = np.zeros_like(source_rgb)
        target_lab = rgb_to_lab(target_rgb)
        target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
        target_std = np.std(target_lab.reshape(-1, 3), axis=0)

        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                continue

            # Extract region
            region_rgb = source_rgb.copy()
            region_rgb[~mask] = 0

            # Convert to LAB
            region_lab = rgb_to_lab(region_rgb)

            # Statistical transfer for this region
            region_mean = np.mean(region_lab[mask], axis=0)
            region_std = np.std(region_lab[mask], axis=0)

            for channel in range(3):
                if region_std[channel] > 1e-6:
                    region_lab[mask, channel] = (
                        (region_lab[mask, channel] - region_mean[channel]) *
                        (target_std[channel] / region_std[channel]) +
                        target_mean[channel]
                    )

            # Convert back and assign
            region_result = lab_to_rgb(region_lab)
            result_rgb[mask] = region_result[mask]

        processing_time = time.time() - start_time

        return TransferResult(
            algorithm_name=self.name,
            result_rgb=result_rgb,
            processing_time=processing_time,
            metadata={
                "n_clusters": self.n_clusters,
                "regions_processed": int(self.n_clusters)
            },
            worker_id=self.worker_id
        )


class WorkerFactory:
    """Factory for creating transfer algorithm workers"""

    @staticmethod
    def create_all_workers() -> list[BaseTransferWorker]:
        """Create instances of all available workers"""
        return [
            ReinhardStatisticalWorker(),
            LinearMappingWorker(),
            HistogramMatchingWorker(),
            LABChannelSpecificWorker(),
            RegionAwareWorker()
        ]

    @staticmethod
    def create_worker(worker_id: str) -> Optional[BaseTransferWorker]:
        """Create a specific worker by ID"""
        workers = {
            "worker_reinhard": ReinhardStatisticalWorker,
            "worker_linear": LinearMappingWorker,
            "worker_histogram": HistogramMatchingWorker,
            "worker_lab_specific": LABChannelSpecificWorker,
            "worker_region": RegionAwareWorker
        }

        worker_class = workers.get(worker_id)
        if worker_class:
            return worker_class()
        return None

    @staticmethod
    def get_worker_info() -> Dict:
        """Get information about all available workers"""
        workers = WorkerFactory.create_all_workers()
        return {
            worker.worker_id: {
                "name": worker.name,
                "specialties": worker.specialties,
                "description": worker.__class__.__doc__.strip() if worker.__class__.__doc__ else ""
            }
            for worker in workers
        }
