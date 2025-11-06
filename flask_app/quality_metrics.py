"""
Quality Metrics and Analysis
=============================

Comprehensive quality assessment for color transfer results including:
- Structural Similarity Index (SSIM)
- Worker Consensus Discrepancy Score (WCDS)
- Perceptual quality metrics
- Regional analysis (edges, textures, smooth areas)
- Style preservation metrics
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegionMetrics:
    """Quality metrics for specific image regions"""
    region_name: str
    ssim_score: float
    pixel_count: int
    percentage_of_image: float
    delta_e_mean: float
    delta_e_std: float


@dataclass
class SSIMMetrics:
    """Complete SSIM analysis"""
    overall_ssim: float
    channel_ssim: Dict[str, float]  # R, G, B channels
    region_metrics: List[RegionMetrics]
    ssim_map: Optional[np.ndarray] = None
    interpretation: str = ""


@dataclass
class WorkerConsensusMetrics:
    """Worker consensus analysis"""
    wcds: float  # Worker Consensus Discrepancy Score (0=perfect, 1=complete disagreement)
    average_agreement: float  # Average SSIM between all worker pairs
    consensus_level: str  # "high", "moderate", "low"
    agreement_matrix: Optional[np.ndarray] = None
    worker_ids: List[str] = None
    outlier_workers: List[str] = None


class QualityMetrics:
    """
    Comprehensive quality analysis for color transfer results.

    Provides SSIM, perceptual metrics, and regional analysis.
    """

    @staticmethod
    def calculate_ssim(
        source: np.ndarray,
        result: np.ndarray,
        multichannel: bool = True,
        return_map: bool = True
    ) -> SSIMMetrics:
        """
        Calculate Structural Similarity Index (SSIM).

        Args:
            source: Source image (RGB, 0-255)
            result: Result image (RGB, 0-255)
            multichannel: Calculate for each channel
            return_map: Return pixel-wise SSIM map

        Returns:
            SSIMMetrics with detailed analysis
        """
        # Ensure images are same size
        if source.shape != result.shape:
            result = cv2.resize(result, (source.shape[1], source.shape[0]))

        # Convert to grayscale for overall SSIM
        source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Calculate overall SSIM
        ssim_score, ssim_map = ssim(
            source_gray,
            result_gray,
            data_range=255,
            full=True
        )

        # Per-channel SSIM
        channel_ssim = {}
        channel_names = ['R', 'G', 'B']
        for i, channel_name in enumerate(channel_names):
            ch_ssim, _ = ssim(
                source[:, :, i],
                result[:, :, i],
                data_range=255,
                full=True
            )
            channel_ssim[channel_name] = float(ch_ssim)

        # Regional analysis
        region_metrics = QualityMetrics._analyze_ssim_regions(
            source_gray,
            result_gray,
            ssim_map
        )

        # Interpretation
        interpretation = QualityMetrics._interpret_ssim(ssim_score)

        return SSIMMetrics(
            overall_ssim=float(ssim_score),
            channel_ssim=channel_ssim,
            region_metrics=region_metrics,
            ssim_map=ssim_map if return_map else None,
            interpretation=interpretation
        )

    @staticmethod
    def _analyze_ssim_regions(
        source_gray: np.ndarray,
        result_gray: np.ndarray,
        ssim_map: np.ndarray
    ) -> List[RegionMetrics]:
        """
        Analyze SSIM in different image regions (edges, textures, smooth).

        Args:
            source_gray: Grayscale source image
            result_gray: Grayscale result image
            ssim_map: SSIM map from skimage

        Returns:
            List of RegionMetrics for different regions
        """
        regions = []
        total_pixels = source_gray.size

        # 1. Edge regions
        edges = cv2.Canny(source_gray, 50, 150)
        edge_mask = edges > 0

        if edge_mask.sum() > 0:
            edge_ssim_values = ssim_map[edge_mask]
            edge_source = source_gray[edge_mask]
            edge_result = result_gray[edge_mask]

            regions.append(RegionMetrics(
                region_name="edges",
                ssim_score=float(edge_ssim_values.mean()),
                pixel_count=int(edge_mask.sum()),
                percentage_of_image=(edge_mask.sum() / total_pixels) * 100,
                delta_e_mean=float(np.abs(edge_source - edge_result).mean()),
                delta_e_std=float(np.abs(edge_source - edge_result).std())
            ))

        # 2. Textured regions (high variance)
        # Calculate local variance using a window
        kernel_size = 5
        variance_map = cv2.blur(source_gray.astype(np.float32) ** 2, (kernel_size, kernel_size)) - \
                      cv2.blur(source_gray.astype(np.float32), (kernel_size, kernel_size)) ** 2

        texture_threshold = np.percentile(variance_map, 75)
        texture_mask = (variance_map > texture_threshold) & (~edge_mask)

        if texture_mask.sum() > 0:
            texture_ssim_values = ssim_map[texture_mask]
            texture_source = source_gray[texture_mask]
            texture_result = result_gray[texture_mask]

            regions.append(RegionMetrics(
                region_name="textures",
                ssim_score=float(texture_ssim_values.mean()),
                pixel_count=int(texture_mask.sum()),
                percentage_of_image=(texture_mask.sum() / total_pixels) * 100,
                delta_e_mean=float(np.abs(texture_source - texture_result).mean()),
                delta_e_std=float(np.abs(texture_source - texture_result).std())
            ))

        # 3. Smooth regions (low variance)
        smooth_mask = (variance_map <= texture_threshold) & (~edge_mask)

        if smooth_mask.sum() > 0:
            smooth_ssim_values = ssim_map[smooth_mask]
            smooth_source = source_gray[smooth_mask]
            smooth_result = result_gray[smooth_mask]

            regions.append(RegionMetrics(
                region_name="smooth_areas",
                ssim_score=float(smooth_ssim_values.mean()),
                pixel_count=int(smooth_mask.sum()),
                percentage_of_image=(smooth_mask.sum() / total_pixels) * 100,
                delta_e_mean=float(np.abs(smooth_source - smooth_result).mean()),
                delta_e_std=float(np.abs(smooth_source - smooth_result).std())
            ))

        return regions

    @staticmethod
    def _interpret_ssim(ssim_score: float) -> str:
        """Interpret SSIM score"""
        if ssim_score >= 0.95:
            return "Excellent - Nearly identical structure"
        elif ssim_score >= 0.85:
            return "Good - High structural similarity"
        elif ssim_score >= 0.70:
            return "Fair - Moderate structural similarity"
        elif ssim_score >= 0.50:
            return "Poor - Low structural similarity"
        else:
            return "Very Poor - Structure significantly changed"

    @staticmethod
    def calculate_perceptual_metrics(
        source: np.ndarray,
        result: np.ndarray
    ) -> Dict:
        """
        Calculate perceptual quality metrics beyond SSIM.

        Args:
            source: Source image (RGB)
            result: Result image (RGB)

        Returns:
            Dictionary with various perceptual metrics
        """
        # Ensure same size
        if source.shape != result.shape:
            result = cv2.resize(result, (source.shape[1], source.shape[0]))

        metrics = {}

        # 1. Peak Signal-to-Noise Ratio (PSNR)
        mse = np.mean((source.astype(float) - result.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
        else:
            psnr = float('inf')
        metrics['psnr'] = float(psnr)

        # 2. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(source.astype(float) - result.astype(float)))
        metrics['mae'] = float(mae)

        # 3. Color histogram correlation
        hist_correlation = QualityMetrics._calculate_histogram_correlation(source, result)
        metrics['histogram_correlation'] = hist_correlation

        # 4. Gradient similarity (edge preservation)
        gradient_similarity = QualityMetrics._calculate_gradient_similarity(source, result)
        metrics['gradient_similarity'] = gradient_similarity

        # 5. Color distribution statistics
        color_stats = QualityMetrics._compare_color_distributions(source, result)
        metrics['color_statistics'] = color_stats

        return metrics

    @staticmethod
    def _calculate_histogram_correlation(source: np.ndarray, result: np.ndarray) -> Dict[str, float]:
        """Calculate histogram correlation for each channel"""
        correlations = {}

        for i, channel_name in enumerate(['R', 'G', 'B']):
            hist_source = cv2.calcHist([source], [i], None, [256], [0, 256]).flatten()
            hist_result = cv2.calcHist([result], [i], None, [256], [0, 256]).flatten()

            # Normalize
            hist_source = hist_source / hist_source.sum()
            hist_result = hist_result / hist_result.sum()

            # Calculate correlation
            correlation, _ = pearsonr(hist_source, hist_result)
            correlations[channel_name] = float(correlation)

        # Overall correlation
        correlations['overall'] = float(np.mean([correlations['R'], correlations['G'], correlations['B']]))

        return correlations

    @staticmethod
    def _calculate_gradient_similarity(source: np.ndarray, result: np.ndarray) -> float:
        """Calculate similarity of image gradients (edge preservation)"""
        # Convert to grayscale
        source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Calculate gradients
        source_gx = cv2.Sobel(source_gray, cv2.CV_64F, 1, 0, ksize=3)
        source_gy = cv2.Sobel(source_gray, cv2.CV_64F, 0, 1, ksize=3)
        source_grad = np.sqrt(source_gx**2 + source_gy**2)

        result_gx = cv2.Sobel(result_gray, cv2.CV_64F, 1, 0, ksize=3)
        result_gy = cv2.Sobel(result_gray, cv2.CV_64F, 0, 1, ksize=3)
        result_grad = np.sqrt(result_gx**2 + result_gy**2)

        # Calculate similarity (correlation)
        correlation, _ = pearsonr(source_grad.flatten(), result_grad.flatten())

        return float(correlation)

    @staticmethod
    def _compare_color_distributions(source: np.ndarray, result: np.ndarray) -> Dict:
        """Compare color distribution statistics"""
        stats = {}

        for i, channel_name in enumerate(['R', 'G', 'B']):
            source_channel = source[:, :, i].flatten()
            result_channel = result[:, :, i].flatten()

            stats[channel_name] = {
                'mean_diff': float(np.abs(source_channel.mean() - result_channel.mean())),
                'std_diff': float(np.abs(source_channel.std() - result_channel.std())),
                'source_mean': float(source_channel.mean()),
                'result_mean': float(result_channel.mean()),
                'source_std': float(source_channel.std()),
                'result_std': float(result_channel.std())
            }

        return stats


class WorkerConsensusAnalyzer:
    """
    Analyze consensus among multiple workers (TSM).

    Calculates Worker Consensus Discrepancy Score (WCDS) and identifies
    outlier workers that disagree with the consensus.
    """

    @staticmethod
    def calculate_wcds(
        worker_results: List[np.ndarray],
        worker_ids: List[str]
    ) -> WorkerConsensusMetrics:
        """
        Calculate Worker Consensus Discrepancy Score (WCDS).

        WCDS ranges from 0 (perfect consensus) to 1 (complete disagreement).

        Args:
            worker_results: List of result images from workers (RGB, 0-255)
            worker_ids: List of worker IDs

        Returns:
            WorkerConsensusMetrics with full analysis
        """
        n_workers = len(worker_results)

        if n_workers < 2:
            raise ValueError("Need at least 2 workers for consensus analysis")

        # Ensure all images are same size
        reference_shape = worker_results[0].shape
        for i in range(1, n_workers):
            if worker_results[i].shape != reference_shape:
                worker_results[i] = cv2.resize(
                    worker_results[i],
                    (reference_shape[1], reference_shape[0])
                )

        # Calculate pairwise agreement matrix (SSIM between all pairs)
        agreement_matrix = np.zeros((n_workers, n_workers))

        for i in range(n_workers):
            for j in range(n_workers):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                elif i < j:
                    # Convert to grayscale for SSIM
                    img1_gray = cv2.cvtColor(worker_results[i], cv2.COLOR_RGB2GRAY)
                    img2_gray = cv2.cvtColor(worker_results[j], cv2.COLOR_RGB2GRAY)

                    # Calculate SSIM
                    ssim_score, _ = ssim(
                        img1_gray,
                        img2_gray,
                        data_range=255,
                        full=True
                    )

                    agreement_matrix[i, j] = ssim_score
                    agreement_matrix[j, i] = ssim_score

        # Calculate average agreement (excluding diagonal)
        mask = np.triu(np.ones((n_workers, n_workers)), k=1).astype(bool)
        avg_agreement = agreement_matrix[mask].mean()

        # WCDS is inverse of agreement
        wcds = 1.0 - avg_agreement

        # Determine consensus level
        if avg_agreement >= 0.9:
            consensus_level = "high"
        elif avg_agreement >= 0.7:
            consensus_level = "moderate"
        else:
            consensus_level = "low"

        # Identify outliers (workers with low average agreement with others)
        outlier_threshold = avg_agreement - 0.15
        outlier_workers = []

        for i in range(n_workers):
            worker_avg_agreement = agreement_matrix[i, :].sum() / (n_workers - 1)
            if worker_avg_agreement < outlier_threshold:
                outlier_workers.append(worker_ids[i])

        logger.info(f"WCDS: {wcds:.3f}, Consensus: {consensus_level}, Outliers: {outlier_workers}")

        return WorkerConsensusMetrics(
            wcds=float(wcds),
            average_agreement=float(avg_agreement),
            consensus_level=consensus_level,
            agreement_matrix=agreement_matrix,
            worker_ids=worker_ids,
            outlier_workers=outlier_workers
        )

    @staticmethod
    def visualize_agreement_matrix(
        consensus_metrics: WorkerConsensusMetrics,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a heatmap visualization of the agreement matrix.

        Args:
            consensus_metrics: WorkerConsensusMetrics from calculate_wcds
            output_path: Optional path to save visualization

        Returns:
            Heatmap image (RGB, 0-255)
        """
        matrix = consensus_metrics.agreement_matrix
        n_workers = matrix.shape[0]

        # Create a colormap visualization
        # Normalize to 0-255
        matrix_normalized = (matrix * 255).astype(np.uint8)

        # Apply colormap (green = high agreement, red = low agreement)
        heatmap = cv2.applyColorMap(matrix_normalized, cv2.COLORMAP_RDY1GN)

        # Resize for better visibility
        scale_factor = max(1, 100 // n_workers)
        heatmap = cv2.resize(
            heatmap,
            (n_workers * scale_factor, n_workers * scale_factor),
            interpolation=cv2.INTER_NEAREST
        )

        # Add grid lines
        for i in range(n_workers + 1):
            pos = i * scale_factor
            cv2.line(heatmap, (pos, 0), (pos, heatmap.shape[0]), (0, 0, 0), 1)
            cv2.line(heatmap, (0, pos), (heatmap.shape[1], pos), (0, 0, 0), 1)

        if output_path:
            cv2.imwrite(output_path, heatmap)
            logger.info(f"Saved agreement matrix visualization to {output_path}")

        return heatmap


def create_side_by_side_comparison(
    images: List[np.ndarray],
    labels: List[str],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of multiple images.

    Args:
        images: List of images (RGB, 0-255)
        labels: List of labels for each image
        output_path: Optional path to save result

    Returns:
        Combined image
    """
    # Ensure all images are same height
    target_height = min(img.shape[0] for img in images)

    resized_images = []
    for img in images:
        if img.shape[0] != target_height:
            aspect_ratio = img.shape[1] / img.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        else:
            resized_images.append(img)

    # Add labels
    labeled_images = []
    for img, label in zip(resized_images, labels):
        # Add white bar at top for label
        bar_height = 40
        bar = np.ones((bar_height, img.shape[1], 3), dtype=np.uint8) * 255

        # Add text
        cv2.putText(
            bar,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        labeled_img = np.vstack([bar, img])
        labeled_images.append(labeled_img)

    # Concatenate horizontally
    combined = np.hstack(labeled_images)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved comparison to {output_path}")

    return combined
