"""
Image Complexity Analyzer for Adaptive Worker Selection
========================================================

This module implements the "Adaptive Intelligence Layer" of TSM.
It analyzes image characteristics to determine complexity and recommends
which workers to activate and how many resources to allocate.

Complexity Metrics:
- Color variance and distribution
- Edge density and texture complexity
- Image size and resolution
- Color palette diversity
- Gradient intensity
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplexityReport:
    """Complete complexity analysis report"""
    overall_complexity: float  # 0-1 scale (0=simple, 1=complex)
    metrics: Dict[str, float]
    recommended_workers: List[str]
    recommended_worker_count: int
    image_characteristics: Dict
    processing_priority: str  # "low", "medium", "high"


class ImageComplexityAnalyzer:
    """
    Analyzes image complexity to enable adaptive worker selection.

    This is the "Project Manager" in TSM that determines resource allocation
    based on task complexity.
    """

    def __init__(self):
        self.complexity_thresholds = {
            "simple": 0.3,      # 0-0.3: simple images (2-3 workers)
            "moderate": 0.6,    # 0.3-0.6: moderate (3-4 workers)
            "complex": 1.0      # 0.6-1.0: complex (all 5 workers)
        }

    def analyze(self, image_rgb: np.ndarray) -> ComplexityReport:
        """
        Perform complete complexity analysis on image.

        Args:
            image_rgb: Input image in RGB format (H, W, 3)

        Returns:
            ComplexityReport with all metrics and recommendations
        """
        metrics = {}

        # 1. Color Variance Analysis
        metrics['color_variance'] = self._calculate_color_variance(image_rgb)

        # 2. Edge Density (texture complexity)
        metrics['edge_density'] = self._calculate_edge_density(image_rgb)

        # 3. Color Palette Diversity
        metrics['color_diversity'] = self._calculate_color_diversity(image_rgb)

        # 4. Gradient Intensity
        metrics['gradient_intensity'] = self._calculate_gradient_intensity(image_rgb)

        # 5. Spatial Complexity (frequency domain)
        metrics['spatial_complexity'] = self._calculate_spatial_complexity(image_rgb)

        # 6. Size/Resolution Factor
        h, w = image_rgb.shape[:2]
        metrics['resolution_factor'] = self._calculate_resolution_factor(h, w)

        # Calculate overall complexity (weighted average)
        overall_complexity = self._calculate_overall_complexity(metrics)

        # Determine recommended workers
        recommended_workers = self._recommend_workers(overall_complexity, metrics)

        # Determine worker count
        recommended_count = self._recommend_worker_count(overall_complexity)

        # Processing priority
        priority = self._determine_priority(overall_complexity, h * w)

        # Image characteristics
        characteristics = self._extract_characteristics(image_rgb, metrics)

        return ComplexityReport(
            overall_complexity=overall_complexity,
            metrics=metrics,
            recommended_workers=recommended_workers,
            recommended_worker_count=recommended_count,
            image_characteristics=characteristics,
            processing_priority=priority
        )

    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """
        Calculate color variance across the image.
        Higher variance = more color diversity = higher complexity
        """
        # Calculate standard deviation for each channel
        std_per_channel = np.std(image, axis=(0, 1))
        # Normalize to 0-1 (255 is max possible std for uint8)
        variance_score = np.mean(std_per_channel) / 128.0
        return float(np.clip(variance_score, 0, 1))

    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection.
        More edges = more texture detail = higher complexity
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate percentage of edge pixels
        edge_density = np.sum(edges > 0) / edges.size

        # Normalize (typically 0-0.3 for most images)
        normalized_density = edge_density / 0.3

        return float(np.clip(normalized_density, 0, 1))

    def _calculate_color_diversity(self, image: np.ndarray) -> float:
        """
        Calculate color palette diversity using histogram analysis.
        More diverse colors = higher complexity
        """
        # Quantize colors to reduce noise
        quantized = (image // 32) * 32

        # Flatten and get unique colors
        h, w, c = image.shape
        pixels = quantized.reshape(-1, 3)

        # Count unique colors
        unique_colors = len(np.unique(pixels, axis=0))

        # Normalize by theoretical maximum (for 8 quantization levels: 8^3 = 512)
        max_unique = 512
        diversity_score = unique_colors / max_unique

        return float(np.clip(diversity_score, 0, 1))

    def _calculate_gradient_intensity(self, image: np.ndarray) -> float:
        """
        Calculate average gradient intensity.
        Strong gradients = smooth transitions = moderate complexity
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize
        avg_gradient = np.mean(gradient_magnitude)
        normalized_gradient = avg_gradient / 128.0  # Typical max for 8-bit images

        return float(np.clip(normalized_gradient, 0, 1))

    def _calculate_spatial_complexity(self, image: np.ndarray) -> float:
        """
        Calculate spatial complexity using frequency domain analysis.
        High frequency content = fine details = higher complexity
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Downsample if too large (FFT is expensive)
        if gray.shape[0] * gray.shape[1] > 512 * 512:
            gray = cv2.resize(gray, (512, 512))

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Calculate high frequency energy
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Create mask for high frequencies (outer regions)
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) / 4)**2

        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)

        # Ratio of high frequency to total
        spatial_complexity = high_freq_energy / (total_energy + 1e-6)

        return float(np.clip(spatial_complexity * 2, 0, 1))

    def _calculate_resolution_factor(self, height: int, width: int) -> float:
        """
        Calculate resolution complexity factor.
        Larger images = more data to process = higher complexity
        """
        total_pixels = height * width

        # Reference: 1920x1080 = 2,073,600 pixels (Full HD)
        reference_pixels = 1920 * 1080

        resolution_factor = total_pixels / reference_pixels

        # Normalize (cap at 4K resolution)
        max_factor = (3840 * 2160) / reference_pixels  # 4K

        normalized = resolution_factor / max_factor

        return float(np.clip(normalized, 0, 1))

    def _calculate_overall_complexity(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted overall complexity score.

        Weights chosen based on impact on transfer algorithm performance.
        """
        weights = {
            'color_variance': 0.20,
            'edge_density': 0.20,
            'color_diversity': 0.20,
            'gradient_intensity': 0.15,
            'spatial_complexity': 0.15,
            'resolution_factor': 0.10
        }

        overall = sum(metrics[key] * weights[key] for key in weights.keys())

        return float(np.clip(overall, 0, 1))

    def _recommend_workers(self, complexity: float, metrics: Dict) -> List[str]:
        """
        Recommend which workers to activate based on complexity and metrics.

        Worker Selection Logic:
        - Always include: Reinhard (general purpose baseline)
        - Low texture: Add Linear
        - High texture: Add Histogram
        - High color diversity: Add LAB-Specific
        - High complexity: Add Region-Aware
        """
        workers = ["worker_reinhard"]  # Always include baseline

        # Linear mapper for simpler images or flat colors
        if complexity < 0.5 or metrics['color_variance'] < 0.4:
            workers.append("worker_linear")

        # Histogram matching for textured/complex images
        if metrics['edge_density'] > 0.4 or metrics['spatial_complexity'] > 0.5:
            workers.append("worker_histogram")

        # LAB-specific for diverse color palettes
        if metrics['color_diversity'] > 0.4:
            workers.append("worker_lab_specific")

        # Region-aware for complex multi-region images
        if complexity > 0.6 or metrics['resolution_factor'] > 0.7:
            workers.append("worker_region")

        return workers

    def _recommend_worker_count(self, complexity: float) -> int:
        """
        Recommend how many workers to use based on overall complexity.

        Simple images: 2-3 workers (fast, efficient)
        Moderate images: 3-4 workers (balanced)
        Complex images: 5 workers (maximum accuracy)
        """
        if complexity < self.complexity_thresholds['simple']:
            return 3  # Simple: use 3 fastest workers
        elif complexity < self.complexity_thresholds['moderate']:
            return 4  # Moderate: use 4 workers
        else:
            return 5  # Complex: use all workers

    def _determine_priority(self, complexity: float, pixel_count: int) -> str:
        """
        Determine processing priority for task queue.

        Priority considers both complexity and size.
        """
        # Simple heuristic
        if complexity > 0.7 or pixel_count > 1920 * 1080 * 2:
            return "high"
        elif complexity > 0.4 or pixel_count > 1920 * 1080:
            return "medium"
        else:
            return "low"

    def _extract_characteristics(self, image: np.ndarray, metrics: Dict) -> Dict:
        """Extract human-readable image characteristics"""
        h, w, c = image.shape

        characteristics = {
            "dimensions": f"{w}x{h}",
            "total_pixels": h * w,
            "aspect_ratio": round(w / h, 2),
            "color_space": "RGB",
            "channels": c
        }

        # Classify image type based on metrics
        if metrics['color_variance'] < 0.3 and metrics['edge_density'] < 0.3:
            characteristics['type'] = "flat_color"
        elif metrics['edge_density'] > 0.6:
            characteristics['type'] = "textured"
        elif metrics['color_diversity'] > 0.6:
            characteristics['type'] = "multi_color"
        elif metrics['gradient_intensity'] > 0.5:
            characteristics['type'] = "gradient_rich"
        else:
            characteristics['type'] = "general"

        return characteristics

    def get_complexity_summary(self, report: ComplexityReport) -> str:
        """Generate human-readable complexity summary"""
        complexity_level = "Simple"
        if report.overall_complexity > 0.6:
            complexity_level = "Complex"
        elif report.overall_complexity > 0.3:
            complexity_level = "Moderate"

        summary = f"""
Image Complexity Analysis
=========================
Complexity Level: {complexity_level} ({report.overall_complexity:.2f})
Image Type: {report.image_characteristics['type']}
Dimensions: {report.image_characteristics['dimensions']}
Processing Priority: {report.processing_priority.upper()}

Recommended Strategy:
- Workers to activate: {len(report.recommended_workers)}
- Workers: {', '.join(report.recommended_workers)}

Detailed Metrics:
- Color Variance: {report.metrics['color_variance']:.2f}
- Edge Density: {report.metrics['edge_density']:.2f}
- Color Diversity: {report.metrics['color_diversity']:.2f}
- Gradient Intensity: {report.metrics['gradient_intensity']:.2f}
- Spatial Complexity: {report.metrics['spatial_complexity']:.2f}
- Resolution Factor: {report.metrics['resolution_factor']:.2f}
"""
        return summary.strip()
