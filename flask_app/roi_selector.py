"""
ROI (Region of Interest) Selection Tool
========================================

Automatic and manual ROI selection for targeted color transfer processing.

Features:
- Automatic ROI detection using saliency, face detection, and edge analysis
- Manual ROI drawing and adjustment
- Cost savings estimation
- Multiple detection strategies
- ROI validation and optimization
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ROI:
    """Region of Interest"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    detection_method: str = "manual"
    area_pixels: int = 0
    area_percentage: float = 0.0


@dataclass
class ROIAnalysis:
    """Complete ROI analysis"""
    primary_roi: ROI
    alternative_rois: List[ROI]
    cost_savings_percentage: float
    processing_time_reduction: float
    image_dimensions: Tuple[int, int]
    detection_confidence: float


class ROISelector:
    """
    Automatic and manual ROI selection for images.

    Combines multiple detection strategies:
    1. Saliency detection (spectral residual)
    2. Face detection (Haar cascades)
    3. Edge density analysis
    4. Color clustering
    """

    def __init__(self):
        """Initialize ROI selector with detection models"""
        # Initialize saliency detector
        try:
            self.saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        except:
            logger.warning("Saliency detector not available")
            self.saliency_detector = None

        # Initialize face detector (Haar cascade)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            logger.warning("Face cascade not available")
            self.face_cascade = None

    def auto_detect_roi(
        self,
        image: np.ndarray,
        method: str = "combined",
        padding_percentage: float = 0.1,
        min_size_percentage: float = 0.1
    ) -> ROIAnalysis:
        """
        Automatically detect region of interest.

        Args:
            image: Input image (RGB, 0-255)
            method: Detection method ("saliency", "face", "edge", "combined")
            padding_percentage: Padding around detected ROI (0-1)
            min_size_percentage: Minimum ROI size as percentage of image (0-1)

        Returns:
            ROIAnalysis with primary and alternative ROIs
        """
        height, width = image.shape[:2]
        min_size_pixels = int(min(width, height) * min_size_percentage)

        rois = []

        # Method 1: Saliency detection
        if method in ["saliency", "combined"] and self.saliency_detector:
            saliency_roi = self._detect_roi_saliency(image, padding_percentage)
            if saliency_roi and saliency_roi.width >= min_size_pixels:
                rois.append(saliency_roi)

        # Method 2: Face detection
        if method in ["face", "combined"] and self.face_cascade:
            face_rois = self._detect_roi_faces(image, padding_percentage)
            rois.extend([roi for roi in face_rois if roi.width >= min_size_pixels])

        # Method 3: Edge density
        if method in ["edge", "combined"]:
            edge_roi = self._detect_roi_edges(image, padding_percentage)
            if edge_roi and edge_roi.width >= min_size_pixels:
                rois.append(edge_roi)

        # Method 4: Color clustering (high saturation regions)
        if method in ["color", "combined"]:
            color_roi = self._detect_roi_color(image, padding_percentage)
            if color_roi and color_roi.width >= min_size_pixels:
                rois.append(color_roi)

        # Select best ROI (highest confidence)
        if not rois:
            # Default to full image
            primary_roi = ROI(
                x=0,
                y=0,
                width=width,
                height=height,
                confidence=0.0,
                detection_method="none",
                area_pixels=width * height,
                area_percentage=100.0
            )
            alternative_rois = []
        else:
            # Sort by confidence
            rois.sort(key=lambda r: r.confidence, reverse=True)
            primary_roi = rois[0]
            alternative_rois = rois[1:5]  # Keep top 5 alternatives

        # Calculate cost savings
        full_area = width * height
        roi_area = primary_roi.width * primary_roi.height
        cost_savings = ((full_area - roi_area) / full_area) * 100

        # Calculate area percentage
        primary_roi.area_pixels = roi_area
        primary_roi.area_percentage = (roi_area / full_area) * 100

        # Estimate processing time reduction (proportional to area reduction)
        time_reduction = cost_savings * 0.8  # Assume 80% linear relationship

        # Overall detection confidence
        detection_confidence = primary_roi.confidence

        logger.info(f"ROI detected: {primary_roi.width}x{primary_roi.height} at ({primary_roi.x}, {primary_roi.y})")
        logger.info(f"Cost savings: {cost_savings:.1f}%, Time reduction: {time_reduction:.1f}%")

        return ROIAnalysis(
            primary_roi=primary_roi,
            alternative_rois=alternative_rois,
            cost_savings_percentage=cost_savings,
            processing_time_reduction=time_reduction,
            image_dimensions=(width, height),
            detection_confidence=detection_confidence
        )

    def _detect_roi_saliency(
        self,
        image: np.ndarray,
        padding: float
    ) -> Optional[ROI]:
        """Detect ROI using saliency detection"""
        if not self.saliency_detector:
            return None

        try:
            # Convert to appropriate format
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Compute saliency
            success, saliency_map = self.saliency_detector.computeSaliency(gray)
            if not success:
                return None

            # Normalize to 0-255
            saliency_map = (saliency_map * 255).astype(np.uint8)

            # Threshold to get salient regions
            threshold = np.percentile(saliency_map, 70)
            _, binary = cv2.threshold(saliency_map, int(threshold), 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)

            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(image.shape[1] - x, w + 2 * pad_x)
            h = min(image.shape[0] - y, h + 2 * pad_y)

            # Calculate confidence based on saliency strength
            roi_saliency = saliency_map[y:y+h, x:x+w].mean()
            confidence = min(1.0, roi_saliency / 255.0)

            return ROI(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                detection_method="saliency"
            )

        except Exception as e:
            logger.error(f"Saliency detection failed: {e}")
            return None

    def _detect_roi_faces(
        self,
        image: np.ndarray,
        padding: float
    ) -> List[ROI]:
        """Detect ROI using face detection"""
        if not self.face_cascade:
            return []

        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            rois = []
            for (x, y, w, h) in faces:
                # Add padding
                pad_x = int(w * padding)
                pad_y = int(h * padding)

                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(image.shape[1] - x, w + 2 * pad_x)
                h = min(image.shape[0] - y, h + 2 * pad_y)

                # Face detection has high confidence
                confidence = 0.95

                rois.append(ROI(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=confidence,
                    detection_method="face"
                ))

            return rois

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def _detect_roi_edges(
        self,
        image: np.ndarray,
        padding: float
    ) -> Optional[ROI]:
        """Detect ROI based on edge density"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)

            # Calculate edge density in grid
            grid_size = 32
            height, width = edges.shape
            max_density = 0
            best_region = None

            for y in range(0, height - grid_size, grid_size // 2):
                for x in range(0, width - grid_size, grid_size // 2):
                    region = edges[y:y+grid_size, x:x+grid_size]
                    density = region.sum() / (grid_size * grid_size * 255)

                    if density > max_density:
                        max_density = density
                        best_region = (x, y)

            if best_region is None:
                return None

            # Expand region to capture full object
            x, y = best_region
            expansion_factor = 3

            x = max(0, x - grid_size)
            y = max(0, y - grid_size)
            w = min(width - x, grid_size * expansion_factor)
            h = min(height - y, grid_size * expansion_factor)

            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)

            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(width - x, w + 2 * pad_x)
            h = min(height - y, h + 2 * pad_y)

            confidence = min(1.0, max_density * 2)

            return ROI(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                detection_method="edge"
            )

        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return None

    def _detect_roi_color(
        self,
        image: np.ndarray,
        padding: float
    ) -> Optional[ROI]:
        """Detect ROI based on high saturation regions"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Get saturation channel
            saturation = hsv[:, :, 1]

            # Threshold high saturation
            threshold = np.percentile(saturation, 70)
            _, binary = cv2.threshold(saturation, int(threshold), 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Get largest high-saturation region
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)

            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(image.shape[1] - x, w + 2 * pad_x)
            h = min(image.shape[0] - y, h + 2 * pad_y)

            # Calculate confidence based on saturation
            roi_saturation = saturation[y:y+h, x:x+w].mean()
            confidence = min(1.0, roi_saturation / 255.0)

            return ROI(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                detection_method="color"
            )

        except Exception as e:
            logger.error(f"Color-based detection failed: {e}")
            return None

    def validate_roi(
        self,
        roi: ROI,
        image_dimensions: Tuple[int, int]
    ) -> Tuple[bool, str]:
        """
        Validate ROI is within bounds and reasonable size.

        Returns:
            (is_valid, error_message)
        """
        width, height = image_dimensions

        # Check bounds
        if roi.x < 0 or roi.y < 0:
            return False, "ROI coordinates cannot be negative"

        if roi.x + roi.width > width or roi.y + roi.height > height:
            return False, "ROI extends beyond image bounds"

        # Check size
        if roi.width < 10 or roi.height < 10:
            return False, "ROI too small (minimum 10x10 pixels)"

        # Check aspect ratio (not too extreme)
        aspect_ratio = roi.width / roi.height
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            return False, "ROI aspect ratio too extreme"

        return True, ""

    def optimize_roi(
        self,
        roi: ROI,
        image_dimensions: Tuple[int, int],
        target_aspect_ratio: Optional[float] = None
    ) -> ROI:
        """
        Optimize ROI by adjusting to preferred aspect ratio or alignment.

        Args:
            roi: Original ROI
            image_dimensions: Image dimensions (width, height)
            target_aspect_ratio: Desired aspect ratio (width/height)

        Returns:
            Optimized ROI
        """
        width, height = image_dimensions

        if target_aspect_ratio:
            # Adjust to target aspect ratio while preserving area
            current_aspect = roi.width / roi.height
            area = roi.width * roi.height

            if current_aspect > target_aspect_ratio:
                # Too wide, increase height
                new_height = int(np.sqrt(area / target_aspect_ratio))
                new_width = int(new_height * target_aspect_ratio)
            else:
                # Too tall, increase width
                new_width = int(np.sqrt(area * target_aspect_ratio))
                new_height = int(new_width / target_aspect_ratio)

            # Center the adjusted ROI
            center_x = roi.x + roi.width // 2
            center_y = roi.y + roi.height // 2

            new_x = max(0, center_x - new_width // 2)
            new_y = max(0, center_y - new_height // 2)

            # Ensure within bounds
            new_x = min(new_x, width - new_width)
            new_y = min(new_y, height - new_height)

            return ROI(
                x=new_x,
                y=new_y,
                width=new_width,
                height=new_height,
                confidence=roi.confidence,
                detection_method=roi.detection_method + "_optimized"
            )

        return roi

    def extract_roi(
        self,
        image: np.ndarray,
        roi: ROI
    ) -> np.ndarray:
        """
        Extract ROI from image.

        Args:
            image: Full image
            roi: ROI to extract

        Returns:
            Cropped image
        """
        return image[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width].copy()

    def visualize_roi(
        self,
        image: np.ndarray,
        roi: ROI,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw ROI rectangle on image.

        Args:
            image: Image to draw on (will be copied)
            roi: ROI to visualize
            color: Rectangle color (RGB)
            thickness: Line thickness

        Returns:
            Image with ROI drawn
        """
        img_copy = image.copy()
        cv2.rectangle(
            img_copy,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            color,
            thickness
        )

        # Add label
        label = f"{roi.detection_method} ({roi.confidence:.2f})"
        cv2.putText(
            img_copy,
            label,
            (roi.x, roi.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        return img_copy
