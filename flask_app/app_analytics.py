#!/usr/bin/env python3
"""
Flask Analytics API Extension
==============================

Adds comprehensive analytics endpoints for cost, performance, and quality metrics.

New endpoints:
- /api/analytics/cost/estimate - Estimate cost before processing
- /api/analytics/cost/history - Get cost history and statistics
- /api/analytics/quality/ssim - Calculate SSIM metrics
- /api/analytics/quality/perceptual - Calculate perceptual metrics
- /api/analytics/worker-consensus - Analyze worker consensus (WCDS)
- /api/analytics/performance/stats - Get performance statistics
"""

from flask import Blueprint, request, jsonify
import numpy as np
import cv2
from pathlib import Path
import logging

from cost_calculator import CostCalculator, EnergyProfile
from quality_metrics import QualityMetrics, WorkerConsensusAnalyzer, create_side_by_side_comparison
from performance_tracker import PerformanceTracker
from roi_selector import ROISelector
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Create Blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# Initialize analytics components
cost_calculator = CostCalculator()
performance_tracker = PerformanceTracker()
roi_selector = ROISelector()


# ============================================================================
# COST ANALYTICS ENDPOINTS
# ============================================================================

@analytics_bp.route('/cost/estimate', methods=['POST'])
def estimate_cost():
    """
    Estimate cost for an operation before executing.

    Body:
    {
        "image_width": 1920,
        "image_height": 1080,
        "processing_mode": "balanced"  // "eco", "balanced", or "max_quality"
    }

    Returns:
        JSON with estimated cost, time, and energy
    """
    data = request.json

    width = data.get('image_width', 1920)
    height = data.get('image_height', 1080)
    mode = data.get('processing_mode', 'balanced')

    image_size_pixels = width * height

    try:
        estimate = cost_calculator.estimate_cost(
            image_size_pixels=image_size_pixels,
            processing_mode=mode
        )

        return jsonify({
            'success': True,
            'estimate': estimate
        })

    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/cost/history', methods=['GET'])
def get_cost_history():
    """
    Get cost history and statistics.

    Query parameters:
    - last_n: Optional, only include last N operations

    Returns:
        JSON with cost statistics and history
    """
    last_n = request.args.get('last_n', type=int)

    try:
        stats = cost_calculator.get_statistics(last_n=last_n)

        # Get recent operations
        recent_ops = cost_calculator.history[-20:] if cost_calculator.history else []
        recent_ops_data = [
            {
                'operation_id': op.operation_id,
                'timestamp': op.timestamp,
                'total_cost_usd': op.total_cost_usd,
                'total_energy_wh': op.total_energy_wh,
                'total_time_seconds': op.total_time_seconds,
                'cost_per_megapixel': op.cost_per_megapixel
            }
            for op in recent_ops
        ]

        return jsonify({
            'success': True,
            'statistics': stats,
            'recent_operations': recent_ops_data,
            'total_operations': len(cost_calculator.history)
        })

    except Exception as e:
        logger.error(f"Failed to get cost history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/cost/recommendations/<operation_id>', methods=['GET'])
def get_cost_recommendations(operation_id):
    """
    Get optimization recommendations for a specific operation.

    Returns:
        JSON with recommendations
    """
    try:
        # Find operation in history
        operation = None
        for op in cost_calculator.history:
            if op.operation_id == operation_id:
                operation = op
                break

        if not operation:
            return jsonify({'success': False, 'error': 'Operation not found'}), 404

        recommendations = cost_calculator.get_optimization_recommendations(operation)

        return jsonify({
            'success': True,
            'operation_id': operation_id,
            'recommendations': recommendations,
            'metrics': {
                'total_cost_usd': operation.total_cost_usd,
                'total_energy_wh': operation.total_energy_wh,
                'total_time_seconds': operation.total_time_seconds,
                'cost_per_megapixel': operation.cost_per_megapixel
            }
        })

    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# QUALITY ANALYTICS ENDPOINTS
# ============================================================================

@analytics_bp.route('/quality/ssim', methods=['POST'])
def calculate_ssim():
    """
    Calculate SSIM between two images.

    Body:
    {
        "source_job_id": "uuid",
        "result_job_id": "uuid",
        "return_map": true
    }

    Returns:
        JSON with SSIM metrics and regional analysis
    """
    data = request.json
    source_job_id = data.get('source_job_id')
    result_job_id = data.get('result_job_id')
    return_map = data.get('return_map', False)

    if not source_job_id or not result_job_id:
        return jsonify({'success': False, 'error': 'Missing required job IDs'}), 400

    try:
        # Load images
        source_img = _load_image_by_job_id(source_job_id)
        result_img = _load_image_by_job_id(result_job_id)

        if source_img is None or result_img is None:
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Calculate SSIM
        ssim_metrics = QualityMetrics.calculate_ssim(
            source_img,
            result_img,
            return_map=return_map
        )

        # Convert to JSON-serializable format
        response = {
            'success': True,
            'ssim': {
                'overall_ssim': ssim_metrics.overall_ssim,
                'channel_ssim': ssim_metrics.channel_ssim,
                'interpretation': ssim_metrics.interpretation,
                'regional_analysis': [
                    {
                        'region': rm.region_name,
                        'ssim': rm.ssim_score,
                        'percentage': rm.percentage_of_image,
                        'pixel_count': rm.pixel_count,
                        'delta_e_mean': rm.delta_e_mean,
                        'delta_e_std': rm.delta_e_std
                    }
                    for rm in ssim_metrics.region_metrics
                ]
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"SSIM calculation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/quality/perceptual', methods=['POST'])
def calculate_perceptual():
    """
    Calculate perceptual quality metrics.

    Body:
    {
        "source_job_id": "uuid",
        "result_job_id": "uuid"
    }

    Returns:
        JSON with PSNR, MAE, histogram correlation, gradient similarity
    """
    data = request.json
    source_job_id = data.get('source_job_id')
    result_job_id = data.get('result_job_id')

    if not source_job_id or not result_job_id:
        return jsonify({'success': False, 'error': 'Missing required job IDs'}), 400

    try:
        # Load images
        source_img = _load_image_by_job_id(source_job_id)
        result_img = _load_image_by_job_id(result_job_id)

        if source_img is None or result_img is None:
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Calculate perceptual metrics
        metrics = QualityMetrics.calculate_perceptual_metrics(source_img, result_img)

        return jsonify({
            'success': True,
            'metrics': metrics
        })

    except Exception as e:
        logger.error(f"Perceptual metrics calculation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/worker-consensus', methods=['POST'])
def calculate_worker_consensus():
    """
    Calculate Worker Consensus Discrepancy Score (WCDS).

    Body:
    {
        "worker_results": [
            {"worker_id": "worker_1", "job_id": "uuid1"},
            {"worker_id": "worker_2", "job_id": "uuid2"},
            ...
        ]
    }

    Returns:
        JSON with WCDS, agreement matrix, and outliers
    """
    data = request.json
    worker_results = data.get('worker_results', [])

    if len(worker_results) < 2:
        return jsonify({'success': False, 'error': 'Need at least 2 workers'}), 400

    try:
        # Load worker result images
        images = []
        worker_ids = []

        for worker_data in worker_results:
            worker_id = worker_data['worker_id']
            job_id = worker_data['job_id']

            img = _load_image_by_job_id(job_id)
            if img is None:
                return jsonify({
                    'success': False,
                    'error': f'Image not found for worker {worker_id}'
                }), 404

            images.append(img)
            worker_ids.append(worker_id)

        # Calculate WCDS
        consensus_metrics = WorkerConsensusAnalyzer.calculate_wcds(images, worker_ids)

        # Convert agreement matrix to list
        agreement_matrix_list = consensus_metrics.agreement_matrix.tolist()

        return jsonify({
            'success': True,
            'wcds': consensus_metrics.wcds,
            'average_agreement': consensus_metrics.average_agreement,
            'consensus_level': consensus_metrics.consensus_level,
            'agreement_matrix': agreement_matrix_list,
            'worker_ids': consensus_metrics.worker_ids,
            'outlier_workers': consensus_metrics.outlier_workers
        })

    except Exception as e:
        logger.error(f"WCDS calculation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# PERFORMANCE ANALYTICS ENDPOINTS
# ============================================================================

@analytics_bp.route('/performance/stats', methods=['GET'])
def get_performance_stats():
    """
    Get worker performance statistics.

    Query parameters:
    - worker_id: Optional, get stats for specific worker

    Returns:
        JSON with performance statistics
    """
    worker_id = request.args.get('worker_id')

    try:
        if worker_id:
            # Get specific worker stats
            stats = performance_tracker.get_worker_statistics(worker_id)
            if not stats:
                return jsonify({'success': False, 'error': 'Worker not found'}), 404

            return jsonify({
                'success': True,
                'worker_id': worker_id,
                'statistics': {
                    'total_executions': stats.total_executions,
                    'successful_executions': stats.successful_executions,
                    'average_delta_e': stats.average_delta_e,
                    'average_processing_time': stats.average_processing_time,
                    'reliability_score': stats.reliability_score,
                    'current_weight': stats.current_weight,
                    'recent_trend': stats.recent_trend,
                    'specialties_performance': stats.specialties_performance
                }
            })
        else:
            # Get all worker stats
            all_stats = performance_tracker.get_all_statistics()

            stats_list = []
            for wid, stats in all_stats.items():
                stats_list.append({
                    'worker_id': wid,
                    'total_executions': stats.total_executions,
                    'successful_executions': stats.successful_executions,
                    'average_delta_e': stats.average_delta_e,
                    'average_processing_time': stats.average_processing_time,
                    'reliability_score': stats.reliability_score,
                    'current_weight': stats.current_weight,
                    'recent_trend': stats.recent_trend
                })

            # Sort by weight
            stats_list.sort(key=lambda x: x['current_weight'], reverse=True)

            return jsonify({
                'success': True,
                'workers': stats_list,
                'total_records': len(performance_tracker.records)
            })

    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/performance/best-workers', methods=['GET'])
def get_best_workers():
    """
    Get best performing workers for a given context.

    Query parameters:
    - n: Number of workers to return (default: 3)
    - target_ral_code: Optional RAL code for context
    - image_type: Optional image type for context

    Returns:
        JSON with ranked workers
    """
    n = request.args.get('n', type=int, default=3)
    target_ral_code = request.args.get('target_ral_code')
    image_type = request.args.get('image_type')

    try:
        context = {}
        if target_ral_code:
            context['target_ral_code'] = target_ral_code
        if image_type:
            context['image_type'] = image_type

        best_workers = performance_tracker.get_best_workers(
            n=n,
            context=context if context else None
        )

        # Get weights
        weights = performance_tracker.get_worker_weights(context if context else None)

        workers_with_weights = [
            {
                'worker_id': worker_id,
                'weight': weights.get(worker_id, 0.0)
            }
            for worker_id in best_workers
        ]

        return jsonify({
            'success': True,
            'best_workers': workers_with_weights,
            'context': context if context else 'none'
        })

    except Exception as e:
        logger.error(f"Failed to get best workers: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/performance/report', methods=['GET'])
def get_performance_report():
    """
    Get human-readable performance report.

    Returns:
        JSON with performance report
    """
    try:
        report = performance_tracker.get_performance_report()

        return jsonify({
            'success': True,
            'report': report
        })

    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROI (REGION OF INTEREST) ENDPOINTS
# ============================================================================

@analytics_bp.route('/roi/auto-detect', methods=['POST'])
def auto_detect_roi():
    """
    Automatically detect ROI in an image.

    Body:
    {
        "job_id": "uuid",
        "method": "combined",  // "saliency", "face", "edge", "color", "combined"
        "padding_percentage": 0.1,
        "min_size_percentage": 0.1
    }

    Returns:
        JSON with detected ROI and alternatives
    """
    data = request.json
    job_id = data.get('job_id')
    method = data.get('method', 'combined')
    padding = data.get('padding_percentage', 0.1)
    min_size = data.get('min_size_percentage', 0.1)

    if not job_id:
        return jsonify({'success': False, 'error': 'Missing job_id'}), 400

    try:
        # Load image
        image = _load_image_by_job_id(job_id)
        if image is None:
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Auto-detect ROI
        analysis = roi_selector.auto_detect_roi(
            image=image,
            method=method,
            padding_percentage=padding,
            min_size_percentage=min_size
        )

        # Convert to JSON-serializable format
        response = {
            'success': True,
            'analysis': {
                'primary_roi': asdict(analysis.primary_roi),
                'alternative_rois': [asdict(roi) for roi in analysis.alternative_rois],
                'cost_savings_percentage': analysis.cost_savings_percentage,
                'processing_time_reduction': analysis.processing_time_reduction,
                'image_dimensions': analysis.image_dimensions,
                'detection_confidence': analysis.detection_confidence
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"ROI auto-detection failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/roi/validate', methods=['POST'])
def validate_roi():
    """
    Validate a manually specified ROI.

    Body:
    {
        "job_id": "uuid",
        "roi": {"x": 100, "y": 100, "width": 500, "height": 500}
    }

    Returns:
        JSON with validation result
    """
    data = request.json
    job_id = data.get('job_id')
    roi_data = data.get('roi')

    if not job_id or not roi_data:
        return jsonify({'success': False, 'error': 'Missing job_id or roi'}), 400

    try:
        # Load image to get dimensions
        image = _load_image_by_job_id(job_id)
        if image is None:
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        height, width = image.shape[:2]

        # Create ROI object
        from roi_selector import ROI
        roi = ROI(
            x=roi_data['x'],
            y=roi_data['y'],
            width=roi_data['width'],
            height=roi_data['height']
        )

        # Validate
        is_valid, error_message = roi_selector.validate_roi(roi, (width, height))

        # Calculate cost savings
        full_area = width * height
        roi_area = roi.width * roi.height
        cost_savings = ((full_area - roi_area) / full_area) * 100

        return jsonify({
            'success': True,
            'is_valid': is_valid,
            'error_message': error_message,
            'cost_savings_percentage': cost_savings,
            'processing_time_reduction': cost_savings * 0.8
        })

    except Exception as e:
        logger.error(f"ROI validation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@analytics_bp.route('/roi/visualize/<job_id>', methods=['POST'])
def visualize_roi(job_id):
    """
    Generate visualization of ROI on image.

    Body:
    {
        "roi": {"x": 100, "y": 100, "width": 500, "height": 500}
    }

    Returns:
        PNG image with ROI overlay
    """
    data = request.json
    roi_data = data.get('roi')

    if not roi_data:
        return jsonify({'success': False, 'error': 'Missing roi'}), 400

    try:
        # Load image
        image = _load_image_by_job_id(job_id)
        if image is None:
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Create ROI object
        from roi_selector import ROI
        roi = ROI(
            x=roi_data['x'],
            y=roi_data['y'],
            width=roi_data['width'],
            height=roi_data['height'],
            detection_method=roi_data.get('detection_method', 'manual'),
            confidence=roi_data.get('confidence', 1.0)
        )

        # Visualize
        visualization = roi_selector.visualize_roi(image, roi)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

        from io import BytesIO
        from flask import send_file
        return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"ROI visualization failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@analytics_bp.route('/dashboard/overview', methods=['GET'])
def get_dashboard_overview():
    """
    Get comprehensive analytics overview for dashboard.

    Returns:
        JSON with cost, performance, and quality metrics
    """
    try:
        # Cost statistics
        cost_stats = cost_calculator.get_statistics(last_n=100)

        # Performance statistics
        all_worker_stats = performance_tracker.get_all_statistics()
        worker_summary = [
            {
                'worker_id': wid,
                'weight': stats.current_weight,
                'avg_delta_e': stats.average_delta_e,
                'reliability': stats.reliability_score,
                'trend': stats.recent_trend
            }
            for wid, stats in all_worker_stats.items()
        ]
        worker_summary.sort(key=lambda x: x['weight'], reverse=True)

        # Recent operations
        recent_costs = cost_calculator.history[-10:] if cost_calculator.history else []
        recent_ops = [
            {
                'operation_id': op.operation_id,
                'timestamp': op.timestamp,
                'cost': op.total_cost_usd,
                'energy_wh': op.total_energy_wh,
                'time': op.total_time_seconds
            }
            for op in recent_costs
        ]

        return jsonify({
            'success': True,
            'overview': {
                'cost_statistics': cost_stats,
                'worker_performance': worker_summary,
                'recent_operations': recent_ops,
                'total_operations': len(cost_calculator.history),
                'total_performance_records': len(performance_tracker.records)
            }
        })

    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _load_image_by_job_id(job_id: str) -> np.ndarray:
    """
    Load image by job ID.

    Searches in both uploads and results folders.

    Returns:
        Image as numpy array (RGB, 0-255) or None if not found
    """
    # Search in uploads
    upload_files = list(Path('uploads').glob(f"{job_id}.*"))
    result_files = list(Path('results').glob(f"{job_id}.*"))

    files = upload_files + result_files

    if not files:
        return None

    # Load first match
    img_path = files[0]
    img = cv2.imread(str(img_path))

    if img is None:
        return None

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def register_analytics_blueprint(app):
    """
    Register analytics blueprint with Flask app.

    Usage:
        from app_analytics import register_analytics_blueprint
        register_analytics_blueprint(app)
    """
    app.register_blueprint(analytics_bp)
    logger.info("✓ Analytics API registered")


if __name__ == '__main__':
    print("This is a Flask Blueprint. Import and register with your main app:")
    print("from app_analytics import register_analytics_blueprint")
    print("register_analytics_blueprint(app)")
