#!/usr/bin/env python3
"""
Flask Color Transfer Web Application - Tom Sawyer Method (TSM) Enhanced
=======================================================================

Production-ready Flask application with Tom Sawyer Method ensemble learning:
- Multiple color transfer algorithm workers
- Adaptive worker selection based on image complexity
- Weighted aggregation using performance history
- Continuous learning and performance tracking
- Parallel asynchronous processing with Celery
- Comprehensive API documentation with Swagger
- All security and logging features

TSM Components:
- Transfer Algorithms: 5 specialized workers
- Complexity Analyzer: Adaptive intelligence layer
- Performance Tracker: Learning and weight optimization
- Aggregation Oracle: Weighted result selection
- TSM Orchestrator: Master conductor
"""

import os
import json
import uuid
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler

from flask import Flask, request, jsonify, render_template, send_file
from flask_swagger_ui import get_swaggerui_blueprint
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
import cv2
from PIL import Image

# Import Celery configuration
from celery_config import make_celery

# Import application modules
from palette_manager import get_palette
from color_utils import (
    rgb_to_lab, delta_e_ciede2000,
    downsample_image, interpret_delta_e
)
from swagger_spec import openapi_spec

# Import TSM components
from tsm_orchestrator import TSMOrchestrator, EnsembleResult
from performance_tracker import PerformanceTracker
from transfer_algorithms import WorkerFactory

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['LOG_FOLDER'] = 'logs'
app.config['DATA_FOLDER'] = 'data'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

# Celery configuration
app.config['CELERY_BROKER_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# TSM configuration
app.config['TSM_MODE'] = os.environ.get('TSM_MODE', 'adaptive')  # adaptive, all, best
app.config['TSM_ENSEMBLE_BLEND'] = os.environ.get('TSM_ENSEMBLE_BLEND', 'false').lower() == 'true'
app.config['TSM_PERFORMANCE_FILE'] = Path(app.config['DATA_FOLDER']) / 'tsm_performance.json'

# Ensure directories exist
for folder_key in ['UPLOAD_FOLDER', 'RESULTS_FOLDER', 'LOG_FOLDER', 'DATA_FOLDER']:
    Path(app.config[folder_key]).mkdir(exist_ok=True, parents=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure comprehensive logging with rotating file handler."""
    log_dir = Path(app.config['LOG_FOLDER'])
    log_dir.mkdir(exist_ok=True)

    log_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Application logs
    file_handler = RotatingFileHandler(
        log_dir / 'app_tsm.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Error logs
    error_handler = RotatingFileHandler(
        log_dir / 'error_tsm.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Suppress noisy libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    app.logger.info("TSM Application logging initialized")

setup_logging()

# ============================================================================
# SECURITY & PERFORMANCE SETUP
# ============================================================================

# CSRF Protection
csrf = CSRFProtect(app)

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=app.config['CELERY_BROKER_URL']
)

# Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

# ============================================================================
# TSM INITIALIZATION
# ============================================================================

# Initialize TSM orchestrator (single instance for application lifecycle)
app.logger.info("Initializing TSM Orchestrator...")
performance_tracker = PerformanceTracker(storage_path=app.config['TSM_PERFORMANCE_FILE'])
tsm_orchestrator = TSMOrchestrator(performance_tracker=performance_tracker, max_workers=5)
app.logger.info("TSM Orchestrator ready")

# Initialize Celery
celery = make_celery(app)
app.logger.info("Celery initialized")

# Load RAL palette
app.logger.info("Loading RAL color palette...")
palette = get_palette()
app.logger.info(f"Loaded {len(palette.colors)} RAL colors")

# ============================================================================
# CELERY TASKS FOR ASYNC TSM PROCESSING
# ============================================================================

@celery.task(bind=True, name='tasks.process_tsm_transfer')
def process_tsm_transfer_celery(
    self,
    job_id: str,
    target_ral_code: str,
    downsample: bool = False,
    tsm_mode: str = 'adaptive',
    use_ensemble_blend: bool = False
):
    """
    Celery task for asynchronous TSM color transfer processing.

    Args:
        self: Celery task instance (bound)
        job_id: Unique job ID for this processing task
        target_ral_code: RAL color code to transfer to
        downsample: Whether to downsample for faster processing
        tsm_mode: TSM mode ('adaptive', 'all', 'best')
        use_ensemble_blend: Whether to create ensemble blend

    Returns:
        Dictionary with success status and results
    """
    try:
        self.update_state(
            state='STARTED',
            meta={'progress': 10, 'status': 'Loading image and target color...'}
        )

        # Get paths
        upload_path = Path(app.config['UPLOAD_FOLDER']) / f"{job_id}.png"
        result_path = Path(app.config['RESULTS_FOLDER']) / f"{job_id}_result.png"

        # Load source image
        source_img = cv2.imread(str(upload_path))
        if source_img is None:
            raise ValueError("Failed to load source image")

        source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

        # Downsample if requested
        if downsample:
            source_rgb = downsample_image(source_rgb)

        self.update_state(
            state='STARTED',
            meta={'progress': 20, 'status': 'Getting target RAL color...'}
        )

        # Get target RAL color
        ral_color = palette.get_color_by_code(target_ral_code)
        if not ral_color:
            raise ValueError(f"RAL code {target_ral_code} not found")

        target_rgb = np.array(ral_color['rgb'], dtype=np.uint8).reshape(1, 1, 3)

        self.update_state(
            state='STARTED',
            meta={'progress': 30, 'status': 'Processing with TSM ensemble...'}
        )

        # Create TSM orchestrator instance for this task
        task_tracker = PerformanceTracker(storage_path=app.config['TSM_PERFORMANCE_FILE'])
        task_orchestrator = TSMOrchestrator(
            performance_tracker=task_tracker,
            max_workers=5
        )

        # Process with TSM
        ensemble_result: EnsembleResult = task_orchestrator.process(
            source_rgb=source_rgb,
            target_rgb=target_rgb,
            target_ral_code=target_ral_code,
            mode=tsm_mode,
            use_ensemble_blend=use_ensemble_blend
        )

        self.update_state(
            state='STARTED',
            meta={'progress': 80, 'status': 'Saving results...'}
        )

        # Save best result
        result_bgr = cv2.cvtColor(ensemble_result.best_result_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_path), result_bgr)

        # Save ensemble blend if created
        ensemble_path = None
        if ensemble_result.ensemble_rgb is not None:
            ensemble_path = Path(app.config['RESULTS_FOLDER']) / f"{job_id}_ensemble.png"
            ensemble_bgr = cv2.cvtColor(ensemble_result.ensemble_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(ensemble_path), ensemble_bgr)

        self.update_state(
            state='STARTED',
            meta={'progress': 90, 'status': 'Finalizing...'}
        )

        # Prepare result
        result_job_id = f"{job_id}_result"

        # Compile comprehensive QC report
        qc_report = {
            'best_worker': ensemble_result.best_worker_id,
            'best_delta_e_mean': ensemble_result.qc_report.get('mean', 0.0),
            'best_delta_e_std': ensemble_result.qc_report.get('std', 0.0),
            'best_delta_e_max': ensemble_result.qc_report.get('max', 0.0),
            'best_delta_e_percentile_95': ensemble_result.qc_report.get('percentile_95', 0.0),
            'all_workers_scores': ensemble_result.qc_report.get('all_workers_scores', {}),
            'workers_executed': [r.worker_id for r in ensemble_result.all_results],
            'complexity_level': ensemble_result.complexity_report.overall_complexity,
            'image_type': ensemble_result.complexity_report.image_characteristics['type'],
            'processing_time_total': ensemble_result.processing_time_total,
            'processing_times_per_worker': ensemble_result.execution_summary['processing_times_per_worker'],
            'ensemble_blend_available': ensemble_result.ensemble_rgb is not None
        }

        return {
            'success': True,
            'result_job_id': result_job_id,
            'result_path': str(result_path),
            'ensemble_path': str(ensemble_path) if ensemble_path else None,
            'qc_report': qc_report,
            'tsm_summary': ensemble_result.execution_summary
        }

    except Exception as e:
        app.logger.error(f"TSM processing failed for job {job_id}: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# VALIDATION & ERROR HANDLING
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_rgb_color(rgb):
    """Validate RGB color values"""
    if not isinstance(rgb, (list, tuple, np.ndarray)):
        raise ValidationError("RGB must be a list, tuple, or array")
    if len(rgb) != 3:
        raise ValidationError("RGB must have exactly 3 values")
    if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in rgb):
        raise ValidationError("RGB values must be numeric")
    if not all(0 <= x <= 255 for x in rgb):
        raise ValidationError("RGB values must be in range [0, 255]")
    return True

def validate_ral_code(ral_code):
    """Validate RAL code format"""
    if not isinstance(ral_code, str):
        raise ValidationError("RAL code must be a string")
    if not ral_code.startswith("RAL_"):
        raise ValidationError("RAL code must start with 'RAL_'")
    return True

def validate_tsm_mode(mode):
    """Validate TSM mode"""
    valid_modes = ['adaptive', 'all', 'best']
    if mode not in valid_modes:
        raise ValidationError(f"TSM mode must be one of: {', '.join(valid_modes)}")
    return True

# Error handlers
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({
        'success': False,
        'error': str(e),
        'error_type': 'validation_error'
    }), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        'success': False,
        'error': 'Resource not found',
        'error_type': 'not_found'
    }), 404

@app.errorhandler(500)
def handle_internal_error(e):
    app.logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'error_type': 'internal_error'
    }), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large (max 50MB)',
        'error_type': 'file_too_large'
    }), 413

# ============================================================================
# API ROUTES - TSM ENHANCED
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/tsm')
def tsm_dashboard():
    """TSM Dashboard page"""
    return render_template('tsm_dashboard.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tsm_enabled': True,
        'workers_available': len(WorkerFactory.create_all_workers()),
        'performance_records': len(performance_tracker.records)
    })

# ============================================================================
# TSM INFORMATION & STATUS ENDPOINTS
# ============================================================================

@app.route('/api/tsm/info', methods=['GET'])
@cache.cached(timeout=300)
def get_tsm_info():
    """
    Get information about TSM system and available workers.

    Returns:
        JSON with TSM configuration and worker details
    """
    try:
        worker_info = WorkerFactory.get_worker_info()
        stats = performance_tracker.get_all_statistics()

        return jsonify({
            'success': True,
            'tsm_config': {
                'mode': app.config['TSM_MODE'],
                'ensemble_blend_enabled': app.config['TSM_ENSEMBLE_BLEND'],
                'max_parallel_workers': 5
            },
            'workers': worker_info,
            'worker_statistics': {
                worker_id: {
                    'weight': stat.current_weight,
                    'executions': stat.total_executions,
                    'avg_delta_e': stat.average_delta_e,
                    'reliability': stat.reliability_score,
                    'trend': stat.recent_trend
                }
                for worker_id, stat in stats.items()
            },
            'total_performance_records': len(performance_tracker.records)
        })

    except Exception as e:
        app.logger.error(f"Failed to get TSM info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tsm/performance', methods=['GET'])
def get_tsm_performance():
    """
    Get detailed TSM performance report.

    Returns:
        JSON with performance metrics and worker rankings
    """
    try:
        report_text = performance_tracker.get_performance_report()

        return jsonify({
            'success': True,
            'performance_report': report_text,
            'statistics': {
                worker_id: {
                    'worker_id': stat.worker_id,
                    'total_executions': stat.total_executions,
                    'successful_executions': stat.successful_executions,
                    'average_delta_e': stat.average_delta_e,
                    'average_processing_time': stat.average_processing_time,
                    'reliability_score': stat.reliability_score,
                    'current_weight': stat.current_weight,
                    'specialties_performance': stat.specialties_performance,
                    'recent_trend': stat.recent_trend
                }
                for worker_id, stat in performance_tracker.get_all_statistics().items()
            }
        })

    except Exception as e:
        app.logger.error(f"Failed to get performance report: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# PALETTE ENDPOINTS (unchanged from async version)
# ============================================================================

@app.route('/api/palette', methods=['GET'])
@cache.cached(timeout=600, query_string=True)
def get_palette_route():
    """Get RAL color palette with optional filtering"""
    try:
        search_query = request.args.get('search', '').strip()

        if search_query:
            colors = palette.search(search_query)
        else:
            colors = palette.colors

        return jsonify({
            'success': True,
            'colors': colors,
            'total': len(colors)
        })

    except Exception as e:
        app.logger.error(f"Failed to get palette: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/palette/<ral_code>', methods=['GET'])
@cache.cached(timeout=600)
def get_color_by_code_route(ral_code):
    """Get specific RAL color by code"""
    try:
        validate_ral_code(ral_code)

        color = palette.get_color_by_code(ral_code)

        if not color:
            return jsonify({
                'success': False,
                'error': f'RAL code {ral_code} not found'
            }), 404

        return jsonify({
            'success': True,
            'color': color
        })

    except ValidationError as e:
        return handle_validation_error(e)
    except Exception as e:
        app.logger.error(f"Failed to get color: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# TSM COLOR TRANSFER ENDPOINTS
# ============================================================================

@app.route('/api/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_image():
    """Upload image for TSM processing"""
    try:
        if 'image' not in request.files:
            raise ValidationError("No image file provided")

        file = request.files['image']

        if file.filename == '':
            raise ValidationError("Empty filename")

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Save file
        filename = secure_filename(f"{job_id}.png")
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename

        # Validate image
        try:
            img = Image.open(file.stream)
            img.verify()
            file.stream.seek(0)  # Reset stream after verify
            img = Image.open(file.stream)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img.save(filepath, 'PNG')

        except Exception as e:
            raise ValidationError(f"Invalid image file: {str(e)}")

        app.logger.info(f"Image uploaded: {job_id}")

        return jsonify({
            'success': True,
            'job_id': job_id
        })

    except ValidationError as e:
        return handle_validation_error(e)
    except Exception as e:
        app.logger.error(f"Upload failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/process/tsm/sync', methods=['POST'])
@limiter.limit("5 per minute")
def process_tsm_sync():
    """
    Process color transfer with TSM synchronously.

    Request JSON:
        {
            "job_id": "uuid",
            "target_ral_code": "RAL_3020",
            "tsm_mode": "adaptive",  # optional: adaptive, all, best
            "use_ensemble_blend": false  # optional
        }

    Returns:
        JSON with result job_id, QC report, and TSM execution summary
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No JSON data provided")

        job_id = data.get('job_id')
        target_ral_code = data.get('target_ral_code')
        tsm_mode = data.get('tsm_mode', app.config['TSM_MODE'])
        use_ensemble_blend = data.get('use_ensemble_blend', app.config['TSM_ENSEMBLE_BLEND'])

        # Validation
        if not job_id:
            raise ValidationError("job_id is required")
        if not target_ral_code:
            raise ValidationError("target_ral_code is required")

        validate_ral_code(target_ral_code)
        validate_tsm_mode(tsm_mode)

        # Check if upload exists
        upload_path = Path(app.config['UPLOAD_FOLDER']) / f"{job_id}.png"
        if not upload_path.exists():
            raise ValidationError(f"Job ID {job_id} not found")

        # Load source image
        source_img = cv2.imread(str(upload_path))
        source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

        # Get target color
        ral_color = palette.get_color_by_code(target_ral_code)
        if not ral_color:
            raise ValidationError(f"RAL code {target_ral_code} not found")

        target_rgb = np.array(ral_color['rgb'], dtype=np.uint8).reshape(1, 1, 3)

        # Process with TSM
        app.logger.info(f"Processing {job_id} with TSM (mode: {tsm_mode})")

        ensemble_result: EnsembleResult = tsm_orchestrator.process(
            source_rgb=source_rgb,
            target_rgb=target_rgb,
            target_ral_code=target_ral_code,
            mode=tsm_mode,
            use_ensemble_blend=use_ensemble_blend
        )

        # Save results
        result_job_id = f"{job_id}_result"
        result_path = Path(app.config['RESULTS_FOLDER']) / f"{result_job_id}.png"

        result_bgr = cv2.cvtColor(ensemble_result.best_result_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_path), result_bgr)

        # Save ensemble blend if created
        ensemble_job_id = None
        if ensemble_result.ensemble_rgb is not None:
            ensemble_job_id = f"{job_id}_ensemble"
            ensemble_path = Path(app.config['RESULTS_FOLDER']) / f"{ensemble_job_id}.png"
            ensemble_bgr = cv2.cvtColor(ensemble_result.ensemble_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(ensemble_path), ensemble_bgr)

        app.logger.info(f"TSM processing complete: {result_job_id}")

        return jsonify({
            'success': True,
            'result_job_id': result_job_id,
            'ensemble_job_id': ensemble_job_id,
            'qc_report': ensemble_result.qc_report,
            'tsm_summary': ensemble_result.execution_summary,
            'complexity_report': {
                'overall_complexity': ensemble_result.complexity_report.overall_complexity,
                'image_type': ensemble_result.complexity_report.image_characteristics['type'],
                'metrics': ensemble_result.complexity_report.metrics,
                'recommended_workers': ensemble_result.complexity_report.recommended_workers
            }
        })

    except ValidationError as e:
        return handle_validation_error(e)
    except Exception as e:
        app.logger.error(f"TSM processing failed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/process/tsm/async', methods=['POST'])
@limiter.limit("5 per minute")
def process_tsm_async():
    """
    Submit TSM color transfer for asynchronous processing.

    Request JSON:
        {
            "job_id": "uuid",
            "target_ral_code": "RAL_3020",
            "downsample": false,
            "tsm_mode": "adaptive",
            "use_ensemble_blend": false
        }

    Returns:
        JSON with task_id for status polling
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No JSON data provided")

        job_id = data.get('job_id')
        target_ral_code = data.get('target_ral_code')
        downsample = data.get('downsample', False)
        tsm_mode = data.get('tsm_mode', app.config['TSM_MODE'])
        use_ensemble_blend = data.get('use_ensemble_blend', app.config['TSM_ENSEMBLE_BLEND'])

        # Validation
        if not job_id:
            raise ValidationError("job_id is required")
        if not target_ral_code:
            raise ValidationError("target_ral_code is required")

        validate_ral_code(target_ral_code)
        validate_tsm_mode(tsm_mode)

        # Check if upload exists
        upload_path = Path(app.config['UPLOAD_FOLDER']) / f"{job_id}.png"
        if not upload_path.exists():
            raise ValidationError(f"Job ID {job_id} not found")

        # Submit task
        task = process_tsm_transfer_celery.apply_async(
            args=[job_id, target_ral_code, downsample, tsm_mode, use_ensemble_blend]
        )

        app.logger.info(f"TSM task submitted: {task.id} for job {job_id}")

        return jsonify({
            'success': True,
            'task_id': task.id,
            'status_url': f'/api/task/{task.id}'
        }), 202

    except ValidationError as e:
        return handle_validation_error(e)
    except Exception as e:
        app.logger.error(f"Failed to submit TSM task: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def task_status(task_id):
    """Check status of async task"""
    try:
        task = process_tsm_transfer_celery.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'state': 'PENDING',
                'status': 'Task is waiting...'
            }
        elif task.state == 'STARTED':
            response = {
                'state': 'STARTED',
                'progress': task.info.get('progress', 0),
                'status': task.info.get('status', 'Processing...')
            }
        elif task.state == 'SUCCESS':
            result = task.result
            response = {
                'state': 'SUCCESS',
                'result': result
            }
        elif task.state == 'FAILURE':
            response = {
                'state': 'FAILURE',
                'error': str(task.info)
            }
        else:
            response = {
                'state': task.state,
                'status': 'Unknown state'
            }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Failed to get task status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download/<job_id>', methods=['GET'])
def download_result(job_id):
    """Download processed image"""
    try:
        result_path = Path(app.config['RESULTS_FOLDER']) / f"{job_id}.png"

        if not result_path.exists():
            return jsonify({
                'success': False,
                'error': 'Result not found'
            }), 404

        return send_file(
            result_path,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{job_id}.png"
        )

    except Exception as e:
        app.logger.error(f"Download failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# SWAGGER UI SETUP
# ============================================================================

SWAGGER_URL = '/api/docs'
API_URL = '/api/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "RAL Color Transfer API - TSM Enhanced",
        'defaultModelsExpandDepth': -1,
        'displayRequestDuration': True
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/api/swagger.json')
def swagger_spec_route():
    """Serve OpenAPI specification"""
    return jsonify(openapi_spec)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    app.logger.info("=" * 60)
    app.logger.info("TSM Color Transfer Application Starting")
    app.logger.info("=" * 60)
    app.logger.info(f"TSM Mode: {app.config['TSM_MODE']}")
    app.logger.info(f"Ensemble Blend: {app.config['TSM_ENSEMBLE_BLEND']}")
    app.logger.info(f"Workers Available: {len(WorkerFactory.create_all_workers())}")
    app.logger.info(f"Performance Records: {len(performance_tracker.records)}")
    app.logger.info("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
