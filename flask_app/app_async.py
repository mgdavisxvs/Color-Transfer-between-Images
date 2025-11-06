#!/usr/bin/env python3
"""
Flask Color Transfer Web Application - Async Version

Production-ready Flask application with:
- Comprehensive Swagger/OpenAPI documentation
- Asynchronous task processing with Celery
- All security and logging features from enhanced version
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

# Import Celery configuration
from celery_config import make_celery
from tasks import process_color_transfer_task, cleanup_old_files_task

# Import application modules
from palette_manager import get_palette
from color_transfer_engine import ColorTransferEngine, QualityControl
from color_utils import (
    rgb_to_lab, delta_e_ciede2000,
    downsample_image, interpret_delta_e
)
from swagger_spec import openapi_spec

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['LOG_FOLDER'] = 'logs'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

# Celery configuration
app.config['CELERY_BROKER_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)
Path(app.config['LOG_FOLDER']).mkdir(exist_ok=True)

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

    file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    error_handler = RotatingFileHandler(
        log_dir / 'error.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.DEBUG if app.debug else logging.INFO)

    app.logger.setLevel(logging.DEBUG if app.debug else logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(error_handler)
    app.logger.addHandler(console_handler)

    if not app.debug:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    app.logger.info("=" * 80)
    app.logger.info("Application starting with async support...")
    app.logger.info(f"Environment: {'Development' if app.debug else 'Production'}")
    app.logger.info("=" * 80)

setup_logging()

# ============================================================================
# SECURITY & PERFORMANCE COMPONENTS
# ============================================================================

csrf = CSRFProtect(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# ============================================================================
# CELERY INITIALIZATION
# ============================================================================

celery = make_celery(app)

# Define Celery tasks with decorator
@celery.task(bind=True, name='tasks.process_color_transfer')
def process_color_transfer_celery(self, job_id, target_ral_code, downsample=False):
    """Celery task wrapper for color transfer processing."""
    return process_color_transfer_task(
        self, job_id, target_ral_code, downsample,
        app_config=app.config
    )

@celery.task(name='tasks.cleanup_old_files')
def cleanup_old_files_celery():
    """Celery task for periodic cleanup."""
    return cleanup_old_files_task(app_config=app.config)

# ============================================================================
# SWAGGER UI CONFIGURATION
# ============================================================================

SWAGGER_URL = '/api/docs'
API_URL = '/api/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "RAL Color Transfer API",
        'layout': "BaseLayout",
        'deepLinking': True,
        'displayRequestDuration': True,
        'filter': True,
        'showExtensions': True,
        'showCommonExtensions': True
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/api/swagger.json')
def swagger_spec_route():
    """Serve OpenAPI specification."""
    return jsonify(openapi_spec)

# ============================================================================
# GLOBAL VARIABLES & VALIDATION
# ============================================================================

palette = None
engine = None
qc = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

class ValidationError(Exception):
    """Custom validation error."""
    pass

def validate_rgb_color(rgb):
    """Validate RGB color values."""
    if not isinstance(rgb, (list, tuple, np.ndarray)):
        raise ValidationError("RGB must be a list, tuple, or array")
    if len(rgb) != 3:
        raise ValidationError("RGB must have exactly 3 values")
    try:
        r, g, b = [int(x) for x in rgb]
    except (ValueError, TypeError):
        raise ValidationError("RGB values must be integers")
    if not all(0 <= x <= 255 for x in [r, g, b]):
        raise ValidationError("RGB values must be in range [0, 255]")

def validate_ral_code(code):
    """Validate RAL color code."""
    if not code or not isinstance(code, str):
        raise ValidationError("RAL code must be a non-empty string")
    if not code.startswith("RAL "):
        raise ValidationError("RAL code must start with 'RAL '")
    color = palette.get_color_by_code(code)
    if not color:
        raise ValidationError(f"RAL code '{code}' not found in palette")

# ============================================================================
# REQUEST/RESPONSE HOOKS
# ============================================================================

@app.before_request
def before_request():
    """Initialize components and log request."""
    global palette, engine, qc

    if palette is None:
        try:
            app.logger.info("Initializing application components...")
            palette = get_palette()
            engine = ColorTransferEngine(downsample_max=2048)
            qc = QualityControl(delta_e_threshold=5.0, acceptance_percentage=95.0)
            app.logger.info(f"✓ Loaded {len(palette.colors)} RAL colors")
        except Exception as e:
            app.logger.error(f"Failed to initialize components: {e}")
            raise

    if not request.path.startswith('/static') and not request.path.startswith('/api/docs'):
        app.logger.info(
            f"{request.method} {request.path} - IP: {request.remote_addr}"
        )

@app.after_request
def after_request(response):
    """Add security headers."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

    if request.path.startswith('/api/'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRFToken'

    return response

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    app.logger.warning(f"Validation error: {str(e)}")
    return jsonify({
        'success': False,
        'error': str(e),
        'error_type': 'validation_error'
    }), 400

@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Resource not found',
            'error_type': 'not_found'
        }), 404
    return render_template('404.html'), 404

@app.errorhandler(413)
def handle_file_too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large (max 50MB)',
        'error_type': 'file_too_large'
    }), 413

@app.errorhandler(429)
def handle_rate_limit(e):
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded. Please try again later.',
        'error_type': 'rate_limit'
    }), 429

@app.errorhandler(500)
def handle_internal_error(e):
    app.logger.exception("Internal server error:")
    return jsonify({
        'success': False,
        'error': 'An internal server error occurred. Please try again later.',
        'error_type': 'internal_error'
    }), 500

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unexpected error occurred:")
    error_msg = str(e) if app.debug else "An unexpected error occurred"
    return jsonify({
        'success': False,
        'error': error_msg,
        'error_type': 'unexpected_error'
    }), 500

# ============================================================================
# ROUTES - MAIN & DOCUMENTATION
# ============================================================================

@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint."""
    celery_ok = False
    try:
        # Check if Celery is responsive
        celery.control.inspect().active()
        celery_ok = True
    except:
        pass

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'palette_loaded': palette is not None,
        'colors_count': len(palette.colors) if palette else 0,
        'celery_connected': celery_ok
    })

# ============================================================================
# ROUTES - ASYNC PROCESSING
# ============================================================================

@app.route('/api/process/async', methods=['POST'])
@limiter.limit("5 per minute")
def process_async():
    """
    Submit color transfer job for asynchronous processing.

    Returns task_id for status checking.
    """
    try:
        data = request.json
        if not data:
            raise ValidationError("No JSON data provided")

        job_id = data.get('job_id')
        target_ral_code = data.get('target_ral_code')
        downsample = data.get('downsample', False)

        if not job_id:
            raise ValidationError("Missing job_id")
        if not target_ral_code:
            raise ValidationError("Missing target_ral_code")

        validate_ral_code(target_ral_code)

        # Submit task to Celery
        task = process_color_transfer_celery.apply_async(
            args=[job_id, target_ral_code, downsample]
        )

        app.logger.info(f"Async task submitted: {task.id} for {job_id} -> {target_ral_code}")

        return jsonify({
            'success': True,
            'task_id': task.id,
            'status_url': f'/api/task/{task.id}',
            'message': 'Task accepted for processing'
        }), 202

    except ValidationError:
        raise
    except Exception as e:
        app.logger.error(f"Error submitting async task: {e}")
        raise

@app.route('/api/task/<task_id>', methods=['GET'])
@limiter.limit("30 per minute")
def task_status(task_id):
    """
    Get status of an asynchronous task.
    """
    try:
        task = process_color_transfer_celery.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'success': True,
                'task_id': task_id,
                'state': 'PENDING',
                'progress': 0,
                'status': 'Task is waiting to be processed...'
            }
        elif task.state == 'STARTED':
            response = {
                'success': True,
                'task_id': task_id,
                'state': 'STARTED',
                'progress': task.info.get('progress', 0),
                'status': task.info.get('status', 'Processing...')
            }
        elif task.state == 'SUCCESS':
            result = task.result
            response = {
                'success': True,
                'task_id': task_id,
                'state': 'SUCCESS',
                'progress': 100,
                'result': result
            }
        elif task.state == 'FAILURE':
            response = {
                'success': False,
                'task_id': task_id,
                'state': 'FAILURE',
                'error': str(task.info),
                'error_type': 'task_failure'
            }
        else:
            response = {
                'success': True,
                'task_id': task_id,
                'state': task.state,
                'status': f'Task state: {task.state}'
            }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error getting task status: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get task status',
            'error_type': 'task_error'
        }), 500

# ============================================================================
# Import remaining routes from enhanced version
# (palette, upload, color matching, download, etc.)
# For brevity, these would be imported or included here
# ============================================================================

# ... (All routes from app_enhanced.py would be included here)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Flask Color Transfer Application - Async Version")
    print("=" * 80)
    print("Features:")
    print("  ✓ Comprehensive Swagger/OpenAPI documentation")
    print("  ✓ Interactive API docs at /api/docs")
    print("  ✓ Asynchronous processing with Celery")
    print("  ✓ Background task management")
    print("  ✓ Comprehensive logging")
    print("  ✓ CSRF protection")
    print("  ✓ Rate limiting")
    print("  ✓ Input validation")
    print("  ✓ Caching")
    print("  ✓ Security headers")
    print("=" * 80)
    print(f"\n🚀 Server starting on http://localhost:5000")
    print(f"📚 API Documentation: http://localhost:5000/api/docs")
    print(f"\n⚠️  Note: Start Redis and Celery worker for async processing:")
    print(f"   redis-server")
    print(f"   celery -A app_async.celery worker --loglevel=info")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
