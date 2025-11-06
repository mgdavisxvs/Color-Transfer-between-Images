"""
Celery Background Tasks

Defines asynchronous tasks for long-running operations.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import cv2

from celery_config import make_celery
from palette_manager import get_palette
from color_transfer_engine import ColorTransferEngine, QualityControl
from color_utils import downsample_image


# Note: celery instance will be created in app_async.py
# This is a placeholder for task definitions


def process_color_transfer_task(celery_instance, job_id, target_ral_code, downsample=False, app_config=None):
    """
    Background task for color transfer processing.

    This function will be decorated with @celery.task in the main app.

    Args:
        job_id: Source image job ID
        target_ral_code: Target RAL color code
        downsample: Whether to downsample image
        app_config: Flask app configuration dict

    Returns:
        dict: Processing results
    """
    # Import here to avoid circular imports
    from app_async import app

    with app.app_context():
        try:
            # Update task state to STARTED
            celery_instance.update_state(
                state='STARTED',
                meta={'progress': 10, 'status': 'Loading image...'}
            )

            # Initialize components
            palette = get_palette()
            engine = ColorTransferEngine(downsample_max=2048)
            qc = QualityControl(delta_e_threshold=5.0, acceptance_percentage=95.0)

            # Find source image
            upload_folder = app_config.get('UPLOAD_FOLDER', 'uploads')
            source_files = list(Path(upload_folder).glob(f"{job_id}.*"))

            if not source_files:
                raise FileNotFoundError(f"Source image not found: {job_id}")

            celery_instance.update_state(
                state='STARTED',
                meta={'progress': 30, 'status': 'Processing color transfer...'}
            )

            source_path = source_files[0]
            source_img = cv2.imread(str(source_path))
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

            if downsample:
                source_img = downsample_image(source_img, max_dimension=1024)

            # Perform color transfer
            result_img, info = engine.transfer_to_ral_color(
                source_img, target_ral_code, method='reinhard'
            )

            celery_instance.update_state(
                state='STARTED',
                meta={'progress': 60, 'status': 'Running quality control...'}
            )

            # Run QC
            ral_color = palette.get_color_by_code(target_ral_code)
            target_rgb = np.array(ral_color['rgb'], dtype=np.uint8)
            qc_report = qc.evaluate(source_img, result_img, target_rgb)

            celery_instance.update_state(
                state='STARTED',
                meta={'progress': 80, 'status': 'Saving results...'}
            )

            # Generate result job ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            import uuid
            result_job_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"

            # Save result
            results_folder = app_config.get('RESULTS_FOLDER', 'results')
            result_filename = f"{result_job_id}.png"
            result_path = os.path.join(results_folder, result_filename)

            result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_path, result_img_bgr)

            # Save QC report
            qc_path = os.path.join(results_folder, f"{result_job_id}_qc.json")
            with open(qc_path, 'w') as f:
                qc_json = {k: v for k, v in qc_report.items() if k != 'delta_e_map'}
                json.dump(qc_json, f, indent=2)

            # Generate CSV report
            qc_csv_path = os.path.join(results_folder, f"{result_job_id}_qc.csv")
            qc.generate_csv_report(qc_report, qc_csv_path)

            # Save heatmap
            heatmap_path = os.path.join(results_folder, f"{result_job_id}_heatmap.png")
            heatmap_normalized = (qc_report['delta_e_map'] / qc_report['delta_e_map'].max() * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            cv2.imwrite(heatmap_path, heatmap_colored)

            celery_instance.update_state(
                state='STARTED',
                meta={'progress': 100, 'status': 'Complete!'}
            )

            # Return results
            return {
                'success': True,
                'result_job_id': result_job_id,
                'ral_info': info,
                'qc_report': {k: v for k, v in qc_report.items() if k != 'delta_e_map'},
                'downloads': {
                    'result_image': f"/api/download/{result_job_id}.png",
                    'qc_json': f"/api/download/{result_job_id}_qc.json",
                    'qc_csv': f"/api/download/{result_job_id}_qc.csv",
                    'heatmap': f"/api/download/{result_job_id}_heatmap.png"
                },
                'completed_at': datetime.now().isoformat()
            }

        except Exception as e:
            # Task failed
            return {
                'success': False,
                'error': str(e),
                'error_type': 'processing_error'
            }


def cleanup_old_files_task(app_config, max_age_hours=24):
    """
    Periodic task to clean up old uploaded and result files.

    Args:
        app_config: Flask app configuration
        max_age_hours: Maximum file age in hours before deletion

    Returns:
        dict: Cleanup statistics
    """
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0

        folders = [
            app_config.get('UPLOAD_FOLDER', 'uploads'),
            app_config.get('RESULTS_FOLDER', 'results')
        ]

        for folder in folders:
            folder_path = Path(folder)
            if not folder_path.exists():
                continue

            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    # Check file modification time
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1

        return {
            'success': True,
            'deleted_count': deleted_count,
            'cutoff_time': cutoff_time.isoformat()
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
