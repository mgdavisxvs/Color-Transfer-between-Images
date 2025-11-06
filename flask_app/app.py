#!/usr/bin/env python3
"""
Flask Color Transfer Web Application

Web interface for precise color transfer using RAL palette with Delta E matching.
"""

import os
import json
import uuid
import zipfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2

from palette_manager import get_palette
from color_transfer_engine import ColorTransferEngine, QualityControl
from color_utils import (
    rgb_to_lab, delta_e_ciede2000, create_delta_e_heatmap,
    downsample_image, interpret_delta_e
)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Initialize components
palette = None
engine = None
qc = None

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_job_id():
    """Generate unique job ID."""
    return str(uuid.uuid4())


@app.before_request
def initialize_app():
    """Initialize global components on first request."""
    global palette, engine, qc

    if palette is None:
        print("Initializing application...")
        palette = get_palette()
        engine = ColorTransferEngine(downsample_max=2048)
        qc = QualityControl(delta_e_threshold=5.0, acceptance_percentage=95.0)
        print("✓ Application initialized")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')


@app.route('/api/palette', methods=['GET'])
def get_palette_data():
    """
    Get RAL palette data.

    Query parameters:
    - search: Optional search query for color names

    Returns:
        JSON with palette colors
    """
    search_query = request.args.get('search', '')

    if search_query:
        colors = palette.search_by_name(search_query)
    else:
        colors = palette.get_all_colors()

    return jsonify({
        'success': True,
        'total': len(colors),
        'colors': colors
    })


@app.route('/api/palette/stats', methods=['GET'])
def get_palette_stats():
    """Get palette statistics."""
    stats = palette.get_color_statistics()
    return jsonify({
        'success': True,
        'statistics': stats
    })


@app.route('/api/color/match', methods=['POST'])
def match_color():
    """
    Find closest RAL color match for given RGB.

    Body:
    {
        "rgb": [255, 0, 0],
        "top_n": 3
    }

    Returns:
        JSON with top N closest matches and Delta E values
    """
    data = request.json
    rgb = np.array(data.get('rgb', [128, 128, 128]), dtype=np.uint8)
    top_n = data.get('top_n', 5)

    matches = palette.find_closest_match(rgb, top_n=top_n)

    results = []
    for color, delta_e in matches:
        results.append({
            'color': color,
            'delta_e': delta_e,
            'interpretation': interpret_delta_e(delta_e)
        })

    return jsonify({
        'success': True,
        'input_rgb': rgb.tolist(),
        'matches': results
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload image file.

    Returns:
        JSON with job_id and file info
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    # Generate job ID and save file
    job_id = generate_job_id()
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    save_filename = f"{job_id}.{file_ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)

    file.save(save_path)

    # Read image to get dimensions
    img = cv2.imread(save_path)
    if img is None:
        os.remove(save_path)
        return jsonify({'success': False, 'error': 'Invalid image file'}), 400

    height, width = img.shape[:2]

    return jsonify({
        'success': True,
        'job_id': job_id,
        'filename': filename,
        'dimensions': {'width': width, 'height': height},
        'size_bytes': os.path.getsize(save_path)
    })


@app.route('/api/process/reinhard', methods=['POST'])
def process_reinhard():
    """
    Process color transfer using Reinhard method.

    Body:
    {
        "job_id": "source-job-id",
        "target_ral_code": "RAL 3000",
        "downsample": true
    }

    Returns:
        JSON with result_job_id and QC report
    """
    data = request.json
    job_id = data.get('job_id')
    target_ral_code = data.get('target_ral_code')
    downsample = data.get('downsample', False)

    if not job_id or not target_ral_code:
        return jsonify({'success': False, 'error': 'Missing required parameters'}), 400

    # Find source image
    source_files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"{job_id}.*"))
    if not source_files:
        return jsonify({'success': False, 'error': 'Source image not found'}), 404

    source_path = source_files[0]
    source_img = cv2.imread(str(source_path))
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    # Downsample if requested
    if downsample:
        source_img = downsample_image(source_img, max_dimension=1024)

    # Perform color transfer
    try:
        result_img, info = engine.transfer_to_ral_color(
            source_img, target_ral_code, method='reinhard'
        )

        # Run QC
        ral_color = palette.get_color_by_code(target_ral_code)
        target_rgb = np.array(ral_color['rgb'], dtype=np.uint8)
        qc_report = qc.evaluate(source_img, result_img, target_rgb)

        # Save result
        result_job_id = generate_job_id()
        result_filename = f"{result_job_id}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, result_img_bgr)

        # Save QC report as JSON
        qc_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_job_id}_qc.json")
        with open(qc_path, 'w') as f:
            # Remove delta_e_map for JSON serialization
            qc_json = {k: v for k, v in qc_report.items() if k != 'delta_e_map'}
            json.dump(qc_json, f, indent=2)

        # Generate CSV report
        qc_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_job_id}_qc.csv")
        qc.generate_csv_report(qc_report, qc_csv_path)

        # Save heatmap
        heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_job_id}_heatmap.png")
        heatmap_normalized = (qc_report['delta_e_map'] / qc_report['delta_e_map'].max() * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_colored)

        return jsonify({
            'success': True,
            'result_job_id': result_job_id,
            'ral_info': info,
            'qc_report': {k: v for k, v in qc_report.items() if k != 'delta_e_map'},
            'downloads': {
                'result_image': f"/api/download/{result_job_id}.png",
                'qc_json': f"/api/download/{result_job_id}_qc.json",
                'qc_csv': f"/api/download/{result_job_id}_qc.csv",
                'heatmap': f"/api/download/{result_job_id}_heatmap.png"
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process/auto-match', methods=['POST'])
def process_auto_match():
    """
    Auto-match source image colors to RAL palette.

    Body:
    {
        "job_id": "source-job-id",
        "num_colors": 5,
        "downsample": true
    }
    """
    data = request.json
    job_id = data.get('job_id')
    num_colors = data.get('num_colors', 5)
    downsample = data.get('downsample', False)

    # Find source image
    source_files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"{job_id}.*"))
    if not source_files:
        return jsonify({'success': False, 'error': 'Source image not found'}), 404

    source_path = source_files[0]
    source_img = cv2.imread(str(source_path))
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    if downsample:
        source_img = downsample_image(source_img, max_dimension=1024)

    try:
        result_img, info = engine.auto_match_to_palette(source_img, num_colors=num_colors)

        # Run QC
        qc_report = qc.evaluate(source_img, result_img)

        # Save result
        result_job_id = generate_job_id()
        result_filename = f"{result_job_id}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, result_img_bgr)

        return jsonify({
            'success': True,
            'result_job_id': result_job_id,
            'match_info': info,
            'qc_report': {k: v for k, v in qc_report.items() if k != 'delta_e_map'},
            'downloads': {
                'result_image': f"/api/download/{result_job_id}.png"
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/delta-e/compute', methods=['POST'])
def compute_delta_e():
    """
    Compute Delta E between two colors.

    Body:
    {
        "color1_rgb": [255, 0, 0],
        "color2_rgb": [0, 0, 255],
        "method": "cie2000"
    }
    """
    data = request.json
    color1 = np.array(data.get('color1_rgb', [0, 0, 0]), dtype=np.uint8)
    color2 = np.array(data.get('color2_rgb', [0, 0, 0]), dtype=np.uint8)

    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)

    delta_e = delta_e_ciede2000(lab1, lab2)

    return jsonify({
        'success': True,
        'delta_e': float(delta_e),
        'interpretation': interpret_delta_e(delta_e),
        'color1_lab': lab1.tolist(),
        'color2_lab': lab2.tolist()
    })


@app.route('/api/preview/<job_id>', methods=['GET'])
def preview_image(job_id):
    """Get preview image (downsampled)."""
    # Check uploads first
    upload_files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"{job_id}.*"))
    result_files = list(Path(app.config['RESULTS_FOLDER']).glob(f"{job_id}.*"))

    files = upload_files + result_files
    if not files:
        return jsonify({'success': False, 'error': 'Image not found'}), 404

    img_path = files[0]
    img = cv2.imread(str(img_path))
    img = downsample_image(img, max_dimension=512)

    # Encode as JPEG for preview
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    from io import BytesIO
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download result file."""
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'}), 404

    return send_file(filepath, as_attachment=True)


@app.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """
    Upload ZIP file with multiple images for batch processing.

    Returns:
        JSON with job_ids for each extracted image
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']

    if not file.filename.endswith('.zip'):
        return jsonify({'success': False, 'error': 'Must be a ZIP file'}), 400

    batch_id = generate_job_id()
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{batch_id}.zip")
    file.save(zip_path)

    job_ids = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue

                filename = os.path.basename(file_info.filename)

                if not allowed_file(filename):
                    continue

                # Extract and save
                job_id = generate_job_id()
                file_ext = filename.rsplit('.', 1)[1].lower()
                save_filename = f"{job_id}.{file_ext}"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)

                with zip_ref.open(file_info) as source, open(save_path, 'wb') as target:
                    target.write(source.read())

                job_ids.append({
                    'job_id': job_id,
                    'original_filename': filename
                })

        os.remove(zip_path)

        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'images': job_ids,
            'total': len(job_ids)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large (max 50MB)'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Flask Color Transfer Application...")
    print("Loading RAL palette...")

    # Pre-initialize
    palette = get_palette()
    engine = ColorTransferEngine()
    qc = QualityControl()

    print(f"✓ Loaded {len(palette.colors)} RAL colors")
    print("\n🚀 Server starting on http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
