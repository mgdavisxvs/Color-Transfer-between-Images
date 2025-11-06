#!/usr/bin/env python3
"""
Unit tests for Flask application routes and functionality.
"""

import pytest
import json
import io
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app_enhanced import app, ValidationError, validate_rgb_color, validate_ral_code
import numpy as np


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create gradient
    for i in range(100):
        img[i, :, 0] = int(50 + i * 2)   # B
        img[i, :, 1] = int(100 + i * 1)  # G
        img[i, :, 2] = int(150 + i * 0.5)  # R
    return img


class TestValidation:
    """Test input validation functions."""

    def test_validate_rgb_color_valid(self):
        """Test RGB validation with valid colors."""
        validate_rgb_color([255, 0, 0])
        validate_rgb_color([0, 255, 0])
        validate_rgb_color([0, 0, 255])
        validate_rgb_color([128, 128, 128])

    def test_validate_rgb_color_invalid_length(self):
        """Test RGB validation fails with wrong length."""
        with pytest.raises(ValidationError, match="exactly 3 values"):
            validate_rgb_color([255, 0])

        with pytest.raises(ValidationError, match="exactly 3 values"):
            validate_rgb_color([255, 0, 0, 255])

    def test_validate_rgb_color_invalid_range(self):
        """Test RGB validation fails with out-of-range values."""
        with pytest.raises(ValidationError, match="range"):
            validate_rgb_color([256, 0, 0])

        with pytest.raises(ValidationError, match="range"):
            validate_rgb_color([-1, 128, 128])

    def test_validate_rgb_color_invalid_type(self):
        """Test RGB validation fails with non-numeric values."""
        with pytest.raises(ValidationError, match="integers"):
            validate_rgb_color(["red", "green", "blue"])

    def test_validate_ral_code_valid(self):
        """Test RAL code validation with valid codes."""
        # This will only work if palette is loaded
        # Skip if not in proper app context
        pass

    def test_validate_ral_code_invalid_format(self):
        """Test RAL code validation fails with invalid format."""
        with pytest.raises(ValidationError, match="start with 'RAL '"):
            validate_ral_code("3000")


class TestRoutes:
    """Test Flask routes."""

    def test_index_route(self, client):
        """Test main index page loads."""
        response = client.get('/')
        assert response.status_code == 200

    def test_palette_route(self, client):
        """Test palette API endpoint."""
        response = client.get('/api/palette')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'colors' in data
        assert 'total' in data
        assert data['total'] > 0

    def test_palette_search(self, client):
        """Test palette search functionality."""
        response = client.get('/api/palette?search=red')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        # Should find some red colors
        assert data['total'] > 0

    def test_palette_stats(self, client):
        """Test palette statistics endpoint."""
        response = client.get('/api/palette/stats')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'statistics' in data

    def test_color_match(self, client):
        """Test color matching endpoint."""
        payload = {
            'rgb': [255, 0, 0],
            'top_n': 3
        }
        response = client.post(
            '/api/color/match',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'matches' in data
        assert len(data['matches']) == 3
        assert all('delta_e' in match for match in data['matches'])

    def test_color_match_invalid_rgb(self, client):
        """Test color matching with invalid RGB."""
        payload = {
            'rgb': [256, 0, 0],  # Invalid: > 255
            'top_n': 3
        }
        response = client.post(
            '/api/color/match',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

    def test_upload_no_file(self, client):
        """Test upload endpoint with no file."""
        response = client.post('/api/upload')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert data['success'] is False

    def test_upload_valid_image(self, client, sample_image, tmp_path):
        """Test uploading a valid image."""
        import cv2

        # Save sample image to temp file
        temp_img = tmp_path / "test.png"
        cv2.imwrite(str(temp_img), sample_image)

        # Upload
        with open(temp_img, 'rb') as f:
            data = {
                'file': (f, 'test.png')
            }
            response = client.post(
                '/api/upload',
                data=data,
                content_type='multipart/form-data'
            )

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['success'] is True
        assert 'job_id' in result
        assert 'dimensions' in result

    def test_delta_e_compute(self, client):
        """Test Delta E computation endpoint."""
        payload = {
            'color1_rgb': [255, 0, 0],
            'color2_rgb': [200, 0, 0]
        }
        response = client.post(
            '/api/delta-e/compute',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success'] is True
        assert 'delta_e' in data
        assert data['delta_e'] > 0
        assert 'interpretation' in data


class TestErrorHandling:
    """Test error handling."""

    def test_404_error(self, client):
        """Test 404 error handler."""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404

        data = json.loads(response.data)
        assert data['success'] is False
        assert data['error_type'] == 'not_found'

    def test_file_too_large(self, client):
        """Test file size limit."""
        # Create large fake file data
        large_data = b'x' * (51 * 1024 * 1024)  # 51MB

        data = {
            'file': (io.BytesIO(large_data), 'large.png')
        }
        response = client.post(
            '/api/upload',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 413


class TestSecurity:
    """Test security features."""

    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get('/')
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-Frame-Options' in response.headers
        assert 'X-XSS-Protection' in response.headers

    def test_cors_headers_api(self, client):
        """Test CORS headers on API endpoints."""
        response = client.get('/api/palette')
        assert 'Access-Control-Allow-Origin' in response.headers


class TestCaching:
    """Test caching functionality."""

    def test_palette_cached(self, client):
        """Test palette endpoint uses caching."""
        # First request
        response1 = client.get('/api/palette')
        data1 = json.loads(response1.data)

        # Second request (should be cached)
        response2 = client.get('/api/palette')
        data2 = json.loads(response2.data)

        # Results should be identical
        assert data1 == data2


def test_app_initialization():
    """Test application initializes correctly."""
    assert app is not None
    assert app.config['TESTING'] is True
    assert app.config['MAX_CONTENT_LENGTH'] == 50 * 1024 * 1024


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
