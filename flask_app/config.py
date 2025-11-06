"""
Flask Application Configuration
"""

import os


class Config:
    """Base configuration."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False

    # File Upload
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

    # Color Transfer
    DOWNSAMPLE_MAX = 2048
    DELTA_E_THRESHOLD = 5.0
    ACCEPTANCE_PERCENTAGE = 95.0

    # Palette
    PALETTE_FILE = 'data/ral.json'


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    # In production, always set SECRET_KEY via environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY')


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    DELTA_E_THRESHOLD = 10.0  # More lenient for tests


# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
