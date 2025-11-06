"""
Celery Configuration for Asynchronous Task Processing

Handles background job execution for long-running operations like
color transfer processing.
"""

import os
from celery import Celery


def make_celery(app):
    """
    Create and configure Celery instance with Flask app context.

    Args:
        app: Flask application instance

    Returns:
        Configured Celery instance
    """
    # Get Redis URL from environment or use default
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

    celery = Celery(
        app.import_name,
        broker=redis_url,
        backend=redis_url
    )

    # Update celery config from Flask config
    celery.conf.update(
        broker_url=redis_url,
        result_backend=redis_url,
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,  # 5 minutes hard limit
        task_soft_time_limit=240,  # 4 minutes soft limit
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=100,
    )

    # Subclass task to run within Flask app context
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    return celery


# Celery beat schedule for periodic tasks (optional)
CELERY_BEAT_SCHEDULE = {
    'cleanup-old-files': {
        'task': 'tasks.cleanup_old_files',
        'schedule': 3600.0,  # Run every hour
    },
}
