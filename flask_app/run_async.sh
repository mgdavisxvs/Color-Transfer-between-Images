#!/bin/bash
#
# Startup script for Flask Color Transfer Application (Async Version)
#

echo "======================================================"
echo "Flask Color Transfer - Async + Swagger Documentation"
echo "======================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Activated"

# Install/update dependencies
echo ""
echo "Installing dependencies (including Celery, Redis, Swagger)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  ✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p uploads results logs data
echo "  ✓ Directories created"

# Check if RAL palette exists
if [ ! -f "data/ral.json" ]; then
    echo ""
    echo "⚠ WARNING: data/ral.json not found!"
    echo "  Please ensure RAL palette data is present."
    exit 1
fi

# Check if Redis is installed
echo ""
echo "Checking Redis availability..."
if ! command -v redis-cli &> /dev/null; then
    echo "  ⚠ WARNING: Redis not found!"
    echo "    Async processing requires Redis. Install with:"
    echo "    - Ubuntu/Debian: sudo apt-get install redis-server"
    echo "    - macOS: brew install redis"
    echo "    - Docker: docker run -d -p 6379:6379 redis:alpine"
    echo ""
    echo "  For sync-only mode, use: python3 app_enhanced.py"
    exit 1
fi

# Check if Redis is running
if ! redis-cli ping &> /dev/null; then
    echo "  ⚠ WARNING: Redis is not running!"
    echo "    Start Redis with:"
    echo "    - systemd: sudo systemctl start redis"
    echo "    - Direct: redis-server"
    echo "    - Docker: docker run -d -p 6379:6379 redis:alpine"
    exit 1
fi
echo "  ✓ Redis is running"

# Set environment variables
export FLASK_APP=app_async.py
export FLASK_ENV=${FLASK_ENV:-development}
export SECRET_KEY=${SECRET_KEY:-dev-secret-key-change-in-production}
export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}

echo ""
echo "Environment:"
echo "  FLASK_APP: $FLASK_APP"
echo "  FLASK_ENV: $FLASK_ENV"
echo "  SECRET_KEY: ${SECRET_KEY:0:20}..."
echo "  REDIS_URL: $REDIS_URL"

# Check if Celery worker should be started
if [ "$1" == "--with-worker" ]; then
    echo ""
    echo "======================================================"
    echo "Starting Celery Worker in background..."
    echo "======================================================"
    celery -A app_async.celery worker --loglevel=info &
    CELERY_PID=$!
    echo "  ✓ Worker started (PID: $CELERY_PID)"
    echo ""
    echo "Note: Worker logs will appear below Flask logs."
    echo "      To stop worker: kill $CELERY_PID"
    sleep 2
fi

# Start server
echo ""
echo "======================================================"
echo "Starting Flask server..."
echo "======================================================"
echo ""
echo "Features enabled:"
echo "  ✓ Comprehensive logging (logs/app.log, logs/error.log)"
echo "  ✓ CSRF protection"
echo "  ✓ Rate limiting (200/day, 50/hour)"
echo "  ✓ Input validation"
echo "  ✓ Caching (5 min default)"
echo "  ✓ Security headers"
echo "  ✓ Global error handling"
echo "  ✓ Asynchronous processing (Celery + Redis)"
echo "  ✓ API Documentation (Swagger UI)"
echo ""
echo "URLs:"
echo "  🌐 Application:  http://localhost:5000"
echo "  📖 API Docs:     http://localhost:5000/api/docs"
echo "  📋 OpenAPI Spec: http://localhost:5000/api/swagger.json"
echo "  💚 Health Check: http://localhost:5000/health"
echo ""
echo "To start Celery worker separately:"
echo "  celery -A app_async.celery worker --loglevel=info"
echo ""
echo "To start Flower (monitoring UI):"
echo "  celery -A app_async.celery flower"
echo "  Then visit: http://localhost:5555"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================"
echo ""

python3 app_async.py
