#!/bin/bash
#
# Startup script for Flask Color Transfer Application (Enhanced Version)
#

echo "=================================================="
echo "Flask Color Transfer - Enhanced Production Version"
echo "=================================================="
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
echo "Installing dependencies..."
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

# Run tests (optional)
if [ "$1" == "--test" ]; then
    echo ""
    echo "Running tests..."
    pytest tests/ -v
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Tests failed! Fix errors before starting server."
        exit 1
    fi
    echo "  ✓ All tests passed"
fi

# Set environment variables
export FLASK_APP=app_enhanced.py
export FLASK_ENV=${FLASK_ENV:-development}
export SECRET_KEY=${SECRET_KEY:-dev-secret-key-change-in-production}

echo ""
echo "Environment:"
echo "  FLASK_APP: $FLASK_APP"
echo "  FLASK_ENV: $FLASK_ENV"
echo "  SECRET_KEY: ${SECRET_KEY:0:20}..."

# Start server
echo ""
echo "=================================================="
echo "Starting Flask server..."
echo "=================================================="
echo ""
echo "Features enabled:"
echo "  ✓ Comprehensive logging (logs/app.log, logs/error.log)"
echo "  ✓ CSRF protection"
echo "  ✓ Rate limiting (200/day, 50/hour)"
echo "  ✓ Input validation"
echo "  ✓ Caching (5 min default)"
echo "  ✓ Security headers"
echo "  ✓ Global error handling"
echo ""
echo "🚀 Server will start on http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

python3 app_enhanced.py
