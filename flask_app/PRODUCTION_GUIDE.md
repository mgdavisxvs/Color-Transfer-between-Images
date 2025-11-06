## Production-Ready Features Guide

This guide covers all production enhancements added to the Flask Color Transfer application.

---

## 🔐 Security Enhancements

### 1. CSRF Protection

**What it does:** Prevents Cross-Site Request Forgery attacks

**Implementation:**
```python
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)
```

**Usage in frontend:**
```html
<meta name="csrf-token" content="{{ csrf_token() }}">

<script>
// Add to AJAX requests
fetch('/api/upload', {
    method: 'POST',
    headers: {
        'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').content
    },
    body: formData
});
</script>
```

**Configuration:**
```python
app.config['WTF_CSRF_ENABLED'] = True  # Enabled by default
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # Token expires in 1 hour
```

---

### 2. Rate Limiting

**What it does:** Prevents abuse by limiting requests per IP address

**Default Limits:**
- Global: 200 requests per day, 50 per hour
- Upload: 10 per minute
- Processing: 5 per minute
- Color matching: 20 per minute

**Example:**
```python
@app.route('/api/upload')
@limiter.limit("10 per minute")
def upload_image():
    # Endpoint limited to 10 uploads per minute per IP
    pass
```

**Custom limits:**
```python
from flask_limiter import Limiter

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"]
)
```

**Response when limit exceeded:**
```json
{
    "success": false,
    "error": "Rate limit exceeded. Please try again later.",
    "error_type": "rate_limit"
}
```

---

### 3. Security Headers

**Automatically added headers:**

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

**What they do:**
- `X-Content-Type-Options`: Prevents MIME-sniffing attacks
- `X-Frame-Options`: Prevents clickjacking
- `X-XSS-Protection`: Enables browser XSS filter
- `HSTS`: Enforces HTTPS connections

---

### 4. Input Validation

**Comprehensive validation for all inputs:**

```python
# RGB color validation
validate_rgb_color([255, 0, 0])  # ✓ Valid
validate_rgb_color([256, 0, 0])  # ✗ ValidationError: out of range

# RAL code validation
validate_ral_code("RAL 3000")  # ✓ Valid
validate_ral_code("3000")      # ✗ ValidationError: must start with 'RAL '

# Image file validation
validate_image_file("image.png")  # Checks:
# - File exists
# - File size within limits
# - Valid image format
# - Readable by OpenCV
# - Dimensions within limits (< 10000x10000)
```

**Error responses:**
```json
{
    "success": false,
    "error": "RGB values must be in range [0, 255]",
    "error_type": "validation_error"
}
```

---

## 📊 Logging System

### Log Files

**Directory structure:**
```
logs/
├── app.log       # All logs (INFO and above)
├── error.log     # Errors only (ERROR and above)
├── app.log.1     # Rotated backup
├── app.log.2
└── ...
```

**Log rotation:**
- Max file size: 10MB
- Backup count: 5 files
- Automatic rotation when size limit reached

### Log Levels

```python
app.logger.debug("Detailed debugging info")      # Development only
app.logger.info("General information")           # Default level
app.logger.warning("Warning messages")           # Potential issues
app.logger.error("Error messages")               # Errors
app.logger.exception("Exception with traceback") # Exceptions
```

### Log Format

```
[2025-11-06 12:34:56] INFO in app:123 - Processing Reinhard transfer: job_abc123 -> RAL 3000
[2025-11-06 12:34:57] INFO in app:245 - Processing complete: result_xyz789 - QC: PASSED - Mean ΔE: 3.42
```

### What Gets Logged

**Every request:**
```
[2025-11-06 12:00:00] INFO in app:85 - POST /api/upload - IP: 192.168.1.1 - User-Agent: Mozilla/5.0...
```

**Processing events:**
```
[2025-11-06 12:00:05] INFO in app:312 - Upload successful: job_20251106_120005_a1b2c3d4 - test.png (1920x1080, 2458240 bytes)
[2025-11-06 12:00:10] INFO in app:445 - Processing Reinhard transfer: job_20251106_120005_a1b2c3d4 -> RAL 3000
[2025-11-06 12:00:15] INFO in app:512 - Processing complete: result_20251106_120015_e5f6g7h8 - QC: PASSED - Mean ΔE: 2.87
```

**Errors:**
```
[2025-11-06 12:05:00] ERROR in app:567 - Error processing Reinhard transfer: Division by zero
Traceback (most recent call last):
  ...
```

---

## ⚡ Performance Optimizations

### 1. Caching

**What gets cached:**
- Palette data (5 minutes)
- Palette statistics (10 minutes)
- Image previews (configurable)

**Configuration:**
```python
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',           # In-memory cache
    'CACHE_DEFAULT_TIMEOUT': 300      # 5 minutes
})
```

**Usage:**
```python
@app.route('/api/palette')
@cache.cached(timeout=300, query_string=True)
def get_palette_data():
    # Cached for 5 minutes, different cache per query string
    pass
```

**Production setup (Redis):**
```python
cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0',
    'CACHE_DEFAULT_TIMEOUT': 600
})
```

---

### 2. Response Compression

**Add gzip compression:**
```python
from flask_compress import Compress

Compress(app)
```

**Reduces response size by 70-90%**

---

## 🛡️ Error Handling

### Error Types

**1. Validation Errors (400)**
```json
{
    "success": false,
    "error": "RGB values must be in range [0, 255]",
    "error_type": "validation_error"
}
```

**2. Not Found (404)**
```json
{
    "success": false,
    "error": "Resource not found",
    "error_type": "not_found"
}
```

**3. File Too Large (413)**
```json
{
    "success": false,
    "error": "File too large (max 50MB)",
    "error_type": "file_too_large"
}
```

**4. Rate Limit (429)**
```json
{
    "success": false,
    "error": "Rate limit exceeded. Please try again later.",
    "error_type": "rate_limit"
}
```

**5. Internal Error (500)**
```json
{
    "success": false,
    "error": "An internal server error occurred. Please try again later.",
    "error_type": "internal_error"
}
```

### Custom Error Handlers

```python
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    app.logger.warning(f"Validation error: {str(e)}")
    return jsonify({
        'success': False,
        'error': str(e),
        'error_type': 'validation_error'
    }), 400
```

---

## 🧪 Testing

### Running Tests

**All tests:**
```bash
pytest flask_app/tests/ -v
```

**With coverage:**
```bash
pytest flask_app/tests/ --cov=. --cov-report=html
```

**Specific test file:**
```bash
pytest flask_app/tests/test_app.py -v
```

**Single test:**
```bash
pytest flask_app/tests/test_app.py::TestRoutes::test_palette_route -v
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_app.py           # Route tests
├── test_color_utils.py   # Color utility tests (original)
└── test_validation.py    # Validation tests
```

### Example Test

```python
def test_upload_valid_image(client, sample_image):
    """Test uploading a valid image."""
    with open('test.png', 'rb') as f:
        data = {'file': (f, 'test.png')}
        response = client.post('/api/upload', data=data)

    assert response.status_code == 200
    result = json.loads(response.data)
    assert result['success'] is True
```

---

## 🚀 Deployment

### Environment Variables

```bash
# Required
export SECRET_KEY="your-secure-secret-key-here"

# Optional
export FLASK_ENV=production
export LOG_LEVEL=INFO
export MAX_CONTENT_LENGTH=52428800  # 50MB
```

### Production Checklist

- [ ] Set secure `SECRET_KEY`
- [ ] Set `FLASK_ENV=production`
- [ ] Enable HTTPS
- [ ] Configure Redis for caching
- [ ] Set up log rotation
- [ ] Configure firewall
- [ ] Set up monitoring
- [ ] Configure backup
- [ ] Test error pages
- [ ] Load test application

### WSGI Server (Gunicorn)

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app_enhanced:app
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /path/to/flask_app/static;
        expires 1y;
    }
}
```

---

## 📈 Monitoring

### Application Metrics

**Log analysis:**
```bash
# Count requests by endpoint
grep "POST /api/process" logs/app.log | wc -l

# Find errors
grep "ERROR" logs/error.log

# Average processing time
grep "Processing complete" logs/app.log | grep -oP 'Mean ΔE: \K[\d.]+' | awk '{sum+=$1} END {print sum/NR}'
```

### Health Check Endpoint

```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'palette_loaded': palette is not None,
        'colors_count': len(palette.colors) if palette else 0
    })
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Rate limit too strict**
```python
# Increase limits
limiter = Limiter(
    default_limits=["500 per day", "100 per hour"]
)
```

**2. Cache issues**
```python
# Clear cache
cache.clear()

# Disable cache for debugging
app.config['CACHE_TYPE'] = 'null'
```

**3. Log file permissions**
```bash
chmod 755 logs/
chmod 644 logs/*.log
```

**4. Memory issues with large images**
```python
# Reduce downsampling threshold
engine = ColorTransferEngine(downsample_max=1024)
```

---

## 📚 Additional Resources

- Flask Security: https://flask.palletsprojects.com/en/3.0.x/security/
- Flask-WTF: https://flask-wtf.readthedocs.io/
- Flask-Limiter: https://flask-limiter.readthedocs.io/
- Logging Best Practices: https://docs.python.org/3/howto/logging.html

---

## 🎯 Best Practices

1. **Always use HTTPS in production**
2. **Rotate logs regularly**
3. **Monitor error rates**
4. **Set rate limits based on actual usage**
5. **Keep dependencies updated**
6. **Use environment variables for secrets**
7. **Test thoroughly before deploying**
8. **Have a rollback plan**
9. **Monitor disk space for uploads/results**
10. **Clean up old files periodically**
