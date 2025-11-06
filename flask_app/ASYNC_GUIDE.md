# Asynchronous Processing & API Documentation Guide

Complete guide for using async features and Swagger API documentation.

---

## 🚀 Overview

The async version adds two major features:

1. **Comprehensive API Documentation** via Swagger/OpenAPI
2. **Asynchronous Processing** via Celery for long-running tasks

---

## 📚 API Documentation (Swagger UI)

### Accessing Documentation

**Interactive Swagger UI:**
```
http://localhost:5000/api/docs
```

**OpenAPI Specification (JSON):**
```
http://localhost:5000/api/swagger.json
```

### Features

✅ **Interactive Testing**: Test all API endpoints directly from browser
✅ **Request/Response Examples**: See example data for all endpoints
✅ **Schema Validation**: View all request/response schemas
✅ **Error Documentation**: Understand all error responses
✅ **Authentication Info**: See rate limits and security requirements

### Using Swagger UI

1. **Navigate to** `http://localhost:5000/api/docs`
2. **Browse endpoints** organized by tags:
   - `palette` - Color palette operations
   - `color-matching` - Delta E calculations
   - `image-processing` - Upload and transfer
   - `async-tasks` - Background job management
   - `system` - Health checks

3. **Test an endpoint**:
   - Click endpoint to expand
   - Click "Try it out"
   - Fill in parameters
   - Click "Execute"
   - View response

### Example: Testing Color Match

```
1. Open Swagger UI
2. Navigate to "color-matching" section
3. Click POST /api/color/match
4. Click "Try it out"
5. Enter JSON:
   {
     "rgb": [255, 0, 0],
     "top_n": 3
   }
6. Click "Execute"
7. View matched RAL colors with Delta E values
```

---

## ⚡ Asynchronous Processing

### Why Async?

**Problems with Synchronous Processing:**
- Long operations block the server
- User must wait for completion
- Timeout errors on slow connections
- Poor scalability

**Benefits of Async Processing:**
- Immediate response (202 Accepted)
- Background execution
- No timeouts
- Better resource utilization
- Scalable to multiple workers

### Architecture

```
User Request → Flask API → Celery Task Queue → Worker Pool
                  ↓                                ↓
            Task ID Returned               Processing Happens
                  ↓                                ↓
         Status Polling ← ← ← ← ← ← ← ← Results Stored
```

**Components:**
- **Flask**: Web API (receives requests, returns task IDs)
- **Redis**: Message broker (queues tasks)
- **Celery**: Task queue manager
- **Workers**: Background processors (can scale horizontally)

---

## 🔧 Setup & Configuration

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**New dependencies:**
- `celery>=5.3.0` - Task queue
- `redis>=5.0.0` - Message broker client
- `flask-swagger-ui>=4.11.1` - API documentation

### 2. Install & Start Redis

**Ubuntu/Debian:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Docker:**
```bash
docker run -d -p 6379:6379 redis:alpine
```

**Verify Redis:**
```bash
redis-cli ping
# Should return: PONG
```

### 3. Start Celery Worker

**In a separate terminal:**
```bash
cd flask_app
celery -A app_async.celery worker --loglevel=info
```

**Expected output:**
```
 -------------- celery@hostname v5.3.0
---- **** -----
--- * ***  * -- Linux-x86_64
-- * - **** ---
- ** ---------- [config]
- ** ---------- .> app:         app_async
- ** ---------- .> transport:   redis://localhost:6379/0
- ** ---------- .> results:     redis://localhost:6379/0
- *** --- * --- .> concurrency: 4 (prefork)
-- ******* ---- .> task events: OFF

[tasks]
  . tasks.cleanup_old_files
  . tasks.process_color_transfer

[2025-11-06 12:00:00,000: INFO/MainProcess] Connected to redis://localhost:6379/0
[2025-11-06 12:00:00,000: INFO/MainProcess] celery@hostname ready.
```

### 4. Start Flask Application

**In another terminal:**
```bash
python app_async.py
```

**Or use the startup script:**
```bash
./run_async.sh
```

---

## 📡 Using Async Endpoints

### Synchronous vs Asynchronous

**Synchronous Processing:**
```bash
POST /api/process/reinhard
```
- Waits for completion
- Returns result immediately
- May timeout on slow operations
- Blocks server resources

**Asynchronous Processing:**
```bash
POST /api/process/async
```
- Returns immediately with task_id
- Processing happens in background
- Poll for status
- Scalable

### Async Workflow

#### Step 1: Submit Task

**Request:**
```bash
curl -X POST http://localhost:5000/api/process/async \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "20251106_120000_a1b2c3d4",
    "target_ral_code": "RAL 3000",
    "downsample": true
  }'
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "task_id": "abc123-def456-789012",
  "status_url": "/api/task/abc123-def456-789012",
  "message": "Task accepted for processing"
}
```

#### Step 2: Poll Task Status

**Request:**
```bash
curl http://localhost:5000/api/task/abc123-def456-789012
```

**Response (PENDING):**
```json
{
  "success": true,
  "task_id": "abc123-def456-789012",
  "state": "PENDING",
  "progress": 0,
  "status": "Task is waiting to be processed..."
}
```

**Response (STARTED):**
```json
{
  "success": true,
  "task_id": "abc123-def456-789012",
  "state": "STARTED",
  "progress": 60,
  "status": "Running quality control..."
}
```

**Response (SUCCESS):**
```json
{
  "success": true,
  "task_id": "abc123-def456-789012",
  "state": "SUCCESS",
  "progress": 100,
  "result": {
    "success": true,
    "result_job_id": "20251106_120100_x1y2z3a4",
    "ral_info": { ... },
    "qc_report": { ... },
    "downloads": { ... }
  }
}
```

#### Step 3: Download Results

```bash
curl -O http://localhost:5000/api/download/20251106_120100_x1y2z3a4.png
```

### JavaScript Example (Frontend)

```javascript
// Submit async task
async function processImageAsync(jobId, ralCode) {
    const response = await fetch('/api/process/async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            job_id: jobId,
            target_ral_code: ralCode,
            downsample: true
        })
    });

    const data = await response.json();
    return data.task_id;
}

// Poll for status
async function pollTaskStatus(taskId) {
    const checkStatus = async () => {
        const response = await fetch(`/api/task/${taskId}`);
        const data = await response.json();

        if (data.state === 'SUCCESS') {
            return data.result;
        } else if (data.state === 'FAILURE') {
            throw new Error(data.error);
        } else {
            // Update progress UI
            updateProgress(data.progress, data.status);

            // Poll again in 2 seconds
            await new Promise(resolve => setTimeout(resolve, 2000));
            return checkStatus();
        }
    };

    return checkStatus();
}

// Complete workflow
async function completeAsyncWorkflow() {
    try {
        // 1. Submit task
        const taskId = await processImageAsync('job_123', 'RAL 3000');
        console.log('Task submitted:', taskId);

        // 2. Poll for completion
        const result = await pollTaskStatus(taskId);
        console.log('Processing complete:', result);

        // 3. Display results
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
    }
}
```

---

## 🔍 Task States

| State | Description | Progress |
|-------|-------------|----------|
| `PENDING` | Task waiting in queue | 0% |
| `STARTED` | Worker picked up task | 10-90% |
| `SUCCESS` | Completed successfully | 100% |
| `FAILURE` | Error occurred | N/A |
| `RETRY` | Retrying after failure | N/A |
| `REVOKED` | Task cancelled | N/A |

### Progress Updates

During `STARTED` state, task reports progress:

```json
{
  "state": "STARTED",
  "progress": 30,
  "status": "Processing color transfer..."
}
```

**Progress milestones:**
- 10% - Loading image
- 30% - Processing color transfer
- 60% - Running quality control
- 80% - Saving results
- 100% - Complete

---

## ⚙️ Configuration

### Environment Variables

```bash
# Redis connection
export REDIS_URL="redis://localhost:6379/0"

# Flask secret key
export SECRET_KEY="your-secret-key"

# Environment
export FLASK_ENV="production"
```

### Celery Configuration

**In `celery_config.py`:**
```python
celery.conf.update(
    task_time_limit=300,          # 5 minutes hard limit
    task_soft_time_limit=240,     # 4 minutes soft limit
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=100 # Restart after 100 tasks
)
```

### Scaling Workers

**Single machine:**
```bash
# Multiple workers
celery -A app_async.celery worker --concurrency=8
```

**Multiple machines:**
```bash
# Machine 1
celery -A app_async.celery worker --hostname=worker1@%h

# Machine 2
celery -A app_async.celery worker --hostname=worker2@%h
```

---

## 📊 Monitoring

### Celery Flower (Web UI)

**Install:**
```bash
pip install flower
```

**Start:**
```bash
celery -A app_async.celery flower
```

**Access:**
```
http://localhost:5555
```

**Features:**
- Real-time worker monitoring
- Task history
- Task execution graphs
- Worker management

### Command Line Monitoring

**Active tasks:**
```bash
celery -A app_async.celery inspect active
```

**Registered tasks:**
```bash
celery -A app_async.celery inspect registered
```

**Worker stats:**
```bash
celery -A app_async.celery inspect stats
```

---

## 🔄 Periodic Tasks

### Cleanup Task

Automatically runs every hour to delete old files.

**Configuration in `celery_config.py`:**
```python
CELERY_BEAT_SCHEDULE = {
    'cleanup-old-files': {
        'task': 'tasks.cleanup_old_files',
        'schedule': 3600.0,  # Every hour
    },
}
```

**Start beat scheduler:**
```bash
celery -A app_async.celery beat --loglevel=info
```

**Or combine with worker:**
```bash
celery -A app_async.celery worker --beat --loglevel=info
```

---

## 🚨 Error Handling

### Task Failures

**Automatic retries:**
```python
@celery.task(bind=True, max_retries=3)
def my_task(self):
    try:
        # Task logic
        pass
    except Exception as exc:
        # Retry in 60 seconds
        raise self.retry(exc=exc, countdown=60)
```

**Failure response:**
```json
{
  "success": false,
  "task_id": "abc123",
  "state": "FAILURE",
  "error": "Error message",
  "error_type": "task_failure"
}
```

### Connection Errors

**Redis down:**
```
ConnectionError: Error connecting to Redis
```

**Solution:**
```bash
sudo systemctl start redis
# or
redis-server
```

**Worker not running:**
```
Task submitted but never starts
```

**Solution:**
```bash
celery -A app_async.celery worker --loglevel=info
```

---

## 📈 Performance

### Benchmarks

**Synchronous Processing:**
- 1920x1080 image: ~2.1s
- Blocks server for entire duration
- 1 request at a time

**Asynchronous Processing:**
- Submit: ~50ms (immediate response)
- Processing: ~2.1s (in background)
- 10+ concurrent requests

### Best Practices

1. **Use async for operations > 1 second**
2. **Poll status every 2-5 seconds**
3. **Implement exponential backoff**
4. **Set reasonable timeouts**
5. **Scale workers based on load**

---

## 🔐 Security

### Rate Limiting

Async endpoint has same rate limits:
```python
@app.route('/api/process/async')
@limiter.limit("5 per minute")
```

### Task ID Security

- Task IDs are UUIDs (non-guessable)
- No authentication required (add if needed)
- Results auto-deleted after 24 hours

---

## 🧪 Testing

### Test Async Endpoint

```bash
# 1. Submit task
curl -X POST http://localhost:5000/api/process/async \
  -H "Content-Type: application/json" \
  -d '{"job_id": "test123", "target_ral_code": "RAL 3000"}'

# 2. Get task ID from response
# 3. Check status
curl http://localhost:5000/api/task/TASK_ID_HERE
```

### Unit Tests

```python
def test_async_processing(client):
    # Submit task
    response = client.post('/api/process/async', json={
        'job_id': 'test123',
        'target_ral_code': 'RAL 3000'
    })

    assert response.status_code == 202
    data = response.json
    assert 'task_id' in data

    # Check status
    task_id = data['task_id']
    response = client.get(f'/api/task/{task_id}')
    assert response.status_code == 200
```

---

## 📚 Additional Resources

- **Celery Documentation**: https://docs.celeryproject.org/
- **Redis Documentation**: https://redis.io/documentation
- **OpenAPI Specification**: https://swagger.io/specification/
- **Flask-RESTX**: https://flask-restx.readthedocs.io/

---

## 🎯 Quick Reference

### Start Everything

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Celery Worker
celery -A app_async.celery worker --loglevel=info

# Terminal 3: Flask App
python app_async.py

# Optional Terminal 4: Celery Beat
celery -A app_async.celery beat --loglevel=info

# Optional Terminal 5: Flower Monitoring
celery -A app_async.celery flower
```

### URLs

- **Application**: http://localhost:5000
- **API Docs**: http://localhost:5000/api/docs
- **OpenAPI Spec**: http://localhost:5000/api/swagger.json
- **Health Check**: http://localhost:5000/health
- **Flower**: http://localhost:5555 (if running)

### Key Endpoints

- `POST /api/process/async` - Submit async task
- `GET /api/task/{task_id}` - Check task status
- `GET /api/docs` - Interactive API documentation
- `GET /health` - Health check

---

## ✅ Troubleshooting

### Issue: Swagger UI not loading

**Solution:**
```bash
pip install flask-swagger-ui
```

### Issue: Tasks stay in PENDING

**Cause:** Worker not running

**Solution:**
```bash
celery -A app_async.celery worker --loglevel=info
```

### Issue: Connection refused to Redis

**Solution:**
```bash
sudo systemctl start redis
# or
redis-server
```

### Issue: Task timeout

**Solution:** Increase time limits in `celery_config.py`:
```python
task_time_limit=600  # 10 minutes
```

---

**Now you have a fully async-enabled, well-documented API!** 🚀
