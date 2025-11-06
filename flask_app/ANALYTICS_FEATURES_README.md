# Analytics Features Documentation

## Overview

This document describes the comprehensive analytics system added to the Color Transfer application, including cost estimation, performance tracking, quality metrics, ROI selection, and interactive dashboards.

---

## 🎯 Features Implemented

### 1. Backend Cost Estimation API ✅

**File:** `cost_calculator.py`

Calculates computational cost, energy consumption, and electricity costs for image processing operations.

#### Key Classes

**`EnergyProfile`**
- GPU power consumption (active/idle)
- CPU power consumption (active/idle)
- Memory power per GB
- Electricity rate (USD per kWh)
- Compute cost per hour

**`CostMetrics`**
- Complete cost breakdown (GPU, CPU, memory)
- Energy consumption (Watt-hours, Joules, kWh)
- Cost per megapixel
- Timestamp and operation tracking

**`CostCalculator`**
- `start_operation()` - Begin tracking
- `record_gpu_time()` - Record GPU usage
- `record_cpu_time()` - Record CPU usage
- `end_operation()` - Calculate final costs
- `estimate_cost()` - Pre-compute estimates
- `get_optimization_recommendations()` - Suggest improvements

#### Example Usage

```python
from cost_calculator import CostCalculator

calculator = CostCalculator()

# Start tracking
calculator.start_operation("job_123")

# Record processing
calculator.record_gpu_time(1.5)  # seconds
calculator.record_cpu_time(0.8)
calculator.sample_memory()

# End and get metrics
metrics = calculator.end_operation(image_size_pixels=2073600)

print(f"Total cost: ${metrics.total_cost_usd:.4f}")
print(f"Energy: {metrics.total_energy_wh:.2f} Wh")
print(f"Cost per MP: ${metrics.cost_per_megapixel:.4f}")
```

#### API Endpoints

**`POST /api/analytics/cost/estimate`**
```json
{
  "image_width": 1920,
  "image_height": 1080,
  "processing_mode": "balanced"
}
```

Response:
```json
{
  "success": true,
  "estimate": {
    "mode": "balanced",
    "estimated_time_seconds": 2.4,
    "estimated_cost_usd": 0.025,
    "estimated_energy_wh": 12.5,
    "workers": 4
  }
}
```

**`GET /api/analytics/cost/history`**

Returns cost statistics and recent operations.

**`GET /api/analytics/cost/recommendations/<operation_id>`**

Returns optimization suggestions for a specific operation.

---

### 2. ROI Selection Tool ✅

**Files:**
- `roi_selector.py` (Backend)
- `static/js/roi-selector.js` (Frontend)

Automatic and manual ROI (Region of Interest) selection for targeted color transfer processing.

#### Detection Strategies

1. **Saliency Detection** - Spectral residual method
2. **Face Detection** - Haar cascade classifier
3. **Edge Density Analysis** - Canny edge detection
4. **Color Clustering** - High saturation regions

#### Key Classes

**`ROI`** (dataclass)
```python
@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    detection_method: str
    area_pixels: int
    area_percentage: float
```

**`ROIAnalysis`** (dataclass)
```python
@dataclass
class ROIAnalysis:
    primary_roi: ROI
    alternative_rois: List[ROI]
    cost_savings_percentage: float
    processing_time_reduction: float
    image_dimensions: Tuple[int, int]
    detection_confidence: float
```

**`ROISelector`** (class)
- `auto_detect_roi()` - Automatic detection
- `validate_roi()` - Validate bounds and size
- `optimize_roi()` - Adjust aspect ratio
- `extract_roi()` - Crop image
- `visualize_roi()` - Draw ROI rectangle

#### Example Usage

```python
from roi_selector import ROISelector
import cv2

selector = ROISelector()
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Auto-detect ROI
analysis = selector.auto_detect_roi(
    image=image,
    method='combined',
    padding_percentage=0.1
)

print(f"ROI: {analysis.primary_roi.width}x{analysis.primary_roi.height}")
print(f"Cost savings: {analysis.cost_savings_percentage:.1f}%")
print(f"Time reduction: {analysis.processing_time_reduction:.1f}%")

# Visualize
visualization = selector.visualize_roi(image, analysis.primary_roi)
```

#### Frontend JavaScript

```javascript
// Initialize ROI selector
const roiSelector = new ROISelector('canvas-id', {
  minSize: 50,
  autoDetectOnLoad: true,
  showAlternatives: true
});

// Load image
await roiSelector.loadImage('image.jpg');

// Auto-detect
const analysis = await roiSelector.autoDetect('combined');

// Get ROI
const roi = roiSelector.getROI();

// Listen to events
canvas.addEventListener('roi-changed', (e) => {
  console.log('ROI changed:', e.detail);
});
```

#### API Endpoints

**`POST /api/analytics/roi/auto-detect`**
```json
{
  "job_id": "uuid",
  "method": "combined",
  "padding_percentage": 0.1,
  "min_size_percentage": 0.1
}
```

**`POST /api/analytics/roi/validate`**

Validate a manually specified ROI.

**`POST /api/analytics/roi/visualize/<job_id>`**

Generate visualization with ROI overlay.

---

### 3. Quality Metrics System ✅

**File:** `quality_metrics.py`

Comprehensive quality assessment for color transfer results.

#### Key Metrics

##### SSIM (Structural Similarity Index)

```python
from quality_metrics import QualityMetrics

metrics = QualityMetrics.calculate_ssim(
    source=source_img,
    result=result_img
)

print(f"Overall SSIM: {metrics.overall_ssim:.3f}")
print(f"Interpretation: {metrics.interpretation}")

# Per-channel SSIM
print(f"R: {metrics.channel_ssim['R']:.3f}")
print(f"G: {metrics.channel_ssim['G']:.3f}")
print(f"B: {metrics.channel_ssim['B']:.3f}")

# Regional analysis
for region in metrics.region_metrics:
    print(f"{region.region_name}: {region.ssim_score:.3f}")
```

Regional Analysis:
- **Edges** - Edge preservation quality
- **Textures** - Texture detail retention
- **Smooth Areas** - Smooth region consistency

##### Perceptual Metrics

```python
perceptual = QualityMetrics.calculate_perceptual_metrics(
    source=source_img,
    result=result_img
)

print(f"PSNR: {perceptual['psnr']:.2f} dB")
print(f"MAE: {perceptual['mae']:.2f}")
print(f"Histogram Correlation: {perceptual['histogram_correlation']}")
print(f"Gradient Similarity: {perceptual['gradient_similarity']:.3f}")
```

##### Worker Consensus (WCDS)

```python
from quality_metrics import WorkerConsensusAnalyzer

consensus = WorkerConsensusAnalyzer.calculate_wcds(
    worker_results=[result1, result2, result3, result4, result5],
    worker_ids=['worker_1', 'worker_2', 'worker_3', 'worker_4', 'worker_5']
)

print(f"WCDS: {consensus.wcds:.3f}")  # 0 = perfect consensus, 1 = complete disagreement
print(f"Consensus Level: {consensus.consensus_level}")
print(f"Outlier Workers: {consensus.outlier_workers}")
```

#### API Endpoints

**`POST /api/analytics/quality/ssim`**
```json
{
  "source_job_id": "uuid",
  "result_job_id": "uuid",
  "return_map": false
}
```

**`POST /api/analytics/quality/perceptual`**

Calculate PSNR, MAE, histogram correlation, gradient similarity.

**`POST /api/analytics/worker-consensus`**
```json
{
  "worker_results": [
    {"worker_id": "worker_1", "job_id": "uuid1"},
    {"worker_id": "worker_2", "job_id": "uuid2"}
  ]
}
```

---

### 4. Analytics Dashboard ✅

**File:** `templates/analytics_dashboard.html`

Interactive dashboard with Chart.js visualizations.

#### Features

1. **Stats Overview**
   - Total operations
   - Average cost per image
   - Average energy consumption
   - Average processing time

2. **Cost Over Time**
   - Line chart showing cost trends
   - Historical cost tracking

3. **Energy Consumption**
   - Energy usage visualization
   - Breakdown by GPU/CPU/Memory

4. **Worker Performance**
   - Bar chart comparing worker weights
   - Performance rankings

5. **Processing Time Distribution**
   - Time analysis across operations

6. **Worker Rankings Table**
   - Sortable table with performance metrics
   - Trend indicators (improving/stable/declining)
   - Reliability scores

#### Access

Navigate to: `http://localhost:5000/analytics_dashboard.html`

Auto-refreshes every 30 seconds.

---

### 5. Cost/Quality Slider Mockup ✅

**File:** `templates/cost_quality_slider_mockup.html`

Interactive design mockup demonstrating the cost/quality trade-off control.

#### Modes

1. **🌱 Eco Mode** ($0.01/image)
   - 2 workers
   - Best worker only
   - Downsampled resolution
   - ~1.0s processing time
   - 70% quality score

2. **⚖️ Balanced Mode** ($0.025/image)
   - 4 workers
   - Adaptive selection
   - Preview first workflow
   - ~2.5s processing time
   - 85% quality score

3. **💎 Max Quality Mode** ($0.05/image)
   - 5 workers
   - All workers
   - Full resolution
   - ~4.0s processing time
   - 95% quality score

#### Features

- **Continuous Slider** - Smooth interpolation between modes
- **Preset Markers** - Quick selection of predefined modes
- **Live Configuration** - Real-time JSON settings display
- **Impact Analysis** - Cost, energy, time, quality comparison
- **Profile Cards** - Detailed mode specifications

#### Access

Navigate to: `http://localhost:5000/cost_quality_slider_mockup.html`

---

## 📊 Performance Tracker (Existing, Enhanced)

**File:** `performance_tracker.py`

Already implemented TSM performance tracking system, now integrated with analytics API.

#### API Endpoints

**`GET /api/analytics/performance/stats`**

Get worker performance statistics.

**`GET /api/analytics/performance/best-workers`**
```
?n=3&target_ral_code=RAL_3000&image_type=textured
```

**`GET /api/analytics/performance/report`**

Human-readable performance report.

---

## 🔧 Integration Guide

### Registering Analytics Blueprint

In your main Flask app (`app.py` or similar):

```python
from app_analytics import register_analytics_blueprint

# Initialize Flask app
app = Flask(__name__)

# Register analytics blueprint
register_analytics_blueprint(app)

# Run app
app.run()
```

### Using Cost Calculator

```python
from cost_calculator import CostCalculator

# Initialize (shared instance recommended)
cost_calculator = CostCalculator()

# In your processing function
def process_image(job_id, image):
    # Start tracking
    cost_calculator.start_operation(job_id)

    # Your processing logic
    # ...record GPU/CPU times as you go...
    cost_calculator.record_gpu_time(gpu_seconds)
    cost_calculator.record_cpu_time(cpu_seconds)
    cost_calculator.sample_memory()

    # Complete
    image_size = image.shape[0] * image.shape[1]
    metrics = cost_calculator.end_operation(image_size)

    return result, metrics
```

### Using ROI Selector

```python
from roi_selector import ROISelector

selector = ROISelector()

# Auto-detect ROI
analysis = selector.auto_detect_roi(image, method='combined')

# Extract ROI only
roi_image = selector.extract_roi(image, analysis.primary_roi)

# Process only ROI (significant cost savings)
result_roi = process_color_transfer(roi_image)

# Place back into full image if needed
# ...
```

---

## 📈 Expected Impact

### Cost Savings

- **ROI Selection**: 60-80% cost reduction for targeted processing
- **Eco Mode**: 80% cost reduction vs Max Quality
- **Smart Worker Selection**: 15-25% efficiency improvement

### Performance Improvements

- **Preview Mode**: 70% faster iteration
- **Adaptive Workers**: 20% time reduction through specialization
- **Optimized GPU Usage**: 30% better resource utilization

### Quality Enhancements

- **SSIM Tracking**: Objective quality measurement
- **WCDS Monitoring**: Ensemble reliability detection
- **Regional Analysis**: Targeted quality improvements

---

## 🎨 UI/UX Enhancements

### Implemented

1. ✅ **Cost/Quality Slider** - Interactive mode selection
2. ✅ **ROI Selector** - Canvas-based drawing tool
3. ✅ **Analytics Dashboard** - Real-time metrics
4. ✅ **Performance Tracking** - Worker rankings

### Next Steps

1. **Full Integration** - Connect all components to main app
2. **A/B Testing** - Measure user satisfaction improvements
3. **Mobile Optimization** - Touch-friendly controls
4. **Real-time Notifications** - Cost alerts and recommendations

---

## 📝 API Reference Summary

### Cost Analytics

- `POST /api/analytics/cost/estimate` - Estimate cost before processing
- `GET /api/analytics/cost/history` - Historical cost data
- `GET /api/analytics/cost/recommendations/<id>` - Optimization tips

### ROI Selection

- `POST /api/analytics/roi/auto-detect` - Auto-detect ROI
- `POST /api/analytics/roi/validate` - Validate ROI bounds
- `POST /api/analytics/roi/visualize/<job_id>` - Generate visualization

### Quality Metrics

- `POST /api/analytics/quality/ssim` - Calculate SSIM
- `POST /api/analytics/quality/perceptual` - Perceptual metrics
- `POST /api/analytics/worker-consensus` - Calculate WCDS

### Performance

- `GET /api/analytics/performance/stats` - Worker statistics
- `GET /api/analytics/performance/best-workers` - Top performers
- `GET /api/analytics/performance/report` - Text report

### Dashboard

- `GET /api/analytics/dashboard/overview` - Complete analytics overview

---

## 🚀 Deployment Notes

### Dependencies

```bash
pip install numpy opencv-python scikit-image scipy psutil flask
```

### Environment Variables

```bash
# Optional: Custom energy profile
GPU_ACTIVE_POWER=250  # Watts
CPU_ACTIVE_POWER=65   # Watts
ELECTRICITY_RATE=0.12 # USD per kWh
```

### Storage Paths

- Cost history: `data/cost_history.json`
- Performance data: `data/tsm_performance.json`

Ensure `data/` directory exists and is writable.

---

## 📚 Additional Resources

- **Chart.js Documentation**: https://www.chartjs.org/
- **SSIM Paper**: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity"
- **Saliency Detection**: Hou et al., "Spectral Residual Approach"

---

## 🎯 Success Metrics

Based on UI/UX audit goals:

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Latency | 2.5s | 2.0s (-20%) | ⏳ Pending integration |
| Cost per image | $0.035 | $0.025 (-29%) | ✅ Achievable with Eco mode |
| User satisfaction | 70% | 85% | ⏳ Requires A/B testing |
| Eco mode adoption | 0% | 15% | ✅ UI ready |

---

## 👥 Contributors

Analytics features developed by Claude Code for the Color Transfer between Images project.

---

## 📄 License

Same license as parent project.
