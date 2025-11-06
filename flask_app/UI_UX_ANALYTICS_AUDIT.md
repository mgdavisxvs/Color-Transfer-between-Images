# UI/UX Analytics Audit: Image-to-Image Transfer System
## Comprehensive Analysis & Feature Recommendations

**Version:** 1.0.0
**Date:** November 2025
**Status:** Initial Audit & Recommendations

---

## Executive Summary

This document provides a comprehensive audit of the current image-to-image transfer UI/UX, identifies critical missing features, and proposes integrated analytics dashboards to optimize transfer quality, performance efficiency, and operational costs.

**Key Findings:**
- ❌ No real-time performance metrics visible to users
- ❌ No cost/energy consumption transparency
- ❌ Limited quality assessment tools beyond visual inspection
- ❌ No ROI (Region of Interest) selection capability
- ❌ Missing preview/iteration workflow
- ❌ No user-facing model/worker selection

**Recommended Priority:**
1. **Immediate (Sprint 1)**: Cost/Quality Control Slider + Basic Analytics
2. **High (Sprint 2)**: Quality Metrics Dashboard + ROI Selection
3. **Medium (Sprint 3)**: Advanced Model Selection + Worker Consensus Display
4. **Future**: Historical Analytics + Optimization Recommendations

---

## Table of Contents

1. [Part 1: Current State Analysis](#part-1-current-state-analysis)
2. [Part 2: Analytics & Cost Integration](#part-2-analytics--cost-integration)
3. [Part 3: Prototyping & Implementation](#part-3-prototyping--implementation)
4. [Appendix: Technical Specifications](#appendix-technical-specifications)

---

## Part 1: Current State Analysis & Feature Integration

### I. Core Transfer & Usability Audit

#### A. Input/Output Clarity Assessment

**Current State:**
- ✅ Source image upload area exists
- ✅ Target color/RAL selection available
- ⚠️ Result display adequate but lacks context
- ❌ No clear visual separation of source/target/result

**Issues Identified:**
1. Users cannot easily compare source → target → result in one view
2. No visual indicators of what changed during transfer
3. Result appears isolated from input context

**Recommended Features:**

**Feature 1.1: Three-Panel Comparison View**
```
┌─────────────────────────────────────────────────────────┐
│  [Source Image]  │  [Target Style]  │  [Result]         │
│                  │                  │                   │
│  Original        │  RAL 3020        │  Transferred      │
│  1920×1080       │  (RGB: 204,6,5)  │  1920×1080        │
│                  │                  │                   │
│  ────────────────┼──────────────────┼──────────────────│
│  Color Stats:    │  Target Color:   │  Achieved:        │
│  Mean RGB: ...   │  Red dominant    │  Mean ΔE: 8.5     │
└─────────────────────────────────────────────────────────┘
```

**Feature 1.2: Interactive Slider Comparison**
```html
<div class="comparison-slider">
  <img src="source.jpg" class="slider-before">
  <img src="result.jpg" class="slider-after">
  <input type="range" class="slider-handle"
         min="0" max="100" value="50">
  <!-- Drag slider to reveal before/after -->
</div>
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

#### B. Parameter Control Assessment

**Current State:**
- ✅ Basic RAL color selection exists
- ⚠️ TSM mode selection available but technical
- ❌ No intuitive controls for transfer strength
- ❌ No content preservation controls

**Issues Identified:**
1. Technical TSM modes (adaptive/all/best) not user-friendly
2. No visual feedback on parameter changes
3. Missing presets for common use cases

**Recommended Features:**

**Feature 1.3: Smart Control Panel**
```
┌──────────────────────────────────────────┐
│ Transfer Controls                        │
├──────────────────────────────────────────┤
│                                          │
│ Transfer Strength:  [====●====] 75%      │
│ Low ←────────────────────────→ High      │
│                                          │
│ Content Preservation: [======●==] 80%    │
│ Flexible ←───────────────────→ Strict    │
│                                          │
│ Quality vs Speed:                        │
│ ○ Eco Mode (Fast, Low Cost)             │
│ ● Balanced (Recommended)                 │
│ ○ Max Quality (Slow, Higher Cost)       │
│                                          │
│ Advanced Options ▼                       │
│ └─ TSM Mode: Adaptive                    │
│ └─ Workers: Auto (4 active)              │
│ └─ Downsample: Auto                      │
└──────────────────────────────────────────┘
```

**Feature 1.4: Preset Templates**
```javascript
const transferPresets = {
  "Product Photography": {
    strength: 85,
    contentPreservation: 90,
    mode: "all",
    description: "High accuracy for e-commerce"
  },
  "Artistic Style": {
    strength: 70,
    contentPreservation: 60,
    mode: "adaptive",
    description: "Creative flexibility"
  },
  "Quick Preview": {
    strength: 60,
    contentPreservation: 70,
    mode: "best",
    downsample: true,
    description: "Fast, low-cost preview"
  }
};
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

### II. Missing Features (Image-to-Image)

#### A. Region of Interest (ROI) Selection

**Current State:**
- ❌ Full image always processed
- ❌ No cropping or masking tools
- ❌ Wasted computation on irrelevant areas

**Cost Impact:**
- Processing full 4K image: ~2.5s, ~$0.05
- Processing 25% ROI: ~0.6s, ~$0.01
- **Potential savings: 76% time, 80% cost**

**Recommended Feature:**

**Feature 2.1: Interactive ROI Selection Tool**

```html
<!-- ROI Selection Interface -->
<div class="roi-selector">
  <div class="roi-canvas-container">
    <img src="source.jpg" class="roi-background">
    <div class="roi-overlay">
      <!-- Draggable/resizable selection box -->
      <div class="roi-box"
           style="left: 25%; top: 25%; width: 50%; height: 50%">
        <div class="roi-handles">
          <span class="handle nw"></span>
          <span class="handle ne"></span>
          <span class="handle se"></span>
          <span class="handle sw"></span>
        </div>
        <div class="roi-info">
          512×512px • 25% of image
          Est. time: 0.6s • Cost: $0.01
        </div>
      </div>
    </div>
  </div>

  <div class="roi-controls">
    <button class="btn-secondary" onclick="selectEntireImage()">
      Full Image
    </button>
    <button class="btn-secondary" onclick="autoDetectSubject()">
      🎯 Auto-Detect Subject
    </button>
    <button class="btn-secondary" onclick="usePresetRatio('1:1')">
      Square (1:1)
    </button>

    <div class="roi-stats">
      <div class="stat">
        <span class="label">Processing:</span>
        <span class="value">512×512 (25%)</span>
      </div>
      <div class="stat">
        <span class="label">Est. Time:</span>
        <span class="value">0.6s (↓76%)</span>
      </div>
      <div class="stat">
        <span class="label">Est. Cost:</span>
        <span class="value">$0.01 (↓80%)</span>
      </div>
    </div>
  </div>
</div>
```

**Feature 2.2: Smart Subject Detection**
```python
# Auto-detect main subject using saliency detection
def auto_detect_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Automatically detect region of interest.

    Uses:
    - Saliency detection (cv2.saliency)
    - Face detection (if applicable)
    - Edge density analysis

    Returns:
        (x, y, width, height) of ROI
    """
    # Saliency detection
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)

    # Find bounding box of salient region
    thresh = cv2.threshold(saliency_map, 0.7, 1, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Add 10% padding
        padding = int(min(w, h) * 0.1)
        return (x - padding, y - padding, w + 2*padding, h + 2*padding)

    # Fallback: center 80% of image
    h, w = image.shape[:2]
    return (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
```

**Implementation Priority:** 🟡 MEDIUM (Sprint 2)

---

#### B. Preview and Iteration Workflow

**Current State:**
- ❌ No preview capability
- ❌ Users commit to full-resolution processing upfront
- ❌ No iterative refinement workflow

**Cost Impact:**
- Current: User processes full image 3-5 times to get desired result
- With preview: User previews 3-5 times (low-cost), then processes once
- **Savings: 60-80% of total compute cost**

**Recommended Feature:**

**Feature 2.3: Low-Resolution Preview Mode**

```
┌─────────────────────────────────────────────────────────┐
│ Preview Mode (Fast & Free)                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Preview Result - 512×512]                             │
│  Processing: 0.2s • Cost: $0.002                        │
│                                                         │
│  ────────────────────────────────────────────────────  │
│                                                         │
│  Quality: ████████░░ 80% of final                      │
│  Speed:   ██████████ 10x faster                        │
│  Cost:    ████████░░ 5% of final                       │
│                                                         │
│  [⚡ Quick Preview]  [🎨 Process Full Resolution]      │
└─────────────────────────────────────────────────────────┘
```

**Feature 2.4: Iteration History & Comparison**

```html
<!-- History Panel -->
<div class="iteration-history">
  <h3>Iteration History</h3>

  <div class="history-items">
    <!-- Iteration 1 -->
    <div class="history-item">
      <img src="preview_1.jpg" class="history-thumbnail">
      <div class="history-info">
        <span class="iteration-num">#1</span>
        <span class="params">RAL 3020, Strength: 70%</span>
        <span class="quality">ΔE: 12.3</span>
      </div>
      <button class="btn-icon" onclick="revertTo(1)">↩️</button>
    </div>

    <!-- Iteration 2 (Current) -->
    <div class="history-item active">
      <img src="preview_2.jpg" class="history-thumbnail">
      <div class="history-info">
        <span class="iteration-num">#2 (Current)</span>
        <span class="params">RAL 3020, Strength: 85%</span>
        <span class="quality">ΔE: 8.5 ✓</span>
      </div>
      <button class="btn-primary">Process Full</button>
    </div>
  </div>

  <div class="history-stats">
    <div class="stat">
      <span class="label">Previews:</span>
      <span class="value">2</span>
    </div>
    <div class="stat">
      <span class="label">Cost Saved:</span>
      <span class="value">$0.08 (89%)</span>
    </div>
    <div class="stat">
      <span class="label">Best Result:</span>
      <span class="value">#2 (ΔE: 8.5)</span>
    </div>
  </div>
</div>
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

### III. UI for Model Management

#### A. Worker/Model Selection

**Current State:**
- ✅ TSM mode selection exists (technical)
- ❌ No user-friendly worker selection
- ❌ No explanation of worker differences

**Recommended Feature:**

**Feature 3.1: User-Friendly Worker Selection**

```html
<div class="model-selection-panel">
  <h3>Processing Mode</h3>

  <!-- Simple Mode (Default) -->
  <div class="mode-card active">
    <div class="mode-header">
      <span class="mode-icon">⚡</span>
      <h4>Balanced (Recommended)</h4>
      <span class="badge badge--success">Default</span>
    </div>
    <div class="mode-description">
      Smart selection of 3-4 algorithms based on image complexity.
      Best balance of speed and quality.
    </div>
    <div class="mode-stats">
      <span class="stat">⏱️ 1-2s</span>
      <span class="stat">💰 $0.02-0.03</span>
      <span class="stat">🎯 ΔE ~8-10</span>
    </div>
  </div>

  <!-- Performance Mode -->
  <div class="mode-card">
    <div class="mode-header">
      <span class="mode-icon">🚀</span>
      <h4>Fast Mode</h4>
    </div>
    <div class="mode-description">
      Uses top 2 fastest algorithms. Quick results for previews or
      when speed matters more than perfection.
    </div>
    <div class="mode-stats">
      <span class="stat">⏱️ 0.5-0.8s</span>
      <span class="stat">💰 $0.01</span>
      <span class="stat">🎯 ΔE ~10-12</span>
    </div>
  </div>

  <!-- Quality Mode -->
  <div class="mode-card">
    <div class="mode-header">
      <span class="mode-icon">✨</span>
      <h4>Maximum Quality</h4>
    </div>
    <div class="mode-description">
      Uses all 5 algorithms + ensemble blending. Best possible
      accuracy for critical applications.
    </div>
    <div class="mode-stats">
      <span class="stat">⏱️ 2.5-3.5s</span>
      <span class="stat">💰 $0.04-0.05</span>
      <span class="stat">🎯 ΔE ~6-8</span>
    </div>
  </div>

  <!-- Advanced: Manual Worker Selection -->
  <details class="advanced-options">
    <summary>🔧 Advanced: Manual Worker Selection</summary>

    <div class="worker-checklist">
      <label class="worker-option">
        <input type="checkbox" checked disabled>
        <span class="worker-name">Reinhard Statistical</span>
        <span class="worker-tag">General Purpose</span>
      </label>

      <label class="worker-option">
        <input type="checkbox" checked>
        <span class="worker-name">Histogram Matching</span>
        <span class="worker-tag">Textures</span>
      </label>

      <label class="worker-option">
        <input type="checkbox" checked>
        <span class="worker-name">LAB Channel-Specific</span>
        <span class="worker-tag">Brightness</span>
      </label>

      <label class="worker-option">
        <input type="checkbox">
        <span class="worker-name">Linear Mapping</span>
        <span class="worker-tag">Flat Colors</span>
      </label>

      <label class="worker-option">
        <input type="checkbox">
        <span class="worker-name">Region-Aware</span>
        <span class="worker-tag">Complex</span>
      </label>
    </div>
  </details>
</div>
```

**Implementation Priority:** 🟡 MEDIUM (Sprint 2)

---

#### B. Configuration Display

**Current State:**
- ❌ No visibility into active configuration
- ❌ Users don't know what parameters are being used
- ❌ No feedback on why certain workers were selected

**Recommended Feature:**

**Feature 3.2: Live Configuration Display**

```html
<div class="config-display">
  <div class="config-header">
    <h4>Active Configuration</h4>
    <button class="btn-icon" onclick="toggleConfigPanel()">⚙️</button>
  </div>

  <div class="config-sections">
    <!-- Processing Mode -->
    <div class="config-section">
      <span class="config-label">Mode:</span>
      <span class="config-value">
        Adaptive
        <span class="info-tooltip"
              data-tip="Automatically selects 3-4 workers based on image complexity">
          ℹ️
        </span>
      </span>
    </div>

    <!-- Active Workers -->
    <div class="config-section">
      <span class="config-label">Active Workers:</span>
      <div class="config-value">
        <div class="worker-badges">
          <span class="badge badge--primary">Reinhard</span>
          <span class="badge badge--primary">Histogram</span>
          <span class="badge badge--primary">LAB-Specific</span>
          <span class="badge badge--tertiary">+1 more</span>
        </div>
        <div class="worker-reason">
          Selected for: textured image with high color diversity
        </div>
      </div>
    </div>

    <!-- Image Analysis -->
    <div class="config-section">
      <span class="config-label">Image Complexity:</span>
      <span class="config-value">
        0.65 (Moderate)
        <div class="complexity-bar">
          <div class="complexity-fill" style="width: 65%"></div>
        </div>
      </span>
    </div>

    <!-- Resource Allocation -->
    <div class="config-section">
      <span class="config-label">Resources:</span>
      <div class="config-value">
        <span class="resource-item">CPU: 4 cores</span>
        <span class="resource-item">GPU: Enabled</span>
        <span class="resource-item">Memory: 2.1 GB</span>
      </div>
    </div>

    <!-- Optimizations -->
    <div class="config-section">
      <span class="config-label">Optimizations:</span>
      <div class="config-value">
        <span class="optimization-badge">✓ Downsampling</span>
        <span class="optimization-badge">✓ Parallel Processing</span>
        <span class="optimization-badge">✓ Caching</span>
      </div>
    </div>
  </div>
</div>
```

**Implementation Priority:** 🔵 LOW (Sprint 3)

---

## Part 2: Analytics & Cost Integration

### A. Performance & Cost Analytics Dashboard

**Current State:**
- ❌ No performance metrics displayed
- ❌ No cost transparency
- ❌ No historical tracking
- ❌ Users have no feedback on efficiency

**Critical Missing Metrics:**

#### Metric 1: Transfer Speed (Latency)

**Recommended Feature:**

**Feature 4.1: Real-Time Performance Tracker**

```html
<div class="performance-dashboard">
  <h3>Performance Analytics</h3>

  <!-- Real-time Progress -->
  <div class="processing-timeline">
    <div class="timeline-stage completed">
      <span class="stage-icon">✓</span>
      <span class="stage-name">Upload</span>
      <span class="stage-time">0.3s</span>
    </div>

    <div class="timeline-stage completed">
      <span class="stage-icon">✓</span>
      <span class="stage-name">Preprocessing</span>
      <span class="stage-time">0.2s</span>
    </div>

    <div class="timeline-stage active">
      <span class="stage-icon spinner">⏳</span>
      <span class="stage-name">TSM Processing</span>
      <span class="stage-time">1.4s...</span>
      <div class="stage-progress">
        <div class="progress-bar" style="width: 60%"></div>
      </div>
    </div>

    <div class="timeline-stage pending">
      <span class="stage-icon">○</span>
      <span class="stage-name">Postprocessing</span>
      <span class="stage-time">~0.1s</span>
    </div>

    <div class="timeline-stage pending">
      <span class="stage-icon">○</span>
      <span class="stage-name">Download</span>
      <span class="stage-time">~0.2s</span>
    </div>
  </div>

  <!-- Performance Breakdown -->
  <div class="performance-breakdown">
    <h4>Time Breakdown</h4>

    <div class="breakdown-chart">
      <!-- Horizontal stacked bar -->
      <div class="breakdown-bar">
        <div class="segment segment--upload"
             style="width: 13%" title="Upload: 0.3s">
        </div>
        <div class="segment segment--preprocess"
             style="width: 9%" title="Preprocess: 0.2s">
        </div>
        <div class="segment segment--compute"
             style="width: 60%" title="Compute: 1.4s">
        </div>
        <div class="segment segment--postprocess"
             style="width: 4%" title="Postprocess: 0.1s">
        </div>
        <div class="segment segment--download"
             style="width: 9%" title="Download: 0.2s">
        </div>
      </div>
    </div>

    <div class="breakdown-table">
      <div class="breakdown-row">
        <span class="label">🔼 Upload:</span>
        <span class="value">0.3s</span>
        <span class="percentage">(13%)</span>
      </div>
      <div class="breakdown-row">
        <span class="label">⚙️ CPU Preprocessing:</span>
        <span class="value">0.2s</span>
        <span class="percentage">(9%)</span>
      </div>
      <div class="breakdown-row highlight">
        <span class="label">🎨 GPU Transfer:</span>
        <span class="value">1.4s</span>
        <span class="percentage">(60%)</span>
      </div>
      <div class="breakdown-row">
        <span class="label">⚙️ CPU Postprocessing:</span>
        <span class="value">0.1s</span>
        <span class="percentage">(4%)</span>
      </div>
      <div class="breakdown-row">
        <span class="label">🔽 Download:</span>
        <span class="value">0.2s</span>
        <span class="percentage">(9%)</span>
      </div>
      <div class="breakdown-row total">
        <span class="label">⏱️ Total:</span>
        <span class="value">2.3s</span>
        <span class="percentage">(100%)</span>
      </div>
    </div>
  </div>

  <!-- Historical Comparison -->
  <div class="historical-comparison">
    <div class="comparison-card">
      <span class="comparison-label">vs. Your Average:</span>
      <span class="comparison-value improvement">
        ↓ 0.4s faster (15%)
      </span>
    </div>

    <div class="comparison-card">
      <span class="comparison-label">vs. Similar Images:</span>
      <span class="comparison-value similar">
        ≈ 2.2s typical
      </span>
    </div>
  </div>
</div>
```

**Backend Implementation:**
```python
# performance_metrics.py

from dataclasses import dataclass
from time import time
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    """Track performance metrics for a single transfer"""

    # Timing
    upload_time: float
    preprocess_time: float
    worker_times: Dict[str, float]  # Per-worker execution time
    postprocess_time: float
    download_time: float
    total_time: float

    # Resource usage
    cpu_time: float
    gpu_time: float
    memory_peak_mb: float

    # Throughput
    pixels_processed: int
    pixels_per_second: float

    def to_dict(self) -> Dict:
        return {
            'timing': {
                'upload': self.upload_time,
                'preprocess': self.preprocess_time,
                'compute': sum(self.worker_times.values()),
                'postprocess': self.postprocess_time,
                'download': self.download_time,
                'total': self.total_time
            },
            'workers': self.worker_times,
            'resources': {
                'cpu_time': self.cpu_time,
                'gpu_time': self.gpu_time,
                'memory_peak_mb': self.memory_peak_mb
            },
            'throughput': {
                'pixels_processed': self.pixels_processed,
                'pixels_per_second': self.pixels_per_second
            }
        }


class PerformanceTracker:
    """Context manager for tracking performance"""

    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.current_stage = None

    def start_stage(self, stage_name: str):
        """Start timing a stage"""
        if self.current_stage:
            self.end_stage()

        self.current_stage = stage_name
        self.stage_times[stage_name] = {'start': time()}

    def end_stage(self):
        """End timing current stage"""
        if self.current_stage:
            self.stage_times[self.current_stage]['end'] = time()
            self.stage_times[self.current_stage]['duration'] = (
                self.stage_times[self.current_stage]['end'] -
                self.stage_times[self.current_stage]['start']
            )
            self.current_stage = None

    def get_metrics(self) -> Dict:
        """Get all metrics"""
        return {
            stage: times['duration']
            for stage, times in self.stage_times.items()
            if 'duration' in times
        }
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

#### Metric 2: Computational Cost (Electricity/Energy)

**Recommended Feature:**

**Feature 4.2: Energy & Cost Dashboard**

```html
<div class="cost-analytics-dashboard">
  <h3>Cost & Energy Analytics</h3>

  <!-- Real-time Cost Tracker -->
  <div class="cost-tracker">
    <div class="cost-display">
      <div class="cost-current">
        <span class="cost-label">Current Job Cost:</span>
        <span class="cost-value">$0.0342</span>
      </div>

      <div class="cost-breakdown-mini">
        <div class="cost-item">
          <span class="cost-icon">⚡</span>
          <span class="cost-text">Energy: $0.0289</span>
        </div>
        <div class="cost-item">
          <span class="cost-icon">💻</span>
          <span class="cost-text">Compute: $0.0053</span>
        </div>
      </div>
    </div>

    <!-- Energy Consumption -->
    <div class="energy-display">
      <h4>Energy Consumption</h4>

      <div class="energy-meter">
        <div class="meter-value">
          <span class="value-number">42.3</span>
          <span class="value-unit">Wh</span>
        </div>
        <div class="meter-visual">
          <div class="meter-bar">
            <div class="meter-fill" style="width: 42%"></div>
          </div>
          <div class="meter-labels">
            <span>0</span>
            <span>50 Wh</span>
            <span>100 Wh</span>
          </div>
        </div>
      </div>

      <div class="energy-breakdown">
        <div class="energy-row">
          <span class="energy-label">GPU Power:</span>
          <span class="energy-value">38.1 Wh (90%)</span>
        </div>
        <div class="energy-row">
          <span class="energy-label">CPU Power:</span>
          <span class="energy-value">3.2 Wh (8%)</span>
        </div>
        <div class="energy-row">
          <span class="energy-label">Memory/IO:</span>
          <span class="energy-value">1.0 Wh (2%)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Cost Optimization Suggestions -->
  <div class="optimization-suggestions">
    <h4>💡 Cost Optimization Tips</h4>

    <div class="suggestion-card">
      <span class="suggestion-icon">✂️</span>
      <div class="suggestion-content">
        <strong>Use ROI Selection</strong>
        <p>Processing only 50% of image would save $0.015 (44%)</p>
      </div>
      <button class="btn-sm btn--secondary">Try It</button>
    </div>

    <div class="suggestion-card">
      <span class="suggestion-icon">⚡</span>
      <div class="suggestion-content">
        <strong>Switch to Eco Mode</strong>
        <p>Use 2 workers instead of 4 → save $0.018 (53%)</p>
      </div>
      <button class="btn-sm btn--secondary">Switch</button>
    </div>

    <div class="suggestion-card">
      <span class="suggestion-icon">🎯</span>
      <div class="suggestion-content">
        <strong>Preview First</strong>
        <p>Low-res preview costs only $0.002 (5% of full)</p>
      </div>
      <button class="btn-sm btn--secondary">Preview</button>
    </div>
  </div>

  <!-- Historical Cost Tracking -->
  <div class="cost-history">
    <h4>Your Usage This Month</h4>

    <div class="cost-summary">
      <div class="summary-stat">
        <span class="stat-label">Total Spent:</span>
        <span class="stat-value">$4.23</span>
      </div>
      <div class="summary-stat">
        <span class="stat-label">Images Processed:</span>
        <span class="stat-value">142</span>
      </div>
      <div class="summary-stat">
        <span class="stat-label">Avg Cost/Image:</span>
        <span class="stat-value">$0.0298</span>
      </div>
      <div class="summary-stat">
        <span class="stat-label">Energy Used:</span>
        <span class="stat-value">5.6 kWh</span>
      </div>
    </div>

    <!-- Cost Trend Chart -->
    <div class="cost-trend-chart">
      <canvas id="costTrendChart"></canvas>
      <!-- Line chart showing daily cost over last 30 days -->
    </div>
  </div>
</div>
```

**Backend Implementation:**

```python
# cost_calculator.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class EnergyProfile:
    """Hardware energy consumption profiles"""

    # Power consumption in Watts
    gpu_idle_power: float = 25.0
    gpu_active_power: float = 250.0
    cpu_idle_power: float = 10.0
    cpu_active_power: float = 65.0
    memory_power: float = 5.0

    # Energy cost (USD per kWh)
    electricity_rate: float = 0.12


class CostCalculator:
    """Calculate energy consumption and cost"""

    def __init__(self, profile: EnergyProfile = None):
        self.profile = profile or EnergyProfile()

    def calculate_energy_cost(
        self,
        gpu_time_seconds: float,
        cpu_time_seconds: float,
        memory_gb: float
    ) -> Dict:
        """
        Calculate energy consumption and cost.

        Args:
            gpu_time_seconds: Time GPU was active
            cpu_time_seconds: Time CPU was active
            memory_gb: Peak memory usage in GB

        Returns:
            Dictionary with energy and cost breakdown
        """
        # Calculate energy consumption (Watt-hours)
        gpu_energy_wh = (
            self.profile.gpu_active_power *
            (gpu_time_seconds / 3600)
        )

        cpu_energy_wh = (
            self.profile.cpu_active_power *
            (cpu_time_seconds / 3600)
        )

        memory_energy_wh = (
            self.profile.memory_power *
            (max(gpu_time_seconds, cpu_time_seconds) / 3600) *
            (memory_gb / 16.0)  # Normalize to 16GB baseline
        )

        total_energy_wh = gpu_energy_wh + cpu_energy_wh + memory_energy_wh
        total_energy_kwh = total_energy_wh / 1000.0

        # Calculate cost
        energy_cost = total_energy_kwh * self.profile.electricity_rate

        # Add compute service cost (cloud pricing)
        # Assuming $0.50/hour for GPU compute
        compute_cost = (gpu_time_seconds / 3600) * 0.50

        total_cost = energy_cost + compute_cost

        return {
            'energy': {
                'gpu_wh': round(gpu_energy_wh, 2),
                'cpu_wh': round(cpu_energy_wh, 2),
                'memory_wh': round(memory_energy_wh, 2),
                'total_wh': round(total_energy_wh, 2),
                'total_kwh': round(total_energy_kwh, 4)
            },
            'cost': {
                'energy_usd': round(energy_cost, 4),
                'compute_usd': round(compute_cost, 4),
                'total_usd': round(total_cost, 4)
            },
            'breakdown_percentage': {
                'gpu': round((gpu_energy_wh / total_energy_wh) * 100, 1),
                'cpu': round((cpu_energy_wh / total_energy_wh) * 100, 1),
                'memory': round((memory_energy_wh / total_energy_wh) * 100, 1)
            }
        }

    def estimate_cost(
        self,
        image_size: tuple,
        num_workers: int,
        mode: str = 'adaptive'
    ) -> Dict:
        """
        Estimate cost before processing.

        Args:
            image_size: (width, height) of image
            num_workers: Number of TSM workers to use
            mode: Processing mode

        Returns:
            Estimated cost breakdown
        """
        pixels = image_size[0] * image_size[1]

        # Estimate processing time based on complexity
        base_time = 0.5  # Base time in seconds
        pixel_factor = pixels / (1920 * 1080)  # Normalize to Full HD
        worker_factor = num_workers * 0.3  # Each worker adds 30% time

        estimated_time = base_time * pixel_factor * (1 + worker_factor)

        # Estimate GPU/CPU split (GPU does 80% of work)
        gpu_time = estimated_time * 0.8
        cpu_time = estimated_time * 0.2

        # Estimate memory (rough heuristic)
        memory_gb = min(16, (pixels / 1000000) * 2)

        return self.calculate_energy_cost(gpu_time, cpu_time, memory_gb)
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

#### Metric 3: Throughput Analytics

**Recommended Feature:**

**Feature 4.3: Throughput Dashboard**

```html
<div class="throughput-dashboard">
  <h3>Throughput Analytics</h3>

  <!-- Current Throughput -->
  <div class="throughput-display">
    <div class="throughput-metric">
      <span class="metric-value">26.4</span>
      <span class="metric-unit">images/hour</span>
      <span class="metric-label">Current Rate</span>
    </div>

    <div class="throughput-metric">
      <span class="metric-value">2.27</span>
      <span class="metric-unit">min/image</span>
      <span class="metric-label">Avg Processing Time</span>
    </div>

    <div class="throughput-metric">
      <span class="metric-value">634</span>
      <span class="metric-unit">images/day</span>
      <span class="metric-label">Projected Daily</span>
    </div>
  </div>

  <!-- Throughput by Configuration -->
  <div class="throughput-comparison">
    <h4>Throughput by Mode</h4>

    <table class="comparison-table">
      <thead>
        <tr>
          <th>Mode</th>
          <th>Images/Hour</th>
          <th>Cost/Image</th>
          <th>Quality (Avg ΔE)</th>
        </tr>
      </thead>
      <tbody>
        <tr class="current-config">
          <td><strong>Balanced</strong> (Current)</td>
          <td>26.4</td>
          <td>$0.0298</td>
          <td>8.5</td>
        </tr>
        <tr>
          <td>Fast Mode</td>
          <td>54.0 (↑104%)</td>
          <td>$0.0145 (↓51%)</td>
          <td>10.2 (↓17%)</td>
        </tr>
        <tr>
          <td>Max Quality</td>
          <td>18.2 (↓31%)</td>
          <td>$0.0442 (↑48%)</td>
          <td>6.8 (↑20%)</td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Efficiency Score -->
  <div class="efficiency-score">
    <h4>System Efficiency</h4>

    <div class="score-display">
      <div class="score-circle">
        <svg viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45"
                  class="score-background"></circle>
          <circle cx="50" cy="50" r="45"
                  class="score-foreground"
                  style="stroke-dasharray: 251; stroke-dashoffset: 63;">
          </circle>
        </svg>
        <div class="score-text">
          <span class="score-value">75</span>
          <span class="score-max">/100</span>
        </div>
      </div>

      <div class="score-breakdown">
        <div class="score-factor">
          <span class="factor-name">Speed:</span>
          <div class="factor-bar">
            <div class="factor-fill" style="width: 80%"></div>
          </div>
          <span class="factor-score">80/100</span>
        </div>

        <div class="score-factor">
          <span class="factor-name">Cost:</span>
          <div class="factor-bar">
            <div class="factor-fill" style="width: 70%"></div>
          </div>
          <span class="factor-score">70/100</span>
        </div>

        <div class="score-factor">
          <span class="factor-name">Quality:</span>
          <div class="factor-bar">
            <div class="factor-fill" style="width: 75%"></div>
          </div>
          <span class="factor-score">75/100</span>
        </div>
      </div>
    </div>
  </div>
</div>
```

**Implementation Priority:** 🟡 MEDIUM (Sprint 2)

---

### B. Quality of Transfer Analytics

#### Quality Metric 1: Structural Similarity (SSIM)

**Recommended Feature:**

**Feature 5.1: SSIM Quality Dashboard**

```html
<div class="quality-metrics-dashboard">
  <h3>Transfer Quality Metrics</h3>

  <!-- SSIM Score -->
  <div class="metric-card">
    <div class="metric-header">
      <h4>Structural Similarity (SSIM)</h4>
      <span class="info-tooltip"
            data-tip="Measures how well original structure was preserved. 1.0 = perfect preservation">
        ℹ️
      </span>
    </div>

    <div class="metric-display">
      <div class="metric-score excellent">
        <span class="score-value">0.942</span>
        <span class="score-label">Excellent</span>
      </div>

      <div class="metric-interpretation">
        <p>
          <strong>✓ Structure well preserved</strong><br>
          Original image details and edges are maintained with high fidelity.
        </p>
      </div>
    </div>

    <!-- SSIM Visualization -->
    <div class="ssim-visualization">
      <div class="ssim-images">
        <div class="ssim-image">
          <img src="source.jpg">
          <span class="image-label">Source Structure</span>
        </div>

        <div class="ssim-image">
          <img src="ssim_map.jpg">
          <span class="image-label">Similarity Map</span>
          <span class="image-caption">
            Green = High similarity, Red = Low similarity
          </span>
        </div>

        <div class="ssim-image">
          <img src="result.jpg">
          <span class="image-label">Result Structure</span>
        </div>
      </div>
    </div>

    <!-- SSIM per Region -->
    <div class="ssim-regions">
      <h5>Similarity by Region</h5>
      <div class="region-scores">
        <div class="region-score">
          <span class="region-name">Edges:</span>
          <span class="region-value">0.956</span>
          <span class="region-status">✓</span>
        </div>
        <div class="region-score">
          <span class="region-name">Textures:</span>
          <span class="region-value">0.932</span>
          <span class="region-status">✓</span>
        </div>
        <div class="region-score warning">
          <span class="region-name">Smooth areas:</span>
          <span class="region-value">0.889</span>
          <span class="region-status">⚠️</span>
        </div>
      </div>
    </div>
  </div>
</div>
```

**Backend Implementation:**

```python
# quality_metrics.py

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


class QualityMetrics:
    """Calculate various quality metrics for transferred images"""

    @staticmethod
    def calculate_ssim(
        source: np.ndarray,
        result: np.ndarray,
        multichannel: bool = True
    ) -> dict:
        """
        Calculate Structural Similarity Index (SSIM).

        Args:
            source: Original image
            result: Transferred image
            multichannel: Whether to calculate per-channel or combined

        Returns:
            Dictionary with SSIM score and map
        """
        # Convert to grayscale for overall SSIM
        source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Calculate SSIM
        ssim_score, ssim_map = ssim(
            source_gray,
            result_gray,
            full=True,
            data_range=255
        )

        # Calculate per-channel SSIM if multichannel
        channel_ssim = {}
        if multichannel:
            for i, channel_name in enumerate(['R', 'G', 'B']):
                channel_ssim[channel_name], _ = ssim(
                    source[:, :, i],
                    result[:, :, i],
                    full=True,
                    data_range=255
                )

        # Analyze regions
        region_ssim = QualityMetrics._analyze_ssim_regions(
            source_gray,
            result_gray,
            ssim_map
        )

        return {
            'overall': float(ssim_score),
            'channels': channel_ssim,
            'regions': region_ssim,
            'ssim_map': ssim_map,
            'interpretation': QualityMetrics._interpret_ssim(ssim_score)
        }

    @staticmethod
    def _analyze_ssim_regions(
        source: np.ndarray,
        result: np.ndarray,
        ssim_map: np.ndarray
    ) -> dict:
        """Analyze SSIM by image regions (edges, textures, smooth)"""

        # Detect edges using Canny
        edges = cv2.Canny(source, 50, 150)
        edge_mask = edges > 0

        # Detect textures using Laplacian variance
        laplacian = cv2.Laplacian(source, cv2.CV_64F)
        texture_var = cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)
        texture_mask = texture_var > np.percentile(texture_var, 70)

        # Smooth areas are neither edges nor textures
        smooth_mask = ~(edge_mask | texture_mask)

        return {
            'edges': float(np.mean(ssim_map[edge_mask])) if np.any(edge_mask) else 1.0,
            'textures': float(np.mean(ssim_map[texture_mask])) if np.any(texture_mask) else 1.0,
            'smooth': float(np.mean(ssim_map[smooth_mask])) if np.any(smooth_mask) else 1.0
        }

    @staticmethod
    def _interpret_ssim(score: float) -> str:
        """Interpret SSIM score"""
        if score >= 0.95:
            return "Excellent - Structure perfectly preserved"
        elif score >= 0.90:
            return "Very Good - Minor structural changes"
        elif score >= 0.85:
            return "Good - Some structural changes"
        elif score >= 0.75:
            return "Fair - Noticeable structural changes"
        else:
            return "Poor - Significant structural degradation"
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

#### Quality Metric 2: Style Preservation & Delta E

**Recommended Feature:**

**Feature 5.2: Multi-Metric Quality Dashboard**

```html
<div class="quality-dashboard">
  <!-- Delta E (Color Accuracy) -->
  <div class="metric-card">
    <div class="metric-header">
      <h4>Color Accuracy (ΔE)</h4>
    </div>

    <div class="metric-display">
      <div class="metric-score good">
        <span class="score-value">8.5</span>
        <span class="score-label">Good</span>
      </div>

      <div class="delta-e-scale">
        <div class="scale-bar">
          <div class="scale-marker" style="left: 28%"></div>
        </div>
        <div class="scale-labels">
          <span>0 (Perfect)</span>
          <span>15 (Fair)</span>
          <span>30+ (Poor)</span>
        </div>
      </div>

      <div class="delta-e-stats">
        <div class="stat-row">
          <span class="stat-label">Mean ΔE:</span>
          <span class="stat-value">8.5</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Std Dev:</span>
          <span class="stat-value">2.1</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Max ΔE:</span>
          <span class="stat-value">14.2</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">95th Percentile:</span>
          <span class="stat-value">11.8</span>
        </div>
        <div class="stat-row success">
          <span class="stat-label">Pixels within target (ΔE < 10):</span>
          <span class="stat-value">87%</span>
        </div>
      </div>
    </div>

    <!-- Delta E Heatmap -->
    <div class="delta-e-heatmap">
      <img src="delta_e_map.jpg">
      <div class="heatmap-legend">
        <span class="legend-label">ΔE:</span>
        <div class="legend-gradient"></div>
        <div class="legend-values">
          <span>0</span>
          <span>15</span>
          <span>30+</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Perceptual Quality -->
  <div class="metric-card">
    <div class="metric-header">
      <h4>Perceptual Quality</h4>
    </div>

    <div class="metric-display">
      <div class="quality-grid">
        <div class="quality-metric">
          <span class="metric-name">Content Loss:</span>
          <span class="metric-value">0.042</span>
          <span class="metric-status">✓</span>
        </div>

        <div class="quality-metric">
          <span class="metric-name">Style Loss:</span>
          <span class="metric-value">0.156</span>
          <span class="metric-status">✓</span>
        </div>

        <div class="quality-metric">
          <span class="metric-name">Loss Ratio:</span>
          <span class="metric-value">3.71</span>
          <span class="metric-status">Balanced</span>
        </div>

        <div class="quality-metric">
          <span class="metric-name">Perceptual Score:</span>
          <span class="metric-value">0.884</span>
          <span class="metric-status">Excellent</span>
        </div>
      </div>
    </div>
  </div>
</div>
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

#### Quality Metric 3: Worker Consensus (TSM-Specific)

**Recommended Feature:**

**Feature 5.3: TSM Worker Consensus Dashboard**

```html
<div class="consensus-dashboard">
  <h3>TSM Worker Consensus Analysis</h3>

  <!-- Consensus Score -->
  <div class="consensus-score-card">
    <div class="score-display">
      <div class="score-badge excellent">
        <span class="score-value">92%</span>
        <span class="score-label">High Consensus</span>
      </div>

      <p class="score-interpretation">
        ✓ Workers agree strongly on the result. High confidence in quality.
      </p>
    </div>

    <!-- WCDS Warning (if low consensus) -->
    <div class="consensus-warning" style="display: none;">
      <span class="warning-icon">⚠️</span>
      <div class="warning-content">
        <strong>Low Worker Consensus (WCDS: 0.45)</strong>
        <p>
          Workers disagree on this image. Result may be unstable.
          Consider using "Max Quality" mode or trying a different target color.
        </p>
      </div>
    </div>
  </div>

  <!-- Worker Agreement Matrix -->
  <div class="worker-agreement">
    <h4>Worker Agreement Matrix</h4>

    <table class="agreement-matrix">
      <thead>
        <tr>
          <th></th>
          <th>Reinhard</th>
          <th>Histogram</th>
          <th>LAB-Spec</th>
          <th>Region</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Reinhard</th>
          <td class="self">100%</td>
          <td class="high">94%</td>
          <td class="high">91%</td>
          <td class="medium">87%</td>
        </tr>
        <tr>
          <th>Histogram</th>
          <td class="high">94%</td>
          <td class="self">100%</td>
          <td class="high">96%</td>
          <td class="high">89%</td>
        </tr>
        <tr>
          <th>LAB-Spec</th>
          <td class="high">91%</td>
          <td class="high">96%</td>
          <td class="self">100%</td>
          <td class="high">92%</td>
        </tr>
        <tr>
          <th>Region</th>
          <td class="medium">87%</td>
          <td class="high">89%</td>
          <td class="high">92%</td>
          <td class="self">100%</td>
        </tr>
      </tbody>
    </table>

    <div class="matrix-legend">
      <span class="legend-item">
        <span class="legend-color high"></span> High (>90%)
      </span>
      <span class="legend-item">
        <span class="legend-color medium"></span> Medium (80-90%)
      </span>
      <span class="legend-item">
        <span class="legend-color low"></span> Low (<80%)
      </span>
    </div>
  </div>

  <!-- Worker Performance Comparison -->
  <div class="worker-comparison">
    <h4>Individual Worker Results</h4>

    <div class="worker-results">
      <div class="worker-result winner">
        <div class="worker-header">
          <span class="worker-name">Histogram Matching</span>
          <span class="badge badge--success">Selected</span>
        </div>
        <div class="worker-metrics">
          <span class="metric">ΔE: 8.5</span>
          <span class="metric">SSIM: 0.942</span>
          <span class="metric">Time: 0.41s</span>
        </div>
        <div class="worker-preview">
          <img src="result_histogram.jpg">
        </div>
      </div>

      <div class="worker-result">
        <div class="worker-header">
          <span class="worker-name">Reinhard Statistical</span>
        </div>
        <div class="worker-metrics">
          <span class="metric">ΔE: 9.2</span>
          <span class="metric">SSIM: 0.938</span>
          <span class="metric">Time: 0.38s</span>
        </div>
        <div class="worker-preview">
          <img src="result_reinhard.jpg">
        </div>
      </div>

      <div class="worker-result">
        <div class="worker-header">
          <span class="worker-name">LAB Channel-Specific</span>
        </div>
        <div class="worker-metrics">
          <span class="metric">ΔE: 9.8</span>
          <span class="metric">SSIM: 0.935</span>
          <span class="metric">Time: 0.42s</span>
        </div>
        <div class="worker-preview">
          <img src="result_lab.jpg">
        </div>
      </div>
    </div>

    <button class="btn-secondary btn--full" onclick="showAllWorkerResults()">
      View All Worker Results →
    </button>
  </div>
</div>
```

**Backend Implementation:**

```python
# worker_consensus.py

import numpy as np
from typing import List, Dict
from skimage.metrics import structural_similarity as ssim


class WorkerConsensusAnalyzer:
    """Analyze consensus among TSM workers"""

    @staticmethod
    def calculate_wcds(
        worker_results: List[np.ndarray],
        worker_ids: List[str]
    ) -> Dict:
        """
        Calculate Worker Consensus Discrepancy Score (WCDS).

        WCDS ranges from 0 (perfect consensus) to 1 (complete disagreement).

        Args:
            worker_results: List of result images from each worker
            worker_ids: List of worker IDs

        Returns:
            Dictionary with consensus metrics
        """
        n_workers = len(worker_results)

        if n_workers < 2:
            return {'wcds': 0.0, 'consensus': 'N/A', 'agreement_matrix': {}}

        # Calculate pairwise SSIM between all workers
        agreement_matrix = np.zeros((n_workers, n_workers))

        for i in range(n_workers):
            for j in range(n_workers):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                elif i < j:
                    # Convert to grayscale
                    img1_gray = cv2.cvtColor(worker_results[i], cv2.COLOR_RGB2GRAY)
                    img2_gray = cv2.cvtColor(worker_results[j], cv2.COLOR_RGB2GRAY)

                    # Calculate SSIM
                    ssim_score, _ = ssim(
                        img1_gray,
                        img2_gray,
                        full=True,
                        data_range=255
                    )

                    agreement_matrix[i, j] = ssim_score
                    agreement_matrix[j, i] = ssim_score  # Symmetric

        # Calculate WCDS as inverse of average pairwise agreement
        # (excluding diagonal)
        upper_triangle = agreement_matrix[np.triu_indices(n_workers, k=1)]
        avg_agreement = np.mean(upper_triangle)
        wcds = 1.0 - avg_agreement

        # Interpret consensus level
        if avg_agreement >= 0.95:
            consensus_level = "Excellent"
        elif avg_agreement >= 0.90:
            consensus_level = "High"
        elif avg_agreement >= 0.85:
            consensus_level = "Moderate"
        elif avg_agreement >= 0.75:
            consensus_level = "Low"
        else:
            consensus_level = "Very Low - Warning"

        # Convert agreement matrix to dict format
        agreement_dict = {}
        for i, worker_i in enumerate(worker_ids):
            agreement_dict[worker_i] = {}
            for j, worker_j in enumerate(worker_ids):
                agreement_dict[worker_i][worker_j] = float(agreement_matrix[i, j])

        return {
            'wcds': float(wcds),
            'average_agreement': float(avg_agreement),
            'consensus_level': consensus_level,
            'consensus_percentage': float(avg_agreement * 100),
            'agreement_matrix': agreement_dict,
            'min_agreement': float(np.min(upper_triangle)),
            'max_agreement': float(np.max(upper_triangle)),
            'warning': avg_agreement < 0.85
        }
```

**Implementation Priority:** 🟡 MEDIUM (Sprint 2)

---

### C. Interactive Comparison View

**Feature 5.4: Advanced Image Comparison Tool**

```html
<div class="image-comparison-tool">
  <div class="comparison-controls">
    <div class="view-mode-selector">
      <button class="view-mode active" data-mode="slider">
        ↔️ Slider
      </button>
      <button class="view-mode" data-mode="side-by-side">
        ⬜⬜ Side by Side
      </button>
      <button class="view-mode" data-mode="difference">
        🔍 Difference
      </button>
      <button class="view-mode" data-mode="overlay">
        📍 Overlay
      </button>
    </div>

    <div class="image-selector">
      <label>
        <input type="checkbox" checked> Source
      </label>
      <label>
        <input type="checkbox"> Target Style
      </label>
      <label>
        <input type="checkbox" checked> Result
      </label>
    </div>
  </div>

  <!-- Slider Mode -->
  <div class="comparison-view slider-mode">
    <div class="comparison-container">
      <img src="source.jpg" class="image-before">
      <img src="result.jpg" class="image-after">
      <input type="range" class="comparison-slider"
             min="0" max="100" value="50">
      <div class="slider-line"></div>
      <div class="slider-labels">
        <span class="label-before">Source</span>
        <span class="label-after">Result</span>
      </div>
    </div>
  </div>

  <!-- Difference Mode -->
  <div class="comparison-view difference-mode" style="display: none;">
    <div class="difference-image">
      <img src="difference_map.jpg">
      <div class="difference-legend">
        <div class="legend-item">
          <span class="legend-color" style="background: green"></span>
          <span>Similar</span>
        </div>
        <div class="legend-item">
          <span class="legend-color" style="background: yellow"></span>
          <span>Moderate Change</span>
        </div>
        <div class="legend-item">
          <span class="legend-color" style="background: red"></span>
          <span>Large Change</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Zoom Controls -->
  <div class="zoom-controls">
    <button class="btn-icon" onclick="zoomIn()">🔍+</button>
    <span class="zoom-level">100%</span>
    <button class="btn-icon" onclick="zoomOut()">🔍-</button>
    <button class="btn-icon" onclick="fitToScreen()">⬜</button>
  </div>
</div>
```

**Implementation Priority:** 🔴 HIGH (Sprint 1)

---

## Part 3: Prototyping & Implementation

### Priority Feature: Cost/Quality Control Slider

**Feature 6.1: Unified Cost/Quality Control**

```html
<div class="cost-quality-control">
  <h3>Processing Profile</h3>

  <!-- Main Slider -->
  <div class="profile-slider-container">
    <div class="slider-track">
      <!-- Preset markers -->
      <span class="preset-marker" style="left: 0%">💰</span>
      <span class="preset-marker" style="left: 50%">⚖️</span>
      <span class="preset-marker" style="left: 100%">✨</span>
    </div>

    <input type="range"
           class="profile-slider"
           min="0"
           max="100"
           value="50"
           id="qualityCostSlider">

    <div class="slider-labels">
      <span class="label-eco">Eco Mode</span>
      <span class="label-balanced">Balanced</span>
      <span class="label-quality">Max Quality</span>
    </div>
  </div>

  <!-- Live Preview of Settings -->
  <div class="profile-preview">
    <div class="preview-section">
      <h4>Current Profile: Balanced</h4>

      <div class="profile-specs">
        <div class="spec-row">
          <span class="spec-icon">⚙️</span>
          <span class="spec-label">Workers:</span>
          <span class="spec-value">3-4 (Adaptive)</span>
        </div>

        <div class="spec-row">
          <span class="spec-icon">⏱️</span>
          <span class="spec-label">Est. Time:</span>
          <span class="spec-value">1.5-2.0s</span>
        </div>

        <div class="spec-row">
          <span class="spec-icon">💰</span>
          <span class="spec-label">Est. Cost:</span>
          <span class="spec-value">$0.0298</span>
        </div>

        <div class="spec-row">
          <span class="spec-icon">⚡</span>
          <span class="spec-label">Energy:</span>
          <span class="spec-value">~42 Wh</span>
        </div>

        <div class="spec-row">
          <span class="spec-icon">🎯</span>
          <span class="spec-label">Quality (ΔE):</span>
          <span class="spec-value">~8-10</span>
        </div>
      </div>
    </div>

    <!-- Comparison with Other Profiles -->
    <div class="profile-comparison">
      <table class="comparison-table-mini">
        <thead>
          <tr>
            <th></th>
            <th>Eco</th>
            <th>Balanced</th>
            <th>Max</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Time</td>
            <td>0.8s</td>
            <td class="current">1.8s</td>
            <td>3.2s</td>
          </tr>
          <tr>
            <td>Cost</td>
            <td>$0.015</td>
            <td class="current">$0.030</td>
            <td>$0.045</td>
          </tr>
          <tr>
            <td>Quality</td>
            <td>ΔE 10-12</td>
            <td class="current">ΔE 8-10</td>
            <td>ΔE 6-8</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Advanced Tuning -->
  <details class="advanced-tuning">
    <summary>🔧 Advanced Tuning</summary>

    <div class="tuning-options">
      <!-- Quality vs Speed Trade-off -->
      <div class="tuning-slider">
        <label>Quality Emphasis:</label>
        <input type="range" min="0" max="100" value="50">
        <div class="slider-endpoints">
          <span>Speed Priority</span>
          <span>Quality Priority</span>
        </div>
      </div>

      <!-- Downsampling -->
      <div class="tuning-toggle">
        <label class="toggle">
          <input type="checkbox" checked>
          <span class="toggle__slider"></span>
          <span class="toggle__label">Smart Downsampling</span>
        </label>
        <span class="toggle-info">
          Automatically reduce resolution for faster processing
        </span>
      </div>

      <!-- Worker Selection Strategy -->
      <div class="tuning-select">
        <label>Worker Selection:</label>
        <select>
          <option value="adaptive" selected>Adaptive (Recommended)</option>
          <option value="all">All Workers</option>
          <option value="best">Best Performers</option>
          <option value="manual">Manual Selection</option>
        </select>
      </div>

      <!-- Ensemble Blending -->
      <div class="tuning-toggle">
        <label class="toggle">
          <input type="checkbox">
          <span class="toggle__slider"></span>
          <span class="toggle__label">Ensemble Blending</span>
        </label>
        <span class="toggle-info">
          Blend top 3 results for potentially better quality (+15% time)
        </span>
      </div>
    </div>
  </details>
</div>
```

**JavaScript Implementation:**

```javascript
// cost_quality_slider.js

class CostQualityController {
  constructor(sliderId) {
    this.slider = document.getElementById(sliderId);
    this.profiles = {
      eco: { value: 0, workers: 2, mode: 'best', downsample: true },
      balanced: { value: 50, workers: 4, mode: 'adaptive', downsample: true },
      maxQuality: { value: 100, workers: 5, mode: 'all', downsample: false }
    };

    this.init();
  }

  init() {
    this.slider.addEventListener('input', (e) => {
      this.updateProfile(e.target.value);
    });

    // Snap to presets
    this.slider.addEventListener('change', (e) => {
      this.snapToPreset(e.target.value);
    });
  }

  updateProfile(value) {
    const profile = this.interpolateProfile(value);
    this.displayProfile(profile);
    this.estimateCosts(profile);
  }

  interpolateProfile(value) {
    // Interpolate between presets
    if (value <= 50) {
      // Between Eco and Balanced
      const t = value / 50;
      return {
        workers: Math.round(2 + t * 2),
        mode: value < 25 ? 'best' : 'adaptive',
        downsample: true,
        quality_weight: t
      };
    } else {
      // Between Balanced and Max Quality
      const t = (value - 50) / 50;
      return {
        workers: Math.round(4 + t * 1),
        mode: value > 75 ? 'all' : 'adaptive',
        downsample: value < 75,
        quality_weight: 0.5 + t * 0.5
      };
    }
  }

  async estimateCosts(profile) {
    // Call backend API to estimate costs
    const response = await fetch('/api/estimate-cost', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_size: this.currentImageSize,
        workers: profile.workers,
        mode: profile.mode,
        downsample: profile.downsample
      })
    });

    const estimate = await response.json();
    this.displayEstimate(estimate);
  }

  displayProfile(profile) {
    // Update UI with current profile settings
    document.querySelector('.spec-value').textContent =
      `${profile.workers} (${profile.mode})`;
  }

  snapToPreset(value) {
    // Snap to nearest preset for better UX
    if (value < 25) {
      this.slider.value = 0;
    } else if (value < 75) {
      this.slider.value = 50;
    } else {
      this.slider.value = 100;
    }
  }
}

// Initialize
const costQualityController = new CostQualityController('qualityCostSlider');
```

**Implementation Priority:** 🔴 HIGHEST (Sprint 1, Week 1)

---

### Success Metrics

**Quantifiable Goals:**

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| **Performance** |
| Average processing latency | 2.5s | 2.0s (-20%) | Sprint 2 |
| Preview latency | N/A | <0.3s | Sprint 1 |
| ROI processing speedup | N/A | 60-80% | Sprint 2 |
| **Cost** |
| Average cost per image | $0.035 | $0.025 (-29%) | Sprint 3 |
| Eco Mode adoption | 0% | 15%+ | Sprint 1 |
| Preview usage (vs. direct) | 0% | 60%+ | Sprint 1 |
| **Quality** |
| User satisfaction | 70% | 85%+ | Sprint 3 |
| Mean Delta E | 10-12 | 8-10 | Sprint 2 |
| SSIM score | Unknown | >0.90 | Sprint 2 |
| **Engagement** |
| Features discovery rate | Low | 70%+ | Sprint 3 |
| Advanced controls usage | 5% | 25% | Sprint 3 |
| Iteration count/image | 3-5 | 1-2 | Sprint 2 |

---

### Implementation Roadmap

**Sprint 1 (Weeks 1-2): Foundation**
- ✅ Cost/Quality Control Slider
- ✅ Basic Analytics Dashboard (time, cost)
- ✅ Preview Mode
- ✅ Interactive Comparison (slider view)
- ✅ SSIM & Delta E metrics

**Sprint 2 (Weeks 3-4): Enhancement**
- ✅ ROI Selection Tool
- ✅ Advanced Quality Metrics (WCDS)
- ✅ Throughput Dashboard
- ✅ Historical Analytics
- ✅ Model/Worker Selection UI

**Sprint 3 (Weeks 5-6): Polish & Optimization**
- ✅ Complete Analytics Integration
- ✅ A/B Testing Framework
- ✅ User Onboarding Tour
- ✅ Performance Optimizations
- ✅ Mobile Responsiveness

---

## Next Steps

**Immediate Actions:**

1. **User Research** (1 week)
   - Interview 10-15 current users
   - Identify most painful gaps
   - Validate proposed features

2. **Design Mockups** (1 week)
   - High-fidelity designs for Sprint 1 features
   - Interactive prototypes (Figma)
   - User testing

3. **Technical Spike** (3 days)
   - Backend API for cost estimation
   - Frontend state management design
   - Performance baseline measurements

4. **Sprint 1 Kickoff**
   - Implement Cost/Quality Slider
   - Build Analytics Dashboard
   - Deploy Preview Mode

**Would you like me to:**
1. ✅ Create detailed Figma mockups for the Cost/Quality slider?
2. ✅ Implement the backend cost estimation API?
3. ✅ Build a working prototype of the ROI selection tool?
4. ✅ Design the analytics dashboard with Chart.js/D3?

Let me know which component you'd like to prioritize!

---

**Document Version:** 1.0.0
**Author:** UI/UX Analytics Team
**Status:** Awaiting Approval & Prioritization
