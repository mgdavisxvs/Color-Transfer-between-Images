/**
 * ROI (Region of Interest) Selector
 * ==================================
 *
 * Interactive ROI selection tool with:
 * - Auto-detection using multiple strategies
 * - Manual drawing and adjustment
 * - Real-time cost savings estimation
 * - Visual feedback
 */

class ROISelector {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      throw new Error(`Canvas with id '${canvasId}' not found`);
    }

    this.ctx = this.canvas.getContext('2d');

    // Configuration
    this.options = {
      minSize: 50,
      maxSize: 5000,
      autoDetectOnLoad: true,
      showAlternatives: true,
      strokeColor: '#00ff00',
      strokeWidth: 3,
      fillOpacity: 0.2,
      ...options
    };

    // State
    this.image = null;
    this.imageData = null;
    this.roi = null;
    this.alternativeROIs = [];
    this.isDrawing = false;
    this.dragStart = null;
    this.mode = 'auto'; // 'auto' or 'manual'
    this.costSavings = 0;
    this.processingTimeReduction = 0;

    // Bind event listeners
    this._bindEvents();
  }

  /**
   * Load image from URL or File
   */
  async loadImage(source) {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.onload = () => {
        this.image = img;

        // Resize canvas to fit image
        const maxWidth = this.canvas.parentElement.clientWidth || 800;
        const scale = Math.min(1, maxWidth / img.width);

        this.canvas.width = img.width * scale;
        this.canvas.height = img.height * scale;
        this.scale = scale;

        // Draw image
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);

        // Auto-detect ROI if enabled
        if (this.options.autoDetectOnLoad) {
          this.autoDetect().then(resolve).catch(reject);
        } else {
          resolve();
        }
      };

      img.onerror = reject;

      if (typeof source === 'string') {
        img.src = source;
      } else if (source instanceof File) {
        const reader = new FileReader();
        reader.onload = (e) => {
          img.src = e.target.result;
        };
        reader.readAsDataURL(source);
      }
    });
  }

  /**
   * Auto-detect ROI using backend API
   */
  async autoDetect(method = 'combined') {
    if (!this.image) {
      throw new Error('No image loaded');
    }

    // Get job ID from canvas data attribute or parent
    const jobId = this.canvas.dataset.jobId;
    if (!jobId) {
      throw new Error('No job ID associated with image');
    }

    try {
      const response = await fetch('/api/roi/auto-detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          job_id: jobId,
          method: method,
          padding_percentage: 0.1,
          min_size_percentage: 0.1
        })
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Auto-detection failed');
      }

      // Set primary ROI
      this.roi = data.analysis.primary_roi;
      this.alternativeROIs = data.analysis.alternative_rois || [];
      this.costSavings = data.analysis.cost_savings_percentage;
      this.processingTimeReduction = data.analysis.processing_time_reduction;

      // Redraw
      this.render();

      // Trigger event
      this._triggerEvent('roi-detected', {
        roi: this.roi,
        alternatives: this.alternativeROIs,
        costSavings: this.costSavings,
        timeReduction: this.processingTimeReduction
      });

      return data.analysis;

    } catch (error) {
      console.error('Auto-detection failed:', error);
      throw error;
    }
  }

  /**
   * Set ROI manually
   */
  setROI(x, y, width, height) {
    // Scale coordinates if needed
    if (this.scale !== 1) {
      x = Math.round(x / this.scale);
      y = Math.round(y / this.scale);
      width = Math.round(width / this.scale);
      height = Math.round(height / this.scale);
    }

    this.roi = {
      x: Math.max(0, x),
      y: Math.max(0, y),
      width: Math.min(width, this.image.width - x),
      height: Math.min(height, this.image.height - y),
      detection_method: 'manual',
      confidence: 1.0
    };

    // Calculate cost savings
    this._calculateCostSavings();

    // Redraw
    this.render();

    // Trigger event
    this._triggerEvent('roi-changed', {
      roi: this.roi,
      costSavings: this.costSavings,
      timeReduction: this.processingTimeReduction
    });
  }

  /**
   * Clear ROI (process full image)
   */
  clearROI() {
    this.roi = null;
    this.alternativeROIs = [];
    this.costSavings = 0;
    this.processingTimeReduction = 0;

    this.render();

    this._triggerEvent('roi-cleared', {});
  }

  /**
   * Get current ROI
   */
  getROI() {
    return this.roi;
  }

  /**
   * Render canvas with ROI overlay
   */
  render() {
    if (!this.image) return;

    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw image
    this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);

    // Draw alternative ROIs
    if (this.options.showAlternatives && this.alternativeROIs.length > 0) {
      this.ctx.strokeStyle = '#ffaa00';
      this.ctx.lineWidth = 1;
      this.ctx.setLineDash([5, 5]);

      for (const altROI of this.alternativeROIs) {
        const scaled = this._scaleROI(altROI);
        this.ctx.strokeRect(scaled.x, scaled.y, scaled.width, scaled.height);
      }

      this.ctx.setLineDash([]);
    }

    // Draw primary ROI
    if (this.roi) {
      const scaled = this._scaleROI(this.roi);

      // Fill
      this.ctx.fillStyle = this.options.strokeColor + Math.round(this.options.fillOpacity * 255).toString(16).padStart(2, '0');
      this.ctx.fillRect(scaled.x, scaled.y, scaled.width, scaled.height);

      // Stroke
      this.ctx.strokeStyle = this.options.strokeColor;
      this.ctx.lineWidth = this.options.strokeWidth;
      this.ctx.strokeRect(scaled.x, scaled.y, scaled.width, scaled.height);

      // Label
      this.ctx.fillStyle = this.options.strokeColor;
      this.ctx.font = 'bold 14px Arial';
      const label = `${this.roi.detection_method} (${(this.roi.confidence * 100).toFixed(0)}%)`;
      this.ctx.fillText(label, scaled.x + 5, scaled.y - 5);

      // Dimensions label
      this.ctx.fillStyle = '#ffffff';
      this.ctx.fillRect(scaled.x, scaled.y + scaled.height - 25, 150, 25);
      this.ctx.fillStyle = '#000000';
      this.ctx.font = '12px monospace';
      this.ctx.fillText(
        `${this.roi.width}×${this.roi.height}px`,
        scaled.x + 5,
        scaled.y + scaled.height - 8
      );
    }
  }

  /**
   * Enable manual drawing mode
   */
  enableManualMode() {
    this.mode = 'manual';
    this.canvas.style.cursor = 'crosshair';
  }

  /**
   * Enable auto-detect mode
   */
  enableAutoMode() {
    this.mode = 'auto';
    this.canvas.style.cursor = 'default';
  }

  /**
   * Select an alternative ROI
   */
  selectAlternative(index) {
    if (index >= 0 && index < this.alternativeROIs.length) {
      this.roi = this.alternativeROIs[index];
      this._calculateCostSavings();
      this.render();

      this._triggerEvent('roi-changed', {
        roi: this.roi,
        costSavings: this.costSavings,
        timeReduction: this.processingTimeReduction
      });
    }
  }

  // ========================================================================
  // PRIVATE METHODS
  // ========================================================================

  _bindEvents() {
    // Mouse events for manual drawing
    this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
    this.canvas.addEventListener('mouseup', (e) => this._onMouseUp(e));
    this.canvas.addEventListener('mouseleave', (e) => this._onMouseUp(e));

    // Touch events for mobile
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.canvas.dispatchEvent(mouseEvent);
    });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      this.canvas.dispatchEvent(mouseEvent);
    });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      const mouseEvent = new MouseEvent('mouseup', {});
      this.canvas.dispatchEvent(mouseEvent);
    });
  }

  _onMouseDown(e) {
    if (this.mode !== 'manual') return;

    const rect = this.canvas.getBoundingClientRect();
    this.dragStart = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
    this.isDrawing = true;
  }

  _onMouseMove(e) {
    if (!this.isDrawing || this.mode !== 'manual') return;

    const rect = this.canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    // Calculate ROI
    const x = Math.min(this.dragStart.x, currentX);
    const y = Math.min(this.dragStart.y, currentY);
    const width = Math.abs(currentX - this.dragStart.x);
    const height = Math.abs(currentY - this.dragStart.y);

    // Temporarily set ROI for visual feedback
    this.roi = {
      x: Math.round(x / this.scale),
      y: Math.round(y / this.scale),
      width: Math.round(width / this.scale),
      height: Math.round(height / this.scale),
      detection_method: 'manual',
      confidence: 1.0
    };

    this.render();
  }

  _onMouseUp(e) {
    if (!this.isDrawing) return;

    this.isDrawing = false;

    if (this.roi && this.roi.width >= this.options.minSize && this.roi.height >= this.options.minSize) {
      this._calculateCostSavings();

      this._triggerEvent('roi-changed', {
        roi: this.roi,
        costSavings: this.costSavings,
        timeReduction: this.processingTimeReduction
      });
    } else {
      // ROI too small, clear it
      this.roi = null;
      this.render();
    }
  }

  _scaleROI(roi) {
    return {
      x: roi.x * this.scale,
      y: roi.y * this.scale,
      width: roi.width * this.scale,
      height: roi.height * this.scale
    };
  }

  _calculateCostSavings() {
    if (!this.image || !this.roi) {
      this.costSavings = 0;
      this.processingTimeReduction = 0;
      return;
    }

    const fullArea = this.image.width * this.image.height;
    const roiArea = this.roi.width * this.roi.height;

    this.costSavings = ((fullArea - roiArea) / fullArea) * 100;
    this.processingTimeReduction = this.costSavings * 0.8; // Assume 80% linear
  }

  _triggerEvent(eventName, detail) {
    const event = new CustomEvent(eventName, {
      detail: detail,
      bubbles: true
    });
    this.canvas.dispatchEvent(event);
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ROISelector;
}
