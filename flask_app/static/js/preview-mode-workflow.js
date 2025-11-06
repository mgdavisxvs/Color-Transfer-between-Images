/**
 * Preview Mode Workflow - Solves "No preview mode"
 * =================================================
 *
 * Features:
 * - Low-resolution preview (512px max)
 * - Side-by-side comparison slider
 * - Iteration history (last 5 attempts)
 * - Upgrade to full resolution
 * - Cost comparison (preview vs full-res)
 */

class PreviewModeWorkflow {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container ${containerId} not found`);
    }

    this.options = {
      maxPreviewDimension: 512,
      maxIterationHistory: 5,
      previewCostMultiplier: 0.23, // Preview is 77% cheaper
      onUpgradeToFullRes: null,
      onIterationSaved: null,
      ...options
    };

    // State
    this.sourceImage = null;
    this.currentPreview = null;
    this.iterationHistory = [];
    this.comparisonMode = 'slider'; // 'slider', 'side-by-side', 'overlay'
    this.sliderPosition = 50;

    this.render();
  }

  /**
   * Set source image
   */
  setSourceImage(imageUrl, jobId) {
    this.sourceImage = { url: imageUrl, jobId };
    this.currentPreview = null;
    this.render();
  }

  /**
   * Set preview result
   */
  setPreviewResult(imageUrl, settings, cost) {
    this.currentPreview = {
      url: imageUrl,
      settings,
      cost,
      timestamp: new Date().toISOString(),
      isPreview: true
    };
    this.render();
  }

  /**
   * Save current preview to iteration history
   */
  saveToHistory() {
    if (!this.currentPreview) {
      return;
    }

    // Add to history
    this.iterationHistory.unshift({
      ...this.currentPreview,
      id: Date.now()
    });

    // Keep only maxIterationHistory
    this.iterationHistory = this.iterationHistory.slice(0, this.options.maxIterationHistory);

    // Callback
    if (this.options.onIterationSaved) {
      this.options.onIterationSaved(this.currentPreview);
    }

    this.render();
  }

  /**
   * Load iteration from history
   */
  loadIteration(id) {
    const iteration = this.iterationHistory.find(it => it.id === id);
    if (iteration) {
      this.currentPreview = iteration;
      this.render();
    }
  }

  /**
   * Upgrade current preview to full resolution
   */
  upgradeToFullRes() {
    if (!this.currentPreview || !this.options.onUpgradeToFullRes) {
      return;
    }

    this.options.onUpgradeToFullRes(this.currentPreview.settings);
  }

  /**
   * Set comparison mode
   */
  setComparisonMode(mode) {
    this.comparisonMode = mode;
    this.render();
  }

  /**
   * Render the component
   */
  render() {
    this.container.innerHTML = `
      <div class="preview-mode-workflow">
        ${!this.sourceImage ? this._renderNoImage() : this._renderWorkflow()}
      </div>
    `;

    this._attachEventListeners();
  }

  /**
   * Render no image state
   */
  _renderNoImage() {
    return `
      <div class="no-image-state">
        <div class="placeholder-icon">🖼️</div>
        <p>Upload an image to start previewing</p>
      </div>
    `;
  }

  /**
   * Render main workflow
   */
  _renderWorkflow() {
    return `
      <!-- Preview Notice -->
      ${this.currentPreview && this.currentPreview.isPreview ? `
        <div class="preview-notice">
          <div class="notice-icon">⚡</div>
          <div class="notice-text">
            <strong>Preview Mode</strong> – Fast, low-cost iteration. Upgrade to full resolution when ready.
          </div>
          <div class="notice-savings">
            Saves ${((1 - this.options.previewCostMultiplier) * 100).toFixed(0)}%
          </div>
        </div>
      ` : ''}

      <!-- Comparison Area -->
      <div class="comparison-area">
        <!-- Comparison Mode Selector -->
        <div class="comparison-toolbar">
          <div class="comparison-modes">
            <button class="mode-toggle ${this.comparisonMode === 'slider' ? 'active' : ''}" data-mode="slider">
              🔀 Slider
            </button>
            <button class="mode-toggle ${this.comparisonMode === 'side-by-side' ? 'active' : ''}" data-mode="side-by-side">
              ↔️ Side-by-Side
            </button>
            <button class="mode-toggle ${this.comparisonMode === 'overlay' ? 'active' : ''}" data-mode="overlay">
              👁️ Overlay
            </button>
          </div>

          ${this.currentPreview ? `
            <button class="save-iteration-btn" onclick="window.previewWorkflow.saveToHistory()">
              💾 Save to History
            </button>
          ` : ''}
        </div>

        <!-- Comparison Viewer -->
        <div class="comparison-viewer">
          ${this._renderComparisonView()}
        </div>
      </div>

      <!-- Iteration History -->
      ${this.iterationHistory.length > 0 ? `
        <div class="iteration-history">
          <h4>📚 Iteration History (${this.iterationHistory.length}/${this.options.maxIterationHistory})</h4>
          <div class="history-grid">
            ${this.iterationHistory.map(iteration => this._renderIterationCard(iteration)).join('')}
          </div>
        </div>
      ` : ''}

      <!-- Upgrade Section -->
      ${this.currentPreview ? `
        <div class="upgrade-section">
          <div class="upgrade-info">
            <div class="upgrade-icon">🚀</div>
            <div class="upgrade-text">
              <h4>Ready for Full Resolution?</h4>
              <p>Upgrade this preview to full quality (original dimensions)</p>
            </div>
          </div>
          <div class="upgrade-costs">
            <div class="cost-item">
              <div class="cost-label">Preview Cost:</div>
              <div class="cost-value preview-cost">$${this.currentPreview.cost.toFixed(3)}</div>
            </div>
            <div class="cost-arrow">→</div>
            <div class="cost-item">
              <div class="cost-label">Full-Res Cost:</div>
              <div class="cost-value fullres-cost">$${(this.currentPreview.cost / this.options.previewCostMultiplier).toFixed(3)}</div>
            </div>
          </div>
          <button class="upgrade-btn" onclick="window.previewWorkflow.upgradeToFullRes()">
            💎 Upgrade to Full Resolution
          </button>
        </div>
      ` : ''}
    `;
  }

  /**
   * Render comparison view based on mode
   */
  _renderComparisonView() {
    if (!this.currentPreview) {
      return `
        <div class="no-preview-state">
          <p>No preview yet. Process an image to see results.</p>
        </div>
      `;
    }

    switch (this.comparisonMode) {
      case 'slider':
        return this._renderSliderComparison();
      case 'side-by-side':
        return this._renderSideBySideComparison();
      case 'overlay':
        return this._renderOverlayComparison();
      default:
        return this._renderSliderComparison();
    }
  }

  /**
   * Render slider comparison
   */
  _renderSliderComparison() {
    return `
      <div class="slider-comparison">
        <div class="comparison-container" id="comparison-container">
          <img src="${this.sourceImage.url}" alt="Source" class="source-image">
          <div class="result-overlay" id="result-overlay" style="width: ${this.sliderPosition}%">
            <img src="${this.currentPreview.url}" alt="Result" class="result-image">
          </div>
          <div class="slider-handle" id="slider-handle" style="left: ${this.sliderPosition}%">
            <div class="handle-bar"></div>
          </div>
        </div>
        <div class="slider-labels">
          <span>Original</span>
          <span>Result</span>
        </div>
      </div>
    `;
  }

  /**
   * Render side-by-side comparison
   */
  _renderSideBySideComparison() {
    return `
      <div class="side-by-side-comparison">
        <div class="comparison-half">
          <img src="${this.sourceImage.url}" alt="Source">
          <div class="image-label">Original</div>
        </div>
        <div class="comparison-half">
          <img src="${this.currentPreview.url}" alt="Result">
          <div class="image-label">Result</div>
        </div>
      </div>
    `;
  }

  /**
   * Render overlay comparison
   */
  _renderOverlayComparison() {
    return `
      <div class="overlay-comparison">
        <div class="overlay-container">
          <img src="${this.sourceImage.url}" alt="Source" class="base-image">
          <img src="${this.currentPreview.url}" alt="Result" class="overlay-image" style="opacity: 0.5">
        </div>
        <div class="opacity-control">
          <label>Overlay Opacity:</label>
          <input type="range" id="overlay-opacity" min="0" max="100" value="50">
          <span id="opacity-value">50%</span>
        </div>
      </div>
    `;
  }

  /**
   * Render iteration card
   */
  _renderIterationCard(iteration) {
    return `
      <div class="iteration-card" data-iteration-id="${iteration.id}">
        <img src="${iteration.url}" alt="Iteration">
        <div class="iteration-info">
          <div class="iteration-time">${new Date(iteration.timestamp).toLocaleTimeString()}</div>
          <div class="iteration-cost">$${iteration.cost.toFixed(3)}</div>
        </div>
        <button class="load-iteration-btn" onclick="window.previewWorkflow.loadIteration(${iteration.id})">
          Load
        </button>
      </div>
    `;
  }

  /**
   * Attach event listeners
   */
  _attachEventListeners() {
    // Comparison mode toggles
    this.container.querySelectorAll('.mode-toggle').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const mode = e.currentTarget.dataset.mode;
        this.setComparisonMode(mode);
      });
    });

    // Slider comparison
    if (this.comparisonMode === 'slider') {
      this._initSliderComparison();
    }

    // Overlay opacity control
    if (this.comparisonMode === 'overlay') {
      const opacitySlider = this.container.querySelector('#overlay-opacity');
      const opacityValue = this.container.querySelector('#opacity-value');
      const overlayImage = this.container.querySelector('.overlay-image');

      if (opacitySlider && overlayImage) {
        opacitySlider.addEventListener('input', (e) => {
          const opacity = e.target.value / 100;
          overlayImage.style.opacity = opacity;
          opacityValue.textContent = e.target.value + '%';
        });
      }
    }
  }

  /**
   * Initialize slider comparison interaction
   */
  _initSliderComparison() {
    const container = this.container.querySelector('#comparison-container');
    const handle = this.container.querySelector('#slider-handle');
    const overlay = this.container.querySelector('#result-overlay');

    if (!container || !handle || !overlay) return;

    let isDragging = false;

    const updateSlider = (clientX) => {
      const rect = container.getBoundingClientRect();
      const x = clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

      this.sliderPosition = percentage;
      handle.style.left = percentage + '%';
      overlay.style.width = percentage + '%';
    };

    // Mouse events
    handle.addEventListener('mousedown', () => {
      isDragging = true;
    });

    document.addEventListener('mousemove', (e) => {
      if (isDragging) {
        updateSlider(e.clientX);
      }
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // Touch events
    handle.addEventListener('touchstart', (e) => {
      isDragging = true;
      e.preventDefault();
    });

    document.addEventListener('touchmove', (e) => {
      if (isDragging) {
        const touch = e.touches[0];
        updateSlider(touch.clientX);
        e.preventDefault();
      }
    });

    document.addEventListener('touchend', () => {
      isDragging = false;
    });

    // Click to position
    container.addEventListener('click', (e) => {
      if (e.target === handle || handle.contains(e.target)) return;
      updateSlider(e.clientX);
    });
  }
}

// Global instance for easy access
window.previewWorkflow = null;

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PreviewModeWorkflow;
}
