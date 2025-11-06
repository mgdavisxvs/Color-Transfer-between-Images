/**
 * Cost Preview Component - Solves "No cost transparency"
 * =======================================================
 *
 * Features:
 * - Pre-process cost estimation
 * - Real-time cost/quality slider
 * - Savings calculator (ROI selection)
 * - Processing time estimate
 * - Energy consumption display
 * - Cost breakdown visualization
 */

class CostPreviewComponent {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container ${containerId} not found`);
    }

    this.options = {
      defaultMode: 'balanced', // 'eco', 'balanced', 'max_quality'
      showEnergyMetrics: true,
      showDetailedBreakdown: false,
      currency: 'USD',
      currencySymbol: '$',
      ...options
    };

    // State
    this.imageSize = null;
    this.roiSelected = false;
    this.roiSavings = 0;
    this.currentMode = this.options.defaultMode;
    this.estimate = null;

    // Mode configurations
    this.modes = {
      eco: {
        name: 'Eco',
        icon: '🌱',
        description: 'Fastest, lowest cost',
        costMultiplier: 0.29, // Relative to balanced
        timeMultiplier: 0.40,
        qualityScore: 70
      },
      balanced: {
        name: 'Balanced',
        icon: '⚖️',
        description: 'Recommended for most uses',
        costMultiplier: 1.0,
        timeMultiplier: 1.0,
        qualityScore: 85
      },
      max_quality: {
        name: 'Max Quality',
        icon: '💎',
        description: 'Best results, slower',
        costMultiplier: 2.0,
        timeMultiplier: 1.6,
        qualityScore: 95
      }
    };

    this.render();
  }

  /**
   * Set image dimensions for cost estimation
   */
  setImageSize(width, height) {
    this.imageSize = { width, height };
    this.updateEstimate();
  }

  /**
   * Set ROI selection status
   */
  setROI(enabled, savingsPercentage = 0) {
    this.roiSelected = enabled;
    this.roiSavings = savingsPercentage;
    this.updateEstimate();
  }

  /**
   * Set processing mode
   */
  setMode(mode) {
    if (!this.modes[mode]) {
      console.error(`Invalid mode: ${mode}`);
      return;
    }
    this.currentMode = mode;
    this.updateEstimate();
  }

  /**
   * Fetch cost estimate from API
   */
  async updateEstimate() {
    if (!this.imageSize) {
      return;
    }

    try {
      const response = await fetch('/api/analytics/cost/estimate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_width: this.imageSize.width,
          image_height: this.imageSize.height,
          processing_mode: this.currentMode
        })
      });

      const data = await response.json();

      if (data.success) {
        this.estimate = data.estimate;

        // Apply ROI savings
        if (this.roiSelected && this.roiSavings > 0) {
          this.estimate.roi_cost = this.estimate.estimated_cost_usd * (1 - this.roiSavings / 100);
          this.estimate.roi_time = this.estimate.estimated_time_seconds * (1 - this.roiSavings / 100);
        }

        this.render();
      }
    } catch (error) {
      console.error('Failed to fetch cost estimate:', error);
    }
  }

  /**
   * Render the component
   */
  render() {
    if (!this.estimate) {
      this.container.innerHTML = this._renderNoData();
      return;
    }

    const mode = this.modes[this.currentMode];
    const cost = this.roiSelected ? this.estimate.roi_cost : this.estimate.estimated_cost_usd;
    const time = this.roiSelected ? this.estimate.roi_time : this.estimate.estimated_time_seconds;
    const energy = this.estimate.estimated_energy_wh;

    this.container.innerHTML = `
      <div class="cost-preview-component">
        <!-- Header -->
        <div class="cost-header">
          <h3>💰 Cost Estimate</h3>
          ${this.roiSelected ? `
            <div class="savings-badge">
              -${this.roiSavings.toFixed(0)}% with ROI Selection
            </div>
          ` : ''}
        </div>

        <!-- Main Cost Display -->
        <div class="cost-main">
          <div class="cost-amount">
            <div class="cost-label">Estimated Cost</div>
            <div class="cost-value">
              ${this.roiSelected ? `
                <span class="original-cost">${this.options.currencySymbol}${this.estimate.estimated_cost_usd.toFixed(3)}</span>
              ` : ''}
              <span class="final-cost">${this.options.currencySymbol}${cost.toFixed(3)}</span>
            </div>
            <div class="cost-breakdown-link">
              <button class="btn-link" onclick="this.closest('.cost-preview-component').querySelector('.detailed-breakdown').classList.toggle('hidden')">
                ${this.options.showDetailedBreakdown ? 'Hide' : 'Show'} breakdown
              </button>
            </div>
          </div>

          <div class="cost-metrics">
            <div class="metric">
              <div class="metric-icon">⏱️</div>
              <div class="metric-value">${time.toFixed(1)}s</div>
              <div class="metric-label">Processing Time</div>
            </div>

            ${this.options.showEnergyMetrics ? `
              <div class="metric">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">${energy.toFixed(1)} Wh</div>
                <div class="metric-label">Energy</div>
              </div>
            ` : ''}

            <div class="metric">
              <div class="metric-icon">${mode.icon}</div>
              <div class="metric-value">${mode.qualityScore}%</div>
              <div class="metric-label">Quality</div>
            </div>
          </div>
        </div>

        <!-- Mode Selector -->
        <div class="mode-selector">
          <div class="mode-label">Quality / Cost Balance:</div>
          <div class="mode-buttons">
            ${Object.entries(this.modes).map(([key, modeData]) => `
              <button
                class="mode-btn ${this.currentMode === key ? 'active' : ''}"
                data-mode="${key}"
              >
                <span class="mode-icon">${modeData.icon}</span>
                <span class="mode-name">${modeData.name}</span>
                <span class="mode-desc">${modeData.description}</span>
              </button>
            `).join('')}
          </div>
        </div>

        <!-- Detailed Breakdown (Hidden by default) -->
        <div class="detailed-breakdown ${this.options.showDetailedBreakdown ? '' : 'hidden'}">
          <h4>Cost Breakdown</h4>
          <div class="breakdown-items">
            <div class="breakdown-item">
              <span class="item-label">Energy Cost:</span>
              <span class="item-value">${this.options.currencySymbol}${(cost * 0.4).toFixed(4)}</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Compute Cost:</span>
              <span class="item-value">${this.options.currencySymbol}${(cost * 0.6).toFixed(4)}</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Workers:</span>
              <span class="item-value">${this.estimate.workers}</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Image Size:</span>
              <span class="item-value">${this.imageSize.width}×${this.imageSize.height}px</span>
            </div>
          </div>
        </div>

        <!-- Comparison with other modes -->
        <div class="mode-comparison">
          <div class="comparison-header">Compare Modes:</div>
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Mode</th>
                <th>Cost</th>
                <th>Time</th>
                <th>Quality</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(this.modes).map(([key, modeData]) => {
                const modeCost = this.estimate.estimated_cost_usd * modeData.costMultiplier;
                const modeTime = this.estimate.estimated_time_seconds * modeData.timeMultiplier;
                const isActive = this.currentMode === key;

                return `
                  <tr class="${isActive ? 'active-row' : ''}">
                    <td>${modeData.icon} ${modeData.name}</td>
                    <td>${this.options.currencySymbol}${modeCost.toFixed(3)}</td>
                    <td>${modeTime.toFixed(1)}s</td>
                    <td>${modeData.qualityScore}%</td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>

        <!-- Trust Badge -->
        <div class="trust-section">
          <div class="trust-icon">🔒</div>
          <div class="trust-text">
            <strong>Transparent Pricing</strong> – You only pay for what you use. No hidden fees.
          </div>
        </div>
      </div>
    `;

    this._attachEventListeners();
  }

  /**
   * Render no data state
   */
  _renderNoData() {
    return `
      <div class="cost-preview-component no-data">
        <div class="placeholder-icon">📊</div>
        <p>Upload an image to see cost estimate</p>
      </div>
    `;
  }

  /**
   * Attach event listeners
   */
  _attachEventListeners() {
    // Mode selection buttons
    this.container.querySelectorAll('.mode-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const mode = e.currentTarget.dataset.mode;
        this.setMode(mode);
      });
    });
  }

  /**
   * Get current estimate
   */
  getEstimate() {
    return this.estimate;
  }

  /**
   * Get selected mode
   */
  getMode() {
    return this.currentMode;
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CostPreviewComponent;
}
