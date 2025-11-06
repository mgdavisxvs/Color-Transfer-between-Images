/**
 * Smart Color Selector - Solves "213 colors = analysis paralysis"
 * ================================================================
 *
 * Features:
 * - Curated "Popular" colors (14 most-used)
 * - Smart search with fuzzy matching
 * - Use-case filters (Corporate, Photography, Fashion, Industrial)
 * - Color picker (RGB → nearest RAL)
 * - Recent colors memory
 * - Lazy loading for full palette
 */

class SmartColorSelector {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container ${containerId} not found`);
    }

    this.options = {
      defaultView: 'popular', // 'popular', 'search', 'all', 'picker'
      showColorPicker: true,
      maxRecent: 5,
      onColorSelected: null,
      preloadFullPalette: false,
      ...options
    };

    // Popular colors (14 most-used RAL colors)
    this.popularColors = [
      { ral_code: "RAL 3000", name: "Flame Red", hex: "#AF2B1E", rgb: [175, 43, 30] },
      { ral_code: "RAL 5002", name: "Ultramarine Blue", hex: "#20214F", rgb: [32, 33, 79] },
      { ral_code: "RAL 5010", name: "Gentian Blue", hex: "#0E294B", rgb: [14, 41, 75] },
      { ral_code: "RAL 5011", name: "Steel Blue", hex: "#231A24", rgb: [35, 26, 36] },
      { ral_code: "RAL 5012", name: "Light Blue", hex: "#256D7B", rgb: [37, 109, 123] },
      { ral_code: "RAL 5015", name: "Sky Blue", hex: "#2874B2", rgb: [40, 116, 178] },
      { ral_code: "RAL 5018", name: "Turquoise Blue", hex: "#0E888C", rgb: [14, 136, 140] },
      { ral_code: "RAL 6011", name: "Reseda Green", hex: "#68825B", rgb: [104, 130, 91] },
      { ral_code: "RAL 7016", name: "Anthracite Grey", hex: "#383E42", rgb: [56, 62, 66] },
      { ral_code: "RAL 7031", name: "Blue Grey", hex: "#474A50", rgb: [71, 74, 80] },
      { ral_code: "RAL 7035", name: "Light Grey", hex: "#D7D7D7", rgb: [215, 215, 215] },
      { ral_code: "RAL 7045", name: "Telegrey 1", hex: "#91969A", rgb: [145, 150, 154] },
      { ral_code: "RAL 9006", name: "White Aluminium", hex: "#A5A5A5", rgb: [165, 165, 165] },
      { ral_code: "RAL 9011", name: "Graphite Black", hex: "#1C1C1C", rgb: [28, 28, 28] }
    ];

    // Use-case categories
    this.categories = {
      corporate: {
        name: "Corporate Branding",
        colors: ["RAL 5010", "RAL 7016", "RAL 3000", "RAL 9011", "RAL 5002"]
      },
      photography: {
        name: "Photography",
        colors: ["RAL 7035", "RAL 7045", "RAL 9006", "RAL 5012", "RAL 6011"]
      },
      fashion: {
        name: "Fashion & Design",
        colors: ["RAL 3000", "RAL 5015", "RAL 5018", "RAL 7031", "RAL 5011"]
      },
      industrial: {
        name: "Industrial",
        colors: ["RAL 7016", "RAL 9011", "RAL 7035", "RAL 5010", "RAL 6011"]
      }
    };

    // State
    this.allColors = null;
    this.recentColors = this._loadRecentColors();
    this.selectedColor = null;
    this.currentView = this.options.defaultView;

    // Initialize
    this.render();

    // Preload full palette if needed
    if (this.options.preloadFullPalette) {
      this.loadFullPalette();
    }
  }

  /**
   * Render the color selector UI
   */
  render() {
    this.container.innerHTML = `
      <div class="smart-color-selector">
        <!-- Header -->
        <div class="selector-header">
          <h3>Select Target Color</h3>
          <div class="selector-tabs">
            <button class="tab-btn ${this.currentView === 'popular' ? 'active' : ''}" data-view="popular">
              ⭐ Popular
            </button>
            <button class="tab-btn ${this.currentView === 'recent' ? 'active' : ''}" data-view="recent">
              🕒 Recent
            </button>
            <button class="tab-btn ${this.currentView === 'category' ? 'active' : ''}" data-view="category">
              📁 By Use Case
            </button>
            <button class="tab-btn ${this.currentView === 'search' ? 'active' : ''}" data-view="search">
              🔍 Search All
            </button>
            ${this.options.showColorPicker ? `
              <button class="tab-btn ${this.currentView === 'picker' ? 'active' : ''}" data-view="picker">
                🎨 Color Picker
              </button>
            ` : ''}
          </div>
        </div>

        <!-- Content Area -->
        <div class="selector-content">
          ${this._renderCurrentView()}
        </div>

        <!-- Selected Color Display -->
        <div class="selected-color-display" id="selected-color-display">
          ${this.selectedColor ? this._renderSelectedColor() : '<p class="placeholder">No color selected</p>'}
        </div>
      </div>
    `;

    this._attachEventListeners();
  }

  /**
   * Render current view content
   */
  _renderCurrentView() {
    switch (this.currentView) {
      case 'popular':
        return this._renderPopularView();
      case 'recent':
        return this._renderRecentView();
      case 'category':
        return this._renderCategoryView();
      case 'search':
        return this._renderSearchView();
      case 'picker':
        return this._renderPickerView();
      default:
        return this._renderPopularView();
    }
  }

  /**
   * Render popular colors view
   */
  _renderPopularView() {
    return `
      <div class="color-grid">
        ${this.popularColors.map(color => this._renderColorCard(color)).join('')}
      </div>
      <div class="view-all-link">
        <button class="btn-link" onclick="document.querySelector('[data-view=search]').click()">
          View all ${this.allColors ? this.allColors.length : '200+'} colors →
        </button>
      </div>
    `;
  }

  /**
   * Render recent colors view
   */
  _renderRecentView() {
    if (this.recentColors.length === 0) {
      return `
        <div class="empty-state">
          <p>No recent colors yet.</p>
          <p class="hint">Colors you select will appear here for quick access.</p>
        </div>
      `;
    }

    return `
      <div class="color-grid">
        ${this.recentColors.map(color => this._renderColorCard(color)).join('')}
      </div>
    `;
  }

  /**
   * Render category view
   */
  _renderCategoryView() {
    return `
      <div class="category-list">
        ${Object.entries(this.categories).map(([key, category]) => `
          <div class="category-section">
            <h4>${category.name}</h4>
            <div class="color-grid compact">
              ${category.colors.map(ralCode => {
                const color = this.popularColors.find(c => c.ral_code === ralCode);
                return color ? this._renderColorCard(color, true) : '';
              }).join('')}
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  /**
   * Render search view
   */
  _renderSearchView() {
    return `
      <div class="search-view">
        <div class="search-bar">
          <input
            type="text"
            id="color-search-input"
            placeholder="Search by name or RAL code (e.g., 'red' or '3000')..."
            autocomplete="off"
          />
          <span class="search-icon">🔍</span>
        </div>
        <div id="search-results" class="search-results">
          ${this.allColors ? `
            <div class="color-grid">
              ${this.allColors.slice(0, 20).map(color => this._renderColorCard(color)).join('')}
            </div>
            <p class="hint">Showing first 20 colors. Type to search.</p>
          ` : `
            <div class="loading-state">
              <div class="spinner"></div>
              <p>Loading full palette...</p>
            </div>
          `}
        </div>
      </div>
    `;
  }

  /**
   * Render color picker view
   */
  _renderPickerView() {
    return `
      <div class="picker-view">
        <div class="picker-controls">
          <label for="color-picker-input">Pick a color:</label>
          <input
            type="color"
            id="color-picker-input"
            value="#AF2B1E"
          />
          <button id="find-nearest-btn" class="btn-primary">Find Nearest RAL Color</button>
        </div>
        <div id="picker-result" class="picker-result">
          <p class="hint">Select a color and click "Find Nearest RAL Color"</p>
        </div>
      </div>
    `;
  }

  /**
   * Render individual color card
   */
  _renderColorCard(color, compact = false) {
    const isSelected = this.selectedColor && this.selectedColor.ral_code === color.ral_code;

    return `
      <div
        class="color-card ${compact ? 'compact' : ''} ${isSelected ? 'selected' : ''}"
        data-ral-code="${color.ral_code}"
        data-color='${JSON.stringify(color)}'
      >
        <div class="color-swatch" style="background-color: ${color.hex}"></div>
        <div class="color-info">
          <div class="color-code">${color.ral_code}</div>
          ${!compact ? `<div class="color-name">${color.name}</div>` : ''}
        </div>
        ${isSelected ? '<div class="selected-badge">✓</div>' : ''}
      </div>
    `;
  }

  /**
   * Render selected color display
   */
  _renderSelectedColor() {
    const color = this.selectedColor;
    return `
      <div class="selected-color-info">
        <div class="selected-swatch" style="background-color: ${color.hex}"></div>
        <div class="selected-details">
          <div class="selected-code">${color.ral_code}</div>
          <div class="selected-name">${color.name}</div>
          <div class="selected-hex">${color.hex}</div>
          <div class="selected-rgb">RGB(${color.rgb.join(', ')})</div>
        </div>
      </div>
    `;
  }

  /**
   * Attach event listeners
   */
  _attachEventListeners() {
    // Tab switching
    this.container.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const view = e.target.dataset.view;
        this.switchView(view);
      });
    });

    // Color card clicks
    this.container.querySelectorAll('.color-card').forEach(card => {
      card.addEventListener('click', (e) => {
        const colorData = e.currentTarget.dataset.color;
        const color = JSON.parse(colorData);
        this.selectColor(color);
      });
    });

    // Search input
    const searchInput = this.container.querySelector('#color-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this._handleSearch(e.target.value);
      });

      // Load full palette when search view is opened
      if (!this.allColors) {
        this.loadFullPalette();
      }
    }

    // Color picker
    const pickerBtn = this.container.querySelector('#find-nearest-btn');
    if (pickerBtn) {
      pickerBtn.addEventListener('click', () => {
        const colorInput = this.container.querySelector('#color-picker-input');
        this._findNearestRAL(colorInput.value);
      });
    }
  }

  /**
   * Switch view
   */
  switchView(view) {
    this.currentView = view;
    this.render();
  }

  /**
   * Select a color
   */
  selectColor(color) {
    this.selectedColor = color;
    this._addToRecent(color);

    // Update selected color display
    const displayEl = this.container.querySelector('#selected-color-display');
    if (displayEl) {
      displayEl.innerHTML = this._renderSelectedColor();
    }

    // Update selected state in cards
    this.container.querySelectorAll('.color-card').forEach(card => {
      card.classList.remove('selected');
      if (card.dataset.ralCode === color.ral_code) {
        card.classList.add('selected');
      }
    });

    // Callback
    if (this.options.onColorSelected) {
      this.options.onColorSelected(color);
    }
  }

  /**
   * Load full palette from API
   */
  async loadFullPalette() {
    try {
      const response = await fetch('/api/palette');
      const data = await response.json();

      if (data.success) {
        this.allColors = data.colors;

        // Update search view if active
        if (this.currentView === 'search') {
          this.render();
        }
      }
    } catch (error) {
      console.error('Failed to load full palette:', error);
    }
  }

  /**
   * Handle search
   */
  _handleSearch(query) {
    if (!this.allColors) {
      return;
    }

    const resultsEl = this.container.querySelector('#search-results');

    if (query.length === 0) {
      // Show first 20 colors
      resultsEl.innerHTML = `
        <div class="color-grid">
          ${this.allColors.slice(0, 20).map(color => this._renderColorCard(color)).join('')}
        </div>
        <p class="hint">Showing first 20 colors. Type to search.</p>
      `;
    } else {
      // Fuzzy search
      const filtered = this.allColors.filter(color => {
        const nameMatch = color.name.toLowerCase().includes(query.toLowerCase());
        const codeMatch = color.ral_code.toLowerCase().includes(query.toLowerCase());
        return nameMatch || codeMatch;
      });

      if (filtered.length === 0) {
        resultsEl.innerHTML = `
          <div class="empty-state">
            <p>No colors found matching "${query}"</p>
            <p class="hint">Try searching by color name or RAL code</p>
          </div>
        `;
      } else {
        resultsEl.innerHTML = `
          <div class="color-grid">
            ${filtered.slice(0, 50).map(color => this._renderColorCard(color)).join('')}
          </div>
          <p class="hint">Found ${filtered.length} color${filtered.length !== 1 ? 's' : ''}</p>
        `;
      }
    }

    // Re-attach click listeners
    resultsEl.querySelectorAll('.color-card').forEach(card => {
      card.addEventListener('click', (e) => {
        const colorData = e.currentTarget.dataset.color;
        const color = JSON.parse(colorData);
        this.selectColor(color);
      });
    });
  }

  /**
   * Find nearest RAL color to RGB
   */
  async _findNearestRAL(hexColor) {
    // Convert hex to RGB
    const rgb = this._hexToRgb(hexColor);

    try {
      const response = await fetch('/api/color/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rgb, top_n: 3 })
      });

      const data = await response.json();

      if (data.success) {
        const resultEl = this.container.querySelector('#picker-result');
        const matches = data.matches;

        resultEl.innerHTML = `
          <div class="picker-matches">
            <h4>Nearest RAL Colors:</h4>
            <div class="color-grid compact">
              ${matches.map(match => this._renderColorCard(match.color, false)).join('')}
            </div>
          </div>
        `;

        // Re-attach click listeners
        resultEl.querySelectorAll('.color-card').forEach(card => {
          card.addEventListener('click', (e) => {
            const colorData = e.currentTarget.dataset.color;
            const color = JSON.parse(colorData);
            this.selectColor(color);
          });
        });
      }
    } catch (error) {
      console.error('Failed to find nearest RAL:', error);
    }
  }

  /**
   * Convert hex to RGB
   */
  _hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
    ] : [0, 0, 0];
  }

  /**
   * Add color to recent
   */
  _addToRecent(color) {
    // Remove if already exists
    this.recentColors = this.recentColors.filter(c => c.ral_code !== color.ral_code);

    // Add to front
    this.recentColors.unshift(color);

    // Keep only maxRecent
    this.recentColors = this.recentColors.slice(0, this.options.maxRecent);

    // Save to localStorage
    localStorage.setItem('recent_colors', JSON.stringify(this.recentColors));
  }

  /**
   * Load recent colors from localStorage
   */
  _loadRecentColors() {
    try {
      const saved = localStorage.getItem('recent_colors');
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      return [];
    }
  }

  /**
   * Get selected color
   */
  getSelectedColor() {
    return this.selectedColor;
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SmartColorSelector;
}
