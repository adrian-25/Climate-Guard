/**
 * ClimateGuard Dashboard — Frontend Logic
 * Handles city selection, API calls, and result rendering.
 */

// ============================================================
// STATE
// ============================================================
const state = {
  cities: [],
  selectedCity: null,
  dateMin: null,
  dateMax: null,
  loading: false,
};

// ============================================================
// DOM REFERENCES
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
  cityGrid:       $('#city-grid'),
  dateInput:      $('#date-input'),
  predictBtn:     $('#predict-btn'),
  results:        $('#results'),
  errorBanner:    $('#error-banner'),
  errorText:      $('#error-text'),
  gaugeFill:      $('#gauge-fill'),
  gaugeValue:     $('#gauge-value'),
  predictionLabel:$('#prediction-label'),
  actualLabel:    $('#actual-label'),
  riskBadge:      $('#risk-badge'),
  riskLevelText:  $('#risk-level-text'),
  detailCity:     $('#detail-city'),
  detailDate:     $('#detail-date'),
  detailPrediction: $('#detail-prediction'),
  detailRules:    $('#detail-rules'),
  warningsCard:   $('#warnings-card'),
  warningsList:   $('#warnings-list'),
  rulesGrid:      $('#rules-grid'),
  recsSection:    $('#recs-section'),
  recsGrid:       $('#recs-grid'),
  modelToggle:    $('#model-info-toggle'),
  modelArrow:     $('#model-info-arrow'),
  modelContent:   $('#model-info-content'),
  modelGrid:      $('#model-info-grid'),
  metricsBar:     $('#metrics-bar'),
};

// ============================================================
// CATEGORY ICONS
// ============================================================
const CATEGORY_ICONS = {
  'hydration':              '💧',
  'outdoor_exposure':       '☀️',
  'cooling':                '❄️',
  'vulnerable_populations': '👶',
  'workplace':              '🏗️',
  'public_awareness':       '📢',
  'emergency_preparedness': '🚨',
  'general':                '📋',
};

// ============================================================
// API
// ============================================================
async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ============================================================
// INITIALIZATION
// ============================================================
async function init() {
  try {
    // Load cities
    state.cities = await api('/cities');
    renderCities();

    // Load model info
    const modelInfo = await api('/model-info');
    renderModelInfo(modelInfo);

    // Initialize map
    initMap();

    // Populate chart city selector
    populateChartCitySelect();

  } catch (err) {
    showError(`Failed to load: ${err.message}`);
  }
}

// ============================================================
// CITY RENDERING
// ============================================================
function renderCities() {
  dom.cityGrid.innerHTML = state.cities.map(city => `
    <div class="city-card" data-city="${city.key}" id="city-${city.key}">
      <div class="city-card__name">${city.name}</div>
      <div class="city-card__meta">${city.state}</div>
      <div class="city-card__region">${city.region}</div>
    </div>
  `).join('');

  // Attach click handlers
  $$('.city-card').forEach(card => {
    card.addEventListener('click', () => selectCity(card.dataset.city));
  });
}

async function selectCity(cityKey) {
  // Update visual state
  $$('.city-card').forEach(c => c.classList.remove('active'));
  $(`#city-${cityKey}`).classList.add('active');

  state.selectedCity = cityKey;

  // Load dates for city
  try {
    const dateInfo = await api(`/dates/${cityKey}`);
    state.dateMin = dateInfo.date_min;
    state.dateMax = dateInfo.date_max;

    dom.dateInput.disabled = false;
    dom.dateInput.min = dateInfo.date_min;
    dom.dateInput.max = dateInfo.date_max;
    dom.dateInput.value = dateInfo.date_min;

    dom.predictBtn.disabled = false;
    hideError();
  } catch (err) {
    showError(err.message);
  }
}

// ============================================================
// PREDICTION
// ============================================================
async function predict() {
  if (!state.selectedCity || !dom.dateInput.value) return;
  if (state.loading) return;

  state.loading = true;
  dom.predictBtn.classList.add('loading');
  dom.predictBtn.disabled = true;
  hideError();
  dom.results.classList.remove('visible');

  try {
    const data = await api('/predict', {
      method: 'POST',
      body: JSON.stringify({
        city: state.selectedCity,
        date: dom.dateInput.value,
      }),
    });

    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    state.loading = false;
    dom.predictBtn.classList.remove('loading');
    dom.predictBtn.disabled = false;
  }
}

// ============================================================
// RESULT RENDERING
// ============================================================
function renderResults(data) {
  const prob = data.prediction?.probability ?? 0;
  const pred = data.prediction?.prediction ?? 0;
  const riskLevel = (data.risk?.level ?? 'LOW').toUpperCase();

  // --- Gauge ---
  animateGauge(prob, riskLevel);

  // --- Prediction label ---
  dom.predictionLabel.textContent = pred === 1
    ? '🔥 Heatwave Tomorrow'
    : '✅ Normal Day Tomorrow';
  dom.predictionLabel.style.color = pred === 1 ? 'var(--risk-extreme)' : 'var(--risk-low)';

  // --- Actual label ---
  if (data.actual) {
    const match = pred === data.actual.heatwave_next_day;
    dom.actualLabel.textContent = `Actual: ${data.actual.label} ${match ? '✓ Correct' : '✗ Missed'}`;
    dom.actualLabel.style.color = match ? 'var(--risk-low)' : 'var(--risk-extreme)';
  } else {
    dom.actualLabel.textContent = '';
  }

  // --- Risk badge ---
  const riskClass = `risk-badge--${riskLevel.toLowerCase()}`;
  dom.riskBadge.className = `risk-badge ${riskClass}`;
  dom.riskLevelText.textContent = riskLevel;

  // --- Score details ---
  const cityInfo = data.city_info || {};
  dom.detailCity.textContent = cityInfo.name || data.input?.city_key || '—';
  dom.detailDate.textContent = data.input?.date || '—';
  dom.detailPrediction.textContent = pred === 1 ? 'Heatwave (1)' : 'Normal (0)';

  const triggeredCount = (data.expert_rules || []).filter(r => r.triggered).length;
  dom.detailRules.textContent = `${triggeredCount} / ${data.expert_rules?.length || 7}`;

  // --- Warnings ---
  const warnings = data.warnings || [];
  if (warnings.length > 0) {
    dom.warningsCard.style.display = 'block';
    dom.warningsList.innerHTML = warnings.map(w =>
      `<div style="font-size:0.82rem;color:var(--risk-moderate);padding:4px 0;">${w}</div>`
    ).join('');
  } else {
    dom.warningsCard.style.display = 'none';
  }

  // --- Expert Rules ---
  renderExpertRules(data.expert_rules || []);

  // --- Recommendations ---
  renderRecommendations(data.recommendations || []);

  // --- Show SHAP section ---
  const shapSection = document.getElementById('shap-section');
  if (shapSection) {
    shapSection.style.display = 'block';
    const shapContainer = document.getElementById('shap-container');
    if (shapContainer) shapContainer.style.display = 'none';
  }

  // Show results
  dom.results.classList.add('visible');

  // Scroll to results
  setTimeout(() => {
    dom.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

// ============================================================
// GAUGE ANIMATION
// ============================================================
function animateGauge(probability, riskLevel) {
  // The gauge arc is 270 degrees (3/4 of a circle)
  // circumference = 2 * PI * 80 ≈ 502.65
  // arc length for 270° = 502.65 * 0.75 ≈ 377
  // offset for empty = 377 (full dashoffset = empty)
  const arcLength = 283; // 75% of circumference = 377, offset starts at 283 for 0%
  const fillOffset = 283 - (probability * 283);

  const colors = {
    LOW:      'var(--risk-low)',
    MODERATE: 'var(--risk-moderate)',
    HIGH:     'var(--risk-high)',
    EXTREME:  'var(--risk-extreme)',
  };

  dom.gaugeFill.style.stroke = colors[riskLevel] || colors.LOW;
  dom.gaugeFill.style.strokeDashoffset = fillOffset;

  // Animate counter
  const target = Math.round(probability * 100);
  let current = 0;
  const duration = 1000;
  const start = performance.now();

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    current = Math.round(target * eased);
    dom.gaugeValue.textContent = `${current}%`;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ============================================================
// EXPERT RULES RENDERING
// ============================================================
function renderExpertRules(rules) {
  dom.rulesGrid.innerHTML = rules.map((rule, i) => {
    const triggered = rule.triggered;
    const severity = (rule.severity || 'info').toLowerCase();
    const cardClass = triggered
      ? `rule-card triggered severity-${severity}`
      : 'rule-card';
    const iconClass = triggered ? 'rule-icon active' : 'rule-icon inactive';
    const iconText = triggered ? '✓' : '○';
    const severityClass = triggered ? severity : 'not-triggered';
    const severityText = triggered ? severity.toUpperCase() : 'NOT TRIGGERED';

    return `
      <div class="${cardClass}" style="animation: fadeSlideUp 0.4s ease ${i * 0.05}s both;">
        <div class="${iconClass}">${iconText}</div>
        <div class="rule-content">
          <div class="rule-content__name">${rule.name || rule.rule_id || `Rule ${i + 1}`}</div>
          <span class="rule-content__severity ${severityClass}">${severityText}</span>
          <div class="rule-content__message">${rule.message || (triggered ? 'Rule triggered.' : 'Conditions not met.')}</div>
        </div>
      </div>
    `;
  }).join('');
}

// ============================================================
// RECOMMENDATIONS RENDERING
// ============================================================
function renderRecommendations(recs) {
  if (!recs || recs.length === 0) {
    dom.recsSection.style.display = 'none';
    return;
  }

  dom.recsSection.style.display = 'block';

  dom.recsGrid.innerHTML = recs.map((rec, i) => {
    const category = (rec.category || 'general').toLowerCase();
    const icon = CATEGORY_ICONS[category] || '📋';
    const displayCat = category.replace(/_/g, ' ');

    return `
      <div class="rec-card" style="animation: fadeSlideUp 0.4s ease ${i * 0.05}s both;">
        <div class="rec-card__category">
          <span class="rec-card__category-icon">${icon}</span>
          ${displayCat}
        </div>
        <div class="rec-card__message">${rec.message || ''}</div>
      </div>
    `;
  }).join('');
}

// ============================================================
// MODEL INFO RENDERING
// ============================================================
function renderModelInfo(info) {
  const fields = [
    { label: 'Model Type',           value: info.model_type },
    { label: 'Feature Count',        value: info.feature_count },
    { label: 'Decision Threshold',   value: info.threshold },
    { label: 'Prediction Type',      value: info.prediction_type },
    { label: 'Training Period',      value: info.training_date_range },
    { label: 'Test Period',          value: info.test_date_range },
    { label: 'Imbalance Strategy',   value: info.imbalance_strategy },
    { label: 'Cities',              value: (info.cities || []).join(', ') },
  ];

  dom.modelGrid.innerHTML = fields.map(f => `
    <div class="model-stat">
      <div class="model-stat__label">${f.label}</div>
      <div class="model-stat__value ${String(f.value).length > 30 ? 'model-stat__value--small' : ''}">${f.value ?? '—'}</div>
    </div>
  `).join('');

  // Test metrics
  const m = info.test_metrics || {};
  const metrics = [
    { label: 'F1 Score',    value: m.f1 },
    { label: 'Precision',   value: m.precision },
    { label: 'Recall',      value: m.recall },
    { label: 'PR-AUC',      value: m.pr_auc },
    { label: 'ROC-AUC',     value: m.roc_auc },
    { label: 'Accuracy',    value: m.accuracy },
  ];

  dom.metricsBar.innerHTML = metrics.map(m => `
    <div class="metric-item">
      <div class="metric-item__value">${m.value != null ? m.value.toFixed(4) : '—'}</div>
      <div class="metric-item__label">${m.label}</div>
    </div>
  `).join('');
}

// ============================================================
// MODEL INFO TOGGLE
// ============================================================
dom.modelToggle.addEventListener('click', () => {
  const isOpen = dom.modelContent.classList.toggle('open');
  dom.modelArrow.classList.toggle('open', isOpen);
});

// ============================================================
// ERROR HANDLING
// ============================================================
function showError(message) {
  dom.errorText.textContent = message;
  dom.errorBanner.classList.add('visible');
}

function hideError() {
  dom.errorBanner.classList.remove('visible');
}

// ============================================================
// EVENT LISTENERS
// ============================================================
dom.predictBtn.addEventListener('click', predict);

dom.dateInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') predict();
});

// SHAP Explain Button
const explainBtn = document.getElementById('explain-btn');
if (explainBtn) {
  explainBtn.addEventListener('click', explainPrediction);
}

// ============================================================
// SHAP EXPLAINABILITY
// ============================================================
let shapChart = null;

async function explainPrediction() {
  if (!state.selectedCity || !dom.dateInput.value) return;

  const btn = document.getElementById('explain-btn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const data = await api('/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ city: state.selectedCity, date: dom.dateInput.value }),
    });

    renderSHAPChart(data);
    document.getElementById('shap-container').style.display = 'block';
  } catch (err) {
    showError(`Explain failed: ${err.message}`);
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderSHAPChart(data) {
  const contributions = data.contributions || [];
  const labels = contributions.map(c => c.feature.replace(/_/g, ' '));
  const values = contributions.map(c => c.shap_value);
  const featureValues = contributions.map(c => c.feature_value);

  const colors = values.map(v => v > 0
    ? 'rgba(16, 185, 129, 0.7)'   // green = pushes toward heatwave
    : 'rgba(239, 68, 68, 0.7)');  // red = pushes toward normal

  const borderColors = values.map(v => v > 0
    ? 'rgba(16, 185, 129, 1)'
    : 'rgba(239, 68, 68, 1)');

  if (shapChart) shapChart.destroy();
  const ctx = document.getElementById('shap-chart').getContext('2d');

  shapChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'SHAP Value',
        data: values,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 15, 35, 0.95)',
          titleColor: '#f0f0f8',
          bodyColor: '#8b8ba7',
          callbacks: {
            label: (ctx) => {
              const i = ctx.dataIndex;
              const dir = values[i] > 0 ? '↑ Heatwave' : '↓ Normal';
              return `SHAP: ${values[i].toFixed(4)} (${dir}) | Value: ${featureValues[i]}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#5a5a7a', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'SHAP Value (impact on prediction)', color: '#5a5a7a' },
        },
        y: {
          ticks: { color: '#9ca3af', font: { size: 11 } },
          grid: { display: false },
        },
      },
    },
  });
}

// ============================================================
// MAP
// ============================================================
let leafletMap = null;

async function initMap() {
  try {
    const mapData = await api('/map-data');

    leafletMap = L.map('map', {
      center: [23.5, 77],
      zoom: 5,
      zoomControl: true,
      attributionControl: true,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 10,
      minZoom: 4,
    }).addTo(leafletMap);

    mapData.forEach(city => {
      const riskColor = city.max_probability >= 0.8 ? '#ef4444'
        : city.max_probability >= 0.6 ? '#f97316'
        : city.max_probability >= 0.3 ? '#f59e0b'
        : '#10b981';

      const riskClass = city.max_probability >= 0.8 ? 'extreme'
        : city.max_probability >= 0.6 ? 'high'
        : city.max_probability >= 0.3 ? 'moderate'
        : 'low';

      const marker = L.circleMarker([city.lat, city.lon], {
        radius: 10 + (city.heatwave_pct * 0.8),
        fillColor: riskColor,
        color: riskColor,
        weight: 2,
        opacity: 0.9,
        fillOpacity: 0.4,
      }).addTo(leafletMap);

      const popupContent = `
        <div class="map-popup__name">${city.name}</div>
        <div class="map-popup__state">${city.state} · ${city.region}</div>
        <div class="map-popup__stat">
          <span class="map-popup__stat-label">Heatwave Days</span>
          <span class="map-popup__stat-value ${riskClass}">${city.heatwave_days}</span>
        </div>
        <div class="map-popup__stat">
          <span class="map-popup__stat-label">Heatwave %</span>
          <span class="map-popup__stat-value ${riskClass}">${city.heatwave_pct}%</span>
        </div>
        <div class="map-popup__stat">
          <span class="map-popup__stat-label">Max Probability</span>
          <span class="map-popup__stat-value ${riskClass}">${(city.max_probability * 100).toFixed(1)}%</span>
        </div>
        <div class="map-popup__stat">
          <span class="map-popup__stat-label">Avg Probability</span>
          <span class="map-popup__stat-value">${(city.avg_probability * 100).toFixed(2)}%</span>
        </div>
      `;

      marker.bindPopup(popupContent, { maxWidth: 260 });

      // Pulsing animation via CSS
      const el = marker.getElement();
      if (el) {
        el.style.transition = 'r 0.3s ease';
      }
    });

    // Fix map rendering in hidden/scrolled containers
    setTimeout(() => leafletMap.invalidateSize(), 200);

  } catch (err) {
    console.error('Map init error:', err);
  }
}

// ============================================================
// TREND CHARTS
// ============================================================
let tempChart = null;
let probChart = null;

function populateChartCitySelect() {
  const select = document.getElementById('chart-city-select');
  if (!select) return;

  state.cities.forEach(city => {
    const opt = document.createElement('option');
    opt.value = city.key;
    opt.textContent = city.name;
    select.appendChild(opt);
  });

  select.addEventListener('change', async (e) => {
    const cityKey = e.target.value;
    if (!cityKey) return;
    await loadTrendCharts(cityKey);
  });
}

async function loadTrendCharts(cityKey) {
  try {
    const data = await api(`/trends/${cityKey}`);
    renderTrendCharts(data);
  } catch (err) {
    console.error('Trend chart error:', err);
  }
}

function renderTrendCharts(trendData) {
  const records = trendData.data;
  // Sample data if too many points (show every Nth point for readability)
  const step = records.length > 400 ? Math.ceil(records.length / 400) : 1;
  const sampled = records.filter((_, i) => i % step === 0);

  const labels = sampled.map(r => r.date);
  const tmax = sampled.map(r => r.tmax);
  const tmin = sampled.map(r => r.tmin);
  const probs = sampled.map(r => r.prob != null ? +(r.prob * 100).toFixed(2) : null);
  const heatwaveBg = sampled.map(r => r.heatwave === 1 ? 'rgba(239, 68, 68, 0.15)' : 'transparent');

  // Chart.js dark theme defaults
  const gridColor = 'rgba(255, 255, 255, 0.04)';
  const tickColor = '#5a5a7a';

  // --- Temperature Chart ---
  if (tempChart) tempChart.destroy();
  const tempCtx = document.getElementById('temp-chart').getContext('2d');

  tempChart = new Chart(tempCtx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Max Temp (°C)',
          data: tmax,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Min Temp (°C)',
          data: tmin,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: tickColor, font: { family: 'Inter', size: 11 } },
        },
        tooltip: {
          backgroundColor: 'rgba(15, 15, 35, 0.95)',
          titleColor: '#f0f0f8',
          bodyColor: '#8b8ba7',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleFont: { family: 'Inter' },
          bodyFont: { family: 'Inter' },
        },
      },
      scales: {
        x: {
          ticks: { color: tickColor, maxTicksLimit: 12, font: { size: 10 } },
          grid: { color: gridColor },
        },
        y: {
          ticks: { color: tickColor, font: { size: 10 } },
          grid: { color: gridColor },
        },
      },
    },
  });

  // --- Probability Chart ---
  if (probChart) probChart.destroy();
  const probCtx = document.getElementById('prob-chart').getContext('2d');

  probChart = new Chart(probCtx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Heatwave Prob (%)',
          data: probs,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Threshold (70%)',
          data: labels.map(() => 70),
          borderColor: 'rgba(239, 68, 68, 0.5)',
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: tickColor, font: { family: 'Inter', size: 11 } },
        },
        tooltip: {
          backgroundColor: 'rgba(15, 15, 35, 0.95)',
          titleColor: '#f0f0f8',
          bodyColor: '#8b8ba7',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleFont: { family: 'Inter' },
          bodyFont: { family: 'Inter' },
        },
      },
      scales: {
        x: {
          ticks: { color: tickColor, maxTicksLimit: 12, font: { size: 10 } },
          grid: { color: gridColor },
        },
        y: {
          min: 0,
          max: 100,
          ticks: { color: tickColor, font: { size: 10 }, callback: v => v + '%' },
          grid: { color: gridColor },
        },
      },
    },
  });
}

// ============================================================
// BOOT
// ============================================================
document.addEventListener('DOMContentLoaded', init);
