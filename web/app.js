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
  mode: 'live',  // 'live' | 'historical'
  liveRefreshTimer: null,
  region: 'india',  // 'india' | 'europe'
};
window.ClimateGuardState = state;

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
// CATEGORY LABELS (text, no emoji)
// ============================================================
const CATEGORY_ICONS = {
  'hydration':              '',
  'outdoor_exposure':       '',
  'cooling':                '',
  'vulnerable_populations': '',
  'workplace':              '',
  'public_awareness':       '',
  'emergency_preparedness': '',
  'general':                '',
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
    // Determine region from URL before loading cities
    const query = new URLSearchParams(location.search);
    state.region = (query.get('region') === 'europe') ? 'europe' : 'india';

    // Load cities for the current region
    state.cities = await api(`/cities?region=${state.region}`);
    renderCities();

    // Load model info (hidden panel — keeps JS intact)
    const modelInfo = await api('/model-info');
    renderModelInfo(modelInfo);

    // Initialize map (India bounds by default; region switch updates it)
    initMap();

    // Populate chart city selector
    populateChartCitySelect();

    // Initialize mode switch
    initModeSwitch();

    // Initialize region switch (visual state; navigation handled by button clicks)
    initRegionSwitch();
    // Apply visual text changes for the current region
    setRegion(state.region, false);

    // Start in Live mode — show live panel, hide historical controls
    const deepLinkedCity = query.get('city');
    setMode(query.get('mode') === 'historical' ? 'historical' : 'live');
    if (deepLinkedCity && state.cities.some(city => city.key === deepLinkedCity)) {
      selectCity(deepLinkedCity);
    }

  } catch (err) {
    showError(`Failed to load: ${err.message}`);
  }
}

// renderCities is defined in the REGION SWITCH section above (region-aware)

async function selectCity(cityKey) {
  // Update visual state
  $$('.city-card').forEach(c => {
    c.classList.remove('active');
    c.setAttribute('aria-pressed', 'false');
  });
  const selected = $(`#city-${cityKey}`);
  if (selected) {
    selected.classList.add('active');
    selected.setAttribute('aria-pressed', 'true');
  }

  state.selectedCity = cityKey;
  const query = new URLSearchParams(location.search);
  query.set('city', cityKey);
  query.set('mode', state.mode);
  history.replaceState(null, '', `/?${query}`);

  // Live mode: fetch live data; Historical mode: load dates
  if (state.mode === 'live') {
    loadLive(cityKey);
    return;
  }

  // Load dates for city (historical mode)
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

  // --- Gauge (runs on hidden SVG, keeps function intact) ---
  animateGauge(prob, riskLevel);

  // --- Probability bar (visible display) ---
  updateProbBar(prob, riskLevel);

  // --- Prediction label (hidden element, kept for JS compat) ---
  dom.predictionLabel.textContent = pred === 1
    ? 'Heatwave tomorrow'
    : 'Normal day tomorrow';
  dom.predictionLabel.style.color = pred === 1 ? 'var(--risk-extreme-text)' : 'var(--risk-low-text)';

  // --- Actual label ---
  if (data.actual) {
    const match = pred === data.actual.heatwave_next_day;
    dom.actualLabel.textContent = `Actual: ${data.actual.label} ${match ? '(correct)' : '(missed)'}`;
    dom.actualLabel.style.color = match ? 'var(--risk-low-text)' : 'var(--risk-extreme-text)';
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
// PROBABILITY BAR UPDATE (visible replacement for gauge)
// ============================================================
function updateProbBar(probability, riskLevel) {
  const pct = Math.round(probability * 100);

  // Large number
  const numEl = document.getElementById('prob-number-display');
  if (numEl) numEl.textContent = pct + '%';

  // Meter bar
  const fill = document.getElementById('prob-bar-fill');
  if (fill) {
    fill.style.width = pct + '%';
    fill.dataset.risk = riskLevel;
  }

  // Meter aria
  const meter = document.getElementById('prob-meter');
  if (meter) meter.setAttribute('aria-valuenow', pct);

  // Plain-language sentence
  const sentence = document.getElementById('prob-sentence');
  if (sentence) {
    const riskMessages = {
      LOW:      'Low probability of a heatwave tomorrow.',
      MODERATE: 'Moderate chance of a heatwave tomorrow — consider preparedness.',
      HIGH:     'High probability of a heatwave tomorrow. Precautions advised.',
      EXTREME:  'Extreme probability of a heatwave tomorrow. Take immediate action.',
    };
    const predText = probability >= 0.70 ? 'Heatwave predicted tomorrow.' : 'Normal day predicted tomorrow.';
    sentence.textContent = `${predText} ${riskMessages[riskLevel] || ''}`;
  }
}


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
    const iconText = triggered ? '&#10003;' : '&ndash;';
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
    ? 'rgba(200, 66, 27, 0.65)'    // accent = pushes toward heatwave
    : 'rgba(92, 85, 80, 0.55)');   // neutral = pushes toward normal

  const borderColors = values.map(v => v > 0
    ? 'rgba(200, 66, 27, 0.9)'
    : 'rgba(92, 85, 80, 0.9)');

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
          backgroundColor: '#FFFFFF',
          titleColor: '#1A1714',
          bodyColor: '#5C5550',
          borderColor: '#E4DED4',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => {
              const i = ctx.dataIndex;
              const dir = values[i] > 0 ? 'increases risk' : 'decreases risk';
              return `SHAP: ${values[i].toFixed(4)} (${dir}) | Value: ${featureValues[i]}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#8C8580', font: { size: 10 } },
          grid: { color: 'rgba(26, 23, 20, 0.06)' },
          title: { display: true, text: 'SHAP value (impact on prediction)', color: '#8C8580' },
        },
        y: {
          ticks: { color: '#5C5550', font: { size: 11 } },
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

const INDIA_BOUNDS  = [[6, 65], [38, 98]];
const EUROPE_BOUNDS = [[35, -12], [45, 8]];  // Iberia + W Mediterranean

function getCurrentBounds() {
  return state.region === 'europe' ? EUROPE_BOUNDS : INDIA_BOUNDS;
}
const OPENFREEMAP_ATTRIBUTION = '<a href="https://openfreemap.org/" target="_blank" rel="noopener noreferrer">OpenFreeMap</a> © <a href="https://openmaptiles.org/" target="_blank" rel="noopener noreferrer">OpenMapTiles</a> Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>';

/*
 * Provider priority is deliberately centralized here. Mappls is first when a
 * public key is configured, but its current Web Maps JS documentation exposes
 * a separate Mappls.Map / mappls.Marker SDK rather than a Leaflet tile layer.
 * Replacing it safely needs a future migration of the Leaflet marker, popup,
 * bounds, and live-update code. Until then it is skipped and the compatible
 * OpenFreeMap vector layer is used; Esri remains a no-WebGL fallback.
 */
const MAP_TILE_PROVIDERS = [
  {
    id: 'mappls',
    name: 'Mappls',
    canUse: config => Boolean(config?.mappls_key),
    add: async () => {
      throw new Error('Mappls currently requires its Web Maps SDK, not a Leaflet-compatible raster tile URL.');
    },
  },
  {
    id: 'openfreemap',
    name: 'OpenFreeMap Positron',
    canUse: () => Boolean(window.L?.maplibreGL && window.maplibregl),
    add: addOpenFreeMapLayer,
  },
  {
    id: 'esri-light-gray',
    name: 'Esri World Light Gray Canvas',
    canUse: () => Boolean(window.L?.tileLayer),
    add: addEsriLightGrayLayer,
  },
];

function waitForMapLibreLoad(layer) {
  return new Promise((resolve, reject) => {
    const glMap = layer.getMaplibreMap();
    const timeout = window.setTimeout(() => reject(new Error('OpenFreeMap timed out while loading.')), 10000);
    const finish = callback => value => {
      window.clearTimeout(timeout);
      callback(value);
    };
    glMap.once('load', finish(resolve));
    glMap.once('error', event => finish(reject)(event?.error || new Error('OpenFreeMap failed to load.')));
  });
}

async function addOpenFreeMapLayer() {
  const layer = L.maplibreGL({
    style: 'https://tiles.openfreemap.org/styles/positron',
    attributionControl: false,
  }).addTo(leafletMap);

  try {
    await waitForMapLibreLoad(layer);
    leafletMap.attributionControl.addAttribution(OPENFREEMAP_ATTRIBUTION);
    return layer;
  } catch (error) {
    leafletMap.removeLayer(layer);
    throw error;
  }
}

async function addEsriLightGrayLayer() {
  return new Promise((resolve, reject) => {
    const layer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: 'Tiles © <a href="https://www.esri.com/" target="_blank" rel="noopener noreferrer">Esri</a> — Esri, DeLorme, NAVTEQ',
        minZoom: 4,
        maxZoom: 10,
      },
    );
    const timeout = window.setTimeout(() => {
      layer.remove();
      reject(new Error('Esri fallback timed out while loading.'));
    }, 10000);
    layer.once('load', () => {
      window.clearTimeout(timeout);
      resolve(layer);
    });
    layer.once('tileerror', event => {
      window.clearTimeout(timeout);
      layer.remove();
      reject(event?.error || new Error('Esri fallback failed to load.'));
    });
    layer.addTo(leafletMap);
  });
}

async function addFirstWorkingMapLayer(config) {
  const failures = [];
  for (const provider of MAP_TILE_PROVIDERS) {
    if (!provider.canUse(config)) continue;
    try {
      await provider.add();
      console.info(`Map provider active: ${provider.name}`);
      return provider.name;
    } catch (error) {
      failures.push(`${provider.name}: ${error.message}`);
      console.warn(`Map provider unavailable: ${provider.name}`, error);
    }
  }
  throw new Error(`No map provider could be initialized. ${failures.join(' ')}`);
}

async function initMap() {
  try {
    const [mapData, config] = await Promise.all([
      api(`/map-data?region=${state.region}`),
      api('/config').catch(() => ({ mappls_key: '' })),
    ]);

    const bounds = getCurrentBounds();
    leafletMap = L.map('map', {
      center: state.region === 'europe' ? [40, -2] : [22.5, 80],
      zoom: state.region === 'europe' ? 5 : 5,
      zoomControl: true,
      attributionControl: true,
      maxBounds: bounds,
      maxBoundsViscosity: 0.8,
      minZoom: 4,
      maxZoom: 10,
    });

    await addFirstWorkingMapLayer(config);
    leafletMap.fitBounds(getCurrentBounds(), { padding: [12, 12] });

    // Dedicated layer group so setRegion can swap markers without touching tiles
    window._markerLayer = L.layerGroup().addTo(leafletMap);

    function addMapMarkers(cities) {
      if (!window._markerLayer) return;
      window._markerLayer.clearLayers();
      cities.forEach(city => {
        const riskColor = city.max_probability >= 0.8 ? '#C84040'
          : city.max_probability >= 0.6 ? '#D47A3A'
          : city.max_probability >= 0.3 ? '#D4A43A'
          : '#A3CEAF';
        const riskClass = city.max_probability >= 0.8 ? 'extreme'
          : city.max_probability >= 0.6 ? 'high'
          : city.max_probability >= 0.3 ? 'moderate'
          : 'low';
        const marker = L.circleMarker([city.lat, city.lon], {
          radius: Math.max(8, 8 + (city.heatwave_pct * 0.8)),
          fillColor: riskColor, color: riskColor,
          weight: 2, opacity: 0.9, fillOpacity: 0.4,
        });
        marker.bindPopup(`
          <div class="map-popup__name">${city.name}</div>
          <div class="map-popup__state">${city.country ? city.country + ' \xb7 ' : ''}${city.state || ''} \xb7 ${city.region}</div>
          <div class="map-popup__stat"><span class="map-popup__stat-label">Heatwave Days</span><span class="map-popup__stat-value ${riskClass}">${city.heatwave_days}</span></div>
          <div class="map-popup__stat"><span class="map-popup__stat-label">Heatwave %</span><span class="map-popup__stat-value ${riskClass}">${city.heatwave_pct}%</span></div>
          <div class="map-popup__stat"><span class="map-popup__stat-label">Max Probability</span><span class="map-popup__stat-value ${riskClass}">${(city.max_probability * 100).toFixed(1)}%</span></div>
          <div class="map-popup__stat"><span class="map-popup__stat-label">Avg Probability</span><span class="map-popup__stat-value">${(city.avg_probability * 100).toFixed(2)}%</span></div>
        `, { maxWidth: 260 });
        window._markerLayer.addLayer(marker);
      });
    }

    // Expose helper so setRegion can call it too
    window._addMapMarkers = addMapMarkers;
    addMapMarkers(mapData);

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

  // Clear existing options except the placeholder
  while (select.options.length > 1) select.remove(1);

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

  // Chart.js light theme defaults
  const gridColor = 'rgba(26, 23, 20, 0.06)';
  const tickColor = '#8C8580';

  // --- Temperature Chart ---
  if (tempChart) tempChart.destroy();
  const tempCtx = document.getElementById('temp-chart').getContext('2d');

  tempChart = new Chart(tempCtx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Max temp (°C)',
          data: tmax,
          borderColor: '#C8421B',
          backgroundColor: 'rgba(200, 66, 27, 0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Min temp (°C)',
          data: tmin,
          borderColor: '#8C8580',
          backgroundColor: 'rgba(140, 133, 128, 0.06)',
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
          labels: { color: tickColor, font: { family: 'IBM Plex Sans', size: 11 } },
        },
        tooltip: {
          backgroundColor: '#FFFFFF',
          titleColor: '#1A1714',
          bodyColor: '#5C5550',
          borderColor: '#E4DED4',
          borderWidth: 1,
          titleFont: { family: 'IBM Plex Sans' },
          bodyFont: { family: 'IBM Plex Sans' },
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
          label: 'Heatwave probability (%)',
          data: probs,
          borderColor: '#C8421B',
          backgroundColor: 'rgba(200, 66, 27, 0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Threshold (70%)',
          data: labels.map(() => 70),
          borderColor: 'rgba(26, 23, 20, 0.35)',
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
          labels: { color: tickColor, font: { family: 'IBM Plex Sans', size: 11 } },
        },
        tooltip: {
          backgroundColor: '#FFFFFF',
          titleColor: '#1A1714',
          bodyColor: '#5C5550',
          borderColor: '#E4DED4',
          borderWidth: 1,
          titleFont: { family: 'IBM Plex Sans' },
          bodyFont: { family: 'IBM Plex Sans' },
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
// MODE SWITCH
// ============================================================
function initModeSwitch() {
  const livBtn  = document.getElementById('mode-live-btn');
  const histBtn = document.getElementById('mode-hist-btn');
  const caption = document.getElementById('mode-caption');
  if (!livBtn || !histBtn) return;

  livBtn.addEventListener('click', () => setMode('live'));
  histBtn.addEventListener('click', () => setMode('historical'));
}

function setMode(mode) {
  state.mode = mode;
  const livBtn  = document.getElementById('mode-live-btn');
  const histBtn = document.getElementById('mode-hist-btn');
  const caption = document.getElementById('mode-caption');
  const histControls = document.getElementById('historical-controls');
  const livePanel    = document.getElementById('live-panel');

  if (mode === 'live') {
    livBtn?.classList.add('mode-btn--active');
    histBtn?.classList.remove('mode-btn--active');
    if (caption) caption.textContent = 'Current conditions from Open-Meteo';
    if (histControls) histControls.style.display = 'none';
    if (livePanel) livePanel.style.display = 'block';
    // Hide historical results
    dom.results.classList.remove('visible');
    if (state.selectedCity) loadLive(state.selectedCity);
  } else {
    histBtn?.classList.add('mode-btn--active');
    livBtn?.classList.remove('mode-btn--active');
    if (caption) caption.textContent = 'Test set (2023–2025)';
    if (histControls) histControls.style.display = 'block';
    if (livePanel) livePanel.style.display = 'none';
    // Stop auto-refresh
    if (state.liveRefreshTimer) { clearInterval(state.liveRefreshTimer); state.liveRefreshTimer = null; }
  }
}

// ============================================================
// LIVE DATA
// ============================================================
async function loadLive(cityKey) {
  const loadingEl = document.getElementById('live-loading');
  const errorEl   = document.getElementById('live-error');
  const contentEl = document.getElementById('live-content');
  if (!loadingEl) return;

  loadingEl.style.display = 'flex';
  if (errorEl)   errorEl.style.display   = 'none';
  if (contentEl) contentEl.style.display = 'none';

  try {
    const data = await api(`/live/${cityKey}`);

    if (data.error) {
      if (errorEl) {
        errorEl.textContent = `Live data unavailable: ${data.error} You can use Historical mode to browse the test set.`;
        errorEl.style.display = 'flex';
      }
      if (data.error_type === 'rate_limited' && !state.liveRefreshTimer) {
        const retryAfter = Math.max(60, Number(data.retry_after_seconds) || 300) * 1000;
        state.liveRefreshTimer = setTimeout(() => {
          state.liveRefreshTimer = null;
          if (document.visibilityState === 'visible' && state.mode === 'live' && state.selectedCity) {
            loadLive(state.selectedCity);
          }
        }, retryAfter);
      }
      loadingEl.style.display = 'none';
      return;
    }

    renderLiveStrip(data);
    renderTodayAdvice(data.days?.[0]);
    renderOfficialAlert(await api(`/official-alerts/${cityKey}`));

    // Last updated timestamp
    const luEl = document.getElementById('live-last-updated');
    if (luEl && data.last_updated) {
      const dt = new Date(data.last_updated);
      const hhmm = dt.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' });
      luEl.textContent = `Updated ${hhmm} IST`;
    }

    if (contentEl) contentEl.style.display = 'block';

    // Start 30-minute auto-refresh if not already running
    if (!state.liveRefreshTimer) {
      state.liveRefreshTimer = setInterval(() => {
        if (document.visibilityState === 'visible' && state.mode === 'live' && state.selectedCity) {
          loadLive(state.selectedCity);
        }
      }, 30 * 60 * 1000);
    }
  } catch (err) {
    if (errorEl) {
      errorEl.textContent = `Could not fetch live data: ${err.message}. You can use Historical mode to browse the test set.`;
      errorEl.style.display = 'flex';
    }
  } finally {
    loadingEl.style.display = 'none';
  }
}

function renderOfficialAlert(alert) {
  const el = document.getElementById('official-alert');
  if (!el) return;
  el.innerHTML = `<strong>Official context:</strong> ${alert.message} <a href="${alert.source_url}" target="_blank" rel="noopener noreferrer">${alert.source_name}</a>.`;
}

function renderLiveStrip(data) {
  const strip = document.getElementById('live-strip');
  if (!strip) return;
  const today = new Date().toISOString().slice(0, 10);

  strip.innerHTML = (data.days || []).map(day => {
    const isToday   = day.date === today;
    const isError   = day.error != null;
    const cardClass = isToday ? 'live-day-card live-day-card--today'
                    : isError ? 'live-day-card live-day-card--error'
                    : 'live-day-card';
    const dateStr = new Date(day.date + 'T00:00:00').toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' });
    const dayLabel = isToday ? `<span class="today-label">Today</span>` : dateStr;

    if (isError) return `<div class="${cardClass}">
      <div class="live-day-date">${dayLabel}</div>
      <div style="font-size: 0.75rem; color: var(--ink-muted);">Data unavailable</div>
    </div>`;

    const risk = day.risk_level || 'LOW';
    const prob = day.probability != null ? (day.probability * 100).toFixed(1) : '—';
    const tmax = day.temperature_max != null ? `${day.temperature_max}°C` : '—';
    const feels = day.apparent_temperature_max != null ? `Feels ${day.apparent_temperature_max}°C` : '';
    const humidity = day.humidity_mean != null ? `Humidity ${day.humidity_mean}%` : '';
    const wind = day.wind_speed_max != null ? `Wind ${day.wind_speed_max} km/h` : '';
    const nRules = (day.triggered_rules || []).length;

    return `<div class="${cardClass}">
      <div class="live-day-date">${dayLabel}</div>
      <div class="live-day-temp">${tmax}</div>
      <div class="live-day-details">${[feels, humidity, wind].filter(Boolean).join(' · ')}</div>
      <div class="live-day-prob">${prob}% probability</div>
      <span class="live-day-risk ${risk}">${risk}</span>
      ${nRules > 0 ? `<div class="live-day-rules">${nRules} rule${nRules > 1 ? 's' : ''} triggered</div>` : ''}
    </div>`;
  }).join('');
}

async function renderTodayAdvice(day) {
  const el = document.getElementById('today-advice');
  if (!el || !day || day.error || !state.selectedCity) return;
  try {
    const response = await api(`/alerts/preview?city=${encodeURIComponent(state.selectedCity)}&level=${encodeURIComponent(day.risk_level || 'LOW')}&lang=en`);
    el.innerHTML = response.tips?.length ? `<strong>What to do today</strong><ul>${response.tips.map(tip => `<li>${tip}</li>`).join('')}</ul>` : '<strong>What to do today</strong> Continue to check IMD and local authority advisories.';
  } catch (_) {
    el.textContent = 'What to do today: continue to check IMD and local authority advisories.';
  }
}

// Manual refresh button
document.addEventListener('DOMContentLoaded', () => {
  const refreshBtn = document.getElementById('live-refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      if (state.selectedCity) loadLive(state.selectedCity);
    });
  }
});

// ============================================================
// REGION SWITCH (India | Europe)
// ============================================================
function initRegionSwitch() {
  const indiaBtn  = document.getElementById('region-india-btn');
  const europeBtn = document.getElementById('region-europe-btn');
  if (!indiaBtn || !europeBtn) return;

  // Set visual active state from current URL (no reload needed here — init() already loaded correct data)
  const urlRegion = new URLSearchParams(location.search).get('region') || 'india';
  if (urlRegion === 'europe') {
    europeBtn.classList.add('region-btn--active');
    indiaBtn.classList.remove('region-btn--active');
  } else {
    indiaBtn.classList.add('region-btn--active');
    europeBtn.classList.remove('region-btn--active');
  }

  // On click: navigate to the same page with the new region param
  // This is the most reliable approach — avoids all stale-closure bugs
  indiaBtn.addEventListener('click', () => {
    if (state.region !== 'india') {
      const q = new URLSearchParams(location.search);
      q.set('region', 'india');
      q.delete('city');
      q.delete('mode');
      location.href = '/?' + q.toString();
    }
  });

  europeBtn.addEventListener('click', () => {
    if (state.region !== 'europe') {
      const q = new URLSearchParams(location.search);
      q.set('region', 'europe');
      q.delete('city');
      q.delete('mode');
      location.href = '/?' + q.toString();
    }
  });
}

async function setRegion(region, reload = true) {
  state.region = region;
  const indiaBtn  = document.getElementById('region-india-btn');
  const europeBtn = document.getElementById('region-europe-btn');

  if (region === 'europe') {
    europeBtn?.classList.add('region-btn--active');
    indiaBtn?.classList.remove('region-btn--active');
    const h1 = document.getElementById('masthead-title');
    if (h1) h1.innerHTML = 'Heatwave risk forecasts<br>for eleven European cities';
    const desc = document.getElementById('masthead-desc');
    if (desc) desc.textContent = 'Select a city to get a 1-day-ahead heatwave prediction using the Europe research model.';
    const meta = document.getElementById('masthead-meta');
    if (meta) meta.textContent = 'Random Forest · 110 features · percentile-based definition · test period 2023–2025 · not an official warning';
    const rulesSubhead = document.querySelector('.rules-section .section-subhead');
    if (rulesSubhead) rulesSubhead.textContent = 'Four contextual rules based on Europe\u2019s relative heat definition. Not official national meteorological alerts.';
  } else {
    indiaBtn?.classList.add('region-btn--active');
    europeBtn?.classList.remove('region-btn--active');
    const h1 = document.getElementById('masthead-title');
    if (h1) h1.innerHTML = 'Heatwave risk forecasts<br>for five Indian cities';
    const desc = document.getElementById('masthead-desc');
    if (desc) desc.textContent = 'Select a city and date to get a 1-day-ahead heatwave prediction from the model.';
    const meta = document.getElementById('masthead-meta');
    if (meta) meta.textContent = 'Random Forest · 110 features · decision threshold 0.70 · test period 2023–2025';
    const rulesSubhead = document.querySelector('.rules-section .section-subhead');
    if (rulesSubhead) rulesSubhead.textContent = 'Seven deterministic rules derived from IMD-inspired heatwave criteria. These supplement the model probability.';
  }

  if (reload) {
    // Navigation is now handled by initRegionSwitch button listeners.
    // This path is kept only for internal calls that update the map/cities
    // without a full page reload (e.g. future programmatic region changes).
    const q = new URLSearchParams(location.search);
    q.set('region', region);
    q.delete('city');
    history.replaceState(null, '', `/?${q}`);
    state.selectedCity = null;
    state.cities = [];
    document.getElementById('results')?.classList.remove('visible');
    hideError();
    try {
      state.cities = await api(`/cities?region=${region}`);
      renderCities();
      populateChartCitySelect();
      if (leafletMap) {
        const bounds = getCurrentBounds();
        leafletMap.setMaxBounds(bounds);
        leafletMap.fitBounds(bounds, { animate: true, padding: [12, 12] });
        if (window._markerLayer) window._markerLayer.clearLayers();
        else window._markerLayer = L.layerGroup().addTo(leafletMap);
        try {
          const mapData = await api(`/map-data?region=${region}`);
          if (window._addMapMarkers) window._addMapMarkers(mapData);
          setTimeout(() => leafletMap.invalidateSize(), 100);
        } catch (mapErr) { console.warn('Map update failed:', mapErr); }
      }
    } catch (err) {
      showError(`Could not load ${region} cities: ${err.message}`);
    }
  }
}

// Override renderCities to group Europe cities by country
function renderCities() {
  const cityGrid = document.getElementById('city-grid');
  if (!cityGrid) return;

  if (state.region === 'europe') {
    // Group by country
    const byCountry = {};
    state.cities.forEach(city => {
      const country = city.country || city.state || 'Other';
      if (!byCountry[country]) byCountry[country] = [];
      byCountry[country].push(city);
    });
    const countryOrder = ['Spain', 'Portugal', 'Andorra', 'Monaco'];
    const orderedCountries = [
      ...countryOrder.filter(c => byCountry[c]),
      ...Object.keys(byCountry).filter(c => !countryOrder.includes(c)),
    ];

    let html = '';
    orderedCountries.forEach(country => {
      html += `<div class="city-group-heading" aria-hidden="true">${country}</div>`;
      html += byCountry[country].map(city => `
        <button class="city-card" data-city="${city.key}" id="city-${city.key}"
                aria-pressed="false" type="button">
          <span class="city-card__check" aria-hidden="true">
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="2 6 5 9 10 3"/>
            </svg>
          </span>
          <div class="city-card__name">${city.name}</div>
          <div class="city-card__meta">${country}</div>
          <div class="city-card__region">${city.region || ''}</div>
        </button>
      `).join('');
    });
    cityGrid.innerHTML = html;
  } else {
    // Original India rendering
    cityGrid.innerHTML = state.cities.map(city => `
      <button class="city-card" data-city="${city.key}" id="city-${city.key}"
              aria-pressed="false" type="button">
        <span class="city-card__check" aria-hidden="true">
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="2 6 5 9 10 3"/>
          </svg>
        </span>
        <div class="city-card__name">${city.name}</div>
        <div class="city-card__meta">${city.state}</div>
        <div class="city-card__region">${city.region}</div>
      </button>
    `).join('');
  }

  $$('.city-card').forEach(card => {
    card.addEventListener('click', () => selectCity(card.dataset.city));
  });
}

// ============================================================
// BOOT
// ============================================================
document.addEventListener('DOMContentLoaded', init);
