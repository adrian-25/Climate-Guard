/**
 * ClimateGuard — Model Performance Page
 * Renders confusion matrix, ROC/PR curves, feature importance, city comparison.
 *
 * Bug fixes (Phase 3):
 *   ROC chart: was type:'line' with labels:fpr → x-axis treated as categories
 *              → fixed to type:'scatter' with {x:fpr,y:tpr} point objects
 *   PR chart:  same issue with labels:recall, data:precision
 *              → fixed to type:'scatter' with {x:recall,y:precision}
 */

const API = (path) => fetch(`/api${path}`).then(r => r.json());

// Light theme chart palette
const CHART = {
  grid:    'rgba(26, 23, 20, 0.07)',
  tick:    '#8C8580',
  accent:  '#C8421B',
  neutral: '#8C8580',
  bg:      'rgba(200, 66, 27, 0.08)',
  font:    'IBM Plex Sans',
};

// ----------------------------------------------------------------
// Humanize feature labels
// ----------------------------------------------------------------
const LABEL_MAP = {
  'temperature_2m_max':        'Max temperature (°C)',
  'temperature_2m_min':        'Min temperature (°C)',
  'temperature_2m_mean':       'Mean temperature (°C)',
  'tmax_departure':            'Max temp departure from normal',
  'tmax_departure_zscore':     'Temp departure z-score',
  'tmax_normal':               'Climatological normal max temp',
  'tmax_delta_1d':             'Tmax change from yesterday',
  'tmax_delta_3d':             'Tmax change over 3 days',
  'tmax_delta_7d':             'Tmax change over 7 days',
  'tmax_slope_3d':             '3-day Tmax trend slope',
  'tmax_slope_7d':             '7-day Tmax trend slope',
  'qualifying_day':            'Qualifying day flag (IMD criterion)',
  'heatwave_lag1':             'Heatwave flag yesterday',
  'et0_fao_evapotranspiration':'Evapotranspiration (ET₀)',
  'relative_humidity_2m_max':  'Max relative humidity',
  'relative_humidity_2m_mean': 'Mean relative humidity',
  'relative_humidity_2m_min':  'Min relative humidity',
  'precipitation_sum':         'Daily precipitation (mm)',
  'wind_speed_10m_max':        'Max wind speed (km/h)',
  'wind_gusts_10m_max':        'Max wind gusts (km/h)',
  'surface_pressure_mean':     'Mean surface pressure (hPa)',
  'shortwave_radiation_sum':   'Shortwave radiation (MJ/m²)',
  'apparent_temperature_max':  'Max apparent temperature',
  'apparent_temperature_mean': 'Mean apparent temperature',
  'apparent_temperature_min':  'Min apparent temperature',
  'city_encoded':              'City (encoded)',
  'is_coastal':                'Coastal city flag',
  'latitude':                  'City latitude',
  'longitude':                 'City longitude',
  'month':                     'Month',
  'month_sin':                 'Month (sine)',
  'month_cos':                 'Month (cosine)',
  'day_of_year':               'Day of year',
  'doy_sin':                   'Day of year (sine)',
  'doy_cos':                   'Day of year (cosine)',
  'season_code':               'Season code',
};

function humanize(featureName) {
  if (LABEL_MAP[featureName]) return LABEL_MAP[featureName];
  // Pattern-based fallback: _lag1 → "(lag 1 day)" etc
  return featureName
    .replace(/_lag(\d+)$/, ' (lag $1 d)')
    .replace(/_roll(\d+)_mean$/, ' ($1-day rolling mean)')
    .replace(/_roll(\d+)_max$/,  ' ($1-day rolling max)')
    .replace(/_roll(\d+)_min$/,  ' ($1-day rolling min)')
    .replace(/_/g, ' ');
}

// ----------------------------------------------------------------
// Shared chart options
// ----------------------------------------------------------------
function scatterOptions(xLabel, yLabel, auc) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: CHART.tick, font: { family: CHART.font, size: 11 } },
      },
      tooltip: {
        backgroundColor: '#FFFFFF',
        titleColor: '#1A1714',
        bodyColor: '#5C5550',
        borderColor: '#E4DED4',
        borderWidth: 1,
        callbacks: {
          label: ctx => `(${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})`,
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        min: 0,
        max: 1,
        ticks: {
          color: CHART.tick,
          font: { size: 10, family: CHART.font },
          callback: v => v.toFixed(1),
        },
        grid: { color: CHART.grid },
        title: { display: true, text: xLabel, color: CHART.tick, font: { size: 11 } },
      },
      y: {
        type: 'linear',
        min: 0,
        max: 1,
        ticks: {
          color: CHART.tick,
          font: { size: 10, family: CHART.font },
          callback: v => v.toFixed(1),
        },
        grid: { color: CHART.grid },
        title: { display: true, text: yLabel, color: CHART.tick, font: { size: 11 } },
      },
    },
  };
}

// ----------------------------------------------------------------
// Main — controlled by region switch at bottom of file
// ----------------------------------------------------------------
async function init_india_only() {
  // Legacy India-only init; kept for backward compatibility
  // Actual init is now done by the region switch DOMContentLoaded handler
}


function renderLiveTrackRecord(data) {
  const el = document.getElementById('live-track-record');
  if (!el) return;
  if (!data || data.status === 'not_enough_data') {
    const count = data?.logged_predictions || 0;
    el.textContent = 'Not enough data yet: ' + count + ' live forecast' + (count === 1 ? '' : 's') + ' logged and no reconciled outcomes. This section updates after forecast dates pass.';
    return;
  }
  el.textContent = data.matched_outcomes + ' reconciled outcomes: precision ' + data.precision.toFixed(4) + ', recall ' + data.recall.toFixed(4) + '. ' + data.note;
}

function renderEvaluation(evaluation) {
  if (!evaluation) {
    const unavailable = document.getElementById('evaluation-unavailable');
    unavailable.hidden = false;
    unavailable.textContent = 'Evaluation artifact is unavailable. Run the documented evaluation script before using these checks.';
    return;
  }
  const metrics = ['f1', 'precision', 'recall', 'brier'];
  const rows = [
    ['Active model', evaluation.active_model],
    ['Persistence baseline', evaluation.baselines.persistence_heatwave_lag1],
    ['IMD-style qualifying-day proxy', evaluation.baselines.imd_style_qualifying_day_proxy],
  ];
  document.getElementById('baseline-comparison').innerHTML = `<table class="comp-table"><thead><tr><th>Approach</th>${metrics.map(metric => `<th class="right">${metric.toUpperCase()}</th>`).join('')}</tr></thead><tbody>${rows.map(([name, value]) => `<tr><td class="comp-city">${name}</td>${metrics.map(metric => `<td class="comp-metric right">${value[metric].toFixed(4)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;

  const points = evaluation.calibration.bin_mean_predicted.map((x, index) => ({ x, y: evaluation.calibration.bin_fraction_positive[index] }));
  new Chart(document.getElementById('calibration-chart'), {
    type: 'scatter',
    data: { datasets: [
      { label: 'Observed frequency', data: points, showLine: true, borderColor: CHART.accent, pointRadius: 3 },
      { label: 'Perfect calibration', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, borderColor: CHART.neutral, borderDash: [5, 5], pointRadius: 0 },
    ] },
    options: scatterOptions('Mean predicted probability', 'Observed heatwave frequency'),
  });
}

// ----------------------------------------------------------------
// Metrics bar
// ----------------------------------------------------------------
function renderMetricsBar(data) {
  const m = data.global_metrics;
  const metrics = [
    { label: 'Recall',        value: m.recall,    color: CHART.accent, primary: true,
      note: 'Fraction of real heatwave days caught' },
    { label: 'Precision',     value: m.precision, color: CHART.accent, primary: true,
      note: 'Fraction of alarms that are real' },
    { label: 'F1 score',      value: m.f1,        color: '#5C5550' },
    { label: 'Accuracy',      value: m.accuracy,  color: '#5C5550' },
    { label: 'ROC-AUC',       value: m.roc_auc,   color: '#5C5550' },
    { label: 'PR-AUC',        value: m.pr_auc,    color: '#5C5550' },
    { label: 'Test samples',  value: data.total_test_samples, color: '#8C8580', isCount: true },
    { label: 'Heatwave days', value: data.positive_samples,  color: CHART.accent, isCount: true },
  ];

  document.getElementById('perf-metrics-bar').innerHTML = metrics.map(m => `
    <div class="perf-metric ${m.primary ? 'perf-metric--primary' : 'perf-metric--secondary'}">
      <div class="perf-metric__value" style="color:${m.color};">
        ${m.isCount ? m.value.toLocaleString() : m.value.toFixed(4)}
      </div>
      <div class="perf-metric__label">${m.label}</div>
    </div>
  `).join('');
}

// ----------------------------------------------------------------
// Confusion matrix
// ----------------------------------------------------------------
function renderConfusionMatrix(cm, data) {
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const el = document.getElementById('confusion-matrix');

  el.innerHTML = `
    <div class="cm-header"></div>
    <div class="cm-header cm-col-header">Predicted: Normal</div>
    <div class="cm-header cm-col-header">Predicted: Heatwave</div>

    <div class="cm-header cm-row-header">Actual: Normal</div>
    <div class="cm-cell cm-tn">
      <div class="cm-cell__value">${cm.tn.toLocaleString()}</div>
      <div class="cm-cell__label">True negative</div>
      <div class="cm-cell__pct">${(cm.tn/total*100).toFixed(1)}%</div>
    </div>
    <div class="cm-cell cm-fp">
      <div class="cm-cell__value">${cm.fp.toLocaleString()}</div>
      <div class="cm-cell__label">False alarm</div>
      <div class="cm-cell__pct">${(cm.fp/total*100).toFixed(1)}%</div>
    </div>

    <div class="cm-header cm-row-header">Actual: Heatwave</div>
    <div class="cm-cell cm-fn">
      <div class="cm-cell__value">${cm.fn.toLocaleString()}</div>
      <div class="cm-cell__label">Missed event</div>
      <div class="cm-cell__pct">${(cm.fn/total*100).toFixed(1)}%</div>
    </div>
    <div class="cm-cell cm-tp">
      <div class="cm-cell__value">${cm.tp.toLocaleString()}</div>
      <div class="cm-cell__label">True positive</div>
      <div class="cm-cell__pct">${(cm.tp/total*100).toFixed(1)}%</div>
    </div>
  `;
}

// ----------------------------------------------------------------
// ROC curve — FIXED: scatter chart with {x:fpr, y:tpr} points
// ----------------------------------------------------------------
function renderROCChart(roc) {
  // Convert parallel arrays → scatter {x,y} points
  const rocPoints = roc.fpr.map((fpr, i) => ({ x: fpr, y: roc.tpr[i] }));
  // Diagonal baseline: (0,0) → (1,1)
  const baseline  = [{ x: 0, y: 0 }, { x: 1, y: 1 }];

  const opts = scatterOptions('False positive rate', 'True positive rate');

  new Chart(document.getElementById('roc-chart'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: `ROC  (AUC = ${roc.auc})`,
          data: rocPoints,
          borderColor: CHART.accent,
          backgroundColor: 'rgba(200, 66, 27, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          showLine: true,
          fill: true,
          tension: 0,
        },
        {
          label: 'Random classifier',
          data: baseline,
          borderColor: 'rgba(26, 23, 20, 0.25)',
          borderWidth: 1,
          borderDash: [5, 5],
          pointRadius: 0,
          showLine: true,
          fill: false,
        },
      ],
    },
    options: opts,
  });
}

// ----------------------------------------------------------------
// PR curve — FIXED: scatter chart with {x:recall, y:precision} points
// ----------------------------------------------------------------
function renderPRChart(pr) {
  const prPoints = pr.recall.map((r, i) => ({ x: r, y: pr.precision[i] }));

  const opts = scatterOptions('Recall', 'Precision');

  new Chart(document.getElementById('pr-chart'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: `PR curve  (AUC = ${pr.auc})`,
        data: prPoints,
        borderColor: '#7A4D00',
        backgroundColor: 'rgba(122, 77, 0, 0.07)',
        borderWidth: 2,
        pointRadius: 0,
        showLine: true,
        fill: true,
        tension: 0,
      }],
    },
    options: opts,
  });
}

// ----------------------------------------------------------------
// Feature importance
// ----------------------------------------------------------------
function renderFeatureImportance(fiData) {
  const features = fiData.features;
  const labels = features.map(f => humanize(f.feature));
  const values = features.map(f => f.importance);

  // Accent color for top 3, neutral for rest
  const colors = values.map((_, i) =>
    i < 3 ? CHART.accent : CHART.neutral
  );
  const bgColors = colors.map(c => c + '28');

  new Chart(document.getElementById('fi-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Feature importance',
        data: values,
        backgroundColor: bgColors,
        borderColor: colors,
        borderWidth: 1.5,
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
            label: ctx => `Importance: ${ctx.raw.toFixed(6)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: CHART.tick, font: { size: 10, family: CHART.font } },
          grid: { color: CHART.grid },
          title: { display: true, text: 'Mean decrease in impurity', color: CHART.tick },
        },
        y: {
          ticks: { color: '#5C5550', font: { size: 11, family: CHART.font } },
          grid: { display: false },
        },
      },
    },
  });
}

// ----------------------------------------------------------------
// City comparison table
// ----------------------------------------------------------------
function renderCityComparison(cities, region = 'india') {
  const el = document.getElementById('city-comparison');

  function fmtMetric(v, heatwaveDays) {
    // Undefined if no positive test cases
    if (heatwaveDays === 0 || v == null) {
      return '<span class="comp-metric comp-na">n/a</span>';
    }
    return `<span class="comp-metric right">${v.toFixed(4)}</span>`;
  }

  el.innerHTML = `
    <table class="comp-table">
      <thead>
        <tr>
          <th>City</th>
          <th>Region</th>
          <th class="right">Test days</th>
          <th class="right">HW days</th>
          <th class="right">Avg Tmax</th>
          <th class="right">Max Tmax</th>
          <th class="right">F1</th>
          <th class="right">Precision</th>
          <th class="right">Recall</th>
          <th class="right">Accuracy</th>
        </tr>
      </thead>
      <tbody>
        ${cities.map(c => `
          <tr>
            <td class="comp-city">${c.name}</td>
            <td>${c.region}</td>
            <td class="comp-metric right">${c.total_days}</td>
            <td class="comp-hw right">${c.heatwave_days}</td>
            <td class="comp-metric right">${c.avg_tmax != null ? c.avg_tmax + '°C' : '—'}</td>
            <td class="comp-hot right">${c.max_tmax != null ? c.max_tmax + '°C' : '—'}</td>
            <td>${fmtMetric(c.f1,        c.heatwave_days)}</td>
            <td>${fmtMetric(c.precision, c.heatwave_days)}</td>
            <td>${fmtMetric(c.recall,    c.heatwave_days)}</td>
            <td class="comp-metric right">${c.accuracy != null ? c.accuracy.toFixed(4) : '—'}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

// ----------------------------------------------------------------
// DL comparison (India only — guarded, handles missing results)
// ----------------------------------------------------------------
function renderDlComparison(data) {
  const section = document.getElementById('dl-comparison-section');
  if (!section) return;

  // Show the section (it is hidden by default)
  section.hidden = false;

  const unavailEl = document.getElementById('dl-unavailable');

  // Guard: 503 / not generated yet
  if (!data || data.status === 'not_generated') {
    if (unavailEl) unavailEl.hidden = false;
    return;
  }

  // ── Summary table (ensemble rows + RF baselines only) ──────────────────────
  const ENSEMBLE_MODELS = ['RF production', 'RF fair', 'GRU raw-seq ens', 'LSTM raw-seq ens', 'GRU feat110 ens'];
  const rows = (data.rows || []).filter(r => ENSEMBLE_MODELS.includes(r.model));

  const colHeaders = ['Model', 'Features', 'Threshold', 'F1', 'Precision', 'Recall', 'PR-AUC'];
  const tableRows = rows.map(r => {
    const note = r.note || '';
    // Extract CI note for DL rows only
    const ciMatch = note.match(/vs RF-prod \u0394F1 95% CI \[([^\]]+)\] includes_zero=\w+ \(([^)]+)\)/);
    const ciNote = ciMatch ? `<br><span style="font-size:0.75rem; color:var(--ink-muted);">vs RF-prod 95% CI [${ciMatch[1]}]: ${ciMatch[2]}</span>` : '';
    const isBaseline = r.model.startsWith('RF');
    return `<tr${isBaseline ? ' style="color:var(--ink-muted);"' : ''}>
      <td class="comp-city">${r.model}</td>
      <td style="font-size:0.8rem;">${r.features}</td>
      <td style="font-size:0.8rem;">${r.threshold_rule}</td>
      <td class="comp-metric right">${r.F1.toFixed(4)}</td>
      <td class="comp-metric right">${r.Precision.toFixed(4)}</td>
      <td class="comp-metric right">${r.Recall.toFixed(4)}</td>
      <td class="comp-metric right">${r['PR-AUC'].toFixed(4)}</td>
    </tr>${ciNote ? `<tr><td colspan="7" style="padding-top:0; padding-bottom:var(--sp-2);">${ciNote}</td></tr>` : ''}`;
  }).join('');

  const tableHtml = `<table class="comp-table">
    <thead><tr>${colHeaders.map(h => `<th${h !== 'Model' && h !== 'Features' && h !== 'Threshold' ? ' class="right"' : ''}>${h}</th>`).join('')}</tr></thead>
    <tbody>${tableRows}</tbody>
  </table>`;

  const tableEl = document.getElementById('dl-summary-table');
  if (tableEl) tableEl.innerHTML = tableHtml;

  // ── Figures ────────────────────────────────────────────────────────────────
  function loadFigure(imgId, unavailId, figureName) {
    const img = document.getElementById(imgId);
    const unavail = document.getElementById(unavailId);
    if (!img) return;
    img.src = `/api/dl/figure/${figureName}`;
    img.style.display = 'block';
    img.onerror = () => {
      img.style.display = 'none';
      if (unavail) unavail.hidden = false;
    };
    img.onload = () => {
      if (unavail) unavail.hidden = true;
    };
  }

  loadFigure('dl-per-city-fig', 'dl-per-city-unavail', 'per_city_f1');
  loadFigure('dl-pr-fig',       'dl-pr-unavail',       'pr_curves');
  loadFigure('dl-ig-fig',       'dl-ig-unavail',       'dl_feature_importance_ig');
}

document.addEventListener('DOMContentLoaded', () => {
  // Read region from URL
  const params = new URLSearchParams(location.search);
  const initialRegion = params.get('region') === 'europe' ? 'europe' : 'india';

  // Region switch buttons
  const indiaBtn  = document.getElementById('perf-india-btn');
  const europeBtn = document.getElementById('perf-europe-btn');
  const note      = document.getElementById('europe-perf-note');
  const subtitle  = document.getElementById('perf-subtitle');

  async function loadRegion(region) {
    // Update button states
    if (region === 'europe') {
      europeBtn?.classList.add('region-btn--active');
      indiaBtn?.classList.remove('region-btn--active');
      if (note)     note.style.display = 'block';
      if (subtitle) subtitle.textContent = 'Evaluation on the held-out temporal test set (2023–2025) across eleven European cities.';
    } else {
      indiaBtn?.classList.add('region-btn--active');
      europeBtn?.classList.remove('region-btn--active');
      if (note)     note.style.display = 'none';
      if (subtitle) subtitle.textContent = 'Evaluation on the held-out temporal test set (2023–2025) across five Indian cities.';
    }
    const url = region === 'europe' ? '/performance?region=europe' : '/performance';
    const [perfData, fiData, cityData] = await Promise.all([
      fetch(`/api/performance?region=${region}`).then(r => r.json()),
      fetch(`/api/feature-importance`).then(r => r.json()),   // always India model importances
      fetch(`/api/city-comparison?region=${region}`).then(r => r.json()),
    ]);
    renderMetricsBar(perfData);
    renderConfusionMatrix(perfData.confusion_matrix, perfData);
    renderROCChart(perfData.roc);
    renderPRChart(perfData.pr);
    if (region === 'india') renderFeatureImportance(fiData);
    renderCityComparison(cityData, region);

    // India-only extras: evaluation, live track record, DL comparison
    if (region === 'india') {
      fetch('/api/evaluation').then(r => r.ok ? r.json() : null).then(renderEvaluation).catch(() => renderEvaluation(null));
      fetch('/api/live-track-record').then(r => r.ok ? r.json() : null).then(renderLiveTrackRecord).catch(() => {});
      fetch('/api/dl/comparison')
        .then(r => r.json())
        .then(renderDlComparison)
        .catch(() => renderDlComparison(null));
    } else {
      // Hide DL section for non-India regions
      const dlSection = document.getElementById('dl-comparison-section');
      if (dlSection) dlSection.hidden = true;
    }
  }

  indiaBtn?.addEventListener('click',  () => loadRegion('india'));
  europeBtn?.addEventListener('click', () => loadRegion('europe'));

  loadRegion(initialRegion);
});
