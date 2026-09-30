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
// Main
// ----------------------------------------------------------------
async function init() {
  const [perfData, fiData, cityData] = await Promise.all([
    API('/performance'),
    API('/feature-importance'),
    API('/city-comparison'),
  ]);

  renderMetricsBar(perfData);
  renderConfusionMatrix(perfData.confusion_matrix, perfData);
  renderROCChart(perfData.roc);
  renderPRChart(perfData.pr);
  renderFeatureImportance(fiData);
  renderCityComparison(cityData);
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
function renderCityComparison(cities) {
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

document.addEventListener('DOMContentLoaded', init);
