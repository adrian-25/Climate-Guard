/**
 * ClimateGuard — Model Performance Page
 * Renders confusion matrix, ROC/PR curves, feature importance, and city comparison.
 */

const API = (path) => fetch(`/api${path}`).then(r => r.json());

const CHART_COLORS = {
  grid: 'rgba(255,255,255,0.04)',
  tick: '#5a5a7a',
  accent: '#6366f1',
  red: '#ef4444',
  amber: '#f59e0b',
  green: '#10b981',
};

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

// ============================================================
// METRICS BAR
// ============================================================
function renderMetricsBar(data) {
  const m = data.global_metrics;
  const metrics = [
    { label: 'F1 Score', value: m.f1, color: '#6366f1' },
    { label: 'Precision', value: m.precision, color: '#8b5cf6' },
    { label: 'Recall', value: m.recall, color: '#a855f7' },
    { label: 'Accuracy', value: m.accuracy, color: '#10b981' },
    { label: 'ROC-AUC', value: m.roc_auc, color: '#f59e0b' },
    { label: 'PR-AUC', value: m.pr_auc, color: '#ef4444' },
    { label: 'Test Samples', value: data.total_test_samples, color: '#64748b', isCount: true },
    { label: 'Heatwave Days', value: data.positive_samples, color: '#ef4444', isCount: true },
  ];

  document.getElementById('perf-metrics-bar').innerHTML = metrics.map(m => `
    <div class="perf-metric">
      <div class="perf-metric__value" style="color: ${m.color};">
        ${m.isCount ? m.value.toLocaleString() : m.value.toFixed(4)}
      </div>
      <div class="perf-metric__label">${m.label}</div>
    </div>
  `).join('');
}

// ============================================================
// CONFUSION MATRIX
// ============================================================
function renderConfusionMatrix(cm, data) {
  const total = cm.tn + cm.fp + cm.fn + cm.tp;
  const el = document.getElementById('confusion-matrix');

  el.innerHTML = `
    <div class="cm-header"></div>
    <div class="cm-header cm-col-header">Predicted Normal</div>
    <div class="cm-header cm-col-header">Predicted Heatwave</div>

    <div class="cm-header cm-row-header">Actual Normal</div>
    <div class="cm-cell cm-tn">
      <div class="cm-cell__value">${cm.tn.toLocaleString()}</div>
      <div class="cm-cell__label">True Negative</div>
      <div class="cm-cell__pct">${(cm.tn/total*100).toFixed(1)}%</div>
    </div>
    <div class="cm-cell cm-fp">
      <div class="cm-cell__value">${cm.fp.toLocaleString()}</div>
      <div class="cm-cell__label">False Positive</div>
      <div class="cm-cell__pct">${(cm.fp/total*100).toFixed(1)}%</div>
    </div>

    <div class="cm-header cm-row-header">Actual Heatwave</div>
    <div class="cm-cell cm-fn">
      <div class="cm-cell__value">${cm.fn.toLocaleString()}</div>
      <div class="cm-cell__label">False Negative</div>
      <div class="cm-cell__pct">${(cm.fn/total*100).toFixed(1)}%</div>
    </div>
    <div class="cm-cell cm-tp">
      <div class="cm-cell__value">${cm.tp.toLocaleString()}</div>
      <div class="cm-cell__label">True Positive</div>
      <div class="cm-cell__pct">${(cm.tp/total*100).toFixed(1)}%</div>
    </div>
  `;
}

// ============================================================
// ROC CURVE
// ============================================================
function renderROCChart(roc) {
  new Chart(document.getElementById('roc-chart'), {
    type: 'line',
    data: {
      labels: roc.fpr,
      datasets: [
        {
          label: `ROC (AUC = ${roc.auc})`,
          data: roc.tpr,
          borderColor: CHART_COLORS.accent,
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
        },
        {
          label: 'Random',
          data: roc.fpr,
          borderColor: 'rgba(255,255,255,0.15)',
          borderWidth: 1,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: chartOptions('False Positive Rate', 'True Positive Rate'),
  });
}

// ============================================================
// PR CURVE
// ============================================================
function renderPRChart(pr) {
  new Chart(document.getElementById('pr-chart'), {
    type: 'line',
    data: {
      labels: pr.recall,
      datasets: [{
        label: `PR (AUC = ${pr.auc})`,
        data: pr.precision,
        borderColor: CHART_COLORS.amber,
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
      }],
    },
    options: chartOptions('Recall', 'Precision'),
  });
}

// ============================================================
// FEATURE IMPORTANCE
// ============================================================
function renderFeatureImportance(fiData) {
  const features = fiData.features;
  const labels = features.map(f => f.feature.replace(/_/g, ' '));
  const values = features.map(f => f.importance);

  // Color gradient from high to low
  const colors = values.map((v, i) => {
    const ratio = i / values.length;
    if (ratio < 0.2) return '#ef4444';
    if (ratio < 0.4) return '#f97316';
    if (ratio < 0.6) return '#f59e0b';
    if (ratio < 0.8) return '#6366f1';
    return '#8b8ba7';
  });

  new Chart(document.getElementById('fi-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance',
        data: values,
        backgroundColor: colors.map(c => c + '33'),
        borderColor: colors,
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
            label: ctx => `Importance: ${ctx.raw.toFixed(6)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: CHART_COLORS.tick, font: { size: 10 } },
          grid: { color: CHART_COLORS.grid },
          title: { display: true, text: 'Feature Importance', color: CHART_COLORS.tick },
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
// CITY COMPARISON
// ============================================================
function renderCityComparison(cities) {
  const el = document.getElementById('city-comparison');

  el.innerHTML = `
    <table class="comp-table">
      <thead>
        <tr>
          <th>City</th>
          <th>Region</th>
          <th>Test Days</th>
          <th>HW Days</th>
          <th>Avg Tmax</th>
          <th>Max Tmax</th>
          <th>F1</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>Accuracy</th>
        </tr>
      </thead>
      <tbody>
        ${cities.map(c => `
          <tr>
            <td class="comp-city">${c.name}</td>
            <td>${c.region}</td>
            <td>${c.total_days}</td>
            <td class="comp-hw">${c.heatwave_days}</td>
            <td>${c.avg_tmax ?? '—'}°C</td>
            <td class="comp-hot">${c.max_tmax ?? '—'}°C</td>
            <td class="comp-metric">${c.f1?.toFixed(4) ?? '—'}</td>
            <td class="comp-metric">${c.precision?.toFixed(4) ?? '—'}</td>
            <td class="comp-metric">${c.recall?.toFixed(4) ?? '—'}</td>
            <td class="comp-metric">${c.accuracy?.toFixed(4) ?? '—'}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

// ============================================================
// SHARED CHART OPTIONS
// ============================================================
function chartOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: CHART_COLORS.tick, font: { family: 'Inter', size: 11 } },
      },
      tooltip: {
        backgroundColor: 'rgba(15, 15, 35, 0.95)',
        titleColor: '#f0f0f8',
        bodyColor: '#8b8ba7',
      },
    },
    scales: {
      x: {
        min: 0, max: 1,
        ticks: { color: CHART_COLORS.tick, font: { size: 10 }, callback: v => v.toFixed(1) },
        grid: { color: CHART_COLORS.grid },
        title: { display: true, text: xLabel, color: CHART_COLORS.tick },
      },
      y: {
        min: 0, max: 1,
        ticks: { color: CHART_COLORS.tick, font: { size: 10 }, callback: v => v.toFixed(1) },
        grid: { color: CHART_COLORS.grid },
        title: { display: true, text: yLabel, color: CHART_COLORS.tick },
      },
    },
  };
}

document.addEventListener('DOMContentLoaded', init);
