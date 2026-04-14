import './style.css'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

// ==========================================
// DATA LOADING
// ==========================================
async function loadJSON(path) {
  const res = await fetch(path)
  return res.json()
}

async function loadAllData() {
  const [mlResults, dlResults, cvResults, dlCvResults, dlHistories, metadata, features, reviewStats] = await Promise.all([
    loadJSON('/data/ml_results.json'),
    loadJSON('/data/dl_results.json'),
    loadJSON('/data/cv_results.json'),
    loadJSON('/data/dl_cv_results.json'),
    loadJSON('/data/dl_histories.json'),
    loadJSON('/data/metadata.json'),
    loadJSON('/data/feature_importance.json'),
    loadJSON('/data/review_stats.json'),
  ])
  return { mlResults, dlResults, cvResults, dlCvResults, dlHistories, metadata, features, reviewStats }
}

// ==========================================
// CHART DEFAULTS
// ==========================================
Chart.defaults.color = '#94a3b8'
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)'
Chart.defaults.font.family = "'Inter', sans-serif"

const COLORS = {
  blue: '#3b82f6', purple: '#8b5cf6', cyan: '#06b6d4',
  green: '#10b981', red: '#ef4444', orange: '#f59e0b',
  pink: '#ec4899', indigo: '#6366f1',
}
const MODEL_COLORS = [COLORS.blue, COLORS.purple, COLORS.green, COLORS.orange, COLORS.red, COLORS.cyan]

// ==========================================
// APP STRUCTURE
// ==========================================
function buildApp() {
  document.getElementById('app').innerHTML = `
    <div class="bg-gradient"></div>
    <div class="app-layout">
      <!-- Sidebar -->
      <aside class="sidebar">
        <div class="sidebar-logo">
          <h1>📊 VNG Sentiment<br>Analysis</h1>
          <div class="subtitle">ML & DL Dashboard</div>
        </div>
        <nav>
          <div class="nav-section">
            <div class="nav-section-label">Overview</div>
            <div class="nav-item active" data-page="overview">
              <span class="icon">🏠</span> Dashboard
            </div>
          </div>
          <div class="nav-section">
            <div class="nav-section-label">Data</div>
            <div class="nav-item" data-page="data">
              <span class="icon">📊</span> Data Analysis
              <span class="badge">3</span>
            </div>
          </div>
          <div class="nav-section">
            <div class="nav-section-label">Models</div>
            <div class="nav-item" data-page="ml">
              <span class="icon">🤖</span> ML Models
              <span class="badge">4</span>
            </div>
            <div class="nav-item" data-page="dl">
              <span class="icon">🧠</span> DL Models
              <span class="badge">2</span>
            </div>
          </div>
          <div class="nav-section">
            <div class="nav-section-label">Analysis</div>
            <div class="nav-item" data-page="comparison">
              <span class="icon">⚡</span> Comparison
            </div>
            <div class="nav-item" data-page="cv">
              <span class="icon">🔄</span> Cross-Validation
            </div>
          </div>
        </nav>
        <div class="sidebar-footer">
          <p>VNG App Reviews<br>25,125 Reviews • 6 Models</p>
        </div>
      </aside>

      <!-- Main -->
      <main class="main-content">
        <div id="page-overview" class="page-section active"></div>
        <div id="page-data" class="page-section"></div>
        <div id="page-ml" class="page-section"></div>
        <div id="page-dl" class="page-section"></div>
        <div id="page-comparison" class="page-section"></div>
        <div id="page-cv" class="page-section"></div>
      </main>
    </div>

    <!-- Modal -->
    <div class="modal-overlay" id="modal">
      <button class="modal-close" id="modal-close">✕</button>
      <div class="modal-content">
        <img id="modal-img" src="" alt="">
        <div class="modal-title" id="modal-title"></div>
      </div>
    </div>
  `
}

// ==========================================
// NAVIGATION
// ==========================================
function setupNavigation(data) {
  const navItems = document.querySelectorAll('.nav-item')
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navItems.forEach(n => n.classList.remove('active'))
      item.classList.add('active')
      const page = item.dataset.page
      document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'))
      document.getElementById(`page-${page}`).classList.add('active')
    })
  })

  // Modal
  document.getElementById('modal').addEventListener('click', () => {
    document.getElementById('modal').classList.remove('active')
  })
  document.getElementById('modal-close').addEventListener('click', () => {
    document.getElementById('modal').classList.remove('active')
  })
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.getElementById('modal').classList.remove('active')
  })
}

function openModal(src, title) {
  document.getElementById('modal-img').src = src
  document.getElementById('modal-title').textContent = title
  document.getElementById('modal').classList.add('active')
}

// ==========================================
// HELPER: Chart image card
// ==========================================
function chartImgCard(src, title, tag) {
  return `
    <div class="chart-img-card" onclick="window.__openModal('/charts/${src}', '${title}')">
      <div class="chart-img-header">
        <div class="chart-img-title">${title}</div>
        <span class="tag tag-${tag}">${tag}</span>
      </div>
      <div class="chart-img-wrap">
        <img src="/charts/${src}" alt="${title}" loading="lazy">
      </div>
    </div>
  `
}

// ==========================================
// PAGE: Overview
// ==========================================
function renderOverview(data) {
  const { mlResults, dlResults, reviewStats, metadata } = data

  // Find best model
  const allModels = { ...mlResults, ...dlResults }
  let bestName = '', bestF1 = 0
  for (const [name, r] of Object.entries(allModels)) {
    if (r.metrics.f1_macro > bestF1) { bestF1 = r.metrics.f1_macro; bestName = name }
  }
  const bestAcc = allModels[bestName]?.metrics?.accuracy || 0

  const el = document.getElementById('page-overview')
  el.innerHTML = `
    <div class="page-header">
      <h2>📊 Dashboard Overview</h2>
      <p>Tổng quan kết quả phân tích cảm xúc VNG App Reviews</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Total Reviews</div>
        <div class="stat-value">${(reviewStats.total_reviews || 25125).toLocaleString()}</div>
        <div class="stat-sub">10 VNG Apps</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Models Trained</div>
        <div class="stat-value">6</div>
        <div class="stat-sub">4 ML + 2 DL</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Best Accuracy</div>
        <div class="stat-value">${(bestAcc * 100).toFixed(1)}%</div>
        <div class="stat-sub">${bestName}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Best F1 Score</div>
        <div class="stat-value">${bestF1.toFixed(4)}</div>
        <div class="stat-sub">${bestName}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Charts</div>
        <div class="stat-value">18</div>
        <div class="stat-sub">Interactive</div>
      </div>
    </div>

    <!-- Sentiment donut + Model comparison -->
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Sentiment Distribution</span>
          <span class="tag tag-data">DATA</span>
        </div>
        <div class="card-body chart-body">
          <canvas id="chart-sentiment-donut" height="260"></canvas>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">All Models — Accuracy</span>
          <span class="tag tag-all">ALL</span>
        </div>
        <div class="card-body chart-body">
          <canvas id="chart-models-acc" height="260"></canvas>
        </div>
      </div>
    </div>

    <!-- Results table -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">All Models Results</span>
        <span class="tag tag-all">ALL</span>
      </div>
      <div class="card-body" style="padding: 0;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Model</th><th>Type</th><th>Accuracy</th><th>F1 Macro</th>
              <th>Precision</th><th>Recall</th>
            </tr>
          </thead>
          <tbody id="table-all-models"></tbody>
        </table>
      </div>
    </div>

    <!-- Feature importance -->
    <div class="section" style="margin-top: 24px;">
      <div class="section-header">
        <div class="section-icon data">🔑</div>
        <div>
          <div class="section-title">Feature Importance</div>
          <div class="section-sub">Top keywords per sentiment class</div>
        </div>
      </div>
      <div class="feature-grid" id="feature-grid"></div>
    </div>
  `

  // Sentiment donut
  const sentDist = reviewStats.sentiment_dist || {}
  new Chart(document.getElementById('chart-sentiment-donut'), {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{
        data: [sentDist.Positive || 0, sentDist.Negative || 0, sentDist.Neutral || 0],
        backgroundColor: [COLORS.green, COLORS.red, COLORS.orange],
        borderWidth: 0,
        hoverOffset: 8,
      }]
    },
    options: {
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, pointStyleWidth: 10 } }
      }
    }
  })

  // Models accuracy bar
  const modelNames = [...Object.keys(mlResults), ...Object.keys(dlResults)]
  const accValues = modelNames.map(n => (allModels[n]?.metrics?.accuracy || 0) * 100)
  new Chart(document.getElementById('chart-models-acc'), {
    type: 'bar',
    data: {
      labels: modelNames,
      datasets: [{
        label: 'Accuracy (%)',
        data: accValues,
        backgroundColor: MODEL_COLORS.slice(0, modelNames.length),
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      scales: { x: { min: 70, max: 85, ticks: { callback: v => v + '%' } } },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ctx.parsed.x.toFixed(2) + '%' } }
      }
    }
  })

  // Table
  const tbody = document.getElementById('table-all-models')
  const allEntries = [
    ...Object.entries(mlResults).map(([n, r]) => ({ name: n, type: 'ML', ...r })),
    ...Object.entries(dlResults).map(([n, r]) => ({ name: n, type: 'DL', ...r })),
  ]
  allEntries.forEach(({ name, type, metrics }) => {
    const isBest = name === bestName
    tbody.innerHTML += `
      <tr class="${isBest ? 'best-row' : ''}">
        <td class="model-name">${name} ${isBest ? '<span class="tag tag-best">BEST</span>' : ''}</td>
        <td><span class="tag tag-${type.toLowerCase()}">${type}</span></td>
        <td class="${isBest ? 'best-value' : ''}">${metrics.accuracy.toFixed(4)}</td>
        <td class="${isBest ? 'best-value' : ''}">${metrics.f1_macro.toFixed(4)}</td>
        <td>${metrics.precision_macro.toFixed(4)}</td>
        <td>${metrics.recall_macro.toFixed(4)}</td>
      </tr>
    `
  })

  // Features
  const featureGrid = document.getElementById('feature-grid')
  const featureData = {
    Negative: { emoji: '😡', words: data.features.Negative || [] },
    Neutral: { emoji: '😐', words: data.features.Neutral || [] },
    Positive: { emoji: '😊', words: data.features.Positive || [] },
  }
  for (const [cls, { emoji, words }] of Object.entries(featureData)) {
    const topWords = words.slice(0, 12).map(w => Array.isArray(w) ? w[0] : w)
    featureGrid.innerHTML += `
      <div class="feature-box ${cls.toLowerCase()}">
        <div class="feature-label">${emoji} ${cls}</div>
        <div class="feature-words">
          ${topWords.map(w => `<span class="feature-word">${w}</span>`).join('')}
        </div>
      </div>
    `
  }
}

// ==========================================
// PAGE: Data Analysis
// ==========================================
function renderDataPage(data) {
  const { reviewStats } = data
  const el = document.getElementById('page-data')
  el.innerHTML = `
    <div class="page-header">
      <h2>📊 Data Analysis</h2>
      <p>Phân bố dữ liệu & phân tích từ khóa</p>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">Star Rating Distribution</span><span class="tag tag-data">DATA</span></div>
        <div class="card-body chart-body"><canvas id="chart-star-dist" height="250"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Reviews per App</span><span class="tag tag-data">DATA</span></div>
        <div class="card-body chart-body"><canvas id="chart-app-dist" height="250"></canvas></div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <div class="card-header"><span class="card-title">Sentiment per App (Stacked)</span><span class="tag tag-data">DATA</span></div>
      <div class="card-body chart-body"><canvas id="chart-app-sentiment" height="300"></canvas></div>
    </div>

    <div class="grid-2">
      ${chartImgCard('wordclouds.png', 'Word Clouds by Sentiment', 'data')}
      ${chartImgCard('data_distribution.png', 'Data Distribution Overview', 'data')}
    </div>
    ${chartImgCard('sentiment_by_app.png', 'Sentiment by App (Detailed)', 'data')}
  `

  // Star dist
  const scoreDist = reviewStats.score_dist || {}
  const starColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#10b981']
  new Chart(document.getElementById('chart-star-dist'), {
    type: 'bar',
    data: {
      labels: ['1⭐', '2⭐', '3⭐', '4⭐', '5⭐'],
      datasets: [{
        data: [1,2,3,4,5].map(s => scoreDist[s] || 0),
        backgroundColor: starColors,
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  })

  // App dist
  const appDist = reviewStats.app_dist || {}
  new Chart(document.getElementById('chart-app-dist'), {
    type: 'bar',
    data: {
      labels: Object.keys(appDist),
      datasets: [{
        data: Object.values(appDist),
        backgroundColor: MODEL_COLORS.concat(MODEL_COLORS),
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true } }
    }
  })

  // App sentiment stacked
  const appSent = reviewStats.app_sentiment || {}
  const apps = Object.keys(appSent)
  new Chart(document.getElementById('chart-app-sentiment'), {
    type: 'bar',
    data: {
      labels: apps,
      datasets: [
        { label: 'Positive', data: apps.map(a => appSent[a]?.Positive || 0), backgroundColor: COLORS.green, borderRadius: 4 },
        { label: 'Negative', data: apps.map(a => appSent[a]?.Negative || 0), backgroundColor: COLORS.red, borderRadius: 4 },
        { label: 'Neutral', data: apps.map(a => appSent[a]?.Neutral || 0), backgroundColor: COLORS.orange, borderRadius: 4 },
      ]
    },
    options: {
      scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
      plugins: { legend: { position: 'top', labels: { usePointStyle: true } } }
    }
  })
}

// ==========================================
// PAGE: ML Models
// ==========================================
function renderMLPage(data) {
  const { mlResults, cvResults } = data
  const el = document.getElementById('page-ml')

  el.innerHTML = `
    <div class="page-header">
      <h2>🤖 Machine Learning Models</h2>
      <p>Naive Bayes • Logistic Regression • SVM • Random Forest — TF-IDF Features</p>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">ML Models — F1 Score Comparison</span><span class="tag tag-ml">ML</span></div>
        <div class="card-body chart-body"><canvas id="chart-ml-f1" height="260"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">ML Models — Radar</span><span class="tag tag-ml">ML</span></div>
        <div class="card-body chart-body"><canvas id="chart-ml-radar" height="260"></canvas></div>
      </div>
    </div>

    <div class="grid-2">
      ${chartImgCard('ml_confusion_matrices.png', 'ML Confusion Matrices', 'ml')}
      ${chartImgCard('ml_model_comparison.png', 'ML Model Comparison', 'ml')}
    </div>

    <div class="grid-2">
      ${chartImgCard('ml_report_naive_bayes.png', 'Naive Bayes Report', 'ml')}
      ${chartImgCard('ml_report_logistic_regression.png', 'Logistic Regression Report', 'ml')}
      ${chartImgCard('ml_report_svm_linear.png', 'SVM (Linear) Report', 'ml')}
      ${chartImgCard('ml_report_random_forest.png', 'Random Forest Report', 'ml')}
    </div>
  `

  // F1 comparison
  const mlNames = Object.keys(mlResults)
  const metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
  const metricLabels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

  new Chart(document.getElementById('chart-ml-f1'), {
    type: 'bar',
    data: {
      labels: mlNames,
      datasets: metrics.map((m, i) => ({
        label: metricLabels[i],
        data: mlNames.map(n => mlResults[n].metrics[m]),
        backgroundColor: MODEL_COLORS[i],
        borderRadius: 4,
      }))
    },
    options: {
      scales: { y: { min: 0.4, max: 0.9 } },
      plugins: { legend: { position: 'top', labels: { usePointStyle: true, pointStyleWidth: 10 } } }
    }
  })

  // Radar
  new Chart(document.getElementById('chart-ml-radar'), {
    type: 'radar',
    data: {
      labels: metricLabels,
      datasets: mlNames.map((n, i) => ({
        label: n,
        data: metrics.map(m => mlResults[n].metrics[m]),
        borderColor: MODEL_COLORS[i],
        backgroundColor: MODEL_COLORS[i] + '20',
        pointBackgroundColor: MODEL_COLORS[i],
        borderWidth: 2,
      }))
    },
    options: {
      scales: { r: { min: 0.4, max: 0.9, ticks: { stepSize: 0.1 } } },
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true } } }
    }
  })
}

// ==========================================
// PAGE: DL Models
// ==========================================
function renderDLPage(data) {
  const { dlResults, dlHistories } = data
  const el = document.getElementById('page-dl')

  el.innerHTML = `
    <div class="page-header">
      <h2>🧠 Deep Learning Models</h2>
      <p>Bidirectional LSTM • CNN (Conv1D) — Word Embeddings</p>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">BiLSTM — Training Loss</span><span class="tag tag-dl">DL</span></div>
        <div class="card-body chart-body"><canvas id="chart-bilstm-loss" height="240"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">BiLSTM — Training Accuracy</span><span class="tag tag-dl">DL</span></div>
        <div class="card-body chart-body"><canvas id="chart-bilstm-acc" height="240"></canvas></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">CNN — Training Loss</span><span class="tag tag-dl">DL</span></div>
        <div class="card-body chart-body"><canvas id="chart-cnn-loss" height="240"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">CNN — Training Accuracy</span><span class="tag tag-dl">DL</span></div>
        <div class="card-body chart-body"><canvas id="chart-cnn-acc" height="240"></canvas></div>
      </div>
    </div>

    <div class="grid-2">
      ${chartImgCard('dl_confusion_matrices.png', 'DL Confusion Matrices', 'dl')}
      ${chartImgCard('dl_model_comparison.png', 'DL Model Comparison', 'dl')}
      ${chartImgCard('dl_report_bilstm.png', 'BiLSTM Classification Report', 'dl')}
      ${chartImgCard('dl_report_cnn.png', 'CNN Classification Report', 'dl')}
    </div>
  `

  // Training histories
  function plotHistory(canvasId, histKey, label, color1, color2) {
    const hist = dlHistories[Object.keys(dlHistories).find(k => canvasId.includes(k.toLowerCase()))]
    if (!hist) return
    const trainData = hist[histKey] || []
    const valData = hist['val_' + histKey] || []
    new Chart(document.getElementById(canvasId), {
      type: 'line',
      data: {
        labels: trainData.map((_, i) => `Epoch ${i + 1}`),
        datasets: [
          { label: `Train ${label}`, data: trainData, borderColor: color1, backgroundColor: color1 + '20', tension: 0.3, pointRadius: 4, fill: true },
          { label: `Val ${label}`, data: valData, borderColor: color2, backgroundColor: color2 + '20', tension: 0.3, pointRadius: 4, fill: true, borderDash: [5, 5] },
        ]
      },
      options: {
        plugins: { legend: { labels: { usePointStyle: true } } },
        interaction: { intersect: false, mode: 'index' },
      }
    })
  }

  plotHistory('chart-bilstm-loss', 'loss', 'Loss', COLORS.blue, COLORS.red)
  plotHistory('chart-bilstm-acc', 'accuracy', 'Accuracy', COLORS.blue, COLORS.green)
  plotHistory('chart-cnn-loss', 'loss', 'Loss', COLORS.purple, COLORS.red)
  plotHistory('chart-cnn-acc', 'accuracy', 'Accuracy', COLORS.purple, COLORS.green)
}

// ==========================================
// PAGE: Comparison
// ==========================================
function renderComparisonPage(data) {
  const { mlResults, dlResults } = data
  const el = document.getElementById('page-comparison')

  el.innerHTML = `
    <div class="page-header">
      <h2>⚡ All Models Comparison</h2>
      <p>ML vs Deep Learning — Performance metrics side by side</p>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <div class="card-header"><span class="card-title">All Models — Multi-Metric Comparison</span><span class="tag tag-all">ALL</span></div>
      <div class="card-body chart-body"><canvas id="chart-all-compare" height="350"></canvas></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">Accuracy vs F1 Score (Scatter)</span><span class="tag tag-all">ALL</span></div>
        <div class="card-body chart-body"><canvas id="chart-scatter" height="280"></canvas></div>
      </div>
      ${chartImgCard('all_models_comparison.png', 'All Models Comparison (Static)', 'all')}
    </div>
  `

  const allModels = { ...mlResults, ...dlResults }
  const names = Object.keys(allModels)
  const metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
  const metricLabels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

  // Grouped bar
  new Chart(document.getElementById('chart-all-compare'), {
    type: 'bar',
    data: {
      labels: names.map(n => (n in mlResults ? 'ML: ' : 'DL: ') + n),
      datasets: metrics.map((m, i) => ({
        label: metricLabels[i],
        data: names.map(n => allModels[n].metrics[m]),
        backgroundColor: MODEL_COLORS[i],
        borderRadius: 6,
      }))
    },
    options: {
      scales: { y: { min: 0.4, max: 0.9, ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, pointStyleWidth: 10 } },
        tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y * 100).toFixed(2) + '%' } }
      }
    }
  })

  // Scatter
  new Chart(document.getElementById('chart-scatter'), {
    type: 'scatter',
    data: {
      datasets: names.map((n, i) => ({
        label: n,
        data: [{ x: allModels[n].metrics.accuracy * 100, y: allModels[n].metrics.f1_macro * 100 }],
        backgroundColor: MODEL_COLORS[i],
        pointRadius: 10,
        pointHoverRadius: 14,
      }))
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'Accuracy (%)' }, min: 78, max: 83 },
        y: { title: { display: true, text: 'F1 Score (%)' }, min: 53, max: 57 }
      },
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: Acc=${ctx.parsed.x.toFixed(2)}%, F1=${ctx.parsed.y.toFixed(2)}%` } }
      }
    }
  })
}

// ==========================================
// PAGE: Cross-Validation
// ==========================================
function renderCVPage(data) {
  const { cvResults, dlCvResults } = data
  const el = document.getElementById('page-cv')

  el.innerHTML = `
    <div class="page-header">
      <h2>🔄 Cross-Validation Results</h2>
      <p>StratifiedKFold (k=5) — Model stability analysis</p>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><span class="card-title">ML — CV Accuracy per Fold</span><span class="tag tag-ml">ML</span></div>
        <div class="card-body chart-body"><canvas id="chart-ml-cv-line" height="260"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">DL — CV Accuracy per Fold</span><span class="tag tag-dl">DL</span></div>
        <div class="card-body chart-body"><canvas id="chart-dl-cv-line" height="260"></canvas></div>
      </div>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <div class="card-header"><span class="card-title">Cross-Validation Summary</span><span class="tag tag-all">ALL</span></div>
      <div class="card-body" style="padding: 0;">
        <table class="data-table">
          <thead>
            <tr><th>Model</th><th>Type</th><th>CV Accuracy</th><th>± Std</th><th>CV F1</th><th>± Std</th></tr>
          </thead>
          <tbody id="table-cv"></tbody>
        </table>
      </div>
    </div>

    <div class="grid-2">
      ${chartImgCard('ml_cv_boxplot.png', 'ML Cross-Validation Box Plot', 'ml')}
      ${chartImgCard('dl_cv_boxplot.png', 'DL Cross-Validation Box Plot', 'dl')}
    </div>
  `

  // ML CV line chart
  const mlCvNames = Object.keys(cvResults).filter(n => cvResults[n].test_accuracy)
  new Chart(document.getElementById('chart-ml-cv-line'), {
    type: 'line',
    data: {
      labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
      datasets: mlCvNames.map((n, i) => ({
        label: n,
        data: cvResults[n].test_accuracy,
        borderColor: MODEL_COLORS[i],
        backgroundColor: MODEL_COLORS[i] + '20',
        tension: 0.3,
        pointRadius: 5,
        pointHoverRadius: 8,
        fill: false,
      }))
    },
    options: {
      scales: { y: { min: 0.77, max: 0.83, ticks: { callback: v => (v * 100).toFixed(1) + '%' } } },
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true } } },
      interaction: { intersect: false, mode: 'index' },
    }
  })

  // DL CV line chart
  const dlCvNames = Object.keys(dlCvResults).filter(n => dlCvResults[n].accuracy)
  if (dlCvNames.length > 0) {
    new Chart(document.getElementById('chart-dl-cv-line'), {
      type: 'line',
      data: {
        labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
        datasets: dlCvNames.map((n, i) => ({
          label: n,
          data: dlCvResults[n].accuracy,
          borderColor: [COLORS.green, COLORS.cyan][i],
          tension: 0.3,
          pointRadius: 5,
          fill: false,
        }))
      },
      options: {
        scales: { y: { min: 0.77, max: 0.83, ticks: { callback: v => (v * 100).toFixed(1) + '%' } } },
        plugins: { legend: { position: 'bottom', labels: { usePointStyle: true } } },
        interaction: { intersect: false, mode: 'index' },
      }
    })
  }

  // CV table
  const tbody = document.getElementById('table-cv')
  for (const [n, cv] of Object.entries(cvResults)) {
    if (cv.mean_test_accuracy) {
      tbody.innerHTML += `<tr>
        <td class="model-name">${n}</td><td><span class="tag tag-ml">ML</span></td>
        <td>${(cv.mean_test_accuracy * 100).toFixed(2)}%</td><td>${(cv.std_test_accuracy * 100).toFixed(2)}%</td>
        <td>${(cv.mean_test_f1 * 100).toFixed(2)}%</td><td>${(cv.std_test_f1 * 100).toFixed(2)}%</td>
      </tr>`
    }
  }
  for (const [n, cv] of Object.entries(dlCvResults)) {
    if (cv.mean_accuracy != null) {
      tbody.innerHTML += `<tr>
        <td class="model-name">${n}</td><td><span class="tag tag-dl">DL</span></td>
        <td>${(cv.mean_accuracy * 100).toFixed(2)}%</td><td>${(cv.std_accuracy * 100).toFixed(2)}%</td>
        <td>${cv.mean_f1 != null ? (cv.mean_f1 * 100).toFixed(2) + '%' : '—'}</td>
        <td>${cv.std_f1 != null ? (cv.std_f1 * 100).toFixed(2) + '%' : '—'}</td>
      </tr>`
    }
  }
}

// ==========================================
// INIT
// ==========================================
async function init() {
  buildApp()

  const data = await loadAllData()

  setupNavigation(data)

  // Expose modal helper for onclick in HTML strings
  window.__openModal = openModal

  // Render all pages
  renderOverview(data)
  renderDataPage(data)
  renderMLPage(data)
  renderDLPage(data)
  renderComparisonPage(data)
  renderCVPage(data)
}

init()
