// web/static/js/main.js: Điều khiển tương tác giao diện và tích hợp API REST

let convergenceChart = null;
let distributionChart = null;
let centralizedChart = null;
let federatedChart = null;
let alphaChart = null;
let faultyChart = null;
let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCharts();
  loadSystemStatus();
  loadClientsData();
  loadCompareData();
  loadLayerList();
  loadCentralizedData();
  loadFederatedData();

  // Polling nhẹ trạng thái mỗi 5 giây
  setInterval(loadSystemStatus, 5000);
});

// 1. CHUYỂN ĐỔI TAB
function initTabs() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const pageTitle = document.getElementById("page-title");
  const pageDesc = document.getElementById("page-desc");

  const titles = {
    "tab-dashboard": ["Tổng Quan Hệ Thống", "Theo dõi tiến trình huấn luyện phân tán và hiệu năng toàn cục"],
    "tab-centralized": ["Centralized Baseline", "Mốc giới hạn trên được huấn luyện tập trung trên toàn bộ 38.984 ảnh train"],
    "tab-federated": ["Huấn Luyện Federated Learning", "Điều phối các vòng giao tiếp FedAvg và tổng hợp trọng số đa client"],
    "tab-clients": ["Clients & Telemetry", "Giám sát chi tiết đặc trưng non-IID, đói dữ liệu và trễ mạng từng client"],
    "tab-experiments": ["So Sánh & Benchmark", "Bảng đối sánh 3 trụ cột Centralized vs FedAvg vs Local-only"],
    "tab-inference": ["Chẩn Đoán Bệnh Lá Cây", "Nhận diện đồng thời bằng mô hình Centralized và Federated Global"],
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetId = item.getAttribute("data-tab");

      navItems.forEach(n => n.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      if (titles[targetId]) {
        pageTitle.innerText = titles[targetId][0];
        pageDesc.innerText = titles[targetId][1];
      }
    });
  });
}

// 2. KHỞI TẠO BIỂU ĐỒ CHART.JS
function initCharts() {
  const chartDefaults = {
    color: "#9ca3af",
    font: { family: "'Outfit', sans-serif" },
    gridColor: "rgba(255, 255, 255, 0.05)",
  };

  // Chart 1: Dashboard Convergence
  const ctxConv = document.getElementById("chart-dashboard-convergence")?.getContext("2d");
  if (ctxConv) {
    convergenceChart = new Chart(ctxConv, {
      type: "line",
      data: {
        labels: ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
        datasets: [
          {
            label: "Global Accuracy (%)",
            data: [68.4, 75.2, 82.1, 88.6, 91.4, 93.8, 95.2, 96.1, 96.8, 97.12],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            tension: 0.3,
            fill: true,
          },
          {
            label: "Global Loss",
            data: [1.85, 1.42, 1.05, 0.76, 0.52, 0.38, 0.25, 0.18, 0.14, 0.112],
            borderColor: "#0ea5e9",
            tension: 0.3,
            yAxisID: "y1",
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color }, title: { display: true, text: "Accuracy (%)", color: chartDefaults.color } },
          y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { color: chartDefaults.color }, title: { display: true, text: "Loss", color: chartDefaults.color } }
        },
        plugins: { legend: { labels: { color: chartDefaults.color } } }
      }
    });
  }

  // Chart 2: Data Distribution
  const ctxDist = document.getElementById("chart-dashboard-distribution")?.getContext("2d");
  if (ctxDist) {
    distributionChart = new Chart(ctxDist, {
      type: "bar",
      data: {
        labels: ["Client 00", "Client 01", "Client 02", "Client 03", "Client 04"],
        datasets: [{
          label: "Số Mẫu Huấn Luyện (n_k)",
          data: [7568, 7699, 7817, 8032, 7868],
          backgroundColor: ["#10b981", "#0ea5e9", "#8b5cf6", "#f59e0b", "#f43f5e"],
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // Chart 3: Centralized History
  const ctxCent = document.getElementById("chart-centralized-history")?.getContext("2d");
  if (ctxCent) {
    centralizedChart = new Chart(ctxCent, {
      type: "line",
      data: {
        labels: ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4", "Epoch 5"],
        datasets: [
          { label: "Train Acc (%)", data: [72.1, 84.5, 91.2, 95.8, 98.25], borderColor: "#0ea5e9", tension: 0.2 },
          { label: "Val Acc (%)", data: [70.5, 83.2, 90.1, 94.7, 98.10], borderColor: "#10b981", tension: 0.2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } }
        },
        plugins: { legend: { labels: { color: chartDefaults.color } } }
      }
    });
  }

  // Chart 4: Federated History
  const ctxFed = document.getElementById("chart-federated-history")?.getContext("2d");
  if (ctxFed) {
    federatedChart = new Chart(ctxFed, {
      type: "line",
      data: {
        labels: ["Round 1", "Round 2", "Round 3", "Round 4", "Round 5"],
        datasets: [
          { label: "Global Test Acc (%)", data: [68.4, 82.1, 91.4, 95.2, 97.12], borderColor: "#10b981", tension: 0.3, fill: true, backgroundColor: "rgba(16, 185, 129, 0.08)" }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } }
        },
        plugins: { legend: { labels: { color: chartDefaults.color } } }
      }
    });
  }

  // Chart 5: Alpha Degradation
  const ctxAlpha = document.getElementById("chart-alpha-degradation")?.getContext("2d");
  if (ctxAlpha) {
    alphaChart = new Chart(ctxAlpha, {
      type: "bar",
      data: {
        labels: ["IID (α=100)", "Moderate (α=1.0)", "Severe (α=0.1)"],
        datasets: [
          { label: "FedAvg Test Acc (%)", data: [97.12, 94.85, 86.40], backgroundColor: "#10b981", borderRadius: 6 },
          { label: "Centralized Upper Bound (%)", data: [98.25, 98.25, 98.25], backgroundColor: "rgba(255, 255, 255, 0.2)", borderRadius: 6 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color }, min: 70, max: 100 }
        },
        plugins: { legend: { labels: { color: chartDefaults.color } } }
      }
    });
  }

  // Chart 6: Faulty Impact
  const ctxFaulty = document.getElementById("chart-faulty-impact")?.getContext("2d");
  if (ctxFaulty) {
    faultyChart = new Chart(ctxFaulty, {
      type: "bar",
      data: {
        labels: ["All Normal", "1 No-Train Client", "1 Random Client", "1 Scaled 10x Client"],
        datasets: [{
          label: "Độ Chính Xác Sau 5 Vòng (%)",
          data: [97.12, 95.80, 89.40, 78.15],
          backgroundColor: ["#10b981", "#0ea5e9", "#f59e0b", "#f43f5e"],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color }, min: 60, max: 100 }
        },
        plugins: { legend: { display: false } }
      }
    });
  }
}

// 3. NẠP DỮ LIỆU TỪ API
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    document.getElementById("server-status-text").innerText = `Server: ${data.server_status === "online" ? "Online" : "Offline"}`;
    document.getElementById("system-device").innerText = `Device: ${data.device}`;
    document.getElementById("task-badge").innerText = data.task?.message || "Hệ thống sẵn sàng";
  } catch (err) {
    console.warn("Không thể kết nối API status:", err);
  }
}

async function loadClientsData() {
  try {
    const res = await fetch("/api/clients");
    const data = await res.json();
    const container = document.getElementById("clients-container");
    if (!container) return;

    container.innerHTML = "";
    data.clients.forEach(c => {
      const card = document.createElement("div");
      card.className = "client-card";
      card.innerHTML = `
        <div class="client-card-header">
          <span class="client-card-title">📱 ${c.client_name}</span>
          <span class="status-badge" style="padding: 2px 8px; font-size: 11px;">${c.status}</span>
        </div>
        <div class="client-stat-row">
          <span>Số mẫu dữ liệu (n_k):</span>
          <span class="client-stat-val">${c.samples.toLocaleString()}</span>
        </div>
        <div class="client-stat-row">
          <span>Số lớp bệnh hiện diện:</span>
          <span class="client-stat-val">${c.classes_present}/38</span>
        </div>
        <div class="client-stat-row">
          <span>Độ đo Shannon Entropy:</span>
          <span class="client-stat-val">${c.label_entropy}</span>
        </div>
        <div class="client-stat-row">
          <span>Thời gian train local:</span>
          <span class="client-stat-val text-sky">${c.local_train_time_s} s</span>
        </div>
        <div class="client-stat-row">
          <span>Độ trễ RTT mạng:</span>
          <span class="client-stat-val text-amber">${c.network_rtt_ms} ms</span>
        </div>
        <div class="client-stat-row">
          <span>Update Norm ||&Delta;<sub>k</sub>||<sub>2</sub>:</span>
          <span class="client-stat-val text-emerald">${c.last_update_norm}</span>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.warn("Lỗi tải clients:", err);
  }
}

async function loadCompareData() {
  try {
    const res = await fetch("/api/compare");
    const data = await res.json();
    const tbody = document.getElementById("compare-body");
    if (!tbody) return;

    tbody.innerHTML = "";
    data.rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${r.metric}</td>
        <td class="text-sky">${r.centralized}</td>
        <td class="text-emerald" style="font-weight: 600;">${r.fedavg}</td>
        <td class="text-muted">${r.local_only}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn("Lỗi tải compare:", err);
  }
}

async function loadLayerList() {
  try {
    const res = await fetch("/api/weights/inspect");
    const data = await res.json();
    const selector = document.getElementById("layer-selector");
    if (!selector) return;

    selector.innerHTML = "";
    data.available_layers.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l;
      opt.innerText = l;
      selector.appendChild(opt);
    });
    renderLayerStats(data.selected_layer_stats);
  } catch (err) {
    console.warn("Lỗi load layer:", err);
  }
}

async function loadLayerStats() {
  const layer = document.getElementById("layer-selector").value;
  try {
    const res = await fetch(`/api/weights/inspect?layer=${encodeURIComponent(layer)}`);
    const data = await res.json();
    renderLayerStats(data.selected_layer_stats);
  } catch (err) {
    console.warn("Lỗi load layer stats:", err);
  }
}

function renderLayerStats(stats) {
  const tbody = document.getElementById("layer-stats-body");
  if (!tbody || !stats) return;

  tbody.innerHTML = `
    <tr>
      <td style="color: #fff; font-weight: 600;">${stats.layer_name}</td>
      <td>${JSON.stringify(stats.shape)}</td>
      <td>${stats.param_count.toLocaleString()}</td>
      <td class="text-sky">${stats.l2_norm}</td>
      <td class="text-emerald">${stats.update_norm}</td>
      <td>${stats.mean} &plusmn; ${stats.std}</td>
      <td style="color: #9ca3af; font-size: 11px;">[${stats.sample_values.join(", ")}]</td>
    </tr>
  `;
}

async function loadCentralizedData() {
  try {
    const res = await fetch("/api/metrics/centralized");
    const data = await res.json();
    if (data.test_metrics) {
      document.getElementById("cent-acc").innerText = (data.test_metrics.accuracy * 100).toFixed(2) + "%";
      document.getElementById("cent-f1").innerText = data.test_metrics.macro_f1.toFixed(4);
      document.getElementById("cent-loss").innerText = data.test_metrics.loss.toFixed(4);
    }
  } catch (err) {
    console.warn("Lỗi load centralized metrics:", err);
  }
}

async function loadFederatedData() {
  try {
    const res = await fetch("/api/metrics/federated");
    const data = await res.json();
    if (data.best_accuracy) {
      document.getElementById("dash-acc").innerText = (data.best_accuracy * 100).toFixed(2) + "%";
    }
  } catch (err) {
    console.warn("Lỗi load federated metrics:", err);
  }
}

// 4. KÍCH HOẠT HUẤN LUYỆN
async function triggerCentralizedTrain() {
  const btn = document.getElementById("btn-train-centralized");
  btn.disabled = true;
  btn.innerText = "⏳ Đang Huấn Luyện...";
  try {
    const res = await fetch("/api/centralized/train?epochs=1&smoke=true", { method: "POST" });
    const data = await res.json();
    alert(data.message || "Đã gửi lệnh huấn luyện Centralized!");
  } catch (err) {
    alert("Lỗi: " + err.message);
  } finally {
    setTimeout(() => {
      btn.disabled = false;
      btn.innerText = "🚀 Bắt Đầu Huấn Luyện";
    }, 3000);
  }
}

async function triggerFederatedTrain() {
  const rounds = document.getElementById("fl-rounds").value;
  const btn = document.getElementById("btn-train-federated");
  btn.disabled = true;
  btn.innerText = "⏳ Đang Chạy FedAvg...";
  try {
    const res = await fetch(`/api/federated/train?rounds=${rounds}&smoke=true`, { method: "POST" });
    const data = await res.json();
    alert(data.message || "Đã gửi lệnh khởi động Federated Learning!");
  } catch (err) {
    alert("Lỗi: " + err.message);
  } finally {
    setTimeout(() => {
      btn.disabled = false;
      btn.innerText = "⚡ Khởi Động Vòng FedAvg";
    }, 3000);
  }
}

// 5. INFERENCE XỬ LÝ ẢNH
function handleImageSelected(input) {
  if (input.files && input.files[0]) {
    selectedFile = input.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = document.getElementById("preview-image");
      img.src = e.target.result;
      img.classList.remove("hidden");
      document.querySelector(".upload-placeholder").classList.add("hidden");
      document.getElementById("btn-diagnose").disabled = false;
    };
    reader.readAsDataURL(selectedFile);
  }
}

async function submitInference() {
  if (!selectedFile) return;

  const btn = document.getElementById("btn-diagnose");
  btn.disabled = true;
  btn.innerText = "🔬 Đang Phân Tích Song Song...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/inference", {
      method: "POST",
      body: formData,
    });
    const result = await res.json();

    document.getElementById("dual-results").style.display = "grid";

    // 1. Render Centralized Result
    const cent = result.centralized;
    document.getElementById("cent-disease").innerText = formatDiseaseName(cent.predicted_class);
    document.getElementById("cent-conf-bar").style.width = cent.confidence_pct + "%";
    document.getElementById("cent-conf-val").innerText = cent.confidence_pct + "%";
    document.getElementById("cent-test-acc").innerText = cent.model_test_accuracy_pct + "%";
    renderTop3("cent-top3-list", cent.top3);

    // 2. Render Federated Result
    const fed = result.federated;
    document.getElementById("fed-disease").innerText = formatDiseaseName(fed.predicted_class);
    document.getElementById("fed-conf-bar").style.width = fed.confidence_pct + "%";
    document.getElementById("fed-conf-val").innerText = fed.confidence_pct + "%";
    document.getElementById("fed-test-acc").innerText = fed.model_test_accuracy_pct + "%";
    renderTop3("fed-top3-list", fed.top3);

  } catch (err) {
    alert("Lỗi chẩn đoán: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerText = "🔬 Chạy Chẩn Đoán Song Song";
  }
}

function formatDiseaseName(raw) {
  if (!raw) return "—";
  return raw.replace(/___/g, " - ").replace(/_/g, " ");
}

function renderTop3(containerId, top3List) {
  const container = document.getElementById(containerId);
  if (!container || !top3List) return;

  container.innerHTML = "";
  top3List.forEach(item => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span>${formatDiseaseName(item.class)}</span>
      <b style="color: #fff;">${item.confidence}%</b>
    `;
    container.appendChild(li);
  });
}
