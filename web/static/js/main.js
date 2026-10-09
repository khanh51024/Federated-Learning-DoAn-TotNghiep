// web/static/js/main.js: Điều khiển tương tác giao diện và tích hợp API REST
// Đáp ứng toàn bộ các yêu cầu kiểm thử và chuẩn hóa dữ liệu thực tế.

const SVG_ICONS = {
  play: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>`,
  zap: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  stop: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>`,
  spinner: `<svg class="svg-icon icon-sm spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>`,
  microscope: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>`,
  server: `<svg class="svg-icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>`,
  clock: `<svg class="svg-icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
  activity: `<svg class="svg-icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
  layers: `<svg class="svg-icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`,
  check: `<svg class="svg-icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`,
  info: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
};

let convergenceChart = null;
let distributionChart = null;
let centralizedChart = null;
let federatedChart = null;
let alphaChart = null;
let faultyChart = null;
let selectedFile = null;
let isChartLibAvailable = true;
let currentTask = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCharts();
  loadSystemStatus();
  loadPublishedDemo();
  loadNetworkMetrics();
  toggleFederatedEngineFields();

  // Polling chu kỳ 4 giây để đồng bộ trạng thái server và tác vụ
  setInterval(loadSystemStatus, 4000);
});

// 1. CHUYỂN ĐỔI TAB & BỐ CỤC
function initTabs() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const pageTitle = document.getElementById("page-title");
  const pageDesc = document.getElementById("page-desc");

  const titles = {
    "tab-dashboard": ["Tổng Quan Hệ Thống", "Theo dõi tiến trình huấn luyện phân tán và hiệu năng toàn cục"],
    "tab-centralized": ["Centralized Baseline", "Validation của checkpoint pretrained v6 trên release PlantVillage + PlantDoc"],
    "tab-federated": ["Huấn Luyện Federated Learning", "Điều phối các vòng giao tiếp FedAvg và tổng hợp trọng số đa client"],
    "tab-clients": ["Clients & Telemetry", "Năm shard đã công bố và phiên Flower localhost riêng"],
    "tab-experiments": ["So Sánh & Benchmark", "Bảng đối sánh 3 trụ cột Centralized vs FedAvg vs Local-only"],
    "tab-inference": ["Chẩn Đoán Bệnh Lá Cây", "So checkpoint Centralized với FedAvg hoặc từng Local-only trên ảnh vùng lá"],
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

// 2. KHỞI TẠO BIỂU ĐỒ CHART.JS (CÓ BẢO VỆ CDN CHỐNG CRASH)
function initCharts() {
  if (typeof Chart === "undefined") {
    console.warn("[Chart.js] Không tìm thấy thư viện Chart.js (CDN lỗi hoặc mạng offline). Bỏ qua vẽ biểu đồ.");
    isChartLibAvailable = false;
    document.querySelectorAll(".chart-container").forEach(c => {
      const notice = document.createElement("div");
      notice.className = "chart-fallback-notice";
      notice.style.padding = "20px";
      notice.style.color = "#9ca3af";
      notice.style.textAlign = "center";
      notice.style.fontSize = "13px";
      notice.innerText = "Biểu đồ đang hoạt động ở chế độ văn bản (CDN Chart.js không khả dụng).";
      c.appendChild(notice);
    });
    return;
  }

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
        labels: [],
        datasets: [
          {
            label: "Global Accuracy (%)",
            data: [],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            tension: 0.3,
            fill: true,
          },
          {
            label: "Global Loss",
            data: [],
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
          data: [],
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
        labels: [],
        datasets: [
          { label: "Train Acc (%)", data: [], borderColor: "#0ea5e9", tension: 0.2 },
          { label: "Val Acc (%)", data: [], borderColor: "#10b981", tension: 0.2 }
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
        labels: [],
        datasets: [
          { label: "Global Test Acc (%)", data: [], borderColor: "#10b981", tension: 0.3, fill: true, backgroundColor: "rgba(16, 185, 129, 0.08)" }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color } },
          y: {
            min: 0,
            max: 100,
            grid: { color: chartDefaults.gridColor },
            ticks: {
              color: chartDefaults.color,
              callback: (v) => v + "%"
            }
          }
        },
        plugins: { legend: { labels: { color: chartDefaults.color } } }
      }
    });
  }

  // Chart 5: Alpha Degradation (Dữ liệu thực nghiệm GĐ2 đã nghiệm thu)
  const ctxAlpha = document.getElementById("chart-alpha-degradation")?.getContext("2d");
  if (ctxAlpha) {
    alphaChart = new Chart(ctxAlpha, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          { label: "FedAvg validation PlantVillage (%)", data: [], backgroundColor: "#10b981", borderRadius: 6 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { color: chartDefaults.color } },
          y: { grid: { color: chartDefaults.gridColor }, ticks: { color: chartDefaults.color }, min: 70, max: 100 }
        },
        plugins: {
          legend: { labels: { color: chartDefaults.color } },
          title: { display: true, text: "Một kịch bản α=0,1; chưa đủ đường suy giảm", color: "#9ca3af", font: { size: 12 } }
        }
      }
    });
  }

  // Chart 6: Faulty Client Impact (Thực nghiệm GĐ2)
  const ctxFaulty = document.getElementById("chart-faulty-impact")?.getContext("2d");
  if (ctxFaulty) {
    faultyChart = new Chart(ctxFaulty, {
      type: "bar",
      data: {
        labels: [],
        datasets: [{
          label: "Độ Chính Xác Sau 5 Vòng (%)",
          data: [],
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
        plugins: {
          legend: { display: false },
          title: { display: true, text: "Chưa có số liệu cùng protocol", color: "#9ca3af", font: { size: 12 } }
        }
      }
    });
  }
}

// 3. NẠP DỮ LIỆU TỪ REST API
let previousTaskStatus = "idle";

async function loadSystemStatus() {
  const statusIndicator = document.getElementById("server-status-text");
  const taskBadge = document.getElementById("task-badge");
  const deviceText = document.getElementById("system-device");
  const btnCent = document.getElementById("btn-train-centralized");
  const btnFed = document.getElementById("btn-train-federated");

  try {
    const res = await fetch("/api/status");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();

    if (statusIndicator) {
      statusIndicator.innerText = `Server: ${data.server_status === "online" ? "Online" : "Offline"}`;
      statusIndicator.parentElement?.classList.remove("offline");
    }
    if (deviceText) {
      const flText = data.flower_server_active ? "Flower: Live" : "Flower: Idle";
      deviceText.innerText = `Device: ${data.device} | ${flText}`;
    }

    currentTask = data.task || { status: "idle", message: "Hệ thống sẵn sàng" };
    const network = data.network_processes || {};
    const dashClients = document.getElementById("dash-clients");
    const dashRtt = document.getElementById("dash-rtt");
    if (dashClients) dashClients.innerText = `${currentTask.type === "federated-network" && currentTask.status === "completed" ? network.accepted_clients_last_round || 0 : network.connected_clients || 0} / 5`;
    if (dashRtt) dashRtt.innerText = Number.isFinite(network.mean_probe_rtt_ms) ? `${Number(network.mean_probe_rtt_ms).toFixed(2)} ms (localhost)` : "—";
    if (taskBadge) {
      taskBadge.innerText = currentTask.message || "Hệ thống sẵn sàng";
      taskBadge.className = "status-badge";
      if (currentTask.status === "running") taskBadge.classList.add("running");
      else if (currentTask.status === "completed") taskBadge.classList.add("success");
      else if (currentTask.status === "failed" || currentTask.status === "cancelled") taskBadge.classList.add("error");
    }

    const centBarBox = document.getElementById("cent-progress-bar-box");
    const centBarFill = document.getElementById("cent-progress-fill");
    const centBarLabel = document.getElementById("cent-progress-label");
    const centBarDetail = document.getElementById("cent-progress-detail");

    const fedBarBox = document.getElementById("fed-progress-bar-box");
    const fedBarFill = document.getElementById("fed-progress-fill");
    const fedBarLabel = document.getElementById("fed-progress-label");
    const fedBarDetail = document.getElementById("fed-progress-detail");

    const btnStopCent = document.getElementById("btn-stop-centralized");
    const btnStopFed = document.getElementById("btn-stop-federated");

    const isRunning = currentTask.status === "running";
    const isCent = currentTask.type === "centralized" || currentTask.experiment === "Centralized";
    const isFed = currentTask.type === "federated" || currentTask.type === "federated-network" || currentTask.experiment === "Federated";

    // Quản lý trạng thái nút và tiến trình theo trạng thái thật của tác vụ nền
    if (isRunning) {
      if (btnCent) {
        btnCent.disabled = true;
        btnCent.innerHTML = `${SVG_ICONS.spinner} Đang Huấn Luyện...`;
      }
      if (btnFed) {
        btnFed.disabled = true;
        btnFed.innerHTML = `${SVG_ICONS.spinner} Đang Điều Phối FedAvg...`;
      }

      if (isCent) {
        if (btnStopCent) {
          btnStopCent.style.display = "inline-flex";
          btnStopCent.disabled = currentTask.cancel_requested || false;
          btnStopCent.innerHTML = currentTask.cancel_requested ? `${SVG_ICONS.spinner} Đang Dừng...` : `${SVG_ICONS.stop} Dừng Huấn Luyện`;
        }
        if (btnStopFed) btnStopFed.style.display = "none";

        if (centBarBox) centBarBox.style.display = "block";
        const pct = Math.max(5, Math.min(100, currentTask.progress || 5));
        if (centBarFill) centBarFill.style.width = `${pct}%`;
        if (centBarLabel) centBarLabel.innerText = `Tiến trình: ${pct}%`;
        if (centBarDetail) centBarDetail.innerText = currentTask.message || "Đang tính toán epoch...";

        if (currentTask.live_progress) {
          renderCentralizedProgress(currentTask.live_progress, true, currentTask);
        }
      } else if (isFed) {
        if (btnStopFed) {
          btnStopFed.style.display = "inline-flex";
          btnStopFed.disabled = currentTask.cancel_requested || false;
          btnStopFed.innerHTML = currentTask.cancel_requested ? `${SVG_ICONS.spinner} Đang Dừng...` : `${SVG_ICONS.stop} Dừng Huấn Luyện`;
        }
        if (btnStopCent) btnStopCent.style.display = "none";

        if (fedBarBox) fedBarBox.style.display = "block";
        const pct = Math.max(5, Math.min(100, currentTask.progress || 5));
        if (fedBarFill) fedBarFill.style.width = `${pct}%`;
        if (fedBarLabel) fedBarLabel.innerText = `Tiến trình: ${pct}%`;
        if (fedBarDetail) fedBarDetail.innerText = currentTask.message || "Đang tổng hợp vòng...";

        if (currentTask.live_progress) {
          renderFederatedProgress(currentTask.live_progress, true, currentTask);
        }
      }
    } else {
      if (btnCent && btnCent.disabled) {
        btnCent.disabled = false;
        btnCent.innerHTML = `${SVG_ICONS.play} Bắt Đầu Huấn Luyện`;
      }
      if (btnFed && btnFed.disabled) {
        btnFed.disabled = false;
        btnFed.innerHTML = `${SVG_ICONS.zap} Khởi Động Vòng FedAvg`;
      }
      if (btnStopCent) btnStopCent.style.display = "none";
      if (btnStopFed) btnStopFed.style.display = "none";

      if (currentTask.status === "completed") {
        if (centBarFill && isCent) centBarFill.style.width = "100%";
        if (centBarLabel && isCent) centBarLabel.innerText = "Tiến trình: 100% (Hoàn thành)";
        if (centBarDetail && isCent) centBarDetail.innerText = currentTask.message || "Huấn luyện Centralized hoàn tất!";
        if (fedBarFill && isFed) fedBarFill.style.width = "100%";
        if (fedBarLabel && isFed) fedBarLabel.innerText = "Tiến trình: 100% (Hoàn thành)";
        if (fedBarDetail && isFed) fedBarDetail.innerText = currentTask.message || "Huấn luyện FedAvg hoàn tất!";
        if (currentTask.type === "federated-network") {
          const badge = document.getElementById("fed-live-badge");
          if (badge) { badge.className = "status-badge success"; badge.textContent = "Flower smoke · hoàn tất"; }
        }
      } else if (currentTask.status === "cancelled") {
        if (centBarLabel && isCent) centBarLabel.innerText = "Tiến trình: Đã dừng theo yêu cầu";
        if (centBarDetail && isCent) centBarDetail.innerText = currentTask.message || "Đã dừng.";
        if (fedBarLabel && isFed) fedBarLabel.innerText = "Tiến trình: Đã dừng theo yêu cầu";
        if (fedBarDetail && isFed) fedBarDetail.innerText = currentTask.message || "Đã dừng.";
      }
    }

    // Tự động làm mới khi tác vụ hoàn tất hoặc bị hủy
    if (previousTaskStatus === "running" && (currentTask.status === "completed" || currentTask.status === "cancelled")) {
      console.log("[Status] Tác vụ hoàn tất/hủy, nạp lại toàn bộ metrics và telemetry...");
      const wasCent = isCent;
      const wasFed = isFed;
      await Promise.all([
        loadPublishedDemo(),
        loadNetworkMetrics(),
      ]);
    }
    previousTaskStatus = currentTask.status || "idle";

  } catch (err) {
    if (statusIndicator) {
      statusIndicator.innerText = "Server: Mất kết nối (Offline)";
      statusIndicator.parentElement?.classList.add("offline");
    }
    if (taskBadge) {
      taskBadge.innerText = "Mất kết nối API";
      taskBadge.className = "status-badge error";
    }
  }
}

async function loadClientsData() {
  try {
    const res = await fetch("/api/clients");
    if (!res.ok) return;
    const data = await res.json();
    const container = document.getElementById("clients-container");
    if (!container) return;

    if (data.clients && data.clients.length > 0) {
      const dashClients = document.getElementById("dash-clients");
      if (dashClients) dashClients.innerText = `${data.clients.length} / ${data.total_clients || data.clients.length}`;

      const validRtts = data.clients.filter(c => typeof c.network_rtt_ms === "number" && c.network_rtt_ms > 0);
      const avgRtt = validRtts.length > 0
        ? (validRtts.reduce((sum, c) => sum + c.network_rtt_ms, 0) / validRtts.length).toFixed(1) + " ms"
        : "N/A (Mô phỏng)";
      const dashRtt = document.getElementById("dash-rtt");
      if (dashRtt) dashRtt.innerText = avgRtt;

      // Cập nhật biểu đồ phân bổ nếu có
      if (distributionChart) {
        distributionChart.data.labels = data.clients.map(c => `Node ${c.client_id}`);
        distributionChart.data.datasets[0].data = data.clients.map(c => c.samples);
        distributionChart.update();
      }
    } else {
      container.innerHTML = `<div class="empty-state p-4 text-center text-muted">Chưa có dữ liệu clients. Hãy chạy FedAvg để thu thập telemetry.</div>`;
      return;
    }

    container.innerHTML = "";
    data.clients.forEach(c => {
      const card = document.createElement("div");
      card.className = "client-card";

      const trainTimeDisplay = c.local_train_time_s !== null && c.local_train_time_s !== undefined
        ? `${c.local_train_time_s} s`
        : `<span class="text-muted">Chờ chạy vòng FedAvg</span>`;

      const normDisplay = c.update_norm !== null && c.update_norm !== undefined
        ? c.update_norm
        : `<span class="text-muted">Chờ tính gradient</span>`;

      card.innerHTML = `
        <div class="client-card-header">
          <div>
            <div class="client-card-title">${c.client_name}</div>
            <div style="margin-top: 6px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
              <span class="badge-role">
                ${c.role_badge}
              </span>
              <span class="badge-contrib">
                Tỷ trọng: ${c.total_dataset_pct}
              </span>
            </div>
          </div>
          <span class="status-badge connected" style="padding: 3px 8px; font-size: 11px; white-space: nowrap; display: inline-flex; align-items: center; gap: 5px;">
            <span class="live-dot"></span> ${c.status}
          </span>
        </div>

        <div class="crop-focus-box">
          <div class="crop-focus-title">${SVG_ICONS.layers} ${c.crop_focus}</div>
          <div class="crop-focus-sub">Phạm vi: ${c.classes_present} / 38 lớp thực vật</div>
        </div>

        <div class="client-stat-row">
          <span style="display: flex; align-items: center; gap: 6px;">${SVG_ICONS.server} Dữ liệu cục bộ:</span>
          <span class="client-stat-val text-sky" style="font-weight: 600;">${Number(c.samples).toLocaleString()} ảnh</span>
        </div>
        <div class="client-stat-row">
          <span style="display: flex; align-items: center; gap: 6px;">${SVG_ICONS.clock} Thời gian huấn luyện:</span>
          <span class="client-stat-val">${trainTimeDisplay}</span>
        </div>
        <div class="client-stat-row">
          <span style="display: flex; align-items: center; gap: 6px;">${SVG_ICONS.activity} Cập nhật trọng số (Δw):</span>
          <span class="client-stat-val text-emerald" style="font-family: var(--font-mono); font-weight: 600;">${normDisplay}</span>
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
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById("compare-body");
    if (!tbody || !data.rows) return;

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
    console.warn("Lỗi tải bảng đối sánh:", err);
  }
}

async function loadLayerList() {
  try {
    const res = await fetch("/api/weights/inspect");
    if (!res.ok) return;
    const data = await res.json();
    const selector = document.getElementById("layer-selector");
    if (!selector) return;

    if (!data.available) {
      selector.innerHTML = `<option value="">— ${data.message} —</option>`;
      const tbody = document.getElementById("layer-stats-body");
      if (tbody) tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">${data.message}</td></tr>`;
      return;
    }

    selector.innerHTML = "";
    data.available_layers.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l;
      opt.innerText = l;
      selector.appendChild(opt);
    });

    renderLayerStats(data.selected_layer_stats, data.has_reference);
  } catch (err) {
    console.warn("Lỗi load layer list:", err);
  }
}

async function loadLayerStats() {
  const selector = document.getElementById("layer-selector");
  if (!selector || !selector.value) return;
  const layer = selector.value;

  try {
    const res = await fetch(`/api/weights/inspect?layer=${encodeURIComponent(layer)}`);
    if (!res.ok) return;
    const data = await res.json();
    renderLayerStats(data.selected_layer_stats, data.has_reference);
  } catch (err) {
    console.warn("Lỗi load layer stats:", err);
  }
}

function renderLayerStats(stats, hasReference) {
  const tbody = document.getElementById("layer-stats-body");
  if (!tbody) return;

  if (!stats) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Chưa có dữ liệu trọng số layer.</td></tr>`;
    return;
  }

  const updateNormDisplay = stats.update_norm !== null && stats.update_norm !== undefined
    ? stats.update_norm
    : `<span class="text-muted" title="Cần có checkpoint vòng trước để đo khoảng cách delta">— (Chưa có reference)</span>`;

  tbody.innerHTML = `
    <tr>
      <td style="color: #fff; font-weight: 600;">${stats.layer_name}</td>
      <td>${JSON.stringify(stats.shape)}</td>
      <td>${Number(stats.param_count).toLocaleString()}</td>
      <td class="text-sky">${stats.l2_norm}</td>
      <td class="text-emerald">${updateNormDisplay}</td>
      <td>${stats.mean} &plusmn; ${stats.std}</td>
      <td style="color: #9ca3af; font-size: 11px;">[${stats.sample_values ? stats.sample_values.join(", ") : ""}]</td>
    </tr>
  `;
}

async function loadCentralizedData(sourceOverride = "canonical") {
  try {
    let res = await fetch(`/api/metrics/centralized?source=${sourceOverride}`);
    if (!res.ok && sourceOverride !== "canonical") {
      res = await fetch("/api/metrics/centralized?source=canonical");
    }
    if (!res.ok) return;
    const data = await res.json();

    if (data.test_metrics) {
      const elAcc = document.getElementById("cent-acc");
      const elF1 = document.getElementById("cent-f1");
      const elLoss = document.getElementById("cent-loss");
      const isVerified = data.scientific_valid === true;

      if (!isVerified && data.source === "canonical") {
        if (elAcc) elAcc.innerText = "UNVERIFIED";
        if (elF1) elF1.innerText = "UNVERIFIED";
        if (elLoss) elLoss.innerText = "UNVERIFIED";
        const badge = document.getElementById("cent-live-badge");
        if (badge) {
          badge.className = "status-badge error";
          badge.innerText = "Historical / UNVERIFIED";
          badge.title = data.scientific_reason || "Canonical provenance chưa xác minh";
        }
      } else {
        const acc = data.test_metrics.accuracy !== undefined ? (data.test_metrics.accuracy * 100).toFixed(2) + "%" : "—";
        const f1 = data.test_metrics.macro_f1 !== undefined ? Number(data.test_metrics.macro_f1).toFixed(4) : "—";
        const loss = data.test_metrics.loss !== undefined ? Number(data.test_metrics.loss).toFixed(4) : "—";
        if (elAcc) elAcc.innerText = acc;
        if (elF1) elF1.innerText = f1;
        if (elLoss) elLoss.innerText = loss;
      }
    }

    if (centralizedChart && data.history && data.history.length > 0) {
      centralizedChart.data.labels = data.history.map(h => `Epoch ${h.epoch}`);
      centralizedChart.data.datasets[0].data = data.history.map(h => Number(((h.train_accuracy || 0) * 100).toFixed(1)));
      centralizedChart.data.datasets[1].data = data.history.map(h => Number(((h.val_accuracy || 0) * 100).toFixed(1)));
      centralizedChart.update();
    }

    if (data.history && data.history.length > 0) {
      renderCentralizedProgress(data.history, false, null);
    }
  } catch (err) {
    console.warn("Lỗi load centralized metrics:", err);
  }
}

function renderCentralizedProgress(history, isLive, currentTask) {
  const tbody = document.getElementById("cent-progress-body");
  const badge = document.getElementById("cent-live-badge");
  if (!tbody) return;

  if (badge) {
    if (isLive) {
      const cur = history ? history.length : 0;
      const tot = (currentTask && currentTask.live_progress && currentTask.live_progress[0]?.total_epochs) || "?";
      badge.className = "status-badge running";
      badge.innerHTML = `<span class="live-dot pulse"></span> Đang huấn luyện (${cur}/${tot} Epochs)`;
    } else if (currentTask && currentTask.status === "cancelled") {
      badge.className = "status-badge error";
      badge.innerHTML = `<span class="live-dot" style="background:#f43f5e;"></span> Đã dừng`;
    } else if (history && history.length > 0) {
      badge.className = "status-badge success";
      badge.innerHTML = `${SVG_ICONS.check} Hoàn tất`;
    } else {
      badge.className = "status-badge";
      badge.innerText = "Sẵn sàng";
    }
  }

  if (!history || history.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Chưa có tiến trình huấn luyện. Bấm "Bắt Đầu Huấn Luyện" để theo dõi.</td></tr>`;
    return;
  }

  tbody.innerHTML = "";
  history.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const trainAcc = formatAcc(item.train_accuracy);
    const trainLoss = formatLoss(item.train_loss);
    const valAcc = formatAcc(item.val_accuracy);
    const valLoss = formatLoss(item.val_loss);
    const f1 = item.val_macro_f1 !== undefined ? formatF1(item.val_macro_f1) : "—";
    const dur = item.epoch_time_s !== undefined ? Number(item.epoch_time_s).toFixed(1) + "s" : "—";

    const isLast = idx === history.length - 1;
    const isCompleted = item.status === "completed" || !isLive || !isLast;
    const isCancelled = isLast && currentTask && currentTask.status === "cancelled";

    let statusText = "Đạt";
    let statusCls = "text-emerald";
    if (isCancelled) {
      statusText = "Đã dừng";
      statusCls = "text-rose";
    } else if (!isCompleted) {
      statusText = "Đang chạy";
      statusCls = "text-amber";
    }

    tr.innerHTML = `
      <td style="font-weight: 600; color: #fff;">Epoch ${item.epoch}</td>
      <td class="text-sky">${trainAcc}</td>
      <td>${trainLoss}</td>
      <td class="text-emerald" style="font-weight: 600;">${valAcc}</td>
      <td>${valLoss}</td>
      <td>${f1}</td>
      <td style="color: #9ca3af;">${dur}</td>
      <td class="${statusCls}">● ${statusText}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadFederatedData(sourceOverride = "canonical") {
  try {
    let res = await fetch(`/api/metrics/federated?source=${sourceOverride}`);
    if (!res.ok && sourceOverride !== "canonical") {
      res = await fetch("/api/metrics/federated?source=canonical");
    }
    if (!res.ok) return;
    const data = await res.json();

    if (data.source === "canonical" && data.scientific_valid !== true) {
      const elDashAcc = document.getElementById("dash-acc");
      const elDashLoss = document.getElementById("dash-loss");
      if (elDashAcc) elDashAcc.innerText = "UNVERIFIED";
      if (elDashLoss) elDashLoss.innerText = "UNVERIFIED";
      const badge = document.getElementById("fed-live-badge");
      if (badge) {
        badge.className = "status-badge error";
        badge.innerText = "Historical / UNVERIFIED";
        badge.title = data.scientific_reason || "Canonical provenance chưa xác minh";
      }
      return;
    }

    if (data.best_accuracy !== undefined) {
      const elDashAcc = document.getElementById("dash-acc");
      if (elDashAcc) {
        const b = data.best_accuracy;
        const bPct = b <= 1.0 && b > 0 ? (b * 100).toFixed(2) : Number(b).toFixed(2);
        elDashAcc.innerText = bPct + "%";
      }
    }

    if (data.history && data.history.length > 0) {
      const rounds = data.history.map(h => `R${h.round}`);
      const accs = data.history.map(h => {
        const raw = h.global_accuracy !== undefined ? h.global_accuracy : (h.accuracy || 0);
        return raw <= 1.0 && raw > 0 ? Number((raw * 100).toFixed(2)) : Number(Number(raw).toFixed(2));
      });
      const losses = data.history.map(h => Number((h.global_loss || 0).toFixed(3)));

      if (federatedChart) {
        federatedChart.data.labels = rounds;
        federatedChart.data.datasets[0].data = accs;
        federatedChart.update();
      }
      if (convergenceChart) {
        convergenceChart.data.labels = rounds;
        convergenceChart.data.datasets[0].data = accs;
        convergenceChart.data.datasets[1].data = losses;
        convergenceChart.update();
      }
      const elDashLoss = document.getElementById("dash-loss");
      if (elDashLoss) {
        const testLoss = data.test_metrics?.loss ?? data.best_val_loss ?? data.history[data.history.length - 1]?.global_loss;
        if (testLoss !== undefined) {
          elDashLoss.innerText = Number(testLoss).toFixed(3);
        }
      }

      renderFederatedProgress(data.history, false, null);
    }
  } catch (err) {
    console.warn("Lỗi load federated metrics:", err);
  }
}

function renderFederatedProgress(history, isLive, currentTask) {
  const tbody = document.getElementById("fed-progress-body");
  const badge = document.getElementById("fed-live-badge");
  if (!tbody) return;

  if (badge) {
    if (isLive) {
      const cur = history ? history.length : 0;
      const tot = (currentTask && currentTask.live_progress && currentTask.live_progress[0]?.total_rounds) || "?";
      badge.className = "status-badge running";
      badge.innerHTML = `<span class="live-dot pulse"></span> Đang giao tiếp (Vòng ${cur}/${tot})`;
    } else if (currentTask && currentTask.status === "cancelled") {
      badge.className = "status-badge error";
      badge.innerHTML = `<span class="live-dot" style="background:#f43f5e;"></span> Đã dừng`;
    } else if (history && history.length > 0) {
      badge.className = "status-badge success";
      badge.innerHTML = `${SVG_ICONS.check} Hoàn tất`;
    } else {
      badge.className = "status-badge";
      badge.innerText = "Sẵn sàng";
    }
  }

  if (!history || history.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Chưa có phiên đang theo dõi. Bấm "Khởi Động Vòng FedAvg" để chạy smoke; audit của phiên hoàn tất nằm ở bảng Flower bên dưới.</td></tr>`;
    return;
  }

  tbody.innerHTML = "";
  history.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const acc = formatAcc(item.global_accuracy);
    const loss = formatLoss(item.global_loss);
    const f1 = item.global_macro_f1 !== undefined ? formatF1(item.global_macro_f1) : "—";
    const agg = item.aggregation_time_ms !== undefined ? Number(item.aggregation_time_ms).toFixed(1) + " ms" : "—";
    const dur = item.round_duration_s !== undefined ? Number(item.round_duration_s).toFixed(1) + "s" : "—";

    const isLast = idx === history.length - 1;
    const isCompleted = item.status === "completed" || !isLive || !isLast;
    const isCancelled = isLast && currentTask && currentTask.status === "cancelled";

    let statusText = "Đã tổng hợp";
    let statusCls = "text-emerald";
    if (isCancelled) {
      statusText = "Đã dừng";
      statusCls = "text-rose";
    } else if (!isCompleted) {
      statusText = "Đang tổng hợp";
      statusCls = "text-amber";
    }

    tr.innerHTML = `
      <td style="font-weight: 600; color: #fff;">Vòng ${item.round}</td>
      <td class="text-emerald" style="font-weight: 600;">${acc}</td>
      <td class="text-sky">${loss}</td>
      <td>${f1}</td>
      <td style="color: #c084fc;">${agg}</td>
      <td style="color: #9ca3af;">${dur}</td>
      <td class="${statusCls}">● ${statusText}</td>
    `;
    tbody.appendChild(tr);
  });
}

function formatAcc(val) {
  if (val === undefined || val === null) return "—";
  const num = Number(val);
  if (isNaN(num)) return "—";
  const pct = num <= 1.0 && num > 0 ? num * 100 : num;
  return pct.toFixed(2) + "%";
}

function formatLoss(val) {
  if (val === undefined || val === null) return "—";
  const num = Number(val);
  if (isNaN(num)) return "—";
  return num.toFixed(4);
}

function formatF1(val) {
  if (val === undefined || val === null) return "—";
  const num = Number(val);
  if (isNaN(num)) return "—";
  return num.toFixed(4);
}

// 4. KÍCH HOẠT HUẤN LUYỆN VỚI VALIDATION VÀ TRUYỀN THAM SỐ ĐẦY ĐỦ
function setCentralizedMode(isDemo) {
  if (!isDemo) return;
  const cardDemo = document.getElementById("mode-cent-demo");
  const cardFull = document.getElementById("mode-cent-full");
  const inputVal = document.getElementById("cent-smoke-val");

  if (cardDemo && cardFull && inputVal) {
    if (isDemo) {
      cardDemo.className = "mode-card active sky";
      cardFull.className = "mode-card";
      inputVal.value = "true";
    } else {
      cardDemo.className = "mode-card";
      cardFull.className = "mode-card active sky";
      inputVal.value = "false";
    }
    loadPublishedDemo();
  }
}

function toggleFederatedEngineFields() {
  const engine = document.getElementById("fl-engine")?.value || "simulation";
  const networkFields = document.getElementById("fl-network-fields");
  const scaleBlock = document.getElementById("fl-scale-mode-block");
  const help = document.getElementById("fl-engine-help");
  const alpha = document.getElementById("fl-alpha");
  const rounds = document.getElementById("fl-rounds");
  const epochs = document.getElementById("fl-epochs");

  if (engine === "network") {
    if (networkFields) networkFields.style.display = "block";
    if (scaleBlock) scaleBlock.style.display = "none";
    if (help) help.innerText = "Demo kỹ thuật: một server và năm tiến trình client localhost, ảnh tổng hợp, cố định hai vòng và một local epoch. RTT đo trên cùng máy.";
    if (alpha) alpha.disabled = true;
    if (rounds) { rounds.value = "2"; rounds.max = "2"; rounds.min = "2"; rounds.disabled = true; }
    if (epochs) { epochs.value = "1"; epochs.max = "1"; epochs.min = "1"; epochs.disabled = true; }
  } else {
    if (networkFields) networkFields.style.display = "none";
    if (scaleBlock) scaleBlock.style.display = "block";
    if (help) help.innerText = "Simulation chạy các client trong cùng máy/process workflow; không được dùng để chứng minh latency mạng thật.";
    if (alpha) alpha.disabled = false;
    if (rounds) { rounds.max = "50"; rounds.min = "1"; rounds.disabled = false; }
    if (epochs) { epochs.max = "10"; epochs.min = "1"; epochs.disabled = false; }
  }
}

async function loadNetworkMetrics() {
  const tbody = document.getElementById("network-metrics-body");
  const clientBody = document.getElementById("network-client-rtt-body");
  const runContext = document.getElementById("network-run-context");
  if (!tbody) return;
  try {
    const res = await fetch("/api/federated/network-metrics");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không đọc được network metrics");
    const finalSummary = document.getElementById("network-final-test-summary");
    if (!data.available || !Array.isArray(data.rounds) || data.rounds.length === 0) {
      if (runContext) runContext.innerText = "Chưa có phiên Flower smoke hoàn tất.";
      tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">Chưa có dữ liệu Flower network thật.</td></tr>`;
      if (clientBody) clientBody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">Chưa có audit per-client.</td></tr>`;
      if (finalSummary) finalSummary.innerText = "Final held-out test: chưa có evaluator.";
      return;
    }
    if (runContext) runContext.innerText = data.historical_run
      ? `Lịch sử smoke đã kiểm: ${String(data.run_dir || "").split(/[\\/]/).pop()} · 2 vòng, 5 client; không phải phiên đang chạy.`
      : "Phiên Flower hiện tại · dữ liệu tổng hợp, không dùng làm kết quả khoa học.";
    tbody.innerHTML = data.rounds.map(r => `
      <tr>
        <td>${r.round}</td>
        <td>${r.accepted_clients ?? "—"}</td>
        <td>${r.suspicious_clients ?? "—"}</td>
        <td>${r.rejected_clients ?? "—"}</td>
        <td>${r.mean_received_update_norm !== undefined ? Number(r.mean_received_update_norm).toFixed(4) : "—"}</td>
        <td>${r.server_val_accuracy !== undefined ? (Number(r.server_val_accuracy) * 100).toFixed(2) + "%" : "—"}</td>
        <td>${r.server_val_loss !== undefined ? Number(r.server_val_loss).toFixed(4) : "—"}</td>
        <td>${r.server_val_macro_f1 !== undefined ? Number(r.server_val_macro_f1).toFixed(4) : "—"}</td>
        <td>${r.mean_probe_rtt_ms !== undefined && Number(r.mean_probe_rtt_ms) >= 0 ? Number(r.mean_probe_rtt_ms).toFixed(2) + " ms" : "—"}</td>
        <td>${r.max_probe_rtt_ms !== undefined && Number(r.max_probe_rtt_ms) >= 0 ? Number(r.max_probe_rtt_ms).toFixed(2) + " ms" : "—"}</td>
        <td>${r.round_fit_elapsed_ms !== undefined ? Number(r.round_fit_elapsed_ms).toFixed(1) + " ms" : "—"}</td>
        <td>${r.estimated_comm_overhead_ms !== undefined ? Number(r.estimated_comm_overhead_ms).toFixed(1) + " ms" : "—"}</td>
      </tr>
    `).join("");

    if (finalSummary) {
      const tm = data.final_test_metrics;
      finalSummary.innerText = tm
        ? `Final held-out test — Accuracy: ${(Number(tm.accuracy) * 100).toFixed(2)}% | Loss: ${Number(tm.loss).toFixed(4)} | Macro-F1: ${Number(tm.macro_f1).toFixed(4)}`
        : "Final held-out test: chưa có evaluator hoặc chưa cấu hình evaluation dataset root.";
    }

    if (clientBody) {
      const clientRows = (data.audit_rounds || []).flatMap(round =>
        (round.clients || []).map(client => ({ round: round.round, ...client }))
      );
      clientBody.innerHTML = clientRows.length ? clientRows.map(row => `
        <tr>
          <td>${row.round}</td>
          <td>Client ${row.client_id}</td>
          <td>${row.probe_rtt_ms !== null && row.probe_rtt_ms !== undefined ? Number(row.probe_rtt_ms).toFixed(2) + " ms" : "—"}</td>
          <td>${row.local_train_time_s !== undefined ? Number(row.local_train_time_s).toFixed(3) + " s" : "—"}</td>
          <td>${row.data_mode || "—"}</td>
          <td>${row.training_seed !== undefined ? row.training_seed : "—"}</td>
          <td title="${row.partition_config_sha256 || ""}">${row.partition_id || "—"}</td>
          <td>${row.received_update_norm !== undefined ? Number(row.received_update_norm).toFixed(4) : "—"}</td>
          <td>${row.status || "—"}</td>
          <td>${row.validation_reason || "—"}</td>
        </tr>
      `).join("") : `<tr><td colspan="10" class="text-center text-muted">Chưa có audit per-client.</td></tr>`;
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">Lỗi đọc network metrics: ${err.message}</td></tr>`;
    if (clientBody) clientBody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">${err.message}</td></tr>`;
    const finalSummary = document.getElementById("network-final-test-summary");
    if (finalSummary) finalSummary.innerText = `Final held-out test: lỗi đọc metrics (${err.message})`;
  }
}

function setFederatedMode(isDemo) {
  if (!isDemo) return;
  const cardDemo = document.getElementById("mode-fed-demo");
  const cardFull = document.getElementById("mode-fed-full");
  const inputVal = document.getElementById("fl-smoke-val");

  if (cardDemo && cardFull && inputVal) {
    if (isDemo) {
      cardDemo.className = "mode-card active";
      cardFull.className = "mode-card";
      inputVal.value = "true";
    } else {
      cardDemo.className = "mode-card";
      cardFull.className = "mode-card active";
      inputVal.value = "false";
    }
    loadPublishedDemo();
  }
}

async function triggerStopTrain() {
  const btnStopCent = document.getElementById("btn-stop-centralized");
  const btnStopFed = document.getElementById("btn-stop-federated");
  if (btnStopCent) { btnStopCent.disabled = true; btnStopCent.innerHTML = `${SVG_ICONS.spinner} Đang Dừng...`; }
  if (btnStopFed) { btnStopFed.disabled = true; btnStopFed.innerHTML = `${SVG_ICONS.spinner} Đang Dừng...`; }

  try {
    const res = await fetch("/api/train/stop", { method: "POST" });
    const data = await res.json();
    const taskBadge = document.getElementById("task-badge");
    if (taskBadge) {
      taskBadge.innerText = data.message || "Đang gửi lệnh dừng...";
      taskBadge.className = "status-badge running";
    }
    await loadSystemStatus();
  } catch (err) {
    alert("Lỗi dừng huấn luyện: " + err.message);
  }
}

async function triggerCentralizedTrain() {
  const btn = document.getElementById("btn-train-centralized");
  const btnStop = document.getElementById("btn-stop-centralized");
  const epochs = document.getElementById("cent-epochs")?.value || 5;
  const seed = document.getElementById("cent-seed")?.value || 42;
  const isSmoke = document.getElementById("cent-smoke-val") ? document.getElementById("cent-smoke-val").value === "true" : true;

  btn.disabled = true;
  btn.innerHTML = `${SVG_ICONS.spinner} Đang Huấn Luyện...`;
  if (btnStop) {
    btnStop.style.display = "inline-flex";
    btnStop.disabled = false;
    btnStop.innerHTML = `${SVG_ICONS.stop} Dừng Huấn Luyện`;
  }

  const barBox = document.getElementById("cent-progress-bar-box");
  const barFill = document.getElementById("cent-progress-fill");
  const barLabel = document.getElementById("cent-progress-label");
  const barDetail = document.getElementById("cent-progress-detail");
  if (barBox) barBox.style.display = "block";
  if (barFill) barFill.style.width = "5%";
  if (barLabel) barLabel.innerText = "Tiến trình: 5%";
  if (barDetail) barDetail.innerText = "Đang khởi tạo mô hình...";

  const tbody = document.getElementById("cent-progress-body");
  if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Đang khởi tạo kiến trúc mạng và phân bổ batch...</td></tr>`;
  const badge = document.getElementById("cent-live-badge");
  if (badge) { badge.className = "status-badge running"; badge.innerHTML = `<span class="live-dot pulse"></span> Đang khởi tạo...`; }

  try {
    const res = await fetch("/api/centralized/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: parseInt(epochs, 10),
        seed: parseInt(seed, 10),
        smoke: isSmoke,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      const errorMsg = data.detail || data.message || "Yêu cầu huấn luyện thất bại";
      alert("Lỗi: " + errorMsg);
      btn.disabled = false;
      btn.innerHTML = `${SVG_ICONS.play} Bắt Đầu Huấn Luyện`;
      if (btnStop) btnStop.style.display = "none";
      if (barBox) barBox.style.display = "none";
      return;
    }

    document.getElementById("task-badge").innerText = data.message;
    await loadSystemStatus();
  } catch (err) {
    alert("Không thể kết nối máy chủ: " + err.message);
    btn.disabled = false;
    btn.innerHTML = `${SVG_ICONS.play} Bắt Đầu Huấn Luyện`;
    if (btnStop) btnStop.style.display = "none";
    if (barBox) barBox.style.display = "none";
  }
}

async function triggerFederatedTrain() {
  const btn = document.getElementById("btn-train-federated");
  const btnStop = document.getElementById("btn-stop-federated");
  const rounds = document.getElementById("fl-rounds")?.value || 5;
  const epochs = document.getElementById("fl-epochs")?.value || 1;
  const seed = document.getElementById("fl-seed")?.value || 42;
  const alpha = document.getElementById("fl-alpha")?.value || 1.0;
  const isSmoke = document.getElementById("fl-smoke-val") ? document.getElementById("fl-smoke-val").value === "true" : true;
  const engine = document.getElementById("fl-engine")?.value || "simulation";
  const serverAddress = document.getElementById("fl-server-address")?.value || "127.0.0.1:8089";
  const numClients = document.getElementById("fl-num-clients")?.value || 5;
  const requireRealClients = document.getElementById("fl-require-real-clients")?.checked || false;
  const evalDatasetRoot = document.getElementById("fl-eval-dataset-root")?.value?.trim() || null;
  const requireRealEvaluation = document.getElementById("fl-require-real-evaluation")?.checked || false;
  const validatorPolicy = document.getElementById("fl-validator-policy")?.value || "observe";
  const maxUpdateNormRaw = document.getElementById("fl-max-update-norm")?.value?.trim() || "";
  const maxUpdateNorm = maxUpdateNormRaw === "" ? null : parseFloat(maxUpdateNormRaw);

  btn.disabled = true;
  btn.innerHTML = `${SVG_ICONS.spinner} Đang Điều Phối FedAvg...`;
  if (btnStop) {
    btnStop.style.display = "inline-flex";
    btnStop.disabled = false;
    btnStop.innerHTML = `${SVG_ICONS.stop} Dừng Huấn Luyện`;
  }

  const barBox = document.getElementById("fed-progress-bar-box");
  const barFill = document.getElementById("fed-progress-fill");
  const barLabel = document.getElementById("fed-progress-label");
  const barDetail = document.getElementById("fed-progress-detail");
  if (barBox) barBox.style.display = "block";
  if (barFill) barFill.style.width = "5%";
  if (barLabel) barLabel.innerText = "Tiến trình: 5%";
  if (barDetail) barDetail.innerText = "Đang khởi tạo 5 trạm client...";

  const tbody = document.getElementById("fed-progress-body");
  if (tbody) tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Đang phân phối trọng số khởi tạo toàn cục tới 5 client...</td></tr>`;
  const badge = document.getElementById("fed-live-badge");
  if (badge) { badge.className = "status-badge running"; badge.innerHTML = `<span class="live-dot pulse"></span> Đang khởi tạo...`; }

  try {
    const res = await fetch("/api/federated/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rounds: parseInt(rounds, 10),
        local_epochs: parseInt(epochs, 10),
        seed: parseInt(seed, 10),
        alpha: parseFloat(alpha),
        smoke: isSmoke,
        engine,
        num_clients: parseInt(numClients, 10),
        server_address: serverAddress,
        require_real_clients: requireRealClients,
        dataset_root: evalDatasetRoot,
        require_real_evaluation: requireRealEvaluation,
        reject_suspicious_updates: validatorPolicy === "reject",
        max_update_norm: maxUpdateNorm,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      const errorMsg = data.detail || data.message || "Yêu cầu huấn luyện FedAvg thất bại";
      alert("Lỗi: " + errorMsg);
      btn.disabled = false;
      btn.innerHTML = `${SVG_ICONS.zap} Khởi Động Vòng FedAvg`;
      if (btnStop) btnStop.style.display = "none";
      if (barBox) barBox.style.display = "none";
      return;
    }

    document.getElementById("task-badge").innerText = data.message;
    await loadSystemStatus();
  } catch (err) {
    alert("Không thể kết nối máy chủ: " + err.message);
    btn.disabled = false;
    btn.innerHTML = `${SVG_ICONS.zap} Khởi Động Vòng FedAvg`;
    if (btnStop) btnStop.style.display = "none";
    if (barBox) barBox.style.display = "none";
  }
}

// 5. INFERENCE XỬ LÝ ẢNH & DUAL MODEL ĐỐI CHIẾU
function handleImageSelected(input) {
  if (input.files && input.files[0]) {
    selectedFile = input.files[0];
    const pvResult = document.getElementById("plantvillage-result");
    if (pvResult) pvResult.textContent = "";
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = document.getElementById("preview-image");
      img.src = e.target.result;
      img.classList.remove("hidden");
      document.querySelector(".upload-placeholder")?.classList.add("hidden");
      document.getElementById("btn-diagnose").disabled = false;
      document.getElementById("btn-plantvillage").disabled = false;
    };
    reader.readAsDataURL(selectedFile);
  }
}

async function loadSampleLeaf(url, label) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const filename = url.split("/").pop();
    selectedFile = new File([blob], filename, { type: blob.type || "image/jpeg" });
    const pvResult = document.getElementById("plantvillage-result");
    if (pvResult) pvResult.textContent = "";

    const img = document.getElementById("preview-image");
    img.src = url;
    img.classList.remove("hidden");
    document.querySelector(".upload-placeholder")?.classList.add("hidden");
    document.getElementById("btn-diagnose").disabled = false;
      document.getElementById("btn-plantvillage").disabled = false;

    // Tự động kích hoạt đối chiếu song song
    await submitInference();
  } catch (err) {
    console.error("Lỗi tải ảnh mẫu:", err);
    alert("Không thể tải ảnh mẫu: " + err.message);
  }
}

async function submitInference() {
  if (!selectedFile) return;

  const btn = document.getElementById("btn-diagnose");
  btn.disabled = true;
  btn.innerHTML = `${SVG_ICONS.spinner} Đang Phân Tích Song Song...`;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const fedSource = document.getElementById("inference-fed-source")?.value || "fedavg";
    const res = await fetch(`/api/inference?federated_source=${encodeURIComponent(fedSource)}`, {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    if (!res.ok) {
      throw new Error(result.detail || result.message || "Lỗi suy luận");
    }

    document.getElementById("dual-results").style.display = "grid";
    const selectedBadge = document.getElementById("selected-model-badge");
    if (selectedBadge) selectedBadge.textContent = result.selected_model_name || "Mô hình được chọn";

    // 1. Render Cột Centralized
    const cent = result.centralized;
    if (cent && cent.ready) {
      document.getElementById("cent-disease").innerText = formatDiseaseName(cent.predicted_class);
      document.getElementById("cent-conf-bar").style.width = cent.confidence_pct + "%";
      document.getElementById("cent-conf-val").innerText = cent.confidence_pct + "%";
      document.getElementById("cent-test-acc").innerText = cent.model_validation
        ? (cent.model_validation.plantvillage.top1_accuracy * 100).toFixed(2) + "%"
        : "—";
      renderTop3("cent-top3-list", cent.top3);
    } else {
      document.getElementById("cent-disease").innerText = cent?.error || "Mô hình Centralized chưa sẵn sàng";
      document.getElementById("cent-conf-bar").style.width = "0%";
      document.getElementById("cent-conf-val").innerText = "—";
      document.getElementById("cent-test-acc").innerText = "Chưa huấn luyện";
      renderTop3("cent-top3-list", []);
    }

    // 2. Render Cột Federated
    const fed = result.federated;
    if (fed && fed.ready) {
      document.getElementById("fed-disease").innerText = formatDiseaseName(fed.predicted_class);
      document.getElementById("fed-conf-bar").style.width = fed.confidence_pct + "%";
      document.getElementById("fed-conf-val").innerText = fed.confidence_pct + "%";
      document.getElementById("fed-test-acc").innerText = fed.model_validation
        ? (fed.model_validation.plantvillage.top1_accuracy * 100).toFixed(2) + "%"
        : "—";
      renderTop3("fed-top3-list", fed.top3);
    } else {
      document.getElementById("fed-disease").innerText = fed?.error || "Mô hình FedAvg chưa sẵn sàng";
      document.getElementById("fed-conf-bar").style.width = "0%";
      document.getElementById("fed-conf-val").innerText = "—";
      document.getElementById("fed-test-acc").innerText = "Chưa huấn luyện";
      renderTop3("fed-top3-list", []);
    }

    // 3. Cảnh báo phân vân / dữ liệu ngoài miền (OOD)
    const advisory = document.getElementById("inference-advisory");
    if (advisory) {
      const centUncertain = cent && cent.is_uncertain;
      const fedUncertain = fed && fed.is_uncertain;
      const lowConf = (cent?.confidence_pct < 60.0) || (fed?.confidence_pct < 60.0);
      const evidenceNote = "Checkpoint đã kiểm SHA-256; số validation từ artifact công bố, chưa tái đánh giá tại máy demo. Xác suất softmax của một ảnh không phải độ chính xác trên ảnh ngoài thực địa. Chỉ dùng ảnh vùng lá đã cắt.";
      if (centUncertain || fedUncertain || lowConf) {
        advisory.innerHTML = `
          <div style="display: flex; gap: 10px; align-items: flex-start;">
            ${SVG_ICONS.info}
            <div>
              <b>Cần kiểm tra ảnh và nhãn dự đoán:</b> ${evidenceNote}
            </div>
          </div>
        `;
        advisory.style.display = "block";
      } else {
        advisory.textContent = evidenceNote;
        advisory.style.display = "block";
      }
    }

  } catch (err) {
    alert("Lỗi chẩn đoán: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `${SVG_ICONS.microscope} Chạy Chẩn Đoán Song Song`;
  }
}

function formatDiseaseName(raw) {
  if (!raw) return "—";
  return raw.replace(/___/g, " - ").replace(/_/g, " ");
}

function renderTop3(containerId, top3List) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = "";
  if (!top3List || top3List.length === 0) {
    container.innerHTML = `<li class="text-muted" style="border:none;">Không có dữ liệu top-3</li>`;
    return;
  }

  top3List.forEach(item => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span>${formatDiseaseName(item.class)}</span>
      <b style="color: #fff;">${item.confidence}%</b>
    `;
    container.appendChild(li);
  });
}

// Independent PlantVillage model, not part of the FedAvg scientific comparison.
async function submitPlantVillageReference() {
  if (!selectedFile) return;
  const btn = document.getElementById("btn-plantvillage");
  const output = document.getElementById("plantvillage-result");
  btn.disabled = true;
  output.textContent = "Đang nhận diện bằng PlantVillage...";
  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const response = await fetch("/api/inference/plantvillage", {method: "POST", body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Inference failed");
    const top = data.top3.map((r, index) => `${index + 1}. ${formatDiseaseName(r.class)}: ${r.confidence}%`).join("\n");
    output.textContent = `Dự đoán: ${formatDiseaseName(data.predicted_class)} (${data.confidence_pct}%)\nTop 3:\n${top}\n${data.is_uncertain ? "⚠ Mô hình không chắc chắn.\n" : ""}Lưu ý: Confidence chưa hiệu chuẩn; kết quả có thể sai trên ảnh thực tế.`;
  } catch (error) {
    output.textContent = `Không thể nhận diện: ${error.message}`;
  } finally {
    btn.disabled = false;
  }
}
