// web/static/js/main.js: Điều khiển tương tác giao diện và tích hợp API REST
// Đáp ứng toàn bộ các yêu cầu kiểm thử và chuẩn hóa dữ liệu thực tế.
//
// Quy ước hiệu năng (xem style.css): chỉ ghi DOM khi giá trị đổi, thanh tiến trình đổi bằng transform,
// không alert() — mọi thông báo đi qua window.toast (toast.js).

const SVG_ICONS = {
  play: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>`,
  zap: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  stop: `<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>`,
  spinner: `<svg class="svg-icon icon-sm spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>`,
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

const STATUS_POLL_MS = 4000;

// ───────── Helper giao diện ─────────
const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

// Escape chuỗi do server trả về trước khi ghép vào innerHTML.
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
}

function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function withAlpha(hex, alpha) {
  if (!/^#[0-9a-f]{6}$/i.test(hex)) return hex;
  return hex + Math.round(alpha * 255).toString(16).padStart(2, "0");
}

// Chỉ ghi khi giá trị đổi: poll 4 giây không còn buộc trình duyệt tính lại layout vô ích.
function setText(el, text) {
  const next = String(text);
  if (el && el.textContent !== next) {
    el.textContent = next;
    delete el.dataset.html; // bộ nhớ đệm của setHtml không còn đúng
  }
}

function setHtml(el, html) {
  if (el && el.dataset.html !== html) {
    el.innerHTML = html;
    el.dataset.html = html;
  }
}

function setClass(el, className) {
  if (el && el.className !== className) el.className = className;
}

function setDisplay(el, display) {
  if (el && el.style.display !== display) el.style.display = display;
}

function setDisabled(el, disabled) {
  if (el && el.disabled !== disabled) el.disabled = disabled;
}

// Thanh tiến trình/độ tin cậy đổi bằng transform (không kích hoạt layout).
function setBar(el, pct) {
  if (!el) return;
  const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
  el.style.transform = `scaleX(${clamped / 100})`;
}

// Số KPI đếm lên bằng rAF; chữ không phải số ("UNVERIFIED", "—") hiện luôn.
function setMetric(el, text) {
  if (!el) return;
  el.classList.remove("is-loading");
  const final = String(text);
  if (el.dataset.final === final) return;
  el.dataset.final = final;
  if (el._raf) cancelAnimationFrame(el._raf);
  const match = /^(-?\d+(?:[.,]\d+)?)(.*)$/.exec(final);
  if (!match || prefersReducedMotion()) {
    el.textContent = final;
    return;
  }
  const decimals = (match[1].split(/[.,]/)[1] || "").length;
  const separator = match[1].includes(",") ? "," : ".";
  const target = parseFloat(match[1].replace(",", "."));
  const suffix = match[2];
  const duration = 700;
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * eased).toFixed(decimals).replace(".", separator) + suffix;
    if (t < 1) {
      el._raf = requestAnimationFrame(tick);
    } else {
      el.textContent = final;
      el._raf = 0;
    }
  };
  el._raf = requestAnimationFrame(tick);
}

// Hiện lần lượt các phần tử (opacity + translateY, composited). Bỏ qua khi người dùng chọn giảm chuyển động.
function revealItems(elements, options = {}) {
  const list = Array.from(elements).filter(Boolean);
  if (prefersReducedMotion() || list.length === 0 || typeof list[0].animate !== "function") return;
  const { stagger = 70, offset = 10, duration = 320 } = options;
  list.forEach((el, index) => {
    el.animate(
      [
        { opacity: 0, transform: `translateY(${offset}px)` },
        { opacity: 1, transform: "translateY(0)" },
      ],
      { duration, delay: index * stagger, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)", fill: "backwards" }
    );
  });
}

// Thông điệp lỗi từ FastAPI: detail có thể là chuỗi, mảng lỗi validate hoặc object.
function formatApiError(data, fallback) {
  if (!data) return fallback;
  const detail = data.detail ?? data.message;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map(item => (item && (item.msg || item.message)) || JSON.stringify(item)).join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

function notify(type, title, message, options) {
  if (window.toast && typeof window.toast[type] === "function") {
    return window.toast[type](title, message, options);
  }
  console[type === "error" ? "error" : "log"](`[${type}] ${title}${message ? ": " + message : ""}`);
  return { close() {} };
}

// ───────── Khởi động ─────────
let statusTimer = null;
let serverOnline = null;
let offlineToast = null;
let layerListRequested = false;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initA11y();
  initDropzone();
  initCharts();
  loadSystemStatus();
  loadPublishedDemo();
  loadNetworkMetrics();
  toggleFederatedEngineFields();

  // Poll trạng thái server/tác vụ; tạm dừng khi tab trình duyệt bị ẩn để không tốn CPU/mạng.
  scheduleStatusPoll();
  document.addEventListener("visibilitychange", () => {
    clearTimeout(statusTimer);
    if (!document.hidden) {
      loadSystemStatus().finally(scheduleStatusPoll);
    }
  });
});

function scheduleStatusPoll() {
  clearTimeout(statusTimer);
  if (document.hidden) return;
  statusTimer = setTimeout(async () => {
    await loadSystemStatus();
    scheduleStatusPoll();
  }, STATUS_POLL_MS);
}

// 1. CHUYỂN ĐỔI TAB & BỐ CỤC
function initTabs() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const pageTitle = document.getElementById("page-title");
  const pageDesc = document.getElementById("page-desc");
  const menu = document.querySelector(".nav-menu");

  const titles = {
    "tab-dashboard": ["Tổng Quan Hệ Thống", "Theo dõi tiến trình huấn luyện phân tán và hiệu năng toàn cục"],
    "tab-centralized": ["Centralized Baseline", "Validation của checkpoint pretrained v6 trên release PlantVillage + PlantDoc"],
    "tab-federated": ["Huấn Luyện Federated Learning", "Điều phối các vòng giao tiếp FedAvg và tổng hợp trọng số đa client"],
    "tab-clients": ["Clients & Telemetry", "Năm shard đã công bố và phiên Flower localhost riêng"],
    "tab-experiments": ["So Sánh & Benchmark", "Bảng đối sánh 3 trụ cột Centralized vs FedAvg vs Local-only"],
    "tab-inference": ["Chẩn Đoán Bệnh Lá Cây", "So checkpoint Centralized với FedAvg hoặc từng Local-only trên ảnh vùng lá"],
  };

  // Thanh chỉ báo trượt theo mục đang chọn (transform + height, không đụng layout của mục).
  let indicator = null;
  if (menu) {
    indicator = document.createElement("span");
    indicator.className = "nav-indicator";
    indicator.setAttribute("aria-hidden", "true");
    menu.prepend(indicator);
    menu.classList.add("has-indicator");
  }

  const placeIndicator = (item, animate) => {
    if (!indicator || !item) return;
    if (!animate) indicator.style.transition = "none";
    indicator.style.height = `${item.offsetHeight}px`;
    indicator.style.transform = `translateY(${item.offsetTop}px)`;
    if (!animate) {
      void indicator.offsetWidth;
      indicator.style.transition = "";
    }
  };

  const activeItem = () => document.querySelector(".nav-item.active");
  placeIndicator(activeItem(), false);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => placeIndicator(activeItem(), false));
  }
  window.addEventListener("resize", () => placeIndicator(activeItem(), false));

  navItems.forEach(item => {
    if (item.classList.contains("active")) item.setAttribute("aria-current", "page");
    item.addEventListener("click", () => {
      const targetId = item.getAttribute("data-tab");

      navItems.forEach(n => {
        n.classList.remove("active");
        n.removeAttribute("aria-current");
      });
      tabPanes.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      item.setAttribute("aria-current", "page");
      placeIndicator(item, true);
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      // Bảng Weight Inspector chỉ cần dữ liệu khi người dùng mở tab Clients (nạp một lần, tránh đọc checkpoint lúc khởi động).
      if (targetId === "tab-clients" && !layerListRequested) {
        layerListRequested = true;
        loadLayerList();
      }

      if (titles[targetId]) {
        setText(pageTitle, titles[targetId][0]);
        setText(pageDesc, titles[targetId][1]);
      }
    });
  });
}

// Thẻ chọn chế độ là <div>: cho phép dùng bàn phím (Enter/Space).
function initA11y() {
  document.querySelectorAll('.mode-option-card[role="button"]').forEach(card => {
    card.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        card.click();
      }
    });
  });
}

// Hiệu ứng viền khi kéo ảnh vào vùng tải (input phủ kín nên sự kiện nổi lên vùng cha).
function initDropzone() {
  const zone = document.getElementById("upload-dropzone");
  if (!zone) return;
  const on = () => zone.classList.add("is-dragover");
  const off = () => zone.classList.remove("is-dragover");
  ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, on));
  ["dragleave", "drop", "dragend"].forEach(name => zone.addEventListener(name, off));
}

// 2. KHỞI TẠO BIỂU ĐỒ CHART.JS (CÓ BẢO VỆ KHI THƯ VIỆN KHÔNG NẠP ĐƯỢC)
function initCharts() {
  if (typeof Chart === "undefined") {
    console.warn("[Chart.js] Không tìm thấy thư viện Chart.js (/static/vendor/chart.umd.js). Bỏ qua vẽ biểu đồ.");
    isChartLibAvailable = false;
    document.querySelectorAll(".chart-container").forEach(c => {
      const notice = document.createElement("div");
      notice.className = "chart-fallback-notice";
      notice.textContent = "Biểu đồ không khả dụng: không nạp được thư viện Chart.js.";
      c.appendChild(notice);
    });
    return;
  }

  // Một nơi cấu hình chung, lấy màu/chữ từ token CSS để biểu đồ luôn khớp giao diện.
  const colors = [1, 2, 3, 4, 5].map(i => cssVar(`--chart-${i}`, "#059669"));
  const textColor = cssVar("--text-secondary", "#46515f");
  Chart.defaults.font.family = cssVar("--font-main", "sans-serif");
  Chart.defaults.font.size = 12;
  Chart.defaults.color = textColor;
  Chart.defaults.borderColor = cssVar("--chart-grid", "rgba(17,24,39,0.08)");
  Chart.defaults.animation.duration = prefersReducedMotion() ? 0 : 450;
  Chart.defaults.elements.line.borderWidth = 2.5;
  Chart.defaults.elements.point.radius = 3;
  Chart.defaults.elements.point.hoverRadius = 5;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.tooltip.backgroundColor = "#111827";
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;

  // Biểu đồ chưa có dữ liệu (ví dụ FedAvg v7 không công bố từng vòng): ghi rõ thay vì để lưới trống khó hiểu.
  Chart.register({
    id: "emptyState",
    afterDraw(chart) {
      const hasData = chart.data.datasets.some(ds => Array.isArray(ds.data) && ds.data.length > 0);
      const area = chart.chartArea;
      if (hasData || !area) return;
      const { ctx } = chart;
      const label = "Chưa có dữ liệu để vẽ";
      const cx = (area.left + area.right) / 2;
      const cy = (area.top + area.bottom) / 2;
      ctx.save();
      ctx.font = `500 13px ${Chart.defaults.font.family}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const width = ctx.measureText(label).width + 28;
      ctx.fillStyle = cssVar("--bg-surface", "#ffffff");
      ctx.strokeStyle = cssVar("--border-color", "#dde2e9");
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - width / 2, cy - 17, width, 34, 17);
      else ctx.rect(cx - width / 2, cy - 17, width, 34);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = cssVar("--text-secondary", "#46515f");
      ctx.fillText(label, cx, cy + 1);
      ctx.restore();
    },
  });

  const noGridX = { grid: { display: false } };
  const base = { responsive: true, maintainAspectRatio: false };
  const subtitle = (text) => ({ display: true, text, color: textColor, font: { size: 12, weight: "500" } });

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
            borderColor: colors[0],
            backgroundColor: withAlpha(colors[0], 0.12),
            tension: 0.3,
            fill: true,
          },
          {
            label: "Global Loss",
            data: [],
            borderColor: colors[1],
            tension: 0.3,
            yAxisID: "y1",
          }
        ]
      },
      options: {
        ...base,
        scales: {
          x: {},
          y: { title: { display: true, text: "Accuracy (%)" } },
          y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Loss" } }
        },
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
          backgroundColor: colors,
          borderRadius: 6,
        }]
      },
      options: {
        ...base,
        scales: { x: noGridX, y: {} },
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
          { label: "Train Acc (%)", data: [], borderColor: colors[1], tension: 0.2 },
          { label: "Val Acc (%)", data: [], borderColor: colors[0], tension: 0.2 }
        ]
      },
      options: { ...base, scales: { x: {}, y: {} } }
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
          { label: "Global Test Acc (%)", data: [], borderColor: colors[0], tension: 0.3, fill: true, backgroundColor: withAlpha(colors[0], 0.1) }
        ]
      },
      options: {
        ...base,
        scales: {
          x: {},
          y: { min: 0, max: 100, ticks: { callback: (v) => v + "%" } }
        },
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
          { label: "FedAvg validation PlantVillage (%)", data: [], backgroundColor: colors[0], borderRadius: 6 }
        ]
      },
      options: {
        ...base,
        scales: { x: noGridX, y: { min: 70, max: 100 } },
        plugins: { title: subtitle("Một kịch bản α=0,1; chưa đủ đường suy giảm") }
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
          backgroundColor: [colors[0], colors[1], colors[3], colors[4]],
          borderRadius: 6
        }]
      },
      options: {
        ...base,
        scales: { x: noGridX, y: { min: 60, max: 100 } },
        plugins: {
          legend: { display: false },
          title: subtitle("Chưa có số liệu cùng protocol")
        }
      }
    });
  }
}

// 3. NẠP DỮ LIỆU TỪ REST API
let previousTaskStatus = "idle";

// Cập nhật khối tiến trình (thanh + nhãn + chi tiết) theo tác vụ nền.
function updateProgressBlock(prefix, show, pct, label, detail) {
  const box = document.getElementById(`${prefix}-progress-bar-box`);
  const fill = document.getElementById(`${prefix}-progress-fill`);
  const labelEl = document.getElementById(`${prefix}-progress-label`);
  const detailEl = document.getElementById(`${prefix}-progress-detail`);
  if (show) setDisplay(box, "block");
  if (pct !== null) setBar(fill, pct);
  if (label !== null) setText(labelEl, label);
  if (detail !== null) setText(detailEl, detail);
}

function notifyTaskFinished(task, wasCent, wasFed) {
  const kind = wasCent ? "Centralized" : (wasFed ? "FedAvg" : "Tác vụ");
  if (task.status === "completed") {
    notify("success", `Huấn luyện ${kind} hoàn tất`, task.message || "");
  } else if (task.status === "cancelled") {
    notify("warning", `Đã dừng huấn luyện ${kind}`, task.message || "Tiến trình đã dừng theo yêu cầu.");
  } else if (task.status === "failed") {
    notify("error", `Huấn luyện ${kind} thất bại`, task.message || "Tác vụ nền báo lỗi.");
  }
}

// Chỉ lỗi mạng/HTTP mới được coi là "mất kết nối"; lỗi dựng giao diện chỉ ghi console, không bật toast sai.
async function loadSystemStatus() {
  const statusIndicator = document.getElementById("server-status-text");
  const taskBadge = document.getElementById("task-badge");

  let data;
  try {
    const res = await fetch("/api/status");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    data = await res.json();
  } catch (err) {
    if (statusIndicator) {
      setText(statusIndicator, "Server: Mất kết nối (Offline)");
      statusIndicator.parentElement?.classList.add("offline");
    }
    if (taskBadge) {
      setText(taskBadge, "Mất kết nối API");
      setClass(taskBadge, "status-badge error");
    }
    if (serverOnline !== false) {
      offlineToast = notify("error", "Mất kết nối máy chủ", "Giao diện sẽ tự thử lại sau vài giây.", { detail: err.message, duration: 0 });
    }
    serverOnline = false;
    return;
  }

  if (serverOnline === false) {
    if (offlineToast) offlineToast.close();
    offlineToast = null;
    notify("success", "Đã kết nối lại máy chủ");
  }
  serverOnline = true;

  try {
    await renderSystemStatus(data);
  } catch (err) {
    console.error("[Status] Lỗi dựng trạng thái hệ thống:", err);
  }
}

async function renderSystemStatus(data) {
  const statusIndicator = document.getElementById("server-status-text");
  const taskBadge = document.getElementById("task-badge");
  const deviceText = document.getElementById("system-device");
  const btnCent = document.getElementById("btn-train-centralized");
  const btnFed = document.getElementById("btn-train-federated");

  {
    if (statusIndicator) {
      setText(statusIndicator, `Server: ${data.server_status === "online" ? "Online" : "Offline"}`);
      statusIndicator.parentElement?.classList.remove("offline");
    }
    if (deviceText) {
      const flText = data.flower_server_active ? "Flower: Live" : "Flower: Idle";
      setText(deviceText, `Device: ${data.device} | ${flText}`);
    }

    currentTask = data.task || { status: "idle", message: "Hệ thống sẵn sàng" };
    const network = data.network_processes || {};
    const dashClients = document.getElementById("dash-clients");
    const dashRtt = document.getElementById("dash-rtt");
    if (dashClients) {
      const shown = currentTask.type === "federated-network" && currentTask.status === "completed"
        ? network.accepted_clients_last_round || 0
        : network.connected_clients || 0;
      setText(dashClients, `${shown} / 5`);
    }
    if (dashRtt) {
      setText(dashRtt, Number.isFinite(network.mean_probe_rtt_ms) ? `${Number(network.mean_probe_rtt_ms).toFixed(2)} ms (localhost)` : "—");
    }
    if (taskBadge) {
      setText(taskBadge, currentTask.message || "Hệ thống sẵn sàng");
      let cls = "status-badge";
      if (currentTask.status === "running") cls += " running";
      else if (currentTask.status === "completed") cls += " success";
      else if (currentTask.status === "failed" || currentTask.status === "cancelled") cls += " error";
      setClass(taskBadge, cls);
    }

    const btnStopCent = document.getElementById("btn-stop-centralized");
    const btnStopFed = document.getElementById("btn-stop-federated");

    const isRunning = currentTask.status === "running";
    const isCent = currentTask.type === "centralized" || currentTask.experiment === "Centralized";
    const isFed = currentTask.type === "federated" || currentTask.type === "federated-network" || currentTask.experiment === "Federated";
    const stopLabel = currentTask.cancel_requested ? `${SVG_ICONS.spinner} Đang Dừng...` : `${SVG_ICONS.stop} Dừng Huấn Luyện`;

    // Quản lý trạng thái nút và tiến trình theo trạng thái thật của tác vụ nền
    if (isRunning) {
      if (btnCent) {
        setDisabled(btnCent, true);
        setHtml(btnCent, `${SVG_ICONS.spinner} Đang Huấn Luyện...`);
      }
      if (btnFed) {
        setDisabled(btnFed, true);
        setHtml(btnFed, `${SVG_ICONS.spinner} Đang Điều Phối FedAvg...`);
      }

      const pct = Math.max(5, Math.min(100, currentTask.progress || 5));
      if (isCent) {
        if (btnStopCent) {
          setDisplay(btnStopCent, "inline-flex");
          setDisabled(btnStopCent, currentTask.cancel_requested || false);
          setHtml(btnStopCent, stopLabel);
        }
        setDisplay(btnStopFed, "none");
        updateProgressBlock("cent", true, pct, `Tiến trình: ${pct}%`, currentTask.message || "Đang tính toán epoch...");
        if (currentTask.live_progress) {
          renderCentralizedProgress(currentTask.live_progress, true, currentTask);
        }
      } else if (isFed) {
        if (btnStopFed) {
          setDisplay(btnStopFed, "inline-flex");
          setDisabled(btnStopFed, currentTask.cancel_requested || false);
          setHtml(btnStopFed, stopLabel);
        }
        setDisplay(btnStopCent, "none");
        updateProgressBlock("fed", true, pct, `Tiến trình: ${pct}%`, currentTask.message || "Đang tổng hợp vòng...");
        if (currentTask.live_progress) {
          renderFederatedProgress(currentTask.live_progress, true, currentTask);
        }
      }
    } else {
      if (btnCent && btnCent.disabled) {
        setDisabled(btnCent, false);
        setHtml(btnCent, `${SVG_ICONS.play} Bắt Đầu Huấn Luyện`);
      }
      if (btnFed && btnFed.disabled) {
        setDisabled(btnFed, false);
        setHtml(btnFed, `${SVG_ICONS.zap} Khởi Động Vòng FedAvg`);
      }
      setDisplay(btnStopCent, "none");
      setDisplay(btnStopFed, "none");

      if (currentTask.status === "completed") {
        if (isCent) updateProgressBlock("cent", false, 100, "Tiến trình: 100% (Hoàn thành)", currentTask.message || "Huấn luyện Centralized hoàn tất!");
        if (isFed) updateProgressBlock("fed", false, 100, "Tiến trình: 100% (Hoàn thành)", currentTask.message || "Huấn luyện FedAvg hoàn tất!");
        if (currentTask.type === "federated-network") {
          const badge = document.getElementById("fed-live-badge");
          if (badge) { setClass(badge, "status-badge success"); setText(badge, "Flower smoke · hoàn tất"); }
        }
      } else if (currentTask.status === "cancelled") {
        if (isCent) updateProgressBlock("cent", false, null, "Tiến trình: Đã dừng theo yêu cầu", currentTask.message || "Đã dừng.");
        if (isFed) updateProgressBlock("fed", false, null, "Tiến trình: Đã dừng theo yêu cầu", currentTask.message || "Đã dừng.");
      } else if (currentTask.status === "failed") {
        if (isCent) updateProgressBlock("cent", false, null, "Tiến trình: Thất bại", currentTask.message || "Tác vụ báo lỗi.");
        if (isFed) updateProgressBlock("fed", false, null, "Tiến trình: Thất bại", currentTask.message || "Tác vụ báo lỗi.");
      }
    }

    // Khi tác vụ vừa kết thúc: thông báo và làm mới số liệu
    if (previousTaskStatus === "running" && currentTask.status !== "running") {
      notifyTaskFinished(currentTask, isCent, isFed);
      if (currentTask.status === "completed" || currentTask.status === "cancelled") {
        console.log("[Status] Tác vụ hoàn tất/hủy, nạp lại toàn bộ metrics và telemetry...");
        await Promise.all([
          loadPublishedDemo(),
          loadNetworkMetrics(),
        ]);
      }
    }
    previousTaskStatus = currentTask.status || "idle";
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
      if (dashClients) setText(dashClients, `${data.clients.length} / ${data.total_clients || data.clients.length}`);

      const validRtts = data.clients.filter(c => typeof c.network_rtt_ms === "number" && c.network_rtt_ms > 0);
      const avgRtt = validRtts.length > 0
        ? (validRtts.reduce((sum, c) => sum + c.network_rtt_ms, 0) / validRtts.length).toFixed(1) + " ms"
        : "N/A (Mô phỏng)";
      const dashRtt = document.getElementById("dash-rtt");
      if (dashRtt) setText(dashRtt, avgRtt);

      // Cập nhật biểu đồ phân bổ nếu có
      if (distributionChart) {
        distributionChart.data.labels = data.clients.map(c => `Node ${c.client_id}`);
        distributionChart.data.datasets[0].data = data.clients.map(c => c.samples);
        distributionChart.update("none");
      }
    } else {
      container.innerHTML = `<div class="empty-state p-4">Chưa có dữ liệu clients. Hãy chạy FedAvg để thu thập telemetry.</div>`;
      return;
    }

    container.replaceChildren();
    container.removeAttribute("aria-busy");
    data.clients.forEach(c => {
      const card = document.createElement("div");
      card.className = "client-card";

      const trainTimeDisplay = c.local_train_time_s !== null && c.local_train_time_s !== undefined
        ? `${esc(c.local_train_time_s)} s`
        : `<span class="text-muted">Chờ chạy vòng FedAvg</span>`;

      const normDisplay = c.update_norm !== null && c.update_norm !== undefined
        ? esc(c.update_norm)
        : `<span class="text-muted">Chờ tính gradient</span>`;

      card.innerHTML = `
        <div class="client-card-header">
          <div>
            <div class="client-card-title">${esc(c.client_name)}</div>
            <div class="client-badges">
              <span class="badge-role">${esc(c.role_badge)}</span>
              <span class="badge-contrib">Tỷ trọng: ${esc(c.total_dataset_pct)}</span>
            </div>
          </div>
          <span class="status-badge connected">
            <span class="live-dot"></span> ${esc(c.status)}
          </span>
        </div>

        <div class="crop-focus-box">
          <div class="crop-focus-title">${SVG_ICONS.layers} ${esc(c.crop_focus)}</div>
          <div class="crop-focus-sub">Phạm vi: ${esc(c.classes_present)} / 38 lớp thực vật</div>
        </div>

        <div class="client-stat-row">
          <span class="client-stat-label">${SVG_ICONS.server} Dữ liệu cục bộ:</span>
          <span class="client-stat-val text-sky fw-600">${Number(c.samples).toLocaleString()} ảnh</span>
        </div>
        <div class="client-stat-row">
          <span class="client-stat-label">${SVG_ICONS.clock} Thời gian huấn luyện:</span>
          <span class="client-stat-val">${trainTimeDisplay}</span>
        </div>
        <div class="client-stat-row">
          <span class="client-stat-label">${SVG_ICONS.activity} Cập nhật trọng số (Δw):</span>
          <span class="client-stat-val text-emerald fw-600">${normDisplay}</span>
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

    tbody.replaceChildren();
    tbody.removeAttribute("aria-busy");
    data.rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="fw-600">${esc(r.metric)}</td>
        <td class="text-sky">${esc(r.centralized)}</td>
        <td class="text-emerald fw-600">${esc(r.fedavg)}</td>
        <td class="text-muted">${esc(r.local_only)}</td>
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const selector = document.getElementById("layer-selector");
    if (!selector) return;

    if (!data.available) {
      selector.innerHTML = `<option value="">— ${esc(data.message)} —</option>`;
      const tbody = document.getElementById("layer-stats-body");
      if (tbody) tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">${esc(data.message)}</td></tr>`;
      layerListRequested = false; // lần mở tab Clients sau sẽ thử nạp lại
      return;
    }

    selector.replaceChildren();
    data.available_layers.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l;
      opt.textContent = l;
      selector.appendChild(opt);
    });

    renderLayerStats(data.selected_layer_stats, data.has_reference);
  } catch (err) {
    console.warn("Lỗi load layer list:", err);
    layerListRequested = false;
    const tbody = document.getElementById("layer-stats-body");
    if (tbody) tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Không đọc được thống kê trọng số: ${esc(err.message)}</td></tr>`;
  }
}

async function loadLayerStats() {
  const selector = document.getElementById("layer-selector");
  if (!selector || !selector.value) return;
  const layer = selector.value;

  try {
    const res = await fetch(`/api/weights/inspect?layer=${encodeURIComponent(layer)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderLayerStats(data.selected_layer_stats, data.has_reference);
  } catch (err) {
    console.warn("Lỗi load layer stats:", err);
    notify("error", "Không tải được thống kê trọng số", `Lớp ${layer}`, { detail: err.message });
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
    ? esc(stats.update_norm)
    : `<span class="text-muted" title="Cần có checkpoint vòng trước để đo khoảng cách delta">— (Chưa có reference)</span>`;

  tbody.innerHTML = `
    <tr>
      <td class="fw-600">${esc(stats.layer_name)}</td>
      <td>${esc(JSON.stringify(stats.shape))}</td>
      <td>${Number(stats.param_count).toLocaleString()}</td>
      <td class="text-sky">${esc(stats.l2_norm)}</td>
      <td class="text-emerald">${updateNormDisplay}</td>
      <td>${esc(stats.mean)} &plusmn; ${esc(stats.std)}</td>
      <td class="text-muted mono">[${stats.sample_values ? esc(stats.sample_values.join(", ")) : ""}]</td>
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
        setMetric(elAcc, "UNVERIFIED");
        setMetric(elF1, "UNVERIFIED");
        setMetric(elLoss, "UNVERIFIED");
        const badge = document.getElementById("cent-live-badge");
        if (badge) {
          setClass(badge, "status-badge error");
          setText(badge, "Historical / UNVERIFIED");
          badge.title = data.scientific_reason || "Canonical provenance chưa xác minh";
        }
      } else {
        const acc = data.test_metrics.accuracy !== undefined ? (data.test_metrics.accuracy * 100).toFixed(2) + "%" : "—";
        const f1 = data.test_metrics.macro_f1 !== undefined ? Number(data.test_metrics.macro_f1).toFixed(4) : "—";
        const loss = data.test_metrics.loss !== undefined ? Number(data.test_metrics.loss).toFixed(4) : "—";
        setMetric(elAcc, acc);
        setMetric(elF1, f1);
        setMetric(elLoss, loss);
      }
    }

    if (centralizedChart && data.history && data.history.length > 0) {
      centralizedChart.data.labels = data.history.map(h => `Epoch ${h.epoch}`);
      centralizedChart.data.datasets[0].data = data.history.map(h => Number(((h.train_accuracy || 0) * 100).toFixed(1)));
      centralizedChart.data.datasets[1].data = data.history.map(h => Number(((h.val_accuracy || 0) * 100).toFixed(1)));
      centralizedChart.update("none");
    }

    if (data.history && data.history.length > 0) {
      renderCentralizedProgress(data.history, false, null);
    }
  } catch (err) {
    console.warn("Lỗi load centralized metrics:", err);
  }
}

// Bảng tiến trình được dựng lại chỉ khi dữ liệu thật sự đổi (poll 4 giây không còn thay DOM vô ích).
function progressSignature(history, isLive, currentTask) {
  const first = currentTask && currentTask.live_progress && currentTask.live_progress[0];
  return JSON.stringify({
    history,
    isLive,
    status: currentTask ? currentTask.status : null,
    total: first ? (first.total_epochs ?? first.total_rounds ?? null) : null,
  });
}

function renderCentralizedProgress(history, isLive, currentTask) {
  const tbody = document.getElementById("cent-progress-body");
  const badge = document.getElementById("cent-live-badge");
  if (!tbody) return;

  const signature = progressSignature(history, isLive, currentTask);

  if (badge) {
    if (isLive) {
      const cur = history ? history.length : 0;
      const tot = (currentTask && currentTask.live_progress && currentTask.live_progress[0]?.total_epochs) || "?";
      setClass(badge, "status-badge running");
      setHtml(badge, `<span class="live-dot pulse"></span> Đang huấn luyện (${cur}/${esc(tot)} Epochs)`);
    } else if (currentTask && currentTask.status === "cancelled") {
      setClass(badge, "status-badge error");
      setHtml(badge, `<span class="live-dot"></span> Đã dừng`);
    } else if (history && history.length > 0) {
      setClass(badge, "status-badge success");
      setHtml(badge, `${SVG_ICONS.check} Hoàn tất`);
    } else {
      setClass(badge, "status-badge");
      setHtml(badge, "Sẵn sàng");
    }
  }

  // Chỉ dựng lại các hàng khi dữ liệu đổi (badge ở trên luôn được đồng bộ vì loadPublishedDemo có thể ghi đè nó).
  if (tbody.dataset.sig === signature) return;
  tbody.dataset.sig = signature;

  if (!history || history.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Chưa có tiến trình huấn luyện. Bấm "Bắt Đầu Huấn Luyện" để theo dõi.</td></tr>`;
    return;
  }

  const frag = document.createDocumentFragment();
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
      <td class="fw-600">Epoch ${esc(item.epoch)}</td>
      <td class="text-sky">${trainAcc}</td>
      <td>${trainLoss}</td>
      <td class="text-emerald fw-600">${valAcc}</td>
      <td>${valLoss}</td>
      <td>${f1}</td>
      <td class="text-muted">${dur}</td>
      <td class="${statusCls}">● ${statusText}</td>
    `;
    frag.appendChild(tr);
  });
  tbody.replaceChildren(frag);
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
      setMetric(document.getElementById("dash-acc"), "UNVERIFIED");
      setMetric(document.getElementById("dash-loss"), "UNVERIFIED");
      const badge = document.getElementById("fed-live-badge");
      if (badge) {
        setClass(badge, "status-badge error");
        setText(badge, "Historical / UNVERIFIED");
        badge.title = data.scientific_reason || "Canonical provenance chưa xác minh";
      }
      return;
    }

    if (data.best_accuracy !== undefined) {
      const b = data.best_accuracy;
      const bPct = b <= 1.0 && b > 0 ? (b * 100).toFixed(2) : Number(b).toFixed(2);
      setMetric(document.getElementById("dash-acc"), bPct + "%");
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
        federatedChart.update("none");
      }
      if (convergenceChart) {
        convergenceChart.data.labels = rounds;
        convergenceChart.data.datasets[0].data = accs;
        convergenceChart.data.datasets[1].data = losses;
        convergenceChart.update("none");
      }
      const testLoss = data.test_metrics?.loss ?? data.best_val_loss ?? data.history[data.history.length - 1]?.global_loss;
      if (testLoss !== undefined) {
        setMetric(document.getElementById("dash-loss"), Number(testLoss).toFixed(3));
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

  const signature = progressSignature(history, isLive, currentTask);

  if (badge) {
    if (isLive) {
      const cur = history ? history.length : 0;
      const tot = (currentTask && currentTask.live_progress && currentTask.live_progress[0]?.total_rounds) || "?";
      setClass(badge, "status-badge running");
      setHtml(badge, `<span class="live-dot pulse"></span> Đang giao tiếp (Vòng ${cur}/${esc(tot)})`);
    } else if (currentTask && currentTask.status === "cancelled") {
      setClass(badge, "status-badge error");
      setHtml(badge, `<span class="live-dot"></span> Đã dừng`);
    } else if (history && history.length > 0) {
      setClass(badge, "status-badge success");
      setHtml(badge, `${SVG_ICONS.check} Hoàn tất`);
    } else {
      setClass(badge, "status-badge");
      setHtml(badge, "Sẵn sàng");
    }
  }

  // Chỉ dựng lại các hàng khi dữ liệu đổi (badge ở trên luôn được đồng bộ vì loadPublishedDemo có thể ghi đè nó).
  if (tbody.dataset.sig === signature) return;
  tbody.dataset.sig = signature;

  if (!history || history.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Chưa có phiên đang theo dõi. Bấm "Khởi Động Vòng FedAvg" để chạy smoke; audit của phiên hoàn tất nằm ở bảng Flower bên dưới.</td></tr>`;
    return;
  }

  const frag = document.createDocumentFragment();
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
      <td class="fw-600">Vòng ${esc(item.round)}</td>
      <td class="text-emerald fw-600">${acc}</td>
      <td class="text-sky">${loss}</td>
      <td>${f1}</td>
      <td class="text-purple">${agg}</td>
      <td class="text-muted">${dur}</td>
      <td class="${statusCls}">● ${statusText}</td>
    `;
    frag.appendChild(tr);
  });
  tbody.replaceChildren(frag);
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
    cardDemo.classList.add("active", "sky");
    cardFull.classList.remove("active", "sky");
    inputVal.value = "true";
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
    setDisplay(networkFields, "block");
    setDisplay(scaleBlock, "none");
    if (help) help.textContent = "Demo kỹ thuật: một server và năm tiến trình client localhost, ảnh tổng hợp, cố định hai vòng và một local epoch. RTT đo trên cùng máy.";
    if (alpha) alpha.disabled = true;
    if (rounds) { rounds.value = "2"; rounds.max = "2"; rounds.min = "2"; rounds.disabled = true; }
    if (epochs) { epochs.value = "1"; epochs.max = "1"; epochs.min = "1"; epochs.disabled = true; }
  } else {
    setDisplay(networkFields, "none");
    setDisplay(scaleBlock, "block");
    if (help) help.textContent = "Simulation chạy các client trong cùng máy/process workflow; không được dùng để chứng minh latency mạng thật.";
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
      if (runContext) setText(runContext, "Chưa có phiên Flower smoke hoàn tất.");
      tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">Chưa có dữ liệu Flower network thật.</td></tr>`;
      if (clientBody) clientBody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">Chưa có audit per-client.</td></tr>`;
      if (finalSummary) setText(finalSummary, "Final held-out test: chưa có evaluator.");
      return;
    }
    if (runContext) setText(runContext, data.historical_run
      ? `Lịch sử smoke đã kiểm: ${String(data.run_dir || "").split(/[\\/]/).pop()} · 2 vòng, 5 client; không phải phiên đang chạy.`
      : "Phiên Flower hiện tại · dữ liệu tổng hợp, không dùng làm kết quả khoa học.");
    tbody.innerHTML = data.rounds.map(r => `
      <tr>
        <td>${esc(r.round)}</td>
        <td>${esc(r.accepted_clients ?? "—")}</td>
        <td>${esc(r.suspicious_clients ?? "—")}</td>
        <td>${esc(r.rejected_clients ?? "—")}</td>
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
      setText(finalSummary, tm
        ? `Final held-out test — Accuracy: ${(Number(tm.accuracy) * 100).toFixed(2)}% | Loss: ${Number(tm.loss).toFixed(4)} | Macro-F1: ${Number(tm.macro_f1).toFixed(4)}`
        : "Final held-out test: chưa có evaluator hoặc chưa cấu hình evaluation dataset root.");
    }

    if (clientBody) {
      const clientRows = (data.audit_rounds || []).flatMap(round =>
        (round.clients || []).map(client => ({ round: round.round, ...client }))
      );
      clientBody.innerHTML = clientRows.length ? clientRows.map(row => `
        <tr>
          <td>${esc(row.round)}</td>
          <td>Client ${esc(row.client_id)}</td>
          <td>${row.probe_rtt_ms !== null && row.probe_rtt_ms !== undefined ? Number(row.probe_rtt_ms).toFixed(2) + " ms" : "—"}</td>
          <td>${row.local_train_time_s !== undefined ? Number(row.local_train_time_s).toFixed(3) + " s" : "—"}</td>
          <td>${esc(row.data_mode || "—")}</td>
          <td>${row.training_seed !== undefined ? esc(row.training_seed) : "—"}</td>
          <td title="${esc(row.partition_config_sha256 || "")}">${esc(row.partition_id || "—")}</td>
          <td>${row.received_update_norm !== undefined ? Number(row.received_update_norm).toFixed(4) : "—"}</td>
          <td>${esc(row.status || "—")}</td>
          <td>${esc(row.validation_reason || "—")}</td>
        </tr>
      `).join("") : `<tr><td colspan="10" class="text-center text-muted">Chưa có audit per-client.</td></tr>`;
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">Lỗi đọc network metrics: ${esc(err.message)}</td></tr>`;
    if (clientBody) clientBody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">${esc(err.message)}</td></tr>`;
    const finalSummary = document.getElementById("network-final-test-summary");
    if (finalSummary) setText(finalSummary, `Final held-out test: lỗi đọc metrics (${err.message})`);
  }
}

function setFederatedMode(isDemo) {
  if (!isDemo) return;
  const cardDemo = document.getElementById("mode-fed-demo");
  const cardFull = document.getElementById("mode-fed-full");
  const inputVal = document.getElementById("fl-smoke-val");

  if (cardDemo && cardFull && inputVal) {
    cardDemo.classList.add("active");
    cardFull.classList.remove("active");
    inputVal.value = "true";
    loadPublishedDemo();
  }
}

async function triggerStopTrain() {
  const btnStopCent = document.getElementById("btn-stop-centralized");
  const btnStopFed = document.getElementById("btn-stop-federated");
  for (const btn of [btnStopCent, btnStopFed]) {
    if (btn) { btn.disabled = true; setHtml(btn, `${SVG_ICONS.spinner} Đang Dừng...`); }
  }

  try {
    const res = await fetch("/api/train/stop", { method: "POST" });
    const data = await res.json().catch(() => null);
    if (!res.ok || !data) throw new Error(formatApiError(data, `HTTP ${res.status}`));
    const taskBadge = document.getElementById("task-badge");
    if (taskBadge) {
      setText(taskBadge, data.message || "Đang gửi lệnh dừng...");
      setClass(taskBadge, "status-badge running");
    }
    if (data.stopped === false) {
      notify("warning", "Không có tác vụ để dừng", data.message || "Hiện không có tiến trình huấn luyện nào đang chạy.");
    } else {
      notify("info", "Đã gửi lệnh dừng", data.message || "Tiến trình sẽ dừng ở điểm kiểm tra gần nhất.");
    }
    await loadSystemStatus();
  } catch (err) {
    for (const btn of [btnStopCent, btnStopFed]) {
      if (btn) { btn.disabled = false; setHtml(btn, `${SVG_ICONS.stop} Dừng Huấn Luyện`); }
    }
    notify("error", "Không dừng được huấn luyện", err.message);
  }
}

// Khôi phục nút/thanh tiến trình khi yêu cầu huấn luyện bị từ chối hoặc lỗi mạng.
function resetTrainUi(btn, btnStop, barBox, idleHtml) {
  if (btn) { btn.disabled = false; setHtml(btn, idleHtml); }
  setDisplay(btnStop, "none");
  setDisplay(barBox, "none");
}

async function triggerCentralizedTrain() {
  const btn = document.getElementById("btn-train-centralized");
  const btnStop = document.getElementById("btn-stop-centralized");
  const epochs = document.getElementById("cent-epochs")?.value || 5;
  const seed = document.getElementById("cent-seed")?.value || 42;
  const isSmoke = document.getElementById("cent-smoke-val") ? document.getElementById("cent-smoke-val").value === "true" : true;

  btn.disabled = true;
  setHtml(btn, `${SVG_ICONS.spinner} Đang Huấn Luyện...`);
  if (btnStop) {
    setDisplay(btnStop, "inline-flex");
    btnStop.disabled = false;
    setHtml(btnStop, `${SVG_ICONS.stop} Dừng Huấn Luyện`);
  }

  const barBox = document.getElementById("cent-progress-bar-box");
  updateProgressBlock("cent", true, 5, "Tiến trình: 5%", "Đang khởi tạo mô hình...");

  const tbody = document.getElementById("cent-progress-body");
  if (tbody) {
    delete tbody.dataset.sig;
    tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">Đang khởi tạo kiến trúc mạng và phân bổ batch...</td></tr>`;
  }
  const badge = document.getElementById("cent-live-badge");
  if (badge) { setClass(badge, "status-badge running"); setHtml(badge, `<span class="live-dot pulse"></span> Đang khởi tạo...`); }

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

    const data = await res.json().catch(() => null);
    if (!res.ok || !data) {
      notify("error", "Không bắt đầu được huấn luyện Centralized", formatApiError(data, `Yêu cầu huấn luyện thất bại (HTTP ${res.status})`));
      resetTrainUi(btn, btnStop, barBox, `${SVG_ICONS.play} Bắt Đầu Huấn Luyện`);
      return;
    }

    previousTaskStatus = "running";
    notify("info", "Đã bắt đầu huấn luyện Centralized", data.message || `${epochs} epoch, seed ${seed}`);
    setText(document.getElementById("task-badge"), data.message || "Đang huấn luyện Centralized");
    await loadSystemStatus();
  } catch (err) {
    notify("error", "Không thể kết nối máy chủ", "Yêu cầu huấn luyện chưa được gửi đi.", { detail: err.message });
    resetTrainUi(btn, btnStop, barBox, `${SVG_ICONS.play} Bắt Đầu Huấn Luyện`);
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
  setHtml(btn, `${SVG_ICONS.spinner} Đang Điều Phối FedAvg...`);
  if (btnStop) {
    setDisplay(btnStop, "inline-flex");
    btnStop.disabled = false;
    setHtml(btnStop, `${SVG_ICONS.stop} Dừng Huấn Luyện`);
  }

  const barBox = document.getElementById("fed-progress-bar-box");
  updateProgressBlock("fed", true, 5, "Tiến trình: 5%", "Đang khởi tạo 5 trạm client...");

  const tbody = document.getElementById("fed-progress-body");
  if (tbody) {
    delete tbody.dataset.sig;
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Đang phân phối trọng số khởi tạo toàn cục tới 5 client...</td></tr>`;
  }
  const badge = document.getElementById("fed-live-badge");
  if (badge) { setClass(badge, "status-badge running"); setHtml(badge, `<span class="live-dot pulse"></span> Đang khởi tạo...`); }

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

    const data = await res.json().catch(() => null);
    if (!res.ok || !data) {
      notify("error", "Không bắt đầu được huấn luyện FedAvg", formatApiError(data, `Yêu cầu huấn luyện FedAvg thất bại (HTTP ${res.status})`));
      resetTrainUi(btn, btnStop, barBox, `${SVG_ICONS.zap} Khởi Động Vòng FedAvg`);
      return;
    }

    previousTaskStatus = "running";
    notify("info", "Đã bắt đầu huấn luyện FedAvg", data.message || `${rounds} vòng, ${epochs} epoch cục bộ`);
    setText(document.getElementById("task-badge"), data.message || "Đang huấn luyện FedAvg");
    await loadSystemStatus();
  } catch (err) {
    notify("error", "Không thể kết nối máy chủ", "Yêu cầu huấn luyện chưa được gửi đi.", { detail: err.message });
    resetTrainUi(btn, btnStop, barBox, `${SVG_ICONS.zap} Khởi Động Vòng FedAvg`);
  }
}

// 5. INFERENCE XỬ LÝ ẢNH & DUAL MODEL ĐỐI CHIẾU
function showSelectedPreview(src) {
  const img = document.getElementById("preview-image");
  img.src = src;
  img.classList.remove("hidden");
  document.querySelector(".upload-placeholder")?.classList.add("hidden");
  document.getElementById("btn-diagnose").disabled = false;
  document.getElementById("btn-plantvillage").disabled = false;
}

function handleImageSelected(input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    if (file.type && !file.type.startsWith("image/")) {
      notify("warning", "Tệp không phải ảnh", `"${file.name}" có định dạng ${file.type}. Hãy chọn ảnh JPG, PNG hoặc WEBP.`);
      input.value = "";
      return;
    }
    selectedFile = file;
    const pvResult = document.getElementById("plantvillage-result");
    if (pvResult) pvResult.textContent = "";
    const reader = new FileReader();
    reader.onload = (e) => showSelectedPreview(e.target.result);
    reader.onerror = () => notify("error", "Không đọc được ảnh", file.name);
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

    showSelectedPreview(url);

    // Tự động kích hoạt đối chiếu song song
    await submitInference();
  } catch (err) {
    console.error("Lỗi tải ảnh mẫu:", err);
    notify("error", "Không thể tải ảnh mẫu", label || "", { detail: err.message });
  }
}

async function submitInference() {
  if (!selectedFile) return;

  const btn = document.getElementById("btn-diagnose");
  const results = document.getElementById("dual-results");
  const skeleton = document.getElementById("inference-skeleton");
  const placeholder = document.getElementById("inference-empty-placeholder");
  const hadResults = results.style.display === "grid";

  btn.disabled = true;
  setHtml(btn, `${SVG_ICONS.spinner} Đang Phân Tích Song Song...`);
  if (hadResults) {
    results.classList.add("is-busy");
  } else {
    setDisplay(placeholder, "none");
    setDisplay(skeleton, "grid");
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const fedSource = document.getElementById("inference-fed-source")?.value || "fedavg";
    const trueClass = document.getElementById("inference-true-class")?.value || "";
    const query = `federated_source=${encodeURIComponent(fedSource)}` + (trueClass ? `&true_class=${encodeURIComponent(trueClass)}` : "");
    const res = await fetch(`/api/inference?${query}`, {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    if (!res.ok) {
      throw new Error(formatApiError(result, "Lỗi suy luận"));
    }

    setDisplay(skeleton, "none");
    setDisplay(placeholder, "none");
    setDisplay(results, "grid");
    void results.offsetWidth; // chốt style ban đầu (thanh = 0) để thanh độ tin cậy chạy từ 0 → giá trị

    const selectedBadge = document.getElementById("selected-model-badge");
    if (selectedBadge) {
      const label = result.federated && result.federated.registry ? result.federated.registry.display_label : "";
      setText(selectedBadge, (result.selected_model_name || "Mô hình được chọn") + (label ? ` · ${label}` : ""));
    }

    // 1. Render Cột Centralized
    const cent = result.centralized;
    if (cent && cent.ready) {
      setText(document.getElementById("cent-disease"), formatDiseaseName(cent.predicted_class));
      setBar(document.getElementById("cent-conf-bar"), cent.confidence_pct);
      setText(document.getElementById("cent-conf-val"), cent.confidence_pct + "%");
      setText(document.getElementById("cent-test-acc"), cent.model_validation
        ? (cent.model_validation.plantvillage.top1_accuracy * 100).toFixed(2) + "%"
        : "—");
      renderTop3("cent-top3-list", cent.top3);
    } else {
      setText(document.getElementById("cent-disease"), cent?.error || "Mô hình Centralized chưa sẵn sàng");
      setBar(document.getElementById("cent-conf-bar"), 0);
      setText(document.getElementById("cent-conf-val"), "—");
      setText(document.getElementById("cent-test-acc"), "Chưa huấn luyện");
      renderTop3("cent-top3-list", []);
    }

    // 2. Render Cột Federated
    const fed = result.federated;
    if (fed && fed.ready) {
      setText(document.getElementById("fed-disease"), formatDiseaseName(fed.predicted_class));
      setBar(document.getElementById("fed-conf-bar"), fed.confidence_pct);
      setText(document.getElementById("fed-conf-val"), fed.confidence_pct + "%");
      setText(document.getElementById("fed-test-acc"), fed.model_validation
        ? (fed.model_validation.plantvillage.top1_accuracy * 100).toFixed(2) + "%"
        : "—");
      renderTop3("fed-top3-list", fed.top3);
    } else {
      setText(document.getElementById("fed-disease"), fed?.error || "Mô hình FedAvg chưa sẵn sàng");
      setBar(document.getElementById("fed-conf-bar"), 0);
      setText(document.getElementById("fed-conf-val"), "—");
      setText(document.getElementById("fed-test-acc"), "Chưa huấn luyện");
      renderTop3("fed-top3-list", []);
    }
    renderShardContext(fed && fed.ready ? result.shard_context : null, result.federated_source, result.selected_model_name);

    // 3. Cảnh báo phân vân / dữ liệu ngoài miền (OOD)
    const advisory = document.getElementById("inference-advisory");
    if (advisory) {
      const centUncertain = cent && cent.is_uncertain;
      const fedUncertain = fed && fed.is_uncertain;
      const lowConf = (cent?.confidence_pct < 60.0) || (fed?.confidence_pct < 60.0);
      const evidenceNote = "Checkpoint đã kiểm SHA-256; số validation từ artifact công bố, chưa tái đánh giá tại máy demo. Xác suất softmax của một ảnh không phải độ chính xác trên ảnh ngoài thực địa. Chỉ dùng ảnh vùng lá đã cắt.";
      if (centUncertain || fedUncertain || lowConf) {
        advisory.innerHTML = `
          <div class="advisory-row">
            ${SVG_ICONS.info}
            <div>
              <b>Cần kiểm tra ảnh và nhãn dự đoán:</b> ${evidenceNote}
            </div>
          </div>
        `;
      } else {
        advisory.textContent = evidenceNote;
      }
      setDisplay(advisory, "block");
    }

    // Hiện kết quả lần lượt: hai thẻ, rồi khuyến nghị
    revealItems([...results.querySelectorAll(".diagnosis-card"), advisory], { stagger: 80 });
    notify("success", "Chẩn đoán hoàn tất", result.selected_model_name ? `So Centralized với ${result.selected_model_name}.` : "");

  } catch (err) {
    setDisplay(skeleton, "none");
    if (!hadResults) setDisplay(placeholder, "block");
    notify("error", "Không thể chẩn đoán ảnh", "Kiểm tra ảnh đầu vào hoặc kết nối rồi thử lại.", { detail: err.message });
  } finally {
    results.classList.remove("is-busy");
    btn.disabled = false;
    setHtml(btn, `${SVG_ICONS.microscope} Chạy Chẩn Đoán Song Song`);
  }
}

function formatDiseaseName(raw) {
  if (!raw) return "—";
  return raw.replace(/___/g, " - ").replace(/_/g, " ");
}

// Giải thích vì sao Local-only sai/đúng: shard huấn luyện của client đó có bao nhiêu ảnh của các lớp top-3.
function renderShardContext(ctx, federatedSource, modelName) {
  const box = document.getElementById("fed-shard-context");
  if (!box) return;
  box.replaceChildren();
  if (!ctx || !Array.isArray(ctx.classes) || ctx.classes.length === 0) {
    box.style.display = "none";
    return;
  }
  const fmt = n => Number(n).toLocaleString("vi-VN");
  const title = document.createElement("div");
  title.className = "shard-context-title";
  title.textContent = ctx.client
    ? `Dữ liệu huấn luyện của ${modelName} (${fmt(ctx.client_n_k)} ảnh, ${ctx.client_classes_with_images}/38 lớp có ảnh):`
    : "Dữ liệu huấn luyện: FedAvg học từ cả 5 client, không client nào gửi ảnh về server:";
  box.appendChild(title);
  for (const item of ctx.classes) {
    const line = document.createElement("div");
    line.className = "shard-context-line";
    const name = formatDiseaseName(item.class);
    const tag = item.is_true_label ? "[Nhãn thật] " : (item.in_top3 ? "[Dự đoán] " : "");
    if (ctx.client) {
      const none = item.client_count === 0;
      line.textContent = none
        ? `• ${tag}${name}: 0 ảnh trong shard này (cả 5 client: ${fmt(item.all_clients_count)}) — client này chưa từng thấy lớp này`
        : `• ${tag}${name}: ${fmt(item.client_count)} ảnh (${item.client_share_pct}% của ${fmt(item.all_clients_count)} ảnh trên cả 5 client)`;
      if (none) line.classList.add("is-warn");
    } else {
      line.textContent = `• ${tag}${name}: ${fmt(item.all_clients_count)} ảnh trên cả 5 client`;
    }
    box.appendChild(line);
  }
  box.style.display = "block";
}

function renderTop3(containerId, top3List) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.replaceChildren();
  if (!top3List || top3List.length === 0) {
    const li = document.createElement("li");
    li.className = "text-muted";
    li.textContent = "Không có dữ liệu top-3";
    container.appendChild(li);
    return;
  }

  top3List.forEach(item => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = formatDiseaseName(item.class);
    const conf = document.createElement("b");
    conf.textContent = `${item.confidence}%`;
    li.append(name, conf);
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
    if (!response.ok) throw new Error(formatApiError(data, "Inference failed"));
    const top = data.top3.map((r, index) => `${index + 1}. ${formatDiseaseName(r.class)}: ${r.confidence}%`).join("\n");
    output.textContent = `Dự đoán: ${formatDiseaseName(data.predicted_class)} (${data.confidence_pct}%)\nTop 3:\n${top}\n${data.is_uncertain ? "⚠ Mô hình không chắc chắn.\n" : ""}Lưu ý: Confidence chưa hiệu chuẩn; kết quả có thể sai trên ảnh thực tế.`;
  } catch (error) {
    output.textContent = `Không thể nhận diện: ${error.message}`;
    notify("error", "Không thể nhận diện bằng PlantVillage", "", { detail: error.message });
  } finally {
    btn.disabled = false;
  }
}
