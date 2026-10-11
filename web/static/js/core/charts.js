// Chart.js: cấu hình chung lấy từ token CSS để biểu đồ luôn khớp giao diện; chỉ tạo khi canvas đang hiện.
import { cssVar, prefersReducedMotion } from "./dom.js";

let configured = false;

export function chartAvailable() {
  return typeof Chart !== "undefined";
}

function configureOnce() {
  if (configured || !chartAvailable()) return;
  configured = true;
  Chart.defaults.font.family = cssVar("--font-main", "sans-serif");
  Chart.defaults.font.size = 12;
  Chart.defaults.color = cssVar("--text-secondary", "#46515f");
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

  // Biểu đồ chưa có dữ liệu: ghi rõ thay vì để lưới trống khó hiểu.
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
}

/** Màu theo trụ cột: Centralized xanh dương, FedAvg xanh lá, Local-only cam (cố định ở mọi nơi). */
export function pillarColors() {
  return {
    centralized: cssVar("--chart-2", "#0284c7"),
    fedavg: cssVar("--chart-1", "#059669"),
    local: cssVar("--chart-4", "#d97706"),
    danger: cssVar("--chart-5", "#e11d48"),
    violet: cssVar("--chart-3", "#7c3aed"),
  };
}

/** Tạo (hoặc thay) biểu đồ trên canvas. Không có Chart.js thì ghi chú văn bản vào khung chứa. */
export function makeChart(canvas, config) {
  if (!canvas) return null;
  if (!chartAvailable()) {
    const host = canvas.parentElement;
    if (host && !host.querySelector(".chart-fallback-notice")) {
      const notice = document.createElement("div");
      notice.className = "chart-fallback-notice";
      notice.textContent = "Biểu đồ không khả dụng: không nạp được thư viện Chart.js.";
      host.appendChild(notice);
    }
    return null;
  }
  configureOnce();
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
  return new Chart(canvas.getContext("2d"), {
    ...config,
    options: { responsive: true, maintainAspectRatio: false, ...(config.options || {}) },
  });
}
