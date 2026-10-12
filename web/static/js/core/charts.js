// Chart.js: cấu hình chung lấy từ token CSS để biểu đồ luôn khớp giao diện; chỉ tạo khi canvas đang hiện.
// makeChart nhận một HÀM trả cấu hình: khi đổi nền sáng/tối, mọi biểu đồ còn trên trang được dựng lại với màu mới.
import { cssVar, prefersReducedMotion } from "./dom.js";

/** Cột mọc từ đáy: 400 ms, easeOutQuart, mỗi cột trễ 30 ms (báo hiệu "dữ liệu vừa được nạp"). Tắt khi giảm chuyển động. */
export function barRise() {
  if (prefersReducedMotion()) return false;
  return {
    duration: 400,
    easing: "easeOutQuart",
    delay: ctx => (ctx.type === "data" && ctx.mode === "default" ? ctx.dataIndex * 30 + ctx.datasetIndex * 15 : 0),
  };
}

/** Đường kẻ vẽ dần theo từng điểm. */
export function lineDraw() {
  if (prefersReducedMotion()) return false;
  return {
    duration: 400,
    easing: "easeOutQuart",
    delay: ctx => (ctx.type === "data" && ctx.mode === "default" ? Math.min(ctx.dataIndex * 10, 500) : 0),
  };
}

let registered = false;
const factories = new Map(); // canvas -> () => config

export function chartAvailable() {
  return typeof Chart !== "undefined";
}

function applyDefaults() {
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
  const tip = Chart.defaults.plugins.tooltip;
  tip.backgroundColor = cssVar("--tooltip-bg", "#1c261d");
  tip.titleColor = cssVar("--tooltip-fg", "#f6f4ec");
  tip.bodyColor = cssVar("--tooltip-fg", "#f6f4ec");
  tip.footerColor = cssVar("--tooltip-fg", "#f6f4ec");
  tip.padding = 12;
  tip.cornerRadius = 8;
  tip.boxPadding = 4;
  Chart.defaults.interaction.mode = "index";
  Chart.defaults.interaction.intersect = false;
}

function registerOnce() {
  if (registered || !chartAvailable()) return;
  registered = true;
  applyDefaults();
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
  window.addEventListener("themechange", () => {
    applyDefaults();
    for (const [canvas, factory] of factories) {
      if (!canvas.isConnected) {
        factories.delete(canvas);
        continue;
      }
      build(canvas, factory, { animate: false }); // đổi nền: dựng lại ngay, không chạy lại hiệu ứng mọc
    }
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
    neutral: cssVar("--text-muted", "#6b7280"),
  };
}

function build(canvas, factory, { animate = true } = {}) {
  const config = factory();
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
  const printing = document.documentElement.dataset.printing === "1"; // đang in: vẽ đồng bộ, không chờ khung hình animation
  return new Chart(canvas.getContext("2d"), {
    ...config,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      ...(config.options || {}),
      ...(printing || !animate ? { animation: false } : {}),
    },
  });
}

/** Tạo (hoặc thay) biểu đồ. `configOrFactory` nên là hàm để biểu đồ đổi màu theo nền; không có Chart.js thì ghi chú. */
export function makeChart(canvas, configOrFactory) {
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
  registerOnce();
  const factory = typeof configOrFactory === "function" ? configOrFactory : () => configOrFactory;
  factories.set(canvas, factory);
  return build(canvas, factory);
}
