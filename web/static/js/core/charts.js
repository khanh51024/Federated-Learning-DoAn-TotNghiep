// Chart.js: cấu hình chung lấy từ token CSS để biểu đồ luôn khớp giao diện; chỉ tạo khi canvas đang hiện.
// makeChart nhận một HÀM trả cấu hình: khi đổi nền sáng/tối, mọi biểu đồ còn trên trang được dựng lại với màu mới.
import { cssVar, prefersReducedMotion, withAlpha } from "./dom.js";

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
  const config = trackHover(factory());
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

// Chart.js (200 KB) được nạp `async` để không chặn app.js: ở mạng chậm nó chiếm ~1 s băng thông ngay trước mọi thứ khác
// và là lý do chính khiến LCP vượt 2,5 s. Hệ quả: makeChart có thể chạy trước lúc thư viện nạp xong. Khi đó biểu đồ được
// xếp hàng và dựng ngay khi script load; script lỗi hoặc quá hạn thì mới ghi chú "không khả dụng".
const CHART_WAIT_MS = 8000;
const waiting = new Map(); // canvas -> factory, chờ Chart.js
let waitStarted = false;
let chartSettled = false; // đã biết kết quả nạp (thành công hoặc hết hạn): không chờ nữa

function chartScript() {
  return document.querySelector('script[src*="chart.umd"]');
}

function showFallback(canvas) {
  const host = canvas.parentElement;
  if (host && !host.querySelector(".chart-fallback-notice")) {
    const notice = document.createElement("div");
    notice.className = "chart-fallback-notice";
    notice.textContent = "Biểu đồ không khả dụng: không nạp được thư viện Chart.js.";
    host.appendChild(notice);
  }
}

function settleWaiting() {
  if (chartSettled) return;
  chartSettled = true;
  const batch = Array.from(waiting);
  waiting.clear();
  for (const [canvas, factory] of batch) {
    if (!canvas.isConnected) continue;
    if (chartAvailable()) {
      registerOnce();
      factories.set(canvas, factory);
      build(canvas, factory);
    } else {
      showFallback(canvas);
    }
  }
}

function waitForChart() {
  if (waitStarted) return;
  waitStarted = true;
  const script = chartScript();
  script.addEventListener("load", settleWaiting, { once: true });
  script.addEventListener("error", settleWaiting, { once: true });
  setTimeout(settleWaiting, CHART_WAIT_MS);
}

/** Biểu đồ nằm trong <details>: dựng khi mục được mở, để hiệu ứng mọc cột chạy lúc người xem nhìn thấy (dựng lúc ẩn
 *  thì hoạt ảnh trôi qua trên canvas rộng 0 và hình cuối bật ra tức thì). Mở lại thì chạy lại hiệu ứng. */
export function makeChartWhenVisible(canvas, configOrFactory) {
  if (!canvas) return null;
  const host = canvas.closest("details");
  if (!host || host.open) return makeChart(canvas, configOrFactory);
  host.addEventListener("toggle", () => { if (host.open) makeChart(canvas, configOrFactory); });
  return null;
}

/** Cột nào đang được rê thì giữ nguyên, các cột còn lại phủ mờ bằng màu nền: chỉ `chart.draw()` (không update/animation). */
const hoverDim = {
  id: "hoverDim",
  afterDatasetsDraw(chart) {
    const hi = chart.$hoverIndex;
    if (hi === null || hi === undefined || chart.config.type !== "bar") return;
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.fillStyle = withAlpha(cssVar("--bg-surface", "#ffffff"), 0.6);
    chart.data.datasets.forEach((_dataset, di) => {
      chart.getDatasetMeta(di).data.forEach((bar, index) => {
        if (index === hi || !bar) return;
        const { x, y, base, width } = bar.getProps(["x", "y", "base", "width"], true);
        ctx.fillRect(x - width / 2, Math.min(y, base), width, Math.abs(base - y));
      });
    });
    ctx.restore();
    void chartArea;
  },
};

function trackHover(config) {
  if (config.type !== "bar") return config;
  const previous = config.options?.onHover;
  return {
    ...config,
    plugins: [...(config.plugins || []), hoverDim],
    options: {
      ...(config.options || {}),
      onHover(event, active, chart) {
        const next = active.length ? active[0].index : null;
        if (chart.$hoverIndex !== next) {
          chart.$hoverIndex = next;
          chart.draw();
        }
        if (previous) previous(event, active, chart);
      },
    },
  };
}

/** Tạo (hoặc thay) biểu đồ. `configOrFactory` nên là hàm để biểu đồ đổi màu theo nền; không có Chart.js thì ghi chú. */
export function makeChart(canvas, configOrFactory) {
  if (!canvas) return null;
  const factory = typeof configOrFactory === "function" ? configOrFactory : () => configOrFactory;
  if (!chartAvailable()) {
    if (!chartSettled && chartScript()) {
      waiting.set(canvas, factory);
      waitForChart();
    } else {
      showFallback(canvas);
    }
    return null;
  }
  registerOnce();
  factories.set(canvas, factory);
  return build(canvas, factory);
}
