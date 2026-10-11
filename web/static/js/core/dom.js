// Tiện ích DOM dùng chung. Quy ước hiệu năng: chỉ ghi khi giá trị đổi; thanh tiến trình đổi bằng transform;
// escape mọi chuỗi từ server trước khi ghép vào innerHTML.

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
}

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
export const byId = id => document.getElementById(id);

export function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

export function cssVar(name, fallback = "") {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export function withAlpha(hex, alpha) {
  if (!/^#[0-9a-f]{6}$/i.test(hex)) return hex;
  return hex + Math.round(alpha * 255).toString(16).padStart(2, "0");
}

export function setText(el, text) {
  const next = String(text);
  if (el && el.textContent !== next) {
    el.textContent = next;
    delete el.dataset.html; // bộ nhớ đệm của setHtml không còn đúng
  }
}

export function setHtml(el, html) {
  if (el && el.dataset.html !== html) {
    el.innerHTML = html;
    el.dataset.html = html;
  }
}

export function setClass(el, className) {
  if (el && el.className !== className) el.className = className;
}

export function setDisplay(el, display) {
  if (el && el.style.display !== display) el.style.display = display;
}

export function setDisabled(el, disabled) {
  if (el && el.disabled !== disabled) el.disabled = disabled;
}

export function toggleHidden(el, hidden) {
  if (el) el.classList.toggle("hidden", !!hidden);
}

// Thanh tiến trình/độ tin cậy đổi bằng transform (không kích hoạt layout).
export function setBar(el, pct) {
  if (!el) return;
  const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
  el.style.transform = `scaleX(${clamped / 100})`;
}

// Số KPI đếm lên bằng rAF; chữ không phải số hiện luôn. Chấp nhận dấu phẩy hoặc chấm thập phân.
export function setMetric(el, text) {
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
  const tick = now => {
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
export function revealItems(elements, options = {}) {
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

// ───────── Định dạng tiếng Việt ─────────
const nf = (digits) => new Intl.NumberFormat("vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const NF = { 0: nf(0), 1: nf(1), 2: nf(2), 4: nf(4) };

export const fmtInt = n => (n === null || n === undefined || Number.isNaN(Number(n)) ? "—" : NF[0].format(Number(n)));
export const fmtNum = (n, digits = 2) => (n === null || n === undefined || Number.isNaN(Number(n)) ? "—" : (NF[digits] || nf(digits)).format(Number(n)));
/** Phân số 0–1 -> "93,12%". */
export const fmtPct = (fraction, digits = 2) => (fraction === null || fraction === undefined ? "—" : `${fmtNum(Number(fraction) * 100, digits)}%`);
/** Số đã là phần trăm 0–100 -> "93,12%". */
export const fmtPctValue = (value, digits = 2) => (value === null || value === undefined ? "—" : `${fmtNum(Number(value), digits)}%`);

export function formatDiseaseName(raw) {
  if (!raw) return "—";
  return String(raw).replace(/___/g, " — ").replace(/_/g, " ");
}

export function cropOf(raw) {
  return String(raw || "").split("___")[0].replace(/_/g, " ");
}

export function diseaseOf(raw) {
  const part = String(raw || "").split("___")[1];
  return part ? part.replace(/_/g, " ") : String(raw || "");
}

// Thông điệp lỗi từ FastAPI: detail có thể là chuỗi, mảng lỗi validate hoặc object.
export function formatApiError(data, fallback) {
  if (!data) return fallback;
  const detail = data.detail ?? data.message;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map(item => (item && (item.msg || item.message)) || JSON.stringify(item)).join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

export function notify(type, title, message, options) {
  if (window.toast && typeof window.toast[type] === "function") {
    return window.toast[type](title, message, options);
  }
  console[type === "error" ? "error" : "log"](`[${type}] ${title}${message ? ": " + message : ""}`);
  return { close() {} };
}

export function skeleton(lines = 3) {
  return Array.from({ length: lines }, () => '<span class="skeleton skeleton-line"></span>').join("");
}

/** Nhãn tiếng Việt ngắn của mô hình (tên đã dùng trong tài liệu). */
export const MODEL_LABELS = {
  centralized: "Centralized v6",
  fedavg: "FedAvg v7",
  local_00: "Local 00",
  local_01: "Local 01",
  local_02: "Local 02",
  local_03: "Local 03",
  local_04: "Local 04",
};

export const MODEL_ORDER = ["centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04"];

export function pillarOf(modelId) {
  if (modelId === "centralized") return "centralized";
  if (modelId === "fedavg") return "fedavg";
  return "local";
}

export function modelPill(modelId, label) {
  return `<span class="model-pill ${pillarOf(modelId)}">${esc(label || MODEL_LABELS[modelId] || modelId)}</span>`;
}

export function clientIdOfModel(modelId) {
  return modelId.startsWith("local_") ? modelId.replace("local_", "client_") : null;
}
