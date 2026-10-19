// Tiện ích DOM dùng chung. Quy ước hiệu năng: chỉ ghi khi giá trị đổi; thanh tiến trình đổi bằng transform;
// escape mọi chuỗi từ server trước khi ghép vào innerHTML.

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
}

// Một nguồn cho điểm gãy bố cục (khớp CSS: @media (max-width: 1100px) chuyển sang một cột).
export const NARROW_QUERY = "(max-width: 1100px)";
export const WIDE_QUERY = "(min-width: 1101px)";

/** Mở sẵn một mục gấp khi màn đủ rộng để bảng không chiếm hết khung (máy chiếu/laptop); điện thoại vẫn gấp. Chỉ chạy một lần khi dựng. */
/**
 * Tiêu đề mục đánh số kiểu báo cáo: "1.1 Bài toán", "A. Kiểm chứng số liệu", kèm một dòng dẫn tùy chọn.
 * `title` và `lead` là HTML viết trong mã (đã an toàn); `no` và `id` được escape.
 */
export function sectionHead(no, title, { id = "", lead = "", tag = "h3" } = {}) {
  const idAttr = id ? ` id="${esc(id)}"` : "";
  return `<${tag} class="panel-title section-title"${idAttr}><span class="sec-no">${esc(no)}</span> ${title}</${tag}>${lead ? `<p class="section-lead">${lead}</p>` : ""}`;
}

export function openOnWide(details) {
  if (details && window.matchMedia?.(WIDE_QUERY).matches) details.open = true;
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

// Tên trường trong body của các endpoint huấn luyện, để lỗi validate 422 của pydantic đọc được bằng tiếng Việt.
const API_FIELD_LABELS = {
  rounds: "Số vòng", local_epochs: "Epoch cục bộ", epochs: "Số epoch", seed: "Seed",
  alpha: "Mức lệch dữ liệu (α)", num_clients: "Số client", max_update_norm: "Ngưỡng L2 norm",
};

function describeValidationItem(item) {
  if (!item || typeof item !== "object") return String(item);
  const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null;
  const name = API_FIELD_LABELS[field] || (typeof field === "string" ? field : "Giá trị");
  const ctx = item.ctx || {};
  switch (item.type) {
    case "greater_than_equal": if (ctx.ge !== undefined) return `${name}: tối thiểu là ${ctx.ge}.`; break;
    case "less_than_equal": if (ctx.le !== undefined) return `${name}: tối đa là ${ctx.le}.`; break;
    case "greater_than": if (ctx.gt !== undefined) return `${name}: phải lớn hơn ${ctx.gt}.`; break;
    case "less_than": if (ctx.lt !== undefined) return `${name}: phải nhỏ hơn ${ctx.lt}.`; break;
    case "missing": return `${name}: còn thiếu.`;
    case "int_parsing": case "int_type": case "int_from_float": case "float_parsing": case "float_type":
      return `${name}: phải là một số hợp lệ.`;
    default: break;
  }
  return item.msg || item.message || JSON.stringify(item);
}

// Thông điệp lỗi từ FastAPI: detail có thể là chuỗi, mảng lỗi validate hoặc object.
export function formatApiError(data, fallback) {
  if (!data) return fallback;
  const detail = data.detail ?? data.message;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map(describeValidationItem).join(" ");
  }
  if (detail && typeof detail === "object") return typeof detail.message === "string" && detail.message ? detail.message : JSON.stringify(detail);
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

// ───────── data-style: thay cho thuộc tính style nội tuyến (CSP style-src 'self' chặn nó trong HTML do innerHTML dựng) ─────────
// Template ghi `data-style="--p:0.4"` hoặc `data-style="transform:scaleX(0.3)"`; ở đây đổi thành CSSOM (CSP cho phép).
// Chỉ nhận biến CSS (--tên) và transform, giá trị chỉ gồm số/%/ngoặc/dấu phẩy: dữ liệu từ server không thể chèn thuộc tính tùy ý.
const STYLE_PROP = /^(--[\w-]+|transform)$/;
const STYLE_VALUE = /^[-\w.%()\s,]+$/;

export function applyStyleData(root = document) {
  const nodes = root.querySelectorAll ? Array.from(root.querySelectorAll("[data-style]")) : [];
  if (root.nodeType === 1 && root.hasAttribute("data-style")) nodes.unshift(root);
  for (const el of nodes) {
    for (const part of el.dataset.style.split(";")) {
      const colon = part.indexOf(":");
      if (colon < 0) continue;
      const prop = part.slice(0, colon).trim();
      const value = part.slice(colon + 1).trim();
      if (STYLE_PROP.test(prop) && STYLE_VALUE.test(value)) el.style.setProperty(prop, value);
    }
    el.removeAttribute("data-style");
  }
}

let styleObserver = null;
/** Áp data-style cho mọi nút được thêm vào trang (chạy ở microtask, trước lần vẽ kế tiếp nên không nháy). Nơi đọc style NGAY sau khi chèn phải gọi applyStyleData trực tiếp. */
export function watchStyleData() {
  if (styleObserver || typeof MutationObserver === "undefined" || !document.documentElement) return;
  styleObserver = new MutationObserver(records => {
    for (const record of records) {
      for (const node of record.addedNodes) if (node.nodeType === 1) applyStyleData(node);
    }
  });
  styleObserver.observe(document.documentElement, { childList: true, subtree: true });
  applyStyleData(document);
}
watchStyleData();
