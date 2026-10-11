// Chế độ trình chiếu cho buổi bảo vệ: chữ/số to hơn, ẩn chi tiết kỹ thuật, phím ← → chuyển bước, nhớ lựa chọn.
import { $, byId } from "./dom.js";
import { ROUTES, activeRouteId, goto, routeIndex } from "./router.js";

const STORAGE_KEY = "cropfed.present";

function readSaved() {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch (_error) {
    return false;
  }
}

function save(on) {
  try {
    window.localStorage.setItem(STORAGE_KEY, on ? "1" : "0");
  } catch (_error) {
    /* private window / bị chặn: chỉ không nhớ được lựa chọn */
  }
}

export function isPresenting() {
  return document.body.classList.contains("present");
}

export function setPresenting(on, { persist = true } = {}) {
  document.body.classList.toggle("present", on);
  const bar = byId("present-bar");
  if (bar) bar.hidden = !on;
  const button = byId("btn-present");
  if (button) button.setAttribute("aria-pressed", String(on));
  if (persist) save(on);
  // Biểu đồ responsive cần biết kích thước mới sau khi đổi cỡ chữ
  window.dispatchEvent(new Event("resize"));
}

function step(delta) {
  const index = routeIndex(activeRouteId());
  const target = ROUTES[index + delta];
  if (target) goto(target.id);
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable === true;
}

/** Phần tử tự dùng phím mũi tên/PageUp/PageDown (nhóm tab/radio, vùng cuộn): không được chuyển bước. */
function ownsArrowKeys(target) {
  return !!(target && target.closest && target.closest('[role="tab"], [role="radio"], [role="tablist"], .heat-wrap, .table-responsive, .gallery-grid, details'));
}

export function initPresent() {
  const steps = byId("present-steps");
  if (steps) {
    steps.innerHTML = ROUTES.map(
      r => `<li><a class="present-step" href="#${r.id}" data-route="${r.id}"><span class="present-step-n">${r.step}</span>${r.label.replace("&", "&amp;")}</a></li>`
    ).join("");
  }

  byId("btn-present")?.addEventListener("click", () => setPresenting(!isPresenting()));
  byId("btn-present-exit")?.addEventListener("click", () => setPresenting(false));

  document.addEventListener("keydown", event => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || isTypingTarget(event.target)) return;
    const key = event.key;
    if (key === "p" || key === "P") {
      event.preventDefault();
      setPresenting(!isPresenting());
    } else if (!isPresenting()) {
      return; // ngoài chế độ trình chiếu, các phím mũi tên giữ nguyên chức năng cuộn/chọn của trình duyệt
    } else if (key === "Escape") {
      setPresenting(false);
    } else if (ownsArrowKeys(event.target)) {
      return;
    } else if (key === "ArrowRight" || key === "PageDown") {
      event.preventDefault();
      step(1);
    } else if (key === "ArrowLeft" || key === "PageUp") {
      event.preventDefault();
      step(-1);
    }
  });

  if (readSaved()) setPresenting(true, { persist: false });
  else if ($("#present-bar")) $("#present-bar").hidden = true;
}
