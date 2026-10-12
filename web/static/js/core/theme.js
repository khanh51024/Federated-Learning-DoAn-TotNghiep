// Nền sáng/tối: mặc định theo hệ điều hành, nhớ lựa chọn của người dùng; phát sự kiện "themechange" để biểu đồ vẽ lại.
// Giá trị ban đầu đã được script nhỏ trong <head> đặt trước khi trang hiện (không nháy nền).
import { byId, cssVar } from "./dom.js";

const KEY = "cropfed.theme";

function stored() {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch (_error) {
    return null;
  }
}

function save(theme) {
  try {
    window.localStorage.setItem(KEY, theme);
  } catch (_error) {
    /* private window / bị chặn: chỉ không nhớ được lựa chọn */
  }
}

export function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

export function applyTheme(theme, { persist = false } = {}) {
  const next = theme === "dark" ? "dark" : "light";
  const changed = currentTheme() !== next;
  const root = document.documentElement;
  if (changed) {
    root.classList.add("theme-switching"); // 2 khung: mọi màu đổi cùng lúc, không trượt từng phần
    requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("theme-switching")));
  }
  root.setAttribute("data-theme", next);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", cssVar("--bg-base", next === "dark" ? "#0e1611" : "#f6f4ec"));
  // nhãn nút nói việc sẽ làm ("Nền tối"/"Nền sáng"), nên không dùng thêm aria-pressed (đọc ra sẽ thành trạng thái sai)
  const label = byId("btn-theme-label");
  if (label) label.textContent = next === "dark" ? "Nền sáng" : "Nền tối";
  if (persist) save(next);
  if (changed) window.dispatchEvent(new CustomEvent("themechange", { detail: { theme: next } }));
}

export function initTheme() {
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  applyTheme(stored() || (media && media.matches ? "dark" : "light"));
  if (media) {
    media.addEventListener("change", event => {
      if (!stored()) applyTheme(event.matches ? "dark" : "light");
    });
  }
  byId("btn-theme")?.addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark", { persist: true }));
  // In luôn dùng nền sáng (chữ sáng trên giấy trắng không đọc được); in xong trả lại nền tối, không ghi đè lựa chọn đã lưu
  let restore = null;
  window.addEventListener("beforeprint", () => {
    if (currentTheme() === "dark") {
      restore = "dark";
      document.documentElement.dataset.printing = "1"; // biểu đồ dựng lại không animation: vẽ xong ngay trước khi chụp trang in
      applyTheme("light");
    }
  });
  window.addEventListener("afterprint", () => {
    if (restore) applyTheme(restore);
    restore = null;
    delete document.documentElement.dataset.printing;
  });
}
