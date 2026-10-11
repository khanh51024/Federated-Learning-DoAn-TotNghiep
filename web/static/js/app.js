// Điểm khởi động: điều hướng 6 bước, chế độ trình chiếu, poll trạng thái server.
// Mọi trang nằm trong js/pages/*.js và được tải lười khi mở lần đầu.
import { $$, byId } from "./core/dom.js";
import { initPresent } from "./core/present.js";
import { initRouter } from "./core/router.js";
import { startStatusPolling } from "./core/status.js";
import { initTheme } from "./core/theme.js";

const MOBILE_QUERY = "(max-width: 1100px)";

function initSidebarToggle() {
  const toggle = byId("mobile-nav-toggle");
  const sidebar = byId("app-sidebar");
  if (!toggle || !sidebar) return;
  const mq = window.matchMedia(MOBILE_QUERY);
  const setOpen = open => {
    sidebar.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    // trên mobile sidebar đóng không được nhận focus; trên màn hình rộng nó luôn hiện nên phải bỏ inert
    if (open || !mq.matches) sidebar.removeAttribute("inert");
    else sidebar.setAttribute("inert", "");
  };
  toggle.addEventListener("click", () => setOpen(!sidebar.classList.contains("open")));
  $$(".nav-item").forEach(item => item.addEventListener("click", () => setOpen(false)));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") setOpen(false);
  });
  mq.addEventListener("change", () => setOpen(sidebar.classList.contains("open")));
  setOpen(false);
}

// Skip link: nhảy tới nội dung mà KHÔNG đổi hash (hash là địa chỉ của bước đang xem).
function initSkipLink() {
  const link = document.querySelector(".skip-link");
  const main = byId("main");
  if (!link || !main) return;
  link.addEventListener("click", event => {
    event.preventDefault();
    main.focus({ preventScroll: false });
  });
}

// Mũi tên trong nhóm tab/radio: di chuyển giữa các lựa chọn và kích hoạt (theo mẫu WAI-ARIA).
function initGroupArrowKeys() {
  document.addEventListener("keydown", event => {
    const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"];
    if (!keys.includes(event.key) || event.ctrlKey || event.metaKey || event.altKey) return;
    const current = event.target.closest && event.target.closest('[role="tab"], [role="radio"]');
    if (!current) return;
    const group = current.closest('[role="tablist"], [role="radiogroup"]');
    if (!group) return;
    const options = Array.from(group.querySelectorAll(current.getAttribute("role") === "tab" ? '[role="tab"]' : '[role="radio"]'))
      .filter(el => !el.disabled && el.offsetParent !== null);
    const index = options.indexOf(current);
    if (index === -1 || options.length < 2) return;
    let next = index;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % options.length;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + options.length) % options.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = options.length - 1;
    event.preventDefault();
    options[next].focus();
    options[next].click();
  });
}

initTheme();
initSidebarToggle();
initSkipLink();
initGroupArrowKeys();
initPresent();
initRouter();
startStatusPolling();
