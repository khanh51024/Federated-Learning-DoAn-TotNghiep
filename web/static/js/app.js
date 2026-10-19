// Điểm khởi động: thanh 6 bước ở đầu trang, poll trạng thái server.
// Mọi trang nằm trong js/pages/*.js và được tải lười khi mở lần đầu.
import { byId } from "./core/dom.js";
import "./core/motion.js";
import { initRouter } from "./core/router.js";
import { startStatusPolling } from "./core/status.js";
import { initTheme } from "./core/theme.js";

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

// Thả ảnh ra ngoài vùng thả sẽ làm trình duyệt mở tệp (file:///…) và dỡ cả ứng dụng. Vì cancel `drop` ở đây cũng hủy cách xử lý mặc định
// của <input type=file>, mọi vùng thả (bước 4, bước 5) phải tự đọc `dataTransfer.files`.
function initDropGuard() {
  const cancel = event => event.preventDefault();
  document.addEventListener("dragover", cancel);
  document.addEventListener("drop", cancel);
}

initTheme();
initDropGuard();
initSkipLink();
initGroupArrowKeys();
initRouter();
startStatusPolling();
