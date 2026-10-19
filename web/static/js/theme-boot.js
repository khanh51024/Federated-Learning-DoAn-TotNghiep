// Đặt nền sáng/tối TRƯỚC khi trình duyệt vẽ lần đầu (không nháy). Tệp cổ điển, không phải module; nạp đồng bộ ở <head>.
// Trước đây là <script> nội tuyến; chuyển ra tệp để CSP của trang chính không cần 'unsafe-inline' cho script.
// Nút đổi nền do core/theme.js đảm nhiệm; tệp này chỉ khởi tạo.
(function () {
  var theme = null;
  try { theme = localStorage.getItem("cropfed.theme"); } catch (e) { /* bị chặn: theo hệ điều hành */ }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.setAttribute("data-theme", theme);
})();
