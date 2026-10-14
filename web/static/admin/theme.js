// Nền sáng/tối của trang quản trị: dùng chung khóa lưu với trang chính. Tệp cổ điển, chạy trước khi vẽ để không nháy nền.
(function () {
  var KEY = "cropfed.theme";
  var theme = null;
  try { theme = localStorage.getItem(KEY); } catch (e) { /* bị chặn: theo hệ điều hành */ }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.setAttribute("data-theme", theme);

  window.addEventListener("DOMContentLoaded", function () {
    var button = document.getElementById("btn-theme");
    var label = document.getElementById("btn-theme-label");
    if (!button) return;
    function sync() {
      if (label) label.textContent = document.documentElement.getAttribute("data-theme") === "dark" ? "Nền sáng" : "Nền tối";
    }
    sync();
    button.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (e) { /* bỏ qua */ }
      sync();
    });
  });
})();
