// web/static/js/toast.js: thông báo dạng toast thay cho alert() của trình duyệt.
// API: toast.success|error|warning|info(title, message?, { detail, duration })  ->  { close() }
// - Không dùng innerHTML với nội dung truyền vào (thông điệp lỗi có thể chứa chuỗi từ server).
// - Lỗi không tự tắt (duration = 0); các loại khác tự tắt sau 5 giây, tạm dừng khi rê chuột/focus.
// - Chỉ animate transform/opacity (xem .toast trong style.css); giảm chuyển động do CSS xử lý.
(function () {
  "use strict";

  var MAX_VISIBLE = 4;
  var DEFAULT_DURATION = { success: 4000, info: 5000, warning: 7000, error: 0 };
  var LEAVE_MS = 220;

  var ICON_PATHS = {
    success: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    warning: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    close: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  };

  var region = null;
  var live = [];

  function svg(name, cls) {
    var ns = "http://www.w3.org/2000/svg";
    var el = document.createElementNS(ns, "svg");
    el.setAttribute("viewBox", "0 0 24 24");
    el.setAttribute("class", "svg-icon " + (cls || "icon-md"));
    el.setAttribute("aria-hidden", "true");
    el.innerHTML = ICON_PATHS[name]; // chuỗi hằng của file này, không phải dữ liệu người dùng
    return el;
  }

  function ensureRegion() {
    if (region && document.body.contains(region)) return region;
    region = document.createElement("div");
    region.className = "toast-region";
    region.setAttribute("role", "region");
    region.setAttribute("aria-label", "Thông báo");
    document.body.appendChild(region);
    return region;
  }

  function remove(entry, immediately) {
    var idx = live.indexOf(entry);
    if (idx === -1) return;
    live.splice(idx, 1);
    clearTimeout(entry.timer);
    var done = function () { if (entry.el.parentNode) entry.el.parentNode.removeChild(entry.el); };
    if (immediately) { done(); return; }
    entry.el.classList.add("is-leaving");
    setTimeout(done, LEAVE_MS + 30);
  }

  function arm(entry) {
    clearTimeout(entry.timer);
    if (!entry.duration) return;
    entry.startedAt = Date.now();
    entry.timer = setTimeout(function () { remove(entry); }, entry.remaining);
  }

  function restartProgress(entry) {
    var bar = entry.el.querySelector(".toast-progress");
    if (!bar) return;
    bar.style.animation = "none";
    void bar.offsetWidth; // một lần reflow để khởi động lại animation khi toast trùng lặp được làm mới
    bar.style.animation = "";
  }

  function show(type, title, message, options) {
    options = options || {};
    var duration = options.duration !== undefined ? options.duration : DEFAULT_DURATION[type];
    var key = [type, title, message || "", options.detail || ""].join("\u0001");

    // Cùng nội dung đang hiện: làm mới thay vì xếp chồng (tránh spam khi mất kết nối liên tục).
    for (var i = 0; i < live.length; i++) {
      if (live[i].key === key) {
        var existing = live[i];
        existing.remaining = existing.duration;
        restartProgress(existing);
        // Đang được rê chuột/focus (timer đã tạm dừng) thì giữ nguyên trạng thái tạm dừng.
        if (!existing.el.matches(":hover, :focus-within")) arm(existing);
        return { close: function () { remove(existing); } };
      }
    }

    var host = ensureRegion();
    var el = document.createElement("div");
    el.className = "toast toast-" + type;
    el.setAttribute("role", type === "error" ? "alert" : "status");
    el.style.setProperty("--toast-duration", duration + "ms");

    var icon = document.createElement("div");
    icon.className = "toast-icon";
    icon.appendChild(svg(type, "icon-lg"));

    var body = document.createElement("div");
    var titleEl = document.createElement("div");
    titleEl.className = "toast-title";
    titleEl.textContent = title;
    body.appendChild(titleEl);
    if (message) {
      var text = document.createElement("div");
      text.className = "toast-text";
      text.textContent = message;
      body.appendChild(text);
    }
    if (options.detail) {
      var detail = document.createElement("div");
      detail.className = "toast-detail";
      detail.textContent = options.detail;
      body.appendChild(detail);
    }

    var close = document.createElement("button");
    close.type = "button";
    close.className = "toast-close";
    close.setAttribute("aria-label", "Đóng thông báo");
    close.appendChild(svg("close", "icon-sm"));

    el.appendChild(icon);
    el.appendChild(body);
    el.appendChild(close);
    if (duration) {
      var bar = document.createElement("div");
      bar.className = "toast-progress";
      el.appendChild(bar);
    }

    var entry = { el: el, key: key, duration: duration, remaining: duration, timer: null, startedAt: 0 };
    close.addEventListener("click", function () { remove(entry); });

    if (duration) {
      var pause = function () {
        if (!entry.timer) return;
        clearTimeout(entry.timer);
        entry.timer = null;
        entry.remaining = Math.max(400, entry.remaining - (Date.now() - entry.startedAt));
      };
      var resume = function () { if (!entry.timer) arm(entry); };
      el.addEventListener("mouseenter", pause);
      el.addEventListener("mouseleave", resume);
      el.addEventListener("focusin", pause);
      el.addEventListener("focusout", resume);
    }

    host.appendChild(el);
    live.push(entry);
    // Quá giới hạn: bỏ toast tự tắt cũ nhất trước; chỉ khi toàn lỗi bền vững mới bỏ cái cũ nhất.
    while (live.length > MAX_VISIBLE) {
      var victim = live.find(function (item) { return item.duration > 0; }) || live[0];
      remove(victim, true);
    }
    arm(entry);
    return { close: function () { remove(entry); } };
  }

  window.toast = {
    success: function (title, message, options) { return show("success", title, message, options); },
    error: function (title, message, options) { return show("error", title, message, options); },
    warning: function (title, message, options) { return show("warning", title, message, options); },
    info: function (title, message, options) { return show("info", title, message, options); },
    dismissAll: function () { live.slice().forEach(function (entry) { remove(entry, true); }); },
  };
})();
