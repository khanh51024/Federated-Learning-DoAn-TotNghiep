// Điều hướng theo hash (#gioi-thieu …): tải lười từng trang, giữ thanh chỉ báo ở menu, cập nhật tiêu đề và nút Trước/Sau.
import { $, $$, byId, notify, prefersReducedMotion, setText } from "./dom.js";
import { enter } from "./motion.js";

export const ROUTES = [
  {
    id: "gioi-thieu", step: 1, group: "Báo cáo", chapter: 1, label: "Giới thiệu",
    title: "Giới thiệu đề tài",
    desc: "",
    load: () => import("../pages/overview.js"),
  },
  {
    id: "du-lieu", step: 2, group: "Báo cáo", chapter: 2, label: "Dữ liệu",
    title: "Dữ liệu và cách chia cho các client",
    desc: "Bộ ảnh dùng để học và kiểm tra, cách chia tập, và vì sao số ảnh mỗi nhóm bệnh khác nhau giữa các client",
    load: () => import("../pages/noniid.js"),
  },
  {
    id: "phuong-phap", step: 3, group: "Báo cáo", chapter: 3, label: "Phương pháp",
    title: "Phương pháp",
    desc: "Ba cách huấn luyện được so sánh, thuật toán FedAvg, thiết lập thí nghiệm và cách đánh giá",
    load: () => import("../pages/methods.js"),
  },
  {
    id: "ket-qua", step: 4, group: "Báo cáo", chapter: 4, label: "Kết quả",
    title: "Kết quả đánh giá mô hình",
    desc: "So sánh độ chính xác của bảy mô hình, kiểm tra trên tập giữ riêng và phân tích các lỗi còn lại",
    load: () => import("../pages/results.js"),
  },
  {
    id: "ket-luan", step: 5, group: "Báo cáo", chapter: 5, label: "Kết luận",
    title: "Kết luận, hạn chế và hướng phát triển",
    desc: "Điều rút ra từ kết quả, những gì nghiên cứu chưa làm được và việc nên làm tiếp",
    load: () => import("../pages/conclusion.js"),
  },
  {
    id: "chan-doan", step: 6, group: "Demo", label: "Chẩn đoán ảnh",
    title: "Chẩn đoán bệnh trên lá cây",
    desc: "Tải ảnh lá cây hoặc chọn ảnh mẫu để xem kết quả dự đoán từ bảy mô hình",
    load: () => import("../pages/diagnosis.js"),
  },
  {
    id: "hoc-lien-ket", step: 7, group: "Demo", label: "Học liên kết",
    title: "Tham gia học liên kết từ trình duyệt",
    desc: "Góp ảnh lá của bạn vào một vòng học liên kết thật, ngay trong trình duyệt",
    load: () => import("../pages/community.js"),
  },
  {
    id: "phu-luc", step: 8, group: "Phụ lục", inNav: false, label: "Phụ lục kỹ thuật",
    title: "Phụ lục kỹ thuật",
    desc: "Kiểm chứng số liệu, nguồn gốc mô hình và kiểm thử hệ thống học liên kết trên máy chủ",
    load: () => import("../pages/appendix.js"),
  },
];

// Địa chỉ cũ (sáu bước) vẫn mở đúng trang: mã tham gia đã in và liên kết đã gửi có dạng /?code=…#huan-luyen.
const ALIASES = {
  "tong-quan": "gioi-thieu",
  "non-iid": "du-lieu",
  "huan-luyen": "hoc-lien-ket",
  "bang-chung": "ket-luan",
};
const resolveId = id => (Object.hasOwn(ALIASES, id) ? ALIASES[id] : id);

/** Các trang nằm trên thanh menu và trong chuỗi Trước/Sau (phụ lục chỉ vào từ chân trang). */
const NAV_ROUTES = ROUTES.filter(r => r.inNav !== false);
const CHAPTERS = ROUTES.filter(r => r.chapter).length;

const DEFAULT_ROUTE = ROUTES[0].id;
const mounted = new Map();
const mounting = new Map(); // id → lời hứa của lượt dựng đang chạy: mở lại trang lúc đang dựng không dựng lần hai (gắn trùng sự kiện)
const listeners = new Set();
let activeId = null;
let token = 0;
let indicator = null;
let firstShow = true; // lần hiển thị đầu: đặt vạch ngay, không trượt từ x = 0

export const routeIndex = id => ROUTES.findIndex(r => r.id === id);

export function currentRouteId() {
  const id = resolveId((location.hash || "").replace(/^#/, ""));
  return ROUTES.some(r => r.id === id) ? id : DEFAULT_ROUTE;
}

export function activeRouteId() {
  return activeId;
}

export function goto(id) {
  if (location.hash !== `#${id}`) location.hash = `#${id}`;
  else show(id);
}

export function onRouteChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// Vạch gạch chân 3 px dưới bước hiện tại: chỉ đổi transform (translateX + scaleX) nên chạy trên compositor.
const INDICATOR_FALLBACK = 100;

function placeIndicator(item, animate) {
  if (!indicator || !item) return;
  if (!animate) indicator.style.transition = "none";
  const base = indicator.offsetWidth || INDICATOR_FALLBACK; // độ rộng cơ sở lấy từ CSS (.nav-indicator), không ghim số trong JS
  indicator.style.transform = `translateX(${item.offsetLeft}px) scaleX(${(item.offsetWidth / base).toFixed(4)})`;
  if (!animate) {
    void indicator.offsetWidth;
    indicator.style.transition = "";
  }
  // màn hẹp: dải bước cuộn ngang; đưa bước hiện tại vào giữa (không dùng scrollIntoView để không cuộn cả trang)
  const nav = item.parentElement;
  if (nav && nav.scrollWidth > nav.clientWidth + 1) {
    const left = item.offsetLeft - (nav.clientWidth - item.offsetWidth) / 2;
    nav.scrollTo({ left: Math.max(0, left), behavior: animate && !prefersReducedMotion() ? "smooth" : "auto" });
  }
}

/** Vị trí của trang trong báo cáo: "Chương 2/5", "Demo 1/2", "Phụ lục". */
export function positionLabel(route) {
  if (route.chapter) return `Chương ${route.chapter}/${CHAPTERS}`;
  if (route.group === "Demo") {
    const demos = ROUTES.filter(r => r.group === "Demo");
    return `Demo ${demos.indexOf(route) + 1}/${demos.length}`;
  }
  return route.group;
}

function renderStepNav(route) {
  const index = NAV_ROUTES.indexOf(route);
  // Phụ lục không nằm trong chuỗi: "Trước" đưa về chương cuối của báo cáo, không có "Sau".
  const prev = index === -1 ? ROUTES.find(r => r.id === "ket-luan") : NAV_ROUTES[index - 1];
  const next = index === -1 ? null : NAV_ROUTES[index + 1];
  const prevEl = byId("step-prev");
  const nextEl = byId("step-next");
  if (prevEl) {
    prevEl.classList.toggle("is-disabled", !prev);
    prevEl.setAttribute("aria-disabled", String(!prev));
    prevEl.tabIndex = prev ? 0 : -1;
    prevEl.href = `#${(prev || route).id}`;
    prevEl.textContent = prev ? `◀ ${prev.label}` : "◀ Trước";
  }
  if (nextEl) {
    const target = next || ROUTES[0];
    nextEl.href = `#${target.id}`;
    nextEl.textContent = next ? `${next.label} ▶` : "Về đầu báo cáo ↺";
    nextEl.hidden = index === -1;
  }
  setText(byId("step-pos"), positionLabel(route));
}

async function show(id) {
  const route = ROUTES.find(r => r.id === id) || ROUTES[0];
  // địa chỉ cũ (#huan-luyen) → địa chỉ mới, không thêm mục vào lịch sử trình duyệt
  const wanted = (location.hash || "").replace(/^#/, "");
  if (Object.hasOwn(ALIASES, wanted)) history.replaceState(null, "", `${location.pathname}${location.search}#${route.id}`);
  const mine = ++token;
  // Sau mỗi `await` người dùng có thể đã sang bước khác: lượt dựng cũ không được chạy onShow, bật lại thanh Trước/Sau hay báo cho người nghe.
  const stale = () => mine !== token;
  // Toast lỗi không tự tắt: lỗi của biểu mẫu ở bước cũ không được theo người dùng sang bước khác và che nội dung
  // (đặc biệt ở 320 px). Dọn ngay khi đổi bước, không đợi trang đích dựng xong. Thông báo trạng thái hệ thống
  // (sticky, vd. mất kết nối) vẫn giữ.
  if (activeId !== null && activeId !== route.id) window.toast?.dismissAll({ keepSticky: true });
  activeId = route.id;
  document.body.dataset.route = route.id;

  // dấu "đã xem" chỉ có nghĩa trong báo cáo (đọc tuần tự); hai trang demo độc lập nhau
  const activeChapter = route.chapter || 0;
  $$(".nav-item").forEach(item => {
    const isActive = item.dataset.route === route.id;
    const itemChapter = ROUTES.find(r => r.id === item.dataset.route)?.chapter || 0;
    item.classList.toggle("active", isActive);
    item.classList.toggle("is-done", itemChapter > 0 && itemChapter < activeChapter);
    if (isActive) {
      item.setAttribute("aria-current", "page");
      placeIndicator(item, firstShow ? false : true);
      firstShow = false;
    } else {
      item.removeAttribute("aria-current");
    }
  });
  if (indicator) indicator.hidden = !$(".nav-item.active"); // phụ lục không có mục trên thanh menu: ẩn vạch chỉ báo
  $$(".tab-pane").forEach(pane => pane.classList.toggle("active", pane.dataset.route === route.id));

  setText(byId("page-title"), route.title);
  setText(byId("page-content-heading"), `Nội dung: ${route.label}`); // tiêu đề h2 ẩn để thứ tự h1 → h2 → h3 không bị nhảy cấp
  setText(byId("route-announcer"), `${positionLabel(route)}: ${route.label}`);
  setText(byId("page-desc"), route.desc);
  setText(byId("step-eyebrow"), route.chapter ? `${positionLabel(route)} · Báo cáo` : positionLabel(route));
  document.body.dataset.group = route.chapter ? "report" : route.group === "Demo" ? "demo" : "appendix"; // CSS: huy hiệu trạng thái máy chủ chỉ hiện ở Demo và Phụ lục
  document.title = `${route.label} — CropFed FL`;
  renderStepNav(route);
  // Thanh Trước/Sau nằm dưới nội dung: trong lúc trang đích chưa dựng xong nó đang ở sát đầu trang, rồi bị đẩy xuống
  // khi nội dung xuất hiện (CLS ~0,1 ở mạng chậm). Ẩn tới khi dựng xong; visibility giữ nguyên chỗ nên không đổi bố cục.
  byId("step-nav")?.classList.add("is-pending");
  window.scrollTo({ top: 0, behavior: "auto" }); // đổi bước: về đầu trang ngay, không cuộn dài làm giật khung hình
  const header = $(".top-header");
  if (header) enter(header, { distance: 6, duration: 280 });

  const pane = byId(`page-${route.id}`);
  pane?.querySelector(":scope > .pane-error")?.remove();
  if (pane) enter(pane, { distance: 6, duration: 240 });
  try {
    let mod = mounted.get(route.id);
    if (!mod) {
      let pending = mounting.get(route.id);
      if (!pending) {
        pending = (async () => {
          const loaded = await route.load();
          await loaded.mount(pane);
          return loaded;
        })();
        mounting.set(route.id, pending);
      }
      try {
        mod = await pending;
        mounted.set(route.id, mod); // chỉ coi là đã dựng khi mount() xong; lỗi thì lần mở sau thử lại
      } finally {
        mounting.delete(route.id);
      }
    }
    if (stale()) return; // trang đã dựng xong (mounted giữ nguyên) nhưng người dùng đã rời đi: lần vào sau sẽ chạy onShow
    if (typeof mod.onShow === "function") await mod.onShow(pane);
  } catch (error) {
    console.error(`[Router] Không dựng được trang ${route.id}:`, error);
    mounted.delete(route.id);
    if (pane && !pane.querySelector(":scope > .pane-error")) {
      // giữ nguyên khung HTML của trang (không xóa) để nút Thử lại dựng lại được
      const box = document.createElement("div");
      box.className = "callout callout-error pane-error";
      box.setAttribute("role", "alert");
      box.innerHTML = '<b>Không dựng được trang này.</b> <button type="button" class="btn btn-secondary btn-sm">Thử lại</button>';
      box.querySelector("button").addEventListener("click", () => show(route.id));
      pane.prepend(box);
    }
    if (!stale()) notify("error", "Không dựng được trang", route.label, { detail: error.message });
  }
  if (stale()) return;
  byId("step-nav")?.classList.remove("is-pending");
  for (const fn of listeners) fn(route);
}

/** Dải bước cuộn ngang: mép nào còn bước bị che thì mờ dần (data-fade), tránh nhãn bị cắt cụt giữa chữ. */
function updateFade() {
  const nav = $(".nav-menu");
  if (!nav) return;
  const start = nav.scrollLeft > 2;
  const end = nav.scrollLeft + nav.clientWidth < nav.scrollWidth - 2;
  const fade = start && end ? "both" : start ? "start" : end ? "end" : "none";
  if (nav.dataset.fade !== fade) nav.dataset.fade = fade;
}

export function initRouter() {
  const menu = $(".nav-menu");
  if (menu) {
    indicator = document.createElement("span");
    indicator.className = "nav-indicator";
    indicator.setAttribute("aria-hidden", "true");
    menu.prepend(indicator);
  }
  const reposition = () => {
    placeIndicator($(".nav-item.active"), false);
    updateFade();
  };
  window.addEventListener("resize", reposition);
  if (menu) menu.addEventListener("scroll", updateFade, { passive: true });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(reposition);
  window.addEventListener("hashchange", () => {
    const hash = resolveId((location.hash || "").replace(/^#/, ""));
    if (hash && !ROUTES.some(r => r.id === hash)) return; // hash không phải trang (vd. #main của skip link): bỏ qua
    show(currentRouteId());
  });
  return show(currentRouteId()).then(reposition);
}
