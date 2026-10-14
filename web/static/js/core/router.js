// Điều hướng theo hash (#tong-quan …): tải lười từng trang, giữ thanh chỉ báo ở menu, cập nhật tiêu đề và nút Trước/Sau.
import { $, $$, byId, notify, prefersReducedMotion, setText } from "./dom.js";
import { enter } from "./motion.js";

export const ROUTES = [
  {
    id: "tong-quan", step: 1, group: "Trình bày", label: "Tổng quan",
    title: "Tổng quan đồ án",
    desc: "",
    load: () => import("../pages/overview.js"),
  },
  {
    id: "non-iid", step: 2, group: "Trình bày", label: "Phân bố dữ liệu",
    title: "Phân bố dữ liệu giữa các client",
    desc: "Dữ liệu được chia cho năm client giả lập. Số ảnh và nhóm bệnh ở mỗi client không đồng đều",
    load: () => import("../pages/noniid.js"),
  },
  {
    id: "ket-qua", step: 3, group: "Trình bày", label: "Kết quả & so sánh",
    title: "Kết quả đánh giá mô hình",
    desc: "So sánh độ chính xác của bảy mô hình, kiểm tra trên tập giữ riêng và phân tích các lỗi còn lại",
    load: () => import("../pages/results.js"),
  },
  {
    id: "chan-doan", step: 4, group: "Demo trực tiếp", label: "Chẩn đoán bệnh lá",
    title: "Chẩn đoán bệnh trên lá cây",
    desc: "Tải ảnh lá cây hoặc chọn ảnh mẫu để xem kết quả dự đoán từ bảy mô hình",
    load: () => import("../pages/diagnosis.js"),
  },
  {
    id: "huan-luyen", step: 5, group: "Demo trực tiếp", label: "Huấn luyện thử",
    title: "Huấn luyện và mô phỏng FedAvg",
    desc: "Theo dõi quá trình huấn luyện liên kết giữa các client và máy chủ. Có thể tham gia từ trình duyệt hoặc chạy mô phỏng trên máy chủ",
    load: () => import("../pages/training.js"),
  },
  {
    id: "bang-chung", step: 6, group: "Minh bạch", label: "Kiểm chứng & giới hạn",
    title: "Kiểm chứng và giới hạn nghiên cứu",
    desc: "Tổng hợp các kết quả đã kiểm tra, giới hạn của nghiên cứu và việc còn lại",
    load: () => import("../pages/evidence.js"),
  },
];

const DEFAULT_ROUTE = ROUTES[0].id;
const mounted = new Map();
const listeners = new Set();
let activeId = null;
let token = 0;
let indicator = null;
let firstShow = true; // lần hiển thị đầu: đặt vạch ngay, không trượt từ x = 0

export const routeIndex = id => ROUTES.findIndex(r => r.id === id);

export function currentRouteId() {
  const id = (location.hash || "").replace(/^#/, "");
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

function renderStepNav(route) {
  const index = routeIndex(route.id);
  const prev = ROUTES[index - 1];
  const next = ROUTES[index + 1];
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
    nextEl.textContent = next ? `${next.label} ▶` : "Về tổng quan ↺";
  }
  setText(byId("step-pos"), `Bước ${route.step}/${ROUTES.length}`);
}

async function show(id) {
  const route = ROUTES.find(r => r.id === id) || ROUTES[0];
  const mine = ++token;
  // Toast lỗi không tự tắt: lỗi của biểu mẫu ở bước cũ không được theo người dùng sang bước khác và che nội dung
  // (đặc biệt ở 320 px). Dọn ngay khi đổi bước, không đợi trang đích dựng xong. Thông báo trạng thái hệ thống
  // (sticky, vd. mất kết nối) vẫn giữ.
  if (activeId !== null && activeId !== route.id) window.toast?.dismissAll({ keepSticky: true });
  activeId = route.id;
  document.body.dataset.route = route.id;

  const activeIndex = routeIndex(route.id);
  $$(".nav-item").forEach(item => {
    const isActive = item.dataset.route === route.id;
    item.classList.toggle("active", isActive);
    item.classList.toggle("is-done", routeIndex(item.dataset.route) < activeIndex);
    if (isActive) {
      item.setAttribute("aria-current", "page");
      placeIndicator(item, firstShow ? false : true);
    } else {
      item.removeAttribute("aria-current");
    }
  });
  firstShow = false;
  $$(".tab-pane").forEach(pane => pane.classList.toggle("active", pane.dataset.route === route.id));

  setText(byId("page-title"), route.title);
  setText(byId("route-announcer"), `Bước ${route.step} trên ${ROUTES.length}: ${route.label}`);
  setText(byId("page-desc"), route.desc);
  setText(byId("step-eyebrow"), `Bước ${route.step}/${ROUTES.length} · ${route.group}`);
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
      mod = await route.load();
      await mod.mount(pane);
      mounted.set(route.id, mod); // chỉ coi là đã dựng khi mount() xong; lỗi thì lần mở sau thử lại
    }
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
    notify("error", "Không dựng được trang", route.label, { detail: error.message });
  }
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
    const hash = (location.hash || "").replace(/^#/, "");
    if (hash && !ROUTES.some(r => r.id === hash)) return; // hash không phải bước (vd. #main của skip link): bỏ qua
    show(currentRouteId());
  });
  return show(currentRouteId()).then(reposition);
}
