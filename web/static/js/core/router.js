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
    id: "non-iid", step: 2, group: "Trình bày", label: "Dữ liệu phân tán",
    title: "Vì sao mỗi nơi tự học là không đủ?",
    desc: "Dữ liệu chia cho năm client và rất lệch: nhiều nhóm gần như chỉ nằm ở một nơi (dữ liệu non-IID)",
    load: () => import("../pages/noniid.js"),
  },
  {
    id: "ket-qua", step: 3, group: "Trình bày", label: "Kết quả & so sánh",
    title: "Kết quả có đáng tin không?",
    desc: "Độ chính xác của bảy mô hình, kiểm tra lại trên tập giữ riêng và phân tích những chỗ mô hình còn sai",
    load: () => import("../pages/results.js"),
  },
  {
    id: "chan-doan", step: 4, group: "Demo trực tiếp", label: "Chẩn đoán bệnh lá",
    title: "Thử chẩn đoán một chiếc lá",
    desc: "Đưa một ảnh qua cả bảy mô hình để thấy mô hình nào nhận ra bệnh, mô hình nào không — và vì sao",
    load: () => import("../pages/diagnosis.js"),
  },
  {
    id: "huan-luyen", step: 5, group: "Demo trực tiếp", label: "Huấn luyện thử",
    title: "Xem FedAvg chạy từng vòng",
    desc: "Một máy chủ và năm client chạy thật trên máy này với ảnh tổng hợp, để minh họa quy trình học liên kết",
    load: () => import("../pages/training.js"),
  },
  {
    id: "bang-chung", step: 6, group: "Minh bạch", label: "Bằng chứng & hạn chế",
    title: "Bằng chứng, hạn chế và hướng phát triển",
    desc: "Những gì đã được kiểm chứng, những điểm còn hạn chế và việc cần làm tiếp theo",
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
  activeId = route.id;

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
  window.scrollTo({ top: 0, behavior: "auto" }); // đổi bước: về đầu trang ngay, không cuộn dài làm giật khung hình
  const header = $(".top-header");
  if (header) enter(header, { distance: 6, duration: 280 });

  const pane = byId(`page-${route.id}`);
  pane?.querySelector(":scope > .pane-error")?.remove();
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
