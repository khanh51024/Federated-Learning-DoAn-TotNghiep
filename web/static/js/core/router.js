// Điều hướng theo hash (#tong-quan …): tải lười từng trang, giữ thanh chỉ báo ở menu, cập nhật tiêu đề và nút Trước/Sau.
import { $, $$, byId, notify, prefersReducedMotion, setText } from "./dom.js";

export const ROUTES = [
  {
    id: "tong-quan", step: 1, group: "Trình bày", label: "Tổng quan",
    title: "Tổng quan: chia sẻ tri thức mà không chia sẻ ảnh",
    desc: "Câu hỏi nghiên cứu, ba cách huấn luyện và kết quả chính trong một màn hình",
    load: () => import("../pages/overview.js"),
  },
  {
    id: "non-iid", step: 2, group: "Trình bày", label: "Dữ liệu non-IID",
    title: "Dữ liệu mỗi nơi lệch nhau thế nào",
    desc: "Năm client, 38 lớp bệnh — vì sao mô hình học riêng không nhận ra bệnh mà nơi mình chưa từng thấy",
    load: () => import("../pages/noniid.js"),
  },
  {
    id: "ket-qua", step: 3, group: "Trình bày", label: "Kết quả chi tiết",
    title: "Kết quả chi tiết",
    desc: "Bảy mô hình trên cùng validation: độ chính xác, PlantDoc theo lớp và độ tin cậy của xác suất",
    load: () => import("../pages/results.js"),
  },
  {
    id: "chan-doan", step: 4, group: "Demo trực tiếp", label: "Chẩn đoán bệnh lá",
    title: "Chẩn đoán bệnh lá cây",
    desc: "Một ảnh, bảy mô hình: mô hình nào nhận ra bệnh, mô hình nào không — và vì sao",
    load: () => import("../pages/diagnosis.js"),
  },
  {
    id: "huan-luyen", step: 5, group: "Demo trực tiếp", label: "Huấn luyện thử",
    title: "Huấn luyện thử với Flower",
    desc: "Một server và năm client chạy thật trên localhost; kiểm tra luồng FedAvg, không phải kết quả khoa học",
    load: () => import("../pages/training.js"),
  },
  {
    id: "bang-chung", step: 6, group: "Minh bạch", label: "Bằng chứng & hạn chế",
    title: "Bằng chứng và hạn chế",
    desc: "Kiểm hash, tái đánh giá, audit rò rỉ dữ liệu — và những gì chưa làm được",
    load: () => import("../pages/evidence.js"),
  },
];

const DEFAULT_ROUTE = ROUTES[0].id;
const mounted = new Map();
const listeners = new Set();
let activeId = null;
let token = 0;
let indicator = null;

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

function placeIndicator(item, animate) {
  if (!indicator || !item) return;
  if (!animate) indicator.style.transition = "none";
  indicator.style.height = `${item.offsetHeight}px`;
  indicator.style.transform = `translateY(${item.offsetTop}px)`;
  if (!animate) {
    void indicator.offsetWidth;
    indicator.style.transition = "";
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

  $$(".nav-item").forEach(item => {
    const isActive = item.dataset.route === route.id;
    item.classList.toggle("active", isActive);
    if (isActive) {
      item.setAttribute("aria-current", "page");
      placeIndicator(item, true);
    } else {
      item.removeAttribute("aria-current");
    }
  });
  $$(".tab-pane").forEach(pane => pane.classList.toggle("active", pane.dataset.route === route.id));
  $$(".present-step").forEach(link => {
    const on = link.dataset.route === route.id;
    link.classList.toggle("is-active", on);
    if (on) link.setAttribute("aria-current", "step");
    else link.removeAttribute("aria-current");
  });

  setText(byId("page-title"), route.title);
  setText(byId("route-announcer"), `Bước ${route.step} trên ${ROUTES.length}: ${route.label}`);
  setText(byId("page-desc"), route.desc);
  setText(byId("step-eyebrow"), `Bước ${route.step}/${ROUTES.length} · ${route.group}`);
  document.title = `${route.label} — CropFed FL`;
  renderStepNav(route);
  window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });

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

export function initRouter() {
  const menu = $(".nav-menu");
  if (menu) {
    indicator = document.createElement("span");
    indicator.className = "nav-indicator";
    indicator.setAttribute("aria-hidden", "true");
    menu.prepend(indicator);
    menu.classList.add("has-indicator");
  }
  const reposition = () => placeIndicator($(".nav-item.active"), false);
  window.addEventListener("resize", reposition);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(reposition);
  window.addEventListener("hashchange", () => {
    const hash = (location.hash || "").replace(/^#/, "");
    if (hash && !ROUTES.some(r => r.id === hash)) return; // hash không phải bước (vd. #main của skip link): bỏ qua
    show(currentRouteId());
  });
  return show(currentRouteId()).then(reposition);
}
