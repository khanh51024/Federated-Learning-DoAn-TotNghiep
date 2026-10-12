// Chuyển động dùng chung: một chỗ định nghĩa thời lượng/easing, tôn trọng "giảm chuyển động", chỉ animate transform/opacity.
// Quy ước: hover 150–220 ms, hiện khối 280–400 ms, dịch chuyển 6–12 px; mỗi hiệu ứng phải báo một trạng thái.
import { prefersReducedMotion } from "./dom.js";

export const DUR = { press: 120, hover: 180, enter: 320, travel: 900 };
export const EASE = {
  out: "cubic-bezier(0.2, 0.8, 0.2, 1)",
  inOut: "cubic-bezier(0.45, 0, 0.25, 1)",
};

/** Có con trỏ chính xác (chuột/bàn di)? Màn cảm ứng thì tắt mọi hiệu ứng bám chuột. */
export function finePointer() {
  return !!(window.matchMedia && window.matchMedia("(hover: hover) and (pointer: fine)").matches);
}

/** WAAPI có kiểm tra: không chạy khi giảm chuyển động (trả null để nơi gọi dùng trạng thái cuối). */
export function play(element, keyframes, options = {}) {
  if (!element || typeof element.animate !== "function" || prefersReducedMotion()) return null;
  return element.animate(keyframes, { duration: DUR.enter, easing: EASE.out, fill: "backwards", ...options });
}

/** Hiện một khối: mờ → rõ kèm dịch lên vài px. */
export function enter(element, { distance = 8, delay = 0, duration = DUR.enter } = {}) {
  return play(
    element,
    [{ opacity: 0, transform: `translateY(${distance}px)` }, { opacity: 1, transform: "translateY(0)" }],
    { delay, duration },
  );
}

/** Hủy mọi hiệu ứng đang chạy bên trong một vùng (trạng thái mặc định của CSS là hình hoàn chỉnh). */
export function cancelWithin(root) {
  if (!root || typeof root.getAnimations !== "function") return;
  for (const animation of root.getAnimations({ subtree: true })) animation.cancel();
}

/** Điểm trên đường Bézier bậc hai p0 → p1 với điều khiển c, n+1 mẫu. */
export function quadPoints(p0, c, p1, n = 16) {
  const out = [];
  for (let i = 0; i <= n; i += 1) {
    const t = i / n;
    const u = 1 - t;
    out.push({
      x: u * u * p0.x + 2 * u * t * c.x + t * t * p1.x,
      y: u * u * p0.y + 2 * u * t * c.y + t * t * p1.y,
    });
  }
  return out;
}

/** Chấm đi dọc theo các điểm đã lấy mẫu: keyframe translate3d chạy trên compositor, luồng chính không làm gì mỗi khung. */
export function travel(dot, points, { scale = 1, radius = 4, delay = 0, duration = DUR.travel } = {}) {
  if (!dot || !points.length) return null;
  const last = points.length - 1;
  const keyframes = points.map((point, index) => ({
    transform: `translate3d(${(point.x * scale - radius).toFixed(1)}px, ${(point.y * scale - radius).toFixed(1)}px, 0)`,
    opacity: index === 0 || index === last ? 0 : 1,
    offset: index / last,
  }));
  return play(dot, keyframes, { delay, duration, easing: EASE.inOut, fill: "backwards" });
}

/** Ánh sáng bám con trỏ trong một vùng: tối đa một rAF mỗi khung, di chuyển bằng translate3d; tắt trên cảm ứng/giảm chuyển động. */
export function followPointer(host, glow) {
  if (!host || !glow || !finePointer() || prefersReducedMotion()) return () => {};
  let frame = 0;
  let x = 0;
  let y = 0;
  let box = null;
  const half = glow.offsetWidth / 2 || 210;

  const paint = () => {
    frame = 0;
    glow.style.transform = `translate3d(${(x - half).toFixed(1)}px, ${(y - half).toFixed(1)}px, 0)`;
  };
  const onEnter = event => {
    if (event.pointerType && event.pointerType !== "mouse") return;
    box = host.getBoundingClientRect();
    glow.classList.add("is-glowing");
  };
  const onMove = event => {
    if (event.pointerType && event.pointerType !== "mouse") return;
    if (!box) box = host.getBoundingClientRect();
    x = event.clientX - box.left;
    y = event.clientY - box.top;
    if (!frame) frame = requestAnimationFrame(paint);
  };
  const onLeave = () => {
    glow.classList.remove("is-glowing");
    box = null;
  };
  const invalidate = () => { box = null; };

  host.addEventListener("pointerenter", onEnter);
  host.addEventListener("pointermove", onMove, { passive: true });
  host.addEventListener("pointerleave", onLeave);
  window.addEventListener("scroll", invalidate, { passive: true });
  window.addEventListener("resize", invalidate, { passive: true });
  return () => {
    host.removeEventListener("pointerenter", onEnter);
    host.removeEventListener("pointermove", onMove);
    host.removeEventListener("pointerleave", onLeave);
    window.removeEventListener("scroll", invalidate);
    window.removeEventListener("resize", invalidate);
    if (frame) cancelAnimationFrame(frame);
  };
}

/** Gọi lại khi người dùng đổi tùy chọn "giảm chuyển động" của hệ điều hành lúc trang đang mở. */
export function onMotionPrefChange(callback) {
  if (!window.matchMedia) return () => {};
  const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  const handler = () => callback(mq.matches);
  mq.addEventListener("change", handler);
  return () => mq.removeEventListener("change", handler);
}

// Một listener thụ động để :active hoạt động cả trên iOS Safari (phản hồi nhấn của nút/thẻ).
document.addEventListener("touchstart", () => {}, { passive: true });
