// Sơ đồ mạng học liên kết dùng chung cho chương Giới thiệu (minh họa) và Phụ lục kỹ thuật (chạy theo trạng thái thật).
// Hình học nằm trong hệ 640×360; dây là đường Bézier bậc hai vẽ bằng SVG tĩnh, còn mọi thứ CHUYỂN ĐỘNG là phần tử HTML
// di chuyển bằng transform (animate phần tử con của SVG không chạy trên compositor). Dây và quỹ đạo chấm dùng cùng hằng số.
import { esc } from "./dom.js";
import { quadPoints, travel, play, EASE } from "./motion.js";

export const VIEW = { w: 640, h: 360 };
const CLIENT_X = 90;
const CLIENT_W = 120;
const SERVER = { x: 400, y: 180, w: 130 };
const MODEL = { x: 562, y: 180, w: 124 };

/** Hình học của n client: điểm đầu/điểm điều khiển/điểm cuối của từng dây về máy chủ. */
export function geometry(count = 5) {
  const top = 40;
  const gap = count > 1 ? (VIEW.h - 2 * top) / (count - 1) : 0;
  const clients = Array.from({ length: count }, (_unused, i) => {
    const y = count > 1 ? top + gap * i : VIEW.h / 2;
    const p0 = { x: CLIENT_X + CLIENT_W / 2, y };
    const p1 = { x: SERVER.x - SERVER.w / 2, y: SERVER.y };
    const c = { x: p0.x + (p1.x - p0.x) * 0.62, y };
    return { x: CLIENT_X, y, p0, c, p1 };
  });
  return { clients, server: SERVER, model: MODEL, modelWire: { p0: { x: SERVER.x + SERVER.w / 2, y: SERVER.y }, p1: { x: MODEL.x - MODEL.w / 2, y: MODEL.y } } };
}

function wirePath(wire) {
  return `M${wire.p0.x} ${wire.p0.y} Q${wire.c.x} ${wire.c.y} ${wire.p1.x} ${wire.p1.y}`;
}

/** "Client 01" → từ "Client" bọc riêng để khung hẹp (≤ 480 px) chỉ còn số, vừa thẻ và không bị dây cắt ngang. */
function clientLabel(label) {
  const match = /^(Client)\s+(.+)$/.exec(String(label));
  return match ? `<span class="fl-word">${esc(match[1])} </span>${esc(match[2])}` : esc(label);
}

/** Markup của sơ đồ. `labels`: nhãn client; `variant`: "hero" (có mô hình chung, chấm) hoặc "live" (huấn luyện thật). */
export function diagramHtml({ count = 5, labels = [], variant = "hero", serverLabel = "FedAvg", serverSub = "gộp trọng số", modelLabel = "Mô hình chung" } = {}) {
  const g = geometry(count);
  const dim = g.clients.map(c => `<path class="fl-wire" d="${wirePath(c)}"/>`).join("")
    + (variant === "hero" ? `<path class="fl-wire" d="M${g.modelWire.p0.x} ${g.modelWire.p0.y} L${g.modelWire.p1.x} ${g.modelWire.p1.y}"/>` : "");
  const lit = g.clients.map((c, i) => `
      <div class="fl-wire-lit" data-i="${i}" aria-hidden="true">
        <svg viewBox="0 0 ${VIEW.w} ${VIEW.h}" preserveAspectRatio="none"><path d="${wirePath(c)}"/></svg>
      </div>`).join("");
  const modelLit = `
      <div class="fl-wire-lit fl-wire-model" aria-hidden="true">
        <svg viewBox="0 0 ${VIEW.w} ${VIEW.h}" preserveAspectRatio="none"><path d="M${g.modelWire.p0.x} ${g.modelWire.p0.y} L${g.modelWire.p1.x} ${g.modelWire.p1.y}"/></svg>
      </div>`;
  const clients = g.clients.map((c, i) => `
      <div class="fl-node fl-client" data-client="${i}" data-style="--x:${c.x};--y:${c.y}">
        <div class="fl-node-card"><span class="fl-node-head"><span class="topo-dot" aria-hidden="true"></span><b>${clientLabel(labels[i] || `Client ${String(i).padStart(2, "0")}`)}</b></span><small class="fl-node-sub">giữ ảnh riêng</small></div>
      </div>`).join("");
  const dots = variant === "hero"
    ? Array.from({ length: count * 2 }, () => '<span class="fl-dot" aria-hidden="true"></span>').join("")
    : Array.from({ length: count }, () => '<span class="fl-dot" aria-hidden="true"></span>').join("");
  return `
    <div class="fl-stage" data-variant="${variant}">
      <svg class="fl-wires" viewBox="0 0 ${VIEW.w} ${VIEW.h}" preserveAspectRatio="none" aria-hidden="true">${dim}</svg>
      ${lit}${variant === "hero" ? modelLit : ""}
      ${clients}
      <div class="fl-node fl-server" id="${variant === "live" ? "topo-server" : "fl-server"}" data-style="--x:${g.server.x};--y:${g.server.y}">
        <div class="fl-node-card"><b>${esc(serverLabel)}</b>${variant === "live"
          ? `<small class="fl-state" id="topo-server-sub">${esc(serverSub)}</small>`
          : `<small class="fl-node-sub">${esc(serverSub)}</small>`}<span class="topo-ring" aria-hidden="true"></span></div>
      </div>
      ${variant === "hero" ? `
      <div class="fl-node fl-model" data-style="--x:${g.model.x};--y:${g.model.y}">
        <div class="fl-node-card"><b>${esc(modelLabel)}</b><small class="fl-node-sub">gửi lại cho mọi client</small></div>
      </div>` : ""}
      <div class="fl-dots" aria-hidden="true">${dots}</div>
    </div>`;
}

function stageScale(stage) {
  return stage.clientWidth / VIEW.w || 1;
}

/** Chạy một đợt "gửi trọng số" từ các client (theo chỉ số) về máy chủ; trả về thời lượng ước tính (ms).
 *  `reverse`: chiều ngược lại (máy chủ → client), dùng chấm dự phòng của hero (nửa sau) để không cắt đợt đi. */
export function pulse(stage, indices, { delay = 0, perClientGap = 110, duration = 900, reverse = false } = {}) {
  if (!stage) return 0;
  const clients = stage.querySelectorAll(".fl-client").length || 5;
  const g = geometry(clients);
  const dots = stage.querySelectorAll(".fl-dot");
  const scale = stageScale(stage);
  indices.forEach((index, order) => {
    const wire = g.clients[index];
    const dot = dots[reverse && dots.length >= clients * 2 ? index + clients : index];
    if (!wire || !dot) return;
    const points = quadPoints(wire.p0, wire.c, wire.p1, 16);
    travel(dot, reverse ? points.reverse() : points, { scale, delay: delay + order * perClientGap, duration });
  });
  return delay + indices.length * perClientGap + duration;
}

/** Nhịp "đập" của máy chủ khi vừa nhận xong trọng số. */
export function beat(serverCard, { delay = 0 } = {}) {
  return play(serverCard, [{ transform: "scale(1)" }, { transform: "scale(1.06)" }, { transform: "scale(1)" }], { delay, duration: 360, easing: EASE.out });
}
