// Sơ đồ động của một vòng trên trang Đo qua mạng: dữ liệu đi xuống, đi lên và mỗi bên làm gì, đúng thứ tự trong server_app.py và client_app.py.
// Hình minh họa thứ tự, không chạy theo thời gian thật; các số trong lời giải thích lấy từ vòng đang xem của lần đo.
import { esc, fmtInt, fmtNum, prefersReducedMotion } from "./dom.js";

const STEP_MS = 4200;
const AWAY_CHECK_MS = 1000;
const ROWS = [65, 165, 265];     // tâm ba thẻ client trong hệ 960×330
const LANE_GAP = 13;
const PACKET = { width: 140, height: 24, from: 252, to: 558 };
const FALLBACK_NOT_SENT = "ảnh, độ chính xác, loss, thời gian học, độ trễ và tốc độ mạng";

const megabytes = bytes => (typeof bytes === "number" ? `${fmtNum(bytes / 1000000, 2)} MB` : "");
const hash = value => (value ? `<span class="mono">${esc(String(value).slice(0, 8))}</span>` : "—");

const STEPS = [
  {
    tag: "Xuống", title: "Máy chủ gửi mô hình chung xuống", motion: "down", server: "Đang gửi", client: "Đang nhận",
    packet: facts => `Mô hình ${megabytes(facts.client?.payload_bytes_down)}`.trim(),
    lines: facts => [
      "Mỗi client nhận cùng một mô hình, kèm số vòng, số epoch, seed và mã băm của mô hình.",
      facts.row ? `Vòng ${esc(facts.row.round)}: mô hình ${hash(facts.row.parent_hash)}, mỗi client nhận ${esc(megabytes(facts.client?.payload_bytes_down) || "—")}.` : "",
    ],
  },
  {
    tag: "Hai chiều", title: "Máy chủ tự đo độ trễ và tốc độ mạng", motion: "both", server: "Đang bấm giờ", client: "Làm theo lệnh",
    packet: () => "Khối byte đo",
    lines: facts => [
      "Client nối vào cổng đo rồi làm theo lệnh của máy chủ. Máy chủ gửi, nhận khối byte và bấm giờ trên đồng hồ của mình, nên client không tự khai được số này.",
      facts.client ? `Client ${esc(facts.client.client_id)}: độ trễ ${esc(fmtNum(facts.client.rtt_ms, 1))} ms, lên ${esc(fmtNum(facts.client.up_mb_s, 2))} MiB/s, xuống ${esc(fmtNum(facts.client.down_mb_s, 2))} MiB/s.` : "",
    ],
  },
  {
    tag: "Trên client", title: "Client học trên ảnh của mình", motion: "learn", server: "Đang chờ", client: "Đang học",
    packet: () => "",
    lines: facts => [
      "Không có gì đi trên đường truyền ở bước này. Ảnh nằm yên trên máy client.",
      facts.client ? `Client ${esc(facts.client.client_id)} học ${esc(fmtInt(facts.client.num_examples))} ảnh, rồi tính mã băm và độ biến thiên của trọng số.` : "Học xong, client tính mã băm và độ biến thiên của trọng số.",
    ],
  },
  {
    tag: "Lên", title: "Client gửi trọng số lên", motion: "up", server: "Đang nhận", client: "Đang gửi",
    packet: facts => `Trọng số ${megabytes(facts.client?.payload_bytes_up)}`.trim(),
    lines: facts => [
      "Gửi: trọng số, số ảnh, độ biến thiên, mã băm trọng số, mã băm mô hình vừa nhận, cùng các mã nhận dạng vòng và dữ liệu.",
      `<b>Không gửi:</b> ${esc(FALLBACK_NOT_SENT)}.`,
      facts.client ? `Client ${esc(facts.client.client_id)} gửi ${esc(megabytes(facts.client.payload_bytes_up) || "—")}, mã băm ${hash(facts.client.client_hash)}.` : "",
    ],
  },
  {
    tag: "Trên máy chủ", title: "Máy chủ kiểm những gì nhận được", motion: "check", server: "Đang kiểm", client: "Đang chờ",
    packet: () => "",
    lines: facts => [
      "Máy chủ băm lại trọng số và tự tính lại độ biến thiên rồi so với số client khai, đồng thời so vòng, seed và danh sách ảnh. Sai một mục là client đó bị loại.",
      facts.client ? `Client ${esc(facts.client.client_id)}: mã băm ${hash(facts.client.client_hash)} và ${hash(facts.client.server_hash)} ${facts.client.hash_ok ? "khớp" : "lệch"}; độ biến thiên khai ${esc(fmtNum(facts.client.update_norm_reported, 3))}, máy chủ tính ${esc(fmtNum(facts.client.update_norm_server, 3))}.` : "",
      facts.tamper ? `Lần thử cố tình sai đã lưu: client ${esc(facts.tamper.client_id)} đổi một con số sau khi đã băm, giống như dữ liệu bị đổi trên đường truyền. Hai mã băm khác nhau nên client bị loại và máy chủ dừng lần đo đó.` : "",
    ],
  },
  {
    tag: "Trên máy chủ", title: "Máy chủ gộp thành mô hình mới", motion: "merge", server: "Đang gộp", client: "Đang chờ",
    packet: () => "",
    lines: facts => [
      "Mô hình mới là trung bình trọng số của các client được nhận, theo tỉ lệ số ảnh. Máy chủ tự chấm trên ảnh riêng mà client không có, rồi vòng sau bắt đầu lại từ bước đầu với mô hình mới.",
      facts.row ? `Vòng ${esc(facts.row.round)}: gộp ${esc(facts.row.accepted ?? "—")} client, mô hình mới ${hash(facts.row.global_hash)}${facts.row.accuracy === null ? "" : `, máy chủ chấm trên tập val được ${esc(fmtNum(facts.row.accuracy * 100, 2))}%`}.` : "",
    ],
  },
];

let host = null;
let index = 0;
let playing = false;
let timer = 0;
let startedAt = 0;
let remaining = STEP_MS;
let facts = { row: null, client: null, tamper: null, numClients: null };

function lanes() {
  return ROWS.map(y => `<line class="flow-lane is-down" x1="250" y1="${y - LANE_GAP}" x2="696" y2="${y - LANE_GAP}" marker-end="url(#flow-arrow-down)"/>
      <line class="flow-lane is-up" x1="700" y1="${y + LANE_GAP}" x2="254" y2="${y + LANE_GAP}" marker-end="url(#flow-arrow-up)"/>`).join("");
}

function clientCards() {
  return ROWS.map((y, row) => `<g class="flow-client">
      <rect class="flow-card" x="700" y="${y - 42}" width="230" height="84" rx="10"/>
      <text class="flow-name" x="716" y="${y - 16}" data-flow-client="${row}">Client</text>
      <text class="flow-state" x="716" y="${y + 6}" data-flow-client-state></text>
      <text class="flow-hint" x="716" y="${y + 27}">Ảnh ở lại máy này</text>
    </g>`).join("");
}

function packets() {
  return ROWS.map(() => `<g class="flow-packet" data-flow-packet>
      <rect class="flow-packet-box" width="${PACKET.width}" height="${PACKET.height}" rx="6"/>
      <text class="flow-packet-text" x="${PACKET.width / 2}" y="16" text-anchor="middle"></text>
    </g>`).join("");
}

function shellHtml() {
  return `<h3 class="panel-title panel-title-sm">Một vòng: dữ liệu đi xuống, đi lên thế nào</h3>
    <div class="flow-controls mt-3">
      <button type="button" class="btn btn-primary btn-sm" data-flow-toggle></button>
      <button type="button" class="btn btn-secondary btn-sm" data-flow-prev>Bước trước</button>
      <button type="button" class="btn btn-secondary btn-sm" data-flow-next>Bước sau</button>
      <span class="field-hint" data-flow-scope></span>
    </div>
    <svg class="flow-svg mt-3" viewBox="0 0 960 330" role="img" aria-label="Sơ đồ một máy chủ và các client. Mô hình đi xuống client, trọng số đi lên máy chủ, ảnh ở lại máy client.">
      <defs>
        <marker id="flow-arrow-down" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="flow-arrow is-down" d="M0 0 L10 5 L0 10 z"/></marker>
        <marker id="flow-arrow-up" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="flow-arrow is-up" d="M0 0 L10 5 L0 10 z"/></marker>
      </defs>
      ${lanes()}
      <text class="flow-hint" x="475" y="${ROWS[0] - LANE_GAP - 18}" text-anchor="middle">xuống</text>
      <text class="flow-hint" x="475" y="${ROWS[0] + LANE_GAP + 27}" text-anchor="middle">lên</text>
      <g class="flow-server">
        <rect class="flow-card" x="30" y="75" width="220" height="180" rx="12"/>
        <text class="flow-name" x="140" y="118" text-anchor="middle">Máy chủ</text>
        <text class="flow-hint" x="140" y="142" text-anchor="middle">VM riêng, không học</text>
        <text class="flow-state" x="140" y="182" text-anchor="middle" data-flow-server-state></text>
        <text class="flow-hint" x="140" y="208" text-anchor="middle" data-flow-server-note></text>
      </g>
      ${clientCards()}
      ${packets()}
    </svg>
    <ol class="flow-steps mt-3">${STEPS.map((step, i) => `<li><button type="button" class="flow-step" data-flow-go="${i}"><span class="step-no">${i + 1}</span><span><b>${esc(step.tag)}</b><small>${esc(step.title)}</small></span></button></li>`).join("")}</ol>
    <div class="flow-caption mt-3" data-flow-caption></div>
    <p class="field-hint mt-2">Hình minh họa thứ tự các bước, tốc độ chạy không phải thời gian thật. Số trong lời giải thích lấy từ lần đo đang xem.</p>`;
}

function clientLabels() {
  const ids = facts.row?.clients?.map(client => client.client_id) || [];
  if (ids.length >= 3) return [ids[0], ids[1], ids[ids.length - 1]].map(id => `Client ${id}`);
  if (ids.length) return ROWS.map((_unused, row) => (ids[row] === undefined ? "" : `Client ${ids[row]}`));
  return ["Client 0", "Client 1", "Client …"];
}

function renderFacts() {
  if (!host) return;
  const labels = clientLabels();
  host.querySelectorAll("[data-flow-client]").forEach(text => {
    const label = labels[Number(text.dataset.flowClient)];
    text.textContent = label;
    text.closest(".flow-client").classList.toggle("is-empty", !label);
  });
  const total = facts.row?.clients?.length || facts.numClients;
  host.querySelector("[data-flow-scope]").textContent = total > 3 ? `Hình vẽ 3 trong ${total} client, các client còn lại làm giống vậy.` : "";
  const step = STEPS[index];
  host.querySelector("[data-flow-caption]").innerHTML = `<h4 class="subhead">Bước ${index + 1}: ${esc(step.title)}</h4>
    ${step.lines(facts).filter(Boolean).map(line => `<p>${line}</p>`).join("")}`;
  const label = step.packet(facts);
  host.querySelectorAll("[data-flow-packet] text").forEach(text => { text.textContent = label; });
  host.querySelector("[data-flow-server-note]").textContent = step.motion === "merge" && facts.row?.global_hash ? `mô hình mới ${facts.row.global_hash.slice(0, 8)}` : step.motion === "check" && facts.client?.server_hash ? `băm lại ra ${facts.client.server_hash.slice(0, 8)}` : "";
}

function cancelMotion() {
  host.querySelectorAll("[data-flow-packet]").forEach(packet => {
    packet.getAnimations().forEach(animation => animation.cancel());
    packet.removeAttribute("transform");
  });
}

// Tọa độ của một gói theo chiều đi, ở dạng translate cho cả Web Animations lẫn hình tĩnh.
function keyframesFor(motion, y) {
  const down = y - LANE_GAP - PACKET.height / 2;
  const up = y + LANE_GAP - PACKET.height / 2;
  const at = (x, lane) => ({ transform: `translate(${x}px, ${lane}px)` });
  if (motion === "down") return [at(PACKET.from, down), at(PACKET.to, down)];
  if (motion === "up") return [at(PACKET.to, up), at(PACKET.from, up)];
  return [at(PACKET.from, down), at(PACKET.to, down), at(PACKET.to, up), at(PACKET.from, up)];   // đo mạng: đi rồi về
}

function runMotion() {
  cancelMotion();
  const motion = STEPS[index].motion;
  if (!["down", "up", "both"].includes(motion)) return;
  host.querySelectorAll("[data-flow-packet]").forEach((packet, row) => {
    if (prefersReducedMotion()) {       // giảm chuyển động: đứng yên giữa đường, chiều đi vẫn đọc được qua làn và lời giải thích
      const lane = motion === "up" ? ROWS[row] + LANE_GAP : ROWS[row] - LANE_GAP;
      packet.setAttribute("transform", `translate(${(PACKET.from + PACKET.to) / 2} ${lane - PACKET.height / 2})`);
      return;
    }
    packet.animate(keyframesFor(motion, ROWS[row]), { duration: motion === "both" ? 3200 : 2400, delay: row * 180, easing: "ease-in-out", fill: "both" });
  });
}

function renderStep() {
  const step = STEPS[index];
  const svg = host.querySelector(".flow-svg");
  svg.dataset.motion = step.motion;
  host.querySelector("[data-flow-server-state]").textContent = step.server;
  host.querySelectorAll("[data-flow-client-state]").forEach(text => { text.textContent = step.client; });
  host.querySelectorAll("[data-flow-go]").forEach(button => {
    const on = Number(button.dataset.flowGo) === index;
    button.classList.toggle("is-current", on);
    if (on) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  renderFacts();
  runMotion();
}

function renderToggle() {
  host.querySelector("[data-flow-toggle]").textContent = playing ? "Tạm dừng" : "Phát";
  // Đang tự chạy thì không đọc lời giải thích liên tục cho trình đọc màn hình.
  host.querySelector("[data-flow-caption]").setAttribute("aria-live", playing ? "off" : "polite");
}

function isVisible() {
  return !document.hidden && host?.isConnected && location.hash === "#flower";
}

function schedule(wait) {
  clearTimeout(timer);
  startedAt = Date.now();
  remaining = wait;
  timer = setTimeout(tick, wait);
}

function tick() {
  if (!playing) return;
  if (!isVisible()) {       // rời trang hoặc ẩn thẻ: không chạy tiếp, quay lại thì đi tiếp
    timer = setTimeout(tick, AWAY_CHECK_MS);
    return;
  }
  index = (index + 1) % STEPS.length;
  renderStep();
  schedule(STEP_MS);
}

function goTo(next) {
  index = (next + STEPS.length) % STEPS.length;
  renderStep();
  if (playing) schedule(STEP_MS);
}

function pause() {
  playing = false;
  clearTimeout(timer);
  remaining = Math.max(0, remaining - (Date.now() - startedAt));
  host.querySelectorAll("[data-flow-packet]").forEach(packet => packet.getAnimations().forEach(animation => animation.pause()));
  renderToggle();
}

function resume() {
  playing = true;
  host.querySelectorAll("[data-flow-packet]").forEach(packet => packet.getAnimations().forEach(animation => animation.play()));
  schedule(remaining > 0 ? remaining : STEP_MS);
  renderToggle();
}

function bind(root) {
  root.addEventListener("click", event => {
    if (event.target.closest("[data-flow-toggle]")) { if (playing) pause(); else resume(); return; }
    if (event.target.closest("[data-flow-prev]")) { goTo(index - 1); return; }
    if (event.target.closest("[data-flow-next]")) { goTo(index + 1); return; }
    const step = event.target.closest("[data-flow-go]");
    if (step) goTo(Number(step.dataset.flowGo));
  });
}

/** Gọi mỗi lần trang có số mới. Khung chỉ dựng khi `root` là phần tử mới (đổi lần đo), nên sơ đồ không giật theo nhịp hỏi máy chủ. */
export function renderFlowDiagram(root, next) {
  if (!root) return;
  facts = next;
  if (root !== host) {
    clearTimeout(timer);
    host = root;
    host.innerHTML = shellHtml();
    bind(host);
    playing = !prefersReducedMotion();
    renderStep();
    renderToggle();
    if (playing) schedule(STEP_MS);
    return;
  }
  renderFacts();
}
