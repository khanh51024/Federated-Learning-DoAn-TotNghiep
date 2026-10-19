// Bước 5 · "Tham gia từ máy của bạn": học liên kết thật. Máy này là một client: ảnh được xử lý và học ngay trong trình duyệt
// (Web Worker + onnxruntime-web), chỉ lớp cuối của mô hình (~156 KB) được gửi lên máy chủ. Giao thức: docs/COMMUNITY_FL.md.
import { getJson, postJson } from "../core/api.js";
import { decodeHead, encodeHead } from "../core/community-math.js";
import {
  byId, esc, fmtInt, fmtNum, fmtPct, formatDiseaseName, notify, setBar, setDisabled, setHtml, setText, toggleHidden,
} from "../core/dom.js";

const WORKER_URL = "/static/js/workers/community-worker.js";
const SESSION_KEY = "cropfed.community.session";
const MAX_FILE_BYTES = 25 * 1024 * 1024;
const POLL_MS = 10000; // thăm dò vòng: người tham gia nhàn rỗi không cần dày hơn (đã gửi xong thì /update trả kết quả ngay)
const MODEL_CACHED_KEY = "cropfed.community.modelCached";
const IDLE_MS = 90000; // worker không gửi gì trong ngần này thời gian thì coi là treo

const S = {
  status: null,
  token: null,
  expiresAt: 0,
  config: null,
  round: null,
  head: null,
  headSha: null,
  items: [],
  nextKey: 1,
  worker: null,
  modelReady: false,
  pending: null,
  busy: null, // "train" | "submit" | null
  trained: null,
  deadlineAt: 0,
  classOptions: "",
  ticks: 0,
  roundSeq: 0, // phản hồi /round đến muộn không được ghi đè phản hồi mới hơn
  roundKnown: false, // đã nhận ít nhất một phản hồi /round trong phiên này
  sub: null, // lần đóng góp gần nhất: { roundId, outcome: null | "merged" | "rejected" | "cancelled", version, notified }
  mounted: false,
};

// ───────── Phiên (sessionStorage chỉ là tiện ích: chặn/ lỗi thì bỏ qua) ─────────
function saveSession() {
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify({ token: S.token, expiresAt: S.expiresAt, sub: S.sub })); } catch { /* bị chặn */ }
}
function loadSession() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null");
    if (saved && saved.token && saved.expiresAt * 1000 > Date.now()) {
      S.token = saved.token;
      S.expiresAt = saved.expiresAt;
      if (saved.sub && Number.isInteger(saved.sub.roundId)) S.sub = saved.sub;
    }
  } catch { /* bị chặn hoặc hỏng */ }
}
function clearSession() {
  try { sessionStorage.removeItem(SESSION_KEY); } catch { /* bị chặn */ }
}

const auth = () => ({ Authorization: `Bearer ${S.token}` });
const pageActive = () => !!byId("page-huan-luyen")?.classList.contains("active");

// ───────── Hiển thị ─────────
function setBadge(text, kind) {
  const badge = byId("comm-badge");
  setText(badge, text);
  badge.className = `status-badge${kind ? ` ${kind}` : ""}`;
}

function showResult(id, kind, html) {
  const box = byId(id);
  box.className = `comm-result${kind ? ` is-${kind}` : ""}`;
  setHtml(box, html);
  toggleHidden(box, false);
}

function pct(value) {
  return value === null || value === undefined ? "—" : fmtPct(value, 1);
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function renderMetricNote() {
  const status = S.status;
  if (!status || !status.metrics) return;
  const base = status.base_metrics?.accuracy;
  setText(byId("comm-metric-note"),
    `Phiên bản ${status.version}: độ chính xác giám sát ${pct(status.metrics.accuracy)}` +
    (status.version > 0 ? ` (mô hình gốc FedAvg v7: ${pct(base)})` : " (mô hình gốc FedAvg v7)") +
    `, trên ${fmtInt(status.metrics.count)} ảnh đã dùng để chọn mô hình gốc — chỉ là tín hiệu theo dõi.`);
}

function renderStatus() {
  const st = S.status;
  byId("comm-panel")?.setAttribute("aria-busy", "false"); // hết giữ chỗ: từ đây chiều cao theo nội dung thật
  toggleHidden(byId("comm-loading"), true);
  const main = byId("comm-main");
  const disabled = byId("comm-disabled");
  if (!st || st.enabled === false) {
    setBadge("Chưa bật", "is-neutral");
    toggleHidden(main, true);
    setText(disabled, st?.reason || "Học liên kết cộng đồng chưa được bật trên máy chủ này.");
    toggleHidden(disabled, false);
    return;
  }
  toggleHidden(disabled, true);
  toggleHidden(main, false);
  const joined = !!S.token;
  toggleHidden(byId("comm-join-box"), joined);
  toggleHidden(byId("comm-joined-box"), !joined);
  toggleHidden(byId("comm-work-box"), !joined);
  if (!st.open) setBadge("Phiên đang đóng", "is-neutral");
  else if (!joined) setBadge("Đang mở", "success");
  else setBadge("Đã tham gia", "connected");
  const unsupported = supportProblem();
  const joinable = st.open && st.joinable && !unsupported;
  setDisabled(byId("btn-comm-join"), !joinable);
  setDisabled(byId("comm-code"), !joinable);
  // Lý do khóa hiện ngay dưới ô nhập mã (không nằm trong khối bị ẩn) và có hướng dẫn "Cách tham gia" cho người mới.
  toggleHidden(byId("comm-closed-note"), joined || !!st.open);
  toggleHidden(byId("comm-howto"), joined);
  setText(byId("comm-unsupported"), unsupported);
  toggleHidden(byId("comm-unsupported"), joined || !unsupported);
  toggleHidden(byId("comm-train-note"), cachedModel());
  setText(byId("comm-caveat"), st.caveat || "");
  const config = st.config || {};
  setText(byId("comm-limit-note"), `Tối thiểu ${config.min_images ?? 5} ảnh, tối đa ${config.max_images ?? 300} ảnh mỗi lần.`);
  renderMetricNote();
  renderControls(); // gồm lý do nút khóa và thanh cột mốc: không để khoảng trống giữa lúc vừa tham gia và lúc có thông tin vòng
  if (!S.classOptions && Array.isArray(st.classes)) {
    S.classOptions = st.classes.map(name => `<option value="${esc(name)}">${esc(formatDiseaseName(name))}</option>`).join("");
    byId("comm-bulk-label").innerHTML = `<option value="">Chọn sau cho từng ảnh</option>${S.classOptions}`;
  }
}

function cachedModel() {
  try { return localStorage.getItem(MODEL_CACHED_KEY) === "1"; } catch { return false; }
}

/** Kiểm khả năng trình duyệt ngay khi mở bước 5 (trước khi người dùng nhập mã). */
function supportProblem() {
  const missing = [];
  if (typeof Worker === "undefined") missing.push("tiến trình nền (Web Worker)");
  if (typeof WebAssembly === "undefined") missing.push("WebAssembly");
  if (typeof OffscreenCanvas === "undefined") missing.push("OffscreenCanvas");
  if (typeof createImageBitmap === "undefined") missing.push("createImageBitmap");
  return missing.length
    ? `Trình duyệt này chưa hỗ trợ: ${missing.join(", ")}. Hãy dùng bản Chrome, Edge, Firefox hoặc Safari mới để tham gia từ máy của bạn.`
    : "";
}

function renderRound() {
  const round = S.round;
  if (!round) {
    ["comm-kpi-round", "comm-kpi-people", "comm-kpi-left", "comm-kpi-version"].forEach(id => setText(byId(id), "—"));
    return;
  }
  setText(byId("comm-kpi-round"), `#${round.round_id}`);
  setText(byId("comm-kpi-people"), `${round.received} / ${round.k_target}`);
  setText(byId("comm-kpi-version"), `v${round.parent_version}`);
  S.deadlineAt = performance.now() + round.seconds_left * 1000;
  tickClock();
}

function tickClock() {
  if (!S.round) return;
  setText(byId("comm-kpi-left"), clock((S.deadlineAt - performance.now()) / 1000));
}

/** Chuỗi <option> của ô chọn nhóm: dựng một lần cho mỗi bộ nhóm (38 nhóm × tối đa 300 ảnh = 11.400 phần tử nếu dựng lại mỗi lần). */
function groupOptionsHtml() {
  const classes = S.status?.classes || [];
  if (S.groupOptionsFor !== classes) {
    S.groupOptionsFor = classes;
    S.groupOptionsHtml = `<option value="">— Chọn nhóm —</option>${classes.map(name => `<option value="${esc(name)}">${esc(formatDiseaseName(name))}</option>`).join("")}`;
  }
  return S.groupOptionsHtml;
}

function itemNode(item) {
  const li = document.createElement("li");
  li.className = "comm-item";
  li.dataset.key = String(item.key);
  li.innerHTML = `
    <img class="comm-thumb" src="${esc(item.url)}" alt="" width="56" height="56" loading="lazy" decoding="async">
    <div class="comm-item-body">
      <div class="comm-item-head">
        <span class="comm-item-name" title="${esc(item.file.name)}">${esc(item.file.name)}</span>
        <span class="comm-item-tools">
          <span class="comm-item-state${item.state === "error" ? " is-error" : ""}" data-state="${item.key}">${esc(itemStateText(item))}</span>
          <button type="button" class="btn btn-ghost btn-sm" data-remove="${item.key}" aria-label="Bỏ ảnh ${esc(item.file.name)}">Bỏ</button>
        </span>
      </div>
      <select class="input-field" data-label="${item.key}" aria-label="Nhóm của ảnh ${esc(item.file.name)}">${groupOptionsHtml()}</select>
    </div>`;
  return li;
}

/** Đồng bộ danh sách với S.items: chỉ thêm nút mới / bỏ nút thừa / cập nhật nhóm, nên ô đang focus không bị mất khi gán hàng loạt. */
function renderItems() {
  const list = byId("comm-items");
  const have = new Map(Array.from(list.children).map(li => [li.dataset.key, li]));
  const wanted = new Set(S.items.map(item => String(item.key)));
  for (const [key, li] of have) if (!wanted.has(key)) li.remove();
  for (const item of S.items) {
    let li = have.get(String(item.key));
    if (!li) {
      li = itemNode(item);
      list.appendChild(li);
    }
    const select = li.querySelector("select[data-label]");
    if (select.value !== item.label) select.value = item.label;
  }
  renderCounts();
}

function itemStateText(item) {
  if (item.state === "error") return item.error || "Không đọc được ảnh";
  if (item.state === "ready") return "Đã trích đặc trưng";
  if (item.state === "working") return "Đang xử lý…";
  return "";
}

function labeledItems() {
  return S.items.filter(item => item.label && item.state !== "error");
}

function renderCounts() {
  const total = S.items.length;
  const labeled = labeledItems().length;
  const groups = new Set(labeledItems().map(item => item.label)).size;
  const min = S.config?.min_images ?? S.status?.config?.min_images ?? 5;
  setText(byId("comm-count"), total === 0
    ? "Chưa có ảnh nào."
    : `${total} ảnh · đã chọn nhóm ${labeled} ảnh (${groups} nhóm)${labeled < min ? ` · cần ít nhất ${min} ảnh có nhóm` : ""}.`);
  renderControls();
}

function renderControls() {
  const busy = !!S.busy;
  const roundOpen = !!S.round && S.round.status === "open";
  const min = S.config?.min_images ?? 5;
  const canTrain = !busy && roundOpen && labeledItems().length >= min && !S.round?.already_submitted;
  setDisabled(byId("btn-comm-train"), !canTrain);
  const trained = S.trained;
  const canSubmit = !busy && roundOpen && !!trained && !trained.stale && trained.roundId === S.round?.round_id
    && !S.round?.already_submitted;
  setDisabled(byId("btn-comm-submit"), !canSubmit);
  setDisabled(byId("comm-files"), busy);
  byId("comm-dropzone")?.classList.toggle("is-disabled", busy);
  setDisabled(byId("btn-comm-bulk"), busy);
  setDisabled(byId("btn-comm-clear"), busy);
  document.querySelectorAll("#comm-items select, #comm-items [data-remove]").forEach(el => setDisabled(el, busy));
  byId("btn-comm-train").setAttribute("aria-busy", String(S.busy === "train"));
  byId("btn-comm-submit").setAttribute("aria-busy", String(S.busy === "submit"));
  setText(byId("comm-train-reason"), canTrain ? "" : trainReason());
  setText(byId("comm-submit-reason"), canSubmit ? "" : submitReason());
  renderFlow();
}

// Mọi nút khóa đều nói vì sao khóa (kể cả sau khi tải lại trang, lúc ảnh đã mất nhưng mã đã gửi).
function trainReason() {
  if (S.busy === "train") return "Đang huấn luyện trên máy này…";
  if (S.busy === "submit") return "Đang gửi cập nhật lên máy chủ…";
  if (!S.round && !S.roundKnown) return "Đang lấy thông tin vòng hiện tại…";
  if (!S.round || S.round.status !== "open") return "Chưa có vòng nào đang mở. Hãy đợi người tổ chức mở phiên hoặc vòng mới.";
  if (S.round.already_submitted) {
    return `Bạn đã gửi cập nhật cho vòng #${S.round.round_id}. Hãy đợi vòng kế tiếp rồi thêm ảnh và huấn luyện lại.`;
  }
  const min = S.config?.min_images ?? 5;
  const count = labeledItems().length;
  if (count < min) return `Cần ít nhất ${min} ảnh có nhóm (hiện có ${count}).`;
  return "";
}

function submitReason() {
  if (S.busy === "submit") return "Đang gửi cập nhật lên máy chủ…";
  if (S.busy) return "Đang huấn luyện; nút này bật khi huấn luyện xong.";
  if (!S.round && !S.roundKnown) return "Đang lấy thông tin vòng hiện tại…";
  if (!S.round || S.round.status !== "open") return "Chưa có vòng nào đang mở.";
  if (S.round.already_submitted) return `Bạn đã gửi cập nhật cho vòng #${S.round.round_id}. Hãy đợi vòng kế tiếp.`;
  if (!S.trained) return "Hãy huấn luyện trên máy này trước khi gửi (nếu bạn vừa đổi ảnh hoặc nhóm thì cần huấn luyện lại).";
  if (S.trained.stale || S.trained.roundId !== S.round.round_id) return "Vòng đã đổi nên kết quả huấn luyện cũ không gửi được. Hãy huấn luyện lại.";
  return "";
}

/** Bốn cột mốc: Đã tham gia → Đã huấn luyện → Đã gửi → Đã gộp, kèm một câu nói bước tiếp theo. */
function renderFlow() {
  const list = byId("comm-steps");
  if (!list) return;
  const sub = S.sub;
  const trainedNow = !!S.trained && !S.trained.stale && S.trained.roundId === S.round?.round_id;
  const done = { joined: !!S.token, trained: !!sub || trainedNow, sent: !!sub, merged: sub?.outcome === "merged" };
  const failed = !!sub && (sub.outcome === "rejected" || sub.outcome === "cancelled");
  let current = true;
  list.querySelectorAll("li").forEach(item => {
    const key = item.dataset.step;
    const isDone = !!done[key];
    const isWarn = key === "merged" && failed;
    item.classList.toggle("is-done", isDone);
    item.classList.toggle("is-warn", isWarn);
    const isCurrent = current && !isDone && !isWarn;
    if (isCurrent) current = false;
    item.classList.toggle("is-current", isCurrent);
    if (isCurrent) item.setAttribute("aria-current", "step"); else item.removeAttribute("aria-current");
  });
  let note = "";
  if (!S.token) note = "";
  else if (!sub) note = trainedNow ? "Đã huấn luyện xong trên máy này. Bấm “Gửi trọng số lên máy chủ” để đóng góp." : "Bước tiếp theo: thêm ảnh lá, chọn nhóm rồi huấn luyện trên máy này.";
  else if (sub.outcome === "merged") note = `Cập nhật của bạn đã góp vào phiên bản v${sub.version} (vòng #${sub.roundId}).`;
  else if (sub.outcome === "rejected") note = `Vòng #${sub.roundId} bị từ chối nên cập nhật của bạn chưa được tính; mô hình cũ được giữ nguyên.`;
  else if (sub.outcome === "cancelled") note = `Vòng #${sub.roundId} đã bị hủy trước khi gộp; cập nhật của bạn chưa được tính.`;
  else note = `Đã gửi cho vòng #${sub.roundId}. Cập nhật của bạn sẽ được gộp khi đủ người hoặc hết giờ.`;
  setText(byId("comm-flow-note"), note);
}

function recordSubmission(roundId) {
  if (S.sub && S.sub.roundId === roundId) return;
  S.sub = { roundId, outcome: null, version: null, notified: false };
  saveSession();
  renderFlow();
}

function settleSubmission(outcome, version) {
  if (!S.sub) return;
  S.sub.outcome = outcome;
  S.sub.version = version ?? null;
  if (!S.sub.notified) {
    S.sub.notified = true;
    if (outcome === "merged") notify("success", "Cập nhật của bạn đã được gộp", `Cập nhật của bạn đã góp vào phiên bản v${version}.`);
    else notify("warning", "Cập nhật chưa được tính", `Vòng #${S.sub.roundId} ${outcome === "rejected" ? "bị từ chối" : "đã bị hủy"}; mô hình cũ được giữ nguyên.`);
  }
  saveSession();
  renderFlow();
}

/** Đối chiếu lịch sử vòng với lần gửi của mình để báo "đã gộp" cho cả người tải lại trang. */
function reconcileSubmission(rows) {
  const sub = S.sub;
  if (!sub || sub.outcome) return;
  const row = rows.find(entry => Number(entry.round_id) === sub.roundId);
  if (!row) return;
  if (row.status === "aggregated") settleSubmission("merged", row.version);
  else if (row.status === "rejected") settleSubmission("rejected", null);
  else if (row.status === "cancelled") settleSubmission("cancelled", null);
  renderFlow();
}

function invalidateTrained() {
  if (!S.trained) return;
  S.trained = null;
  toggleHidden(byId("comm-local-result"), true); // khung "Huấn luyện xong…" đã cũ sau khi đổi ảnh hoặc nhóm
}

function setProgress(label, detail, fraction) {
  toggleHidden(byId("comm-progress-box"), false);
  setText(byId("comm-progress-label"), label);
  setText(byId("comm-progress-detail"), detail || "");
  setBar(byId("comm-progress-fill"), Math.round((fraction || 0) * 100));
}

// ───────── Worker ─────────
function resetWorker(reason) {
  const worker = S.worker;
  const pending = S.pending;
  S.worker = null;
  S.modelReady = false;
  S.pending = null;
  forgetFeatures(); // đặc trưng nằm trong bộ nhớ worker: mất theo worker, cờ phía giao diện phải được xóa cùng
  if (worker) { try { worker.terminate(); } catch { /* đã dừng */ } }
  if (pending) {
    clearTimeout(pending.timer);
    pending.reject(new Error(reason));
  }
}

function forgetFeatures() {
  S.items.forEach(item => {
    if (item.state === "error") return;
    item.hasFeature = false;
    item.state = "new";
  });
  renderStates();
}

function ensureWorker() {
  if (S.worker) return S.worker;
  const worker = new Worker(WORKER_URL, { type: "module" });
  worker.onmessage = event => {
    if (S.worker !== worker) return; // worker cũ đã bị thay thế: bỏ qua tin muộn
    const message = event.data || {};
    const pending = S.pending;
    if (!pending) return;
    pending.arm(); // còn sống: gia hạn đồng hồ canh
    if (message.type === "error") {
      S.pending = null;
      clearTimeout(pending.timer);
      pending.reject(Object.assign(new Error(message.message), { code: message.code }));
    } else if (pending.until.includes(message.type)) {
      S.pending = null;
      clearTimeout(pending.timer);
      pending.resolve(message);
    } else if (pending.onEvent) {
      pending.onEvent(message);
    }
  };
  worker.onerror = event => {
    if (S.worker !== worker) return;
    resetWorker(event.message || "Tiến trình nền gặp lỗi (trình duyệt có thể quá cũ).");
  };
  S.worker = worker;
  return worker;
}

function callWorker(message, until, onEvent, transfer = []) {
  const worker = ensureWorker();
  if (S.pending) return Promise.reject(new Error("Tiến trình nền đang bận."));
  return new Promise((resolve, reject) => {
    const pending = { until, onEvent, resolve, reject, timer: 0, arm: null };
    pending.arm = () => {
      clearTimeout(pending.timer);
      pending.timer = setTimeout(() => resetWorker("Tiến trình xử lý ảnh không phản hồi. Hãy thử lại (nếu lặp lại, tải lại trang)."), IDLE_MS);
    };
    S.pending = pending;
    pending.arm();
    worker.postMessage(message, transfer);
  });
}

async function prepareModel() {
  if (S.modelReady) return;
  const model = S.status?.feature_model;
  if (!model) throw new Error("Máy chủ chưa công bố mô hình đặc trưng.");
  await callWorker({ type: "init", model: { url: model.url, sha256: model.sha256, bytes: model.bytes } }, ["ready"], event => {
    if (event.type !== "progress") return;
    if (event.phase === "download") {
      const total = event.total || model.bytes;
      setProgress(event.cached ? "Mô hình đặc trưng đã có sẵn trong trình duyệt" : "Đang tải mô hình đặc trưng",
        `${fmtNum(event.done / 1e6, 1)} / ${fmtNum(total / 1e6, 1)} MB`, total ? event.done / total : 0);
    } else if (event.phase === "runtime") {
      setProgress("Đang tải bộ chạy mô hình", `${fmtNum(event.done / 1e6, 1)} / ${fmtNum(event.total / 1e6, 1)} MB`,
        event.total ? Math.min(1, event.done / event.total) : 0);
    } else if (event.phase === "session") {
      setProgress("Đang khởi động mô hình", "Có thể mất vài chục giây trên mạng chậm", 0.97);
    }
  });
  S.modelReady = true;
  try { localStorage.setItem(MODEL_CACHED_KEY, "1"); } catch { /* bị chặn */ }
  toggleHidden(byId("comm-train-note"), true);
}

async function extractFeatures(items) {
  const total = items.length;
  const byKey = new Map(items.map(item => [String(item.key), item]));
  items.forEach(item => { item.state = "working"; });
  renderStates();
  await callWorker(
    { type: "extract", items: items.map(item => ({ key: String(item.key), file: item.file })) },
    ["extract_done"],
    event => {
      const item = byKey.get(event.key);
      if (event.type === "feature" && item) { item.state = "ready"; item.hasFeature = true; }
      if (event.type === "feature_error" && item) { item.state = "error"; item.error = "Không đọc được ảnh"; item.hasFeature = false; }
      if (event.type === "feature" || event.type === "feature_error") {
        setProgress("Đang trích đặc trưng từ ảnh của bạn", `${event.index} / ${total}`, event.index / total);
        const label = document.querySelector(`[data-state="${event.key}"]`);
        if (label && item) { setText(label, itemStateText(item)); label.classList.toggle("is-error", item.state === "error"); }
      }
    },
  );
}

function renderStates() {
  S.items.forEach(item => {
    const label = document.querySelector(`[data-state="${item.key}"]`);
    if (label) { setText(label, itemStateText(item)); label.classList.toggle("is-error", item.state === "error"); }
  });
}

// ───────── Máy chủ ─────────
let throttledAt = 0;
function noteThrottled() {
  if (Date.now() - throttledAt < 60000) return;
  throttledAt = Date.now();
  notify("warning", "Máy chủ đang giới hạn tần suất", "Có nhiều yêu cầu cùng lúc từ mạng của bạn; trang sẽ tự thử lại sau ít giây.");
}

function errorText(error) {
  return error && error.message ? error.message : "Đã xảy ra lỗi không xác định.";
}

// Lỗi mạng khi worker tải mô hình hiện nguyên văn tiếng Anh của trình duyệt ("Failed to fetch"): đổi sang câu có đường ra.
function trainErrorText(error) {
  const message = errorText(error);
  if (/failed to fetch|networkerror|load failed|network request failed/i.test(message)) {
    return "Không tải được mô hình đặc trưng. Hãy kiểm tra kết nối mạng rồi bấm Huấn luyện lại.";
  }
  return message;
}

async function loadStatus() {
  try {
    S.status = await getJson("/api/community/status", { timeoutMs: 15000 });
  } catch (error) {
    if (!S.status) S.status = { enabled: false, reason: errorText(error) };
    else if (error.status === 429) noteThrottled();
    // đã có trạng thái trước đó: giữ nguyên (một lần lỗi mạng/429 không được làm giao diện hiện "chưa bật")
    return;
  }
  renderStatus();
  if (S.status.enabled !== false) await loadHistory();
}

async function loadHistory() {
  const body = byId("comm-history-body");
  try {
    const data = await getJson("/api/community/history?limit=8", { timeoutMs: 15000 });
    const rows = data.rounds || [];
    reconcileSubmission(rows);
    if (rows.length === 0) {
      body.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Chưa có vòng nào hoàn tất.</td></tr>';
      return;
    }
    body.innerHTML = rows.map(row => {
      const before = row.parent_metrics?.accuracy;
      const after = row.metrics?.accuracy;
      let outcome = "Đã hủy";
      if (row.status === "aggregated") outcome = `<span class="text-emerald fw-600">Đã nhận → phiên bản ${esc(row.version)}</span>`;
      else if (row.status === "rejected") outcome = `<span class="text-rose">Bị từ chối: ${esc((row.messages || []).join("; "))}</span>`;
      return `<tr role="row">
        <td role="cell" data-label="Vòng" class="fw-600">#${esc(row.round_id)}</td><td role="cell" data-label="Người">${esc(row.clients)}</td><td role="cell" data-label="Ảnh">${esc(row.examples)}</td><td role="cell" data-label="Kết quả">${outcome}</td>
        <td role="cell" data-label="Độ chính xác giám sát" class="mono">${row.metrics ? `${pct(before)} → ${pct(after)}` : "—"}</td></tr>`;
    }).join("");
  } catch (error) {
    body.innerHTML = `<tr><td colspan="5" class="text-center text-muted">${esc(errorText(error))}</td></tr>`;
  }
}

function signOut(message) {
  S.token = null;
  S.round = null;
  S.head = null;
  S.headSha = null;
  S.trained = null;
  S.sub = null;
  S.roundKnown = false;
  toggleHidden(byId("comm-local-result"), true);
  toggleHidden(byId("comm-submit-result"), true);
  clearSession();
  renderStatus();
  renderRound();
  renderControls();
  if (message) notify("warning", "Đã rời phiên", message);
}

/** Trả về true nếu nhận được phản hồi từ máy chủ (kể cả khi phiên đóng); false nếu lỗi mạng/quá thời gian. */
async function refreshRound() {
  if (!S.token) return false;
  const seq = ++S.roundSeq;
  try {
    const data = await getJson(`/api/community/round?have=${encodeURIComponent(S.headSha || "")}`, { headers: auth(), timeoutMs: 20000 });
    if (seq !== S.roundSeq) return true; // đã có yêu cầu mới hơn: bỏ phản hồi cũ để chỉ số không nhảy ngược
    S.roundKnown = true;
    S.config = data.config || S.config;
    if (!data.open || !data.round) {
      S.round = null;
      renderRound();
      renderControls();
      return true;
    }
    if (data.head) {
      S.head = decodeHead(data.head);
      S.headSha = data.round.parent_sha;
    }
    const previous = S.round?.round_id;
    S.round = { ...data.round, already_submitted: data.already_submitted };
    // Tải lại trang sau khi đã gửi: mã đã dùng cho vòng này nhưng trình duyệt không còn nhớ. Ghi lại để theo dõi "đã gộp".
    if (data.already_submitted) recordSubmission(S.round.round_id);
    if (S.trained && S.trained.roundId !== S.round.round_id) S.trained.stale = true;
    if (previous !== undefined && previous !== S.round.round_id) await loadStatus();
    renderRound();
    renderControls();
    return true;
  } catch (error) {
    if (error.status === 429) noteThrottled();
    if (error.status === 401) signOut("Phiên tham gia đã hết hạn. Hãy nhập lại mã.");
    return error.status > 0;
  }
}

async function join(event) {
  event.preventDefault();
  const input = byId("comm-code");
  const code = input.value.trim();
  if (!code) { input.focus(); return; }
  setDisabled(byId("btn-comm-join"), true);
  try {
    const data = await postJson("/api/community/join", { code }, { timeoutMs: 20000 });
    S.token = data.token;
    S.expiresAt = data.expires_at;
    saveSession();
    input.value = "";
    notify("success", "Đã tham gia", "Máy của bạn là một client của phiên học liên kết cộng đồng.");
    await loadStatus();
    await refreshRound();
  } catch (error) {
    notify("error", "Không tham gia được", errorText(error));
    input.focus();
  } finally {
    setDisabled(byId("btn-comm-join"), !(S.status?.open && S.status?.joinable));
  }
}

// ───────── Ảnh ─────────
function addFiles(fileList) {
  const max = S.config?.max_images ?? S.status?.config?.max_images ?? 300;
  const group = byId("comm-bulk-label").value; // chọn nhóm TRƯỚC: ảnh thêm vào được gán sẵn, vẫn sửa được từng ảnh
  const rejected = [];
  let added = 0;
  for (const file of Array.from(fileList)) {
    if (!file.type.startsWith("image/")) { rejected.push(`${file.name}: không phải ảnh`); continue; }
    if (file.size > MAX_FILE_BYTES) { rejected.push(`${file.name}: lớn hơn ${MAX_FILE_BYTES / 1048576} MB`); continue; }
    if (S.items.some(item => item.file.name === file.name && item.file.size === file.size && item.file.lastModified === file.lastModified)) continue;
    if (S.items.length >= max) { rejected.push(`${file.name}: vượt ${max} ảnh`); continue; }
    S.items.push({ key: S.nextKey++, file, url: URL.createObjectURL(file), label: group, state: "new", hasFeature: false });
    added += 1;
  }
  if (group && added) notify("info", "Đã gán nhóm cho ảnh mới", `${formatDiseaseName(group)}: ${added} ảnh. Đổi ở từng ảnh nếu có ảnh thuộc nhóm khác.`);
  if (rejected.length) notify("warning", "Có ảnh bị bỏ qua", rejected.slice(0, 5).join("; "));
  invalidateTrained();
  renderItems();
}

function removeItem(key) {
  const index = S.items.findIndex(item => item.key === key);
  if (index < 0) return;
  URL.revokeObjectURL(S.items[index].url);
  S.items.splice(index, 1);
  if (S.worker && S.modelReady && !S.pending) S.worker.postMessage({ type: "forget", keys: [String(key)] });
  invalidateTrained();
  renderItems();
}

function clearItems() {
  S.items.forEach(item => URL.revokeObjectURL(item.url));
  S.items = [];
  if (S.worker && S.modelReady && !S.pending) S.worker.postMessage({ type: "forget" });
  invalidateTrained();
  byId("comm-files").value = "";
  toggleHidden(byId("comm-local-result"), true);
  renderItems();
}

// ───────── Huấn luyện và gửi ─────────
async function trainLocal() {
  if (S.busy) return;
  S.busy = "train";
  toggleHidden(byId("comm-local-result"), true);
  toggleHidden(byId("comm-submit-result"), true);
  renderControls();
  try {
    await refreshRound();
    if (!S.round || !S.head) throw new Error("Hiện chưa có vòng nào đang mở để đóng góp.");
    if (S.round.already_submitted) throw new Error("Bạn đã gửi cập nhật cho vòng này. Hãy đợi vòng kế tiếp.");
    const classes = S.status.classes;
    const min = S.config?.min_images ?? 5;
    const config = { ...(S.config || {}), seed: Math.floor(Math.random() * 2 ** 31) };
    let result = null;
    // Lần đầu có thể gặp "thiếu đặc trưng" nếu tiến trình xử lý ảnh vừa được khởi động lại: xóa cờ rồi trích lại một lần, không bắt người dùng xóa ảnh.
    for (let attempt = 0; result === null; attempt += 1) {
      await prepareModel();
      const need = labeledItems().filter(item => !item.hasFeature);
      if (need.length) await extractFeatures(need);
      const usable = labeledItems().filter(item => item.hasFeature);
      if (usable.length < min) throw new Error(`Chỉ ${usable.length} ảnh dùng được; cần ít nhất ${min} ảnh có nhóm và đọc được.`);
      setProgress("Đang huấn luyện lớp cuối trên máy này", `0 / ${config.epochs} epoch`, 0);
      try {
        result = await callWorker(
          {
            type: "train", keys: usable.map(item => String(item.key)), labels: usable.map(item => classes.indexOf(item.label)),
            weight: S.head.weight, bias: S.head.bias, config,
          },
          ["trained"],
          event => {
            if (event.type === "progress" && event.phase === "train") {
              setProgress("Đang huấn luyện lớp cuối trên máy này", `${event.done} / ${event.total} epoch`, event.done / event.total);
            }
          },
        );
      } catch (error) {
        if (error.code === "missing_features" && attempt === 0) { forgetFeatures(); continue; }
        throw error;
      }
    }
    if (result.nTrain < min) {
      throw new Error(`Chỉ ${result.nTrain} ảnh được dùng để học (phần còn lại giữ để kiểm tra), cần ít nhất ${min}. Hãy thêm ảnh.`);
    }
    if (S.sub && S.sub.roundId !== S.round.round_id) { S.sub = null; saveSession(); } // bắt đầu một lần đóng góp mới
    S.trained = {
      weight: result.weight, bias: result.bias, roundId: S.round.round_id, parentSha: S.round.parent_sha, stale: false,
      nTrain: result.nTrain, labelCounts: result.labelCounts, dpApplied: result.dpApplied,
      local: {
        epochs: config.epochs, lr: config.lr, batch: config.batch, n_train: result.nTrain, n_holdout: result.nHoldout,
        acc_before: result.accBefore, acc_after: result.accAfter, loss_before: result.lossBefore, loss_after: result.lossAfter,
        train_ms: result.trainMs,
      },
    };
    renderTrainResult(result);
    toggleHidden(byId("comm-progress-box"), true);
  } catch (error) {
    toggleHidden(byId("comm-progress-box"), true);
    showResult("comm-local-result", "error", `<b>Chưa huấn luyện được.</b> ${esc(trainErrorText(error))}`);
  } finally {
    S.busy = null;
    renderControls();
  }
}

function renderTrainResult(result) {
  const clip = S.config?.dp?.enabled ? S.config.dp.clip : null;
  const lines = [`<li>Đã học trên <b>${result.nTrain}</b> ảnh · ${S.config.epochs} epoch, tốc độ học ${fmtNum(S.config.lr, 3)}.</li>`];
  let kind = "";
  if (result.nHoldout > 0) {
    lines.push(`<li>Độ chính xác trên <b>${result.nHoldout}</b> ảnh giữ lại (không dùng để học): <b>${pct(result.accBefore)}</b> → <b>${pct(result.accAfter)}</b>.</li>`);
    if (result.accAfter < result.accBefore) {
      kind = "warn";
      lines.push("<li>Mô hình sau khi học không tốt hơn trên ảnh giữ lại. Hãy kiểm tra lại nhóm đã chọn hoặc thêm ảnh; máy chủ sẽ kiểm tra trước khi nhận.</li>");
    }
  } else {
    lines.push(`<li>Chưa đủ ảnh để giữ lại một phần kiểm tra (cần từ 15 ảnh). Độ chính xác trên chính ảnh học: ${pct(result.trainAccBefore)} → ${pct(result.trainAccAfter)} — con số này luôn tăng nên không nói lên chất lượng.</li>`);
  }
  lines.push(`<li>Độ lớn thay đổi so với mô hình hiện hành: ${fmtNum(result.norm, 3)}${result.clipped ? " (đã cắt theo cấu hình riêng tư)" : ""}. Máy chủ có thể cắt thêm nếu quá lớn.</li>`);
  if (clip !== null) lines.push(`<li>Đã bật bảo vệ riêng tư tùy chọn: cắt chuẩn ${fmtNum(clip, 2)} và thêm nhiễu Gaussian. Đây chưa phải bảo đảm differential privacy hình thức.</li>`);
  lines.push(`<li>Thời gian học: ${fmtNum(result.trainMs / 1000, 2)} s. Chưa có gì được gửi đi; bấm “Gửi trọng số lên máy chủ” khi sẵn sàng.</li>`);
  showResult("comm-local-result", kind, `<b>Huấn luyện xong trên máy này.</b><ul>${lines.join("")}</ul>`);
}

async function submitUpdate() {
  const trained = S.trained;
  if (!trained || S.busy) return;
  S.busy = "submit";
  renderControls();
  try {
    const data = await postJson("/api/community/update", {
      round_id: trained.roundId, parent_sha: trained.parentSha, head: encodeHead(trained.weight, trained.bias),
      num_examples: trained.nTrain, label_counts: trained.labelCounts, local: trained.local, dp_applied: trained.dpApplied,
    }, { headers: auth(), timeoutMs: 60000 });
    invalidateTrained();
    recordSubmission(trained.roundId);
    const lines = [`<li>Máy chủ đã nhận cập nhật của bạn (độ lớn thay đổi ${fmtNum(data.norm, 3)}${data.clipped ? `, đã cắt xuống ${fmtNum(data.norm_used, 3)}` : ""}).</li>`];
    let kind = "";
    if (data.aggregated && data.result) {
      const r = data.result;
      if (r.status === "aggregated") {
        lines.push(`<li><b>Vòng #${r.round_id} đã gộp</b> từ ${r.clients} người, ${r.examples} ảnh → phiên bản mới <b>${r.version}</b>: độ chính xác giám sát ${pct(r.parent_metrics?.accuracy)} → ${pct(r.metrics?.accuracy)}.</li>`);
        settleSubmission("merged", r.version);
      } else if (r.status === "cancelled") {
        kind = "warn";
        lines.push("<li>Vòng này đã bị hủy trước khi gộp (phiên đóng hoặc lỗi nội bộ); cập nhật của bạn chưa được tính. Hãy thử lại ở vòng sau.</li>");
        settleSubmission("cancelled", null);
      } else {
        kind = "warn";
        lines.push(`<li><b>Vòng #${r.round_id} bị từ chối</b> và mô hình cũ được giữ nguyên: ${esc((r.messages || []).join("; ") || "không đạt cổng kiểm tra")}.</li>`);
        settleSubmission("rejected", null);
      }
    } else if (data.round) {
      lines.push(`<li>Vòng #${data.round.round_id}: đã có ${data.round.received} / ${data.round.k_target} người. Vòng sẽ gộp khi đủ người hoặc hết giờ.</li>`);
    }
    showResult("comm-submit-result", kind, `<b>Đã gửi.</b><ul>${lines.join("")}</ul>`);
    if (!S.sub?.outcome) notify("success", "Đã gửi trọng số", "Ảnh của bạn vẫn ở trên máy.");
    await refreshRound();
    await loadStatus();
  } catch (error) {
    await handleSubmitError(error, trained);
  } finally {
    S.busy = null;
    renderControls();
  }
}

async function handleSubmitError(error, trained) {
  const code = error.body?.detail?.code;
  if (code === "stale_round") {
    if (S.trained) S.trained.stale = true;
    await refreshRound();
    showResult("comm-submit-result", "warn", `<b>Vòng đã đổi.</b> ${esc(errorText(error))} Ảnh đã trích đặc trưng nên huấn luyện lại rất nhanh.`);
    return;
  }
  if (code === "already_submitted") {
    invalidateTrained();
    recordSubmission(trained.roundId);
    showResult("comm-submit-result", "warn", `<b>Bạn đã gửi cho vòng này.</b> ${esc(errorText(error))}`);
    return;
  }
  if (code === "bad_token" || error.status === 401) {
    signOut("Phiên tham gia không còn hiệu lực.");
    showResult("comm-submit-result", "error", `<b>Máy chủ không nhận cập nhật.</b> ${esc(errorText(error))}`);
    return;
  }
  if (error.status === 429) {
    showResult("comm-submit-result", "warn", "<b>Máy chủ đang giới hạn tần suất yêu cầu.</b> Cập nhật chưa được gửi. Hãy đợi ít giây rồi bấm “Gửi trọng số lên máy chủ” lại.");
    return;
  }
  if (!error.status || error.status >= 500) {
    // Không biết máy chủ đã ghi nhận hay chưa (mất phản hồi, quá thời gian, 5xx): hỏi lại trước khi báo, kẻo báo "không nhận" khi đã nhận.
    const reached = await refreshRound();
    if (S.round?.already_submitted) {
      invalidateTrained();
      recordSubmission(trained.roundId);
      showResult("comm-submit-result", "", "<b>Máy chủ đã nhận cập nhật của bạn.</b> Phản hồi bị mất trên đường truyền nhưng cập nhật đã được lưu; số liệu sẽ cập nhật sau vài giây. Không cần gửi lại.");
      await loadStatus();
    } else if (reached) {
      showResult("comm-submit-result", "error", `<b>Máy chủ chưa nhận cập nhật.</b> ${esc(errorText(error))} Hãy bấm “Gửi trọng số lên máy chủ” để gửi lại.`);
    } else {
      showResult("comm-submit-result", "warn", "<b>Chưa nhận được phản hồi từ máy chủ.</b> Cập nhật có thể đã được lưu; hãy xem chỉ số “Người đã gửi” và đừng gửi lại ngay.");
    }
    return;
  }
  showResult("comm-submit-result", "error", `<b>Máy chủ không nhận cập nhật.</b> ${esc(errorText(error))}`);
}

// ───────── Gắn sự kiện ─────────
function poll() {
  S.ticks += 1;
  if (!pageActive() || document.hidden || S.busy) return;
  if (S.token) {
    refreshRound();
    if (S.ticks % 6 === 0) loadStatus(); // gồm cả lịch sử; một lần lỗi không làm mất trạng thái đang có
  } else if (S.ticks % 2 === 0) {
    loadStatus();
  }
}

export async function mount() {
  if (S.mounted) return;
  S.mounted = true;
  loadSession();
  byId("comm-join-form").addEventListener("submit", join);
  byId("comm-code").addEventListener("input", event => { event.currentTarget.value = event.currentTarget.value.toUpperCase(); });
  let leaveTimer = 0;
  byId("btn-comm-leave").addEventListener("click", event => {
    const button = event.currentTarget;
    if (!leaveTimer) {
      setText(button, "Bấm lần nữa để rời");
      toggleHidden(byId("comm-leave-hint"), false);
      leaveTimer = setTimeout(() => { leaveTimer = 0; setText(button, "Rời phiên"); toggleHidden(byId("comm-leave-hint"), true); }, 6000);
      return;
    }
    clearTimeout(leaveTimer);
    leaveTimer = 0;
    setText(button, "Rời phiên");
    toggleHidden(byId("comm-leave-hint"), true);
    clearItems();
    signOut("");
    loadStatus();
  });
  byId("comm-files").addEventListener("change", event => { addFiles(event.currentTarget.files); event.currentTarget.value = ""; });
  // Kéo thả vào vùng chọn ảnh. `drop` bị app.js chặn mặc định (initDropGuard) nên phải tự đọc dataTransfer.files.
  const dropzone = byId("comm-dropzone");
  ["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault();
    if (!byId("comm-files").disabled) dropzone.classList.add("is-dragover");
  }));
  ["dragleave", "dragend"].forEach(name => dropzone.addEventListener(name, () => dropzone.classList.remove("is-dragover")));
  dropzone.addEventListener("drop", event => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
    if (!byId("comm-files").disabled && event.dataTransfer?.files?.length) addFiles(event.dataTransfer.files);
  });
  byId("btn-comm-clear").addEventListener("click", clearItems);
  byId("btn-comm-bulk").addEventListener("click", () => {
    const label = byId("comm-bulk-label").value;
    if (!label) { notify("warning", "Chưa chọn nhóm", "Chọn một nhóm ở ô bên cạnh rồi bấm lại."); return; }
    S.items.forEach(item => { if (!item.label) item.label = label; });
    invalidateTrained();
    renderItems();
  });
  byId("comm-items").addEventListener("change", event => {
    const select = event.target.closest("select[data-label]");
    if (!select) return;
    const item = S.items.find(entry => String(entry.key) === select.dataset.label);
    if (item) { item.label = select.value; invalidateTrained(); renderCounts(); }
  });
  byId("comm-items").addEventListener("click", event => {
    const button = event.target.closest("[data-remove]");
    if (button) removeItem(Number(button.dataset.remove));
  });
  byId("btn-comm-train").addEventListener("click", trainLocal);
  byId("btn-comm-submit").addEventListener("click", submitUpdate);
  // Link tham gia từ trang quản trị có dạng /?code=XXXXX-XXXXX#huan-luyen: điền sẵn mã rồi xóa khỏi thanh địa chỉ để mã
  // không nằm lại trong lịch sử trình duyệt.
  const params = new URLSearchParams(location.search);
  const preset = params.get("code");
  let presetFilled = false;
  if (preset !== null) {
    params.delete("code");
    const rest = params.toString();
    try { history.replaceState(null, "", `${location.pathname}${rest ? `?${rest}` : ""}${location.hash}`); } catch { /* bỏ qua */ }
    if (!S.token) {
      const input = byId("comm-code");
      input.value = preset.toUpperCase().replace(/[^A-Z0-9-]/g, "").slice(0, 14);
      presetFilled = !!input.value;
    }
  }
  setInterval(tickClock, 1000);
  setInterval(poll, POLL_MS);
  await loadStatus();
  if (presetFilled) {
    if (S.status?.open) notify("info", "Đã điền sẵn mã tham gia", "Bấm “Tham gia” để vào phiên.");
    else notify("info", "Đã điền sẵn mã tham gia", "Phiên đang đóng; nút Tham gia sẽ bật khi người tổ chức mở phiên.");
  }
  if (S.token) await refreshRound();
}
