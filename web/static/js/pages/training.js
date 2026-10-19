// Bước 5 · Huấn luyện thử: chạy FedAvg qua Flower (5 client localhost) hoặc Centralized, theo dõi trực tiếp.
import { getJson, postJson } from "../core/api.js";
import { clearCredential, readCredential, storedAuthHeader } from "../core/admin-auth.js";
import { currentTask, markTaskStarted, onStatus, pollStatus, setPollInterval } from "../core/status.js";
import { beat, diagramHtml, pulse } from "../core/flnet.js";
import { mount as mountCommunity } from "./community.js";
import {
  applyStyleData, $$, byId, esc, fmtNum, formatApiError, notify, prefersReducedMotion, setBar, setClass, setDisabled, setDisplay, setHtml,
  setText, toggleHidden,
} from "../core/dom.js";

const SPINNER = '<svg class="svg-icon icon-sm spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>';
const PLAY = '<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
const LOCK = '<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>';
const STOP = '<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>';

const state = { mode: "federated", layerLoaded: false, starting: false };
// Chính sách huấn luyện của máy chủ (từ /api/status): có cần đăng nhập quản trị không và các trần tham số.
const policy = { requiresAdmin: false, limits: {} };
const FORM_MAX = { rounds: 50, localEpochs: 10, centralizedEpochs: 50 }; // trần của biểu mẫu khi máy chủ chưa nêu trần
const topo = { lastRound: null }; // số vòng đã hoàn tất ở lần poll trước (null: chưa biết, không chạy nhịp khi mới mở trang)
const last = { network: {} }; // lần poll gần nhất: để đổi chế độ không phải đợi chu kỳ poll sau

// Ba loại bảng: Flower thật (network), FedAvg mô phỏng trong tiến trình web (simulation), Centralized.
// Không có cột "độ chính xác": phần chạy trên máy chủ dùng ảnh nhiễu ngẫu nhiên nên nó đứng yên ở 1/38 và không mang thông tin.
const COLS = { network: 7, simulation: 5, centralized: 5 };
const HEAD = {
  network: "<tr><th>Vòng</th><th>Client hợp lệ / đáng ngờ / bị loại</th><th>Client học lâu nhất</th><th>Ảnh mỗi client</th><th>Tham số học được đổi</th><th>Mã băm mô hình chung</th><th>Trạng thái</th></tr>",
  simulation: "<tr><th>Vòng</th><th>Loss mô hình chung</th><th>Thời gian gộp</th><th>Thời gian cả vòng</th><th>Trạng thái</th></tr>",
  centralized: "<tr><th>Epoch</th><th>Loss (học)</th><th>Loss (kiểm tra)</th><th>Thời gian</th><th>Trạng thái</th></tr>",
};
const EMPTY_TEXT = {
  network: 'Chưa có phiên đang theo dõi. Bấm "Bắt đầu" để chạy; nhật ký chi tiết từng client nằm ở mục bên dưới.',
  simulation: 'Chưa có phiên đang theo dõi. Bấm "Bắt đầu" để chạy.',
  centralized: 'Chưa có tiến trình huấn luyện. Bấm "Bắt đầu" để theo dõi.',
};
const emptyRow = (kind, text) => `<tr><td colspan="${COLS[kind]}" class="text-center text-muted">${text}</td></tr>`;

/** Số rất nhỏ (0,0028) vẫn đọc được: giữ `digits` chữ số có nghĩa thay vì làm tròn về 0. */
function fmtSig(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n === 0 ? "0" : n.toLocaleString("vi-VN", { maximumSignificantDigits: digits });
}
const shortHash = value => (typeof value === "string" && value.length >= 12 ? value.slice(0, 12) : "—");
const formatDec = (val, digits = 4) => (val === undefined || val === null || Number.isNaN(Number(val)) ? "—" : fmtNum(Number(val), digits));

// ───────── Bảng tiến trình ─────────
function kindOfTask(task) {
  if (!task) return null;
  if (task.type === "centralized" || task.experiment === "Centralized") return "centralized";
  if (task.type === "federated" || task.type === "federated-network" || task.experiment === "Federated") return "federated";
  return null;
}

/** Loại bảng của một tác vụ: network (Flower thật) | simulation (trong tiến trình web) | centralized. */
function tableKind(task) {
  const kind = kindOfTask(task);
  if (kind === "centralized") return "centralized";
  if (kind === "federated") return task?.type === "federated-network" ? "network" : "simulation";
  return null;
}

/** Loại bảng theo lựa chọn hiện tại của người dùng (khi chưa có tác vụ đang chạy). */
function modeKind() {
  if (state.mode === "centralized") return "centralized";
  return byId("fl-engine")?.value === "simulation" ? "simulation" : "network";
}

/** Các dòng của bảng: Flower lấy từ progress.json của máy chủ (live_progress luôn rỗng ở engine này). */
function rowsFor(kind, task, network) {
  if (kind !== "network") return task?.live_progress || [];
  const rows = Array.isArray(network.rounds_brief) ? [...network.rounds_brief] : [];
  const inProgress = Number(network.round_in_progress || 0);
  if (task?.status === "running" && inProgress > rows.length) rows.push({ round: inProgress, _live: true });
  return rows;
}

function signature(history, isLive, task) {
  const first = task?.live_progress?.[0];
  return JSON.stringify({ history, isLive, status: task?.status ?? null, total: first ? (first.total_epochs ?? first.total_rounds ?? null) : null });
}

function networkRowHtml(item, label, cls) {
  const clients = Array.isArray(item.clients) ? item.clients : [];
  const times = clients.map(c => Number(c.local_train_time_s)).filter(Number.isFinite);
  const examples = clients.map(c => Number(c.num_examples)).filter(Number.isFinite);
  const relative = clients.map(c => Number(c.learnable_update_relative)).filter(Number.isFinite);
  const meanRelative = relative.length ? relative.reduce((a, b) => a + b, 0) / relative.length : null;
  const norm = item.mean_received_learnable_update_norm;
  const learned = norm === null || norm === undefined ? "—" : `${fmtSig(norm)} (${fmtSig(meanRelative === null ? 0 : meanRelative * 100)}% của ‖W‖)`;
  return `
    <td class="fw-600">Vòng ${esc(item.round)}</td>
    <td>${item._live ? "—" : `${esc(item.accepted_clients ?? "—")} / ${esc(item.suspicious_clients ?? 0)} / ${esc(item.rejected_clients ?? 0)}`}</td>
    <td class="text-muted">${times.length ? `${fmtNum(Math.max(...times), 1)} s` : "—"}</td>
    <td>${examples.length ? `${Math.min(...examples)}–${Math.max(...examples)}` : "—"}</td>
    <td class="text-sky">${item._live ? "—" : learned}</td>
    <td><span class="hash-chip" title="${esc(item.global_model_hash || "")}">${item._live ? "—" : shortHash(item.global_model_hash)}</span></td>
    <td class="${cls}">● ${label}</td>`;
}

function renderProgress(kind, history, isLive, task, network = {}) {
  const body = byId("train-progress-body");
  const head = byId("train-progress-head");
  const badge = byId("train-live-badge");
  const sig = `${kind}|${signature(history, isLive, task)}`;

  if (head.dataset.kind !== kind) {
    head.innerHTML = HEAD[kind];
    head.dataset.kind = kind;
  }
  const first = task?.live_progress?.[0];
  if (isLive) {
    // Flower thật không ghi live_progress: số vòng đã xong lấy từ progress.json (network.round_completed), tổng vòng từ cấu hình tác vụ.
    const fromNetwork = kind === "network";
    const done = fromNetwork ? Number(network.round_completed || 0) : history.length;
    const total = (kind === "centralized" ? first?.total_epochs : first?.total_rounds)
      || (fromNetwork ? Number(task?.config?.rounds) || Number(byId("fl-rounds")?.value) || "?" : "?");
    setClass(badge, "status-badge running");
    setHtml(badge, `<span class="live-dot pulse"></span> ${kind === "centralized" ? `Đang huấn luyện (${history.length}/${esc(total)} epoch)` : `Đang giao tiếp (đã xong ${done}/${esc(total)} vòng)`}`);
  } else if (task?.status === "cancelled") {
    setClass(badge, "status-badge error");
    setHtml(badge, '<span class="live-dot"></span> Đã dừng');
  } else if (history?.length) {
    setClass(badge, "status-badge success");
    setHtml(badge, "✓ Hoàn tất");
  } else {
    setClass(badge, "status-badge");
    setHtml(badge, "Sẵn sàng");
  }

  if (body.dataset.sig === sig) return;
  body.dataset.sig = sig;
  if (!history || history.length === 0) {
    body.innerHTML = isLive
      ? emptyRow(kind, `Đang chạy — chờ kết quả ${kind === "centralized" ? "epoch" : "vòng"} đầu tiên…`)
      : emptyRow(kind, EMPTY_TEXT[kind]);
    return;
  }
  const frag = document.createDocumentFragment();
  history.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const isLast = idx === history.length - 1;
    const done = !item._live && (item.status === "completed" || item.status === undefined || !isLive || !isLast);
    const cancelled = isLast && task?.status === "cancelled";
    const federated = kind !== "centralized";
    let label = federated ? "Đã gộp" : "Đã xong";
    let cls = "text-emerald";
    if (cancelled) { label = "Đã dừng"; cls = "text-rose"; }
    else if (!done) { label = federated ? "Đang học" : "Đang chạy"; cls = "text-amber"; }
    if (kind === "network") {
      tr.innerHTML = networkRowHtml(item, label, cls);
    } else if (kind === "simulation") {
      tr.innerHTML = `
        <td class="fw-600">Vòng ${esc(item.round)}</td>
        <td class="text-sky">${formatDec(item.global_loss)}</td>
        <td class="text-purple">${item.aggregation_time_ms !== undefined ? `${fmtNum(item.aggregation_time_ms, 1)} ms` : "—"}</td>
        <td class="text-muted">${item.round_duration_s !== undefined ? `${fmtNum(item.round_duration_s, 1)} s` : "—"}</td>
        <td class="${cls}">● ${label}</td>`;
    } else {
      tr.innerHTML = `
        <td class="fw-600">Epoch ${esc(item.epoch)}</td>
        <td class="text-sky">${formatDec(item.train_loss)}</td>
        <td>${formatDec(item.val_loss)}</td>
        <td class="text-muted">${item.epoch_time_s !== undefined ? `${fmtNum(item.epoch_time_s, 1)} s` : "—"}</td>
        <td class="${cls}">● ${label}</td>`;
    }
    frag.appendChild(tr);
  });
  body.replaceChildren(frag);
}

/** Khung "Bằng chứng máy đã tính thật" của phiên Flower: tiến trình, vòng đã gộp, chuỗi mã băm mô hình chung. */
function renderProof(network) {
  const box = byId("train-proof");
  const processes = Array.isArray(network.processes) ? network.processes : [];
  const rounds = Array.isArray(network.rounds_brief) ? network.rounds_brief : [];
  if (!processes.length && !rounds.length) {
    toggleHidden(box, true);
    return;
  }
  toggleHidden(box, false);
  const running = processes.filter(p => p.running).length;
  const codes = [...new Set(processes.filter(p => !p.running).map(p => p.exit_code))].join(", ");
  setText(byId("proof-procs"), !processes.length ? "—" : running ? `${running}/${processes.length} đang chạy` : `${processes.length}/${processes.length} đã thoát (mã ${codes})`);
  setText(byId("proof-rounds"), String(rounds.length));
  const lastRound = rounds[rounds.length - 1];
  const hashEl = byId("proof-hash");
  setText(hashEl, lastRound ? `${String(lastRound.global_model_hash || "").slice(0, 16)}…` : "—");
  if (hashEl) hashEl.title = lastRound?.global_model_hash || "";
  const chained = rounds.length > 1 && rounds.every((r, i) => i === 0 || r.parent_model_hash === rounds[i - 1].global_model_hash);
  const relative = (lastRound?.clients || []).map(c => Number(c.learnable_update_relative)).filter(Number.isFinite);
  const meanRelative = relative.length ? relative.reduce((a, b) => a + b, 0) / relative.length : null;
  const parts = [];
  if (rounds.length > 1) parts.push(chained ? `Chuỗi mã băm khớp qua ${rounds.length} vòng: mô hình chung của vòng trước chính là mô hình cha của vòng sau.` : "Cảnh báo: chuỗi mã băm giữa các vòng không khớp.");
  if (meanRelative !== null) parts.push(`Tham số học được chỉ đổi ${fmtSig(meanRelative * 100)}% so với trọng số cha — đúng kỳ vọng khi ảnh là nhiễu ngẫu nhiên; chuẩn cập nhật lớn hơn trong nhật ký chủ yếu là bộ đệm BatchNorm trôi.`);
  setText(byId("proof-note"), parts.join(" "));
}

/** KPI và sơ đồ theo chế độ: Centralized không có client/RTT, nên không hiện số của phiên Flower trước. */
function renderViews(task, network) {
  const running = task?.status === "running";
  const kind = running ? (tableKind(task) ?? modeKind()) : modeKind();
  const centralized = kind === "centralized";
  toggleHidden(byId("topology"), centralized);
  toggleHidden(byId("kpi-fed"), kind !== "network");
  toggleHidden(byId("kpi-cent"), !centralized);
  if (kind === "network") renderProof(network || {});
  else toggleHidden(byId("train-proof"), true);
  if (centralized) {
    const rows = tableKind(task) === "centralized" ? (task.live_progress || []) : [];
    const total = rows[0]?.total_epochs;
    setText(byId("dash-epochs"), rows.length ? `${rows.length}${total ? ` / ${total}` : ""}` : "—");
    const lastRow = rows[rows.length - 1];
    setText(byId("dash-epoch-time"), lastRow?.epoch_time_s !== undefined ? `${fmtNum(lastRow.epoch_time_s, 1)} s` : "—");
  }
}

/** Client "đang kết nối" theo trạng thái máy chủ: chỉ tính khi tác vụ hiện tại là học liên kết (tránh số cũ của phiên Flower trước). */
function clientView(task, network) {
  const kind = kindOfTask(task);
  const isNetwork = task?.type === "federated-network";
  const finished = isNetwork && task?.status === "completed";
  const otherRunning = kind === "centralized" && task?.status === "running";
  const connected = otherRunning ? 0 : finished ? Number(network.accepted_clients_last_round || 0) : Number(network.connected_clients || 0);
  return { kind, isNetwork, finished, connected, federated: kind === "federated" };
}

// ───────── Sơ đồ máy chủ + client (theo trạng thái THẬT) ─────────
// Mọi trạng thái lấy từ /api/status: vòng sáng khi đang chạy, dây sáng theo số client đã kết nối, chấm chạy về máy chủ
// đúng lúc một vòng vừa hoàn tất (round_completed tăng). Không có hoạt ảnh hẹn giờ hay bịa ra.
function renderTopology(task, network) {
  const stage = byId("topology")?.querySelector(".fl-stage");
  if (!stage) return;
  const view = clientView(task, network);
  const running = task?.status === "running" && view.federated;
  const { finished, connected } = view;
  const serverUp = !!network.server_running && !(view.kind === "centralized" && task?.status === "running");
  const server = stage.querySelector(".fl-server");
  server.classList.toggle("is-on", running || serverUp);
  server.classList.toggle("is-running", running);

  // Số vòng đã xong: Flower thật → round_completed; engine trong tiến trình → độ dài live_progress; Centralized (epoch) KHÔNG phải vòng.
  const live = view.federated ? task?.live_progress : null;
  const total = live?.[0]?.total_rounds;
  const completed = view.isNetwork ? Number(network.round_completed || 0) : (live?.length || 0);
  const inProgress = view.isNetwork ? Number(network.round_in_progress || 0) : 0;
  let sub = "chưa chạy";
  if (view.kind === "centralized" && task?.status === "running") sub = "không dùng ở chế độ này";
  else if (running) sub = live?.length ? `vòng ${live.length}${total ? `/${total}` : ""}` : (inProgress ? `đang ở vòng ${inProgress}` : "đang khởi động");
  else if (finished) sub = "đã hoàn tất";
  else if (serverUp) sub = "đang chờ client";
  setText(byId("topo-server-sub"), sub);
  byId("topology").dataset.phase = sub === "chưa chạy" ? "idle" : "active"; // CSS thu gọn sơ đồ khi chưa có phiên

  stage.querySelectorAll(".fl-client").forEach((el, index) => {
    const on = index < connected;
    el.classList.toggle("is-on", on && !finished);
    el.classList.toggle("is-done", on && finished);
    el.classList.toggle("is-training", running && on && inProgress > completed);
  });
  stage.querySelectorAll(".fl-wire-lit[data-i]").forEach((el, index) => el.classList.toggle("is-on", index < connected));

  // nhịp: chỉ khi một vòng thật vừa xong và bước 5 đang hiện (poll chậm có thể gộp hai vòng thành một nhịp)
  const visible = byId("page-huan-luyen")?.classList.contains("active");
  if (topo.lastRound !== null && completed > topo.lastRound && visible && view.federated) {
    // Flower thật: đúng số client được chấp nhận ở vòng vừa xong (0 → không có gì để chạy về máy chủ, không nháy).
    // Engine trong tiến trình mô phỏng đủ năm shard mỗi vòng.
    const nodes = stage.querySelectorAll(".fl-client").length;
    const delivered = view.isNetwork ? Math.min(nodes, Number(network.accepted_clients_last_round || 0)) : nodes;
    if (delivered > 0) {
      const end = pulse(stage, Array.from({ length: delivered }, (_unused, i) => i), { perClientGap: 90, duration: 800 });
      beat(server.querySelector(".fl-node-card"), { delay: Math.max(0, end - 120) });
    }
  }
  topo.lastRound = completed;
  // poll nhanh khi đang chạy và đang xem bước 5: sơ đồ bám sát trạng thái thật
  setPollInterval(running && visible ? 1500 : 8000);
}

// ───────── Cập nhật từ poll trạng thái ─────────
function applyStatus(data, { task, network, finished, stale }) {
  if (data?.training) applyPolicy(data.training);
  if (stale) return; // poll đầu tiên sau khi bấm Bắt đầu có thể còn trả tác vụ cũ: không vẽ lại trạng thái cũ
  last.network = network;
  const kind = kindOfTask(task);
  const kindOfTable = tableKind(task);
  const running = task.status === "running";
  const start = byId("btn-train-start");
  const stop = byId("btn-train-stop");

  setText(byId("dash-clients"), `${clientView(task, network).connected} / 5`);
  setText(byId("dash-rtt"), Number.isFinite(network.mean_probe_rtt_ms) ? `${fmtNum(network.mean_probe_rtt_ms, 2)} ms (localhost)` : "—");
  renderTopology(task, network);
  renderViews(task, network);

  const pct = Math.max(5, Math.min(100, task.progress || 5));
  if (running) {
    setDisabled(start, true);
    setHtml(start, `${SPINNER} ${kind === "centralized" ? "Đang huấn luyện…" : "Đang điều phối FedAvg…"}`);
    setDisplay(stop, "inline-flex");
    setDisabled(stop, !!task.cancel_requested);
    setHtml(stop, task.cancel_requested ? `${SPINNER} Đang dừng…` : `${STOP} Dừng tiến trình`);
    toggleHidden(byId("train-progress-box"), false);
    setBar(byId("train-progress-fill"), pct);
    setText(byId("train-progress-label"), `Tiến trình: ${pct}%`);
    setText(byId("train-progress-detail"), task.message || "Đang tính toán…");
    setText(byId("train-monitor-desc"), kind === "centralized" ? "Huấn luyện Centralized đang chạy." : "Máy chủ và năm client đang chạy trên localhost.");
    if (kindOfTable) renderProgress(kindOfTable, rowsFor(kindOfTable, task, network), true, task, network);
    byId("train-progress-fill").classList.toggle("fed-fill", kind !== "centralized");
    return;
  }

  renderStartIdle();
  setDisplay(stop, "none");
  if (task.status === "completed") {
    setBar(byId("train-progress-fill"), 100);
    setText(byId("train-progress-label"), "Tiến trình: 100% (hoàn thành)");
    setText(byId("train-progress-detail"), task.message || "Hoàn tất.");
    setText(byId("train-monitor-desc"), "Phiên gần nhất đã hoàn tất.");
  } else if (task.status === "cancelled") {
    setText(byId("train-progress-label"), "Tiến trình: đã dừng theo yêu cầu");
    setText(byId("train-progress-detail"), task.message || "Đã dừng.");
  } else if (task.status === "failed") {
    setText(byId("train-progress-label"), "Tiến trình: thất bại");
    setText(byId("train-progress-detail"), task.message || "Tác vụ báo lỗi.");
  }
  if (finished) loadNetworkMetrics();
  // Bảng hiển thị phiên gần nhất cùng loại với chế độ đang chọn (kể cả khi vừa mở lại trang sau khi phiên đã xong).
  if (kindOfTable && (finished || kindOfTable === modeKind())) {
    const rows = rowsFor(kindOfTable, task, network);
    if (finished || rows.length) renderProgress(kindOfTable, rows, false, task, network);
  }
}

// ───────── Telemetry Flower (phiên gần nhất) ─────────
export async function loadNetworkMetrics() {
  const tbody = byId("network-metrics-body");
  const clientBody = byId("network-client-rtt-body");
  const runContext = byId("network-run-context");
  const summary = byId("network-final-test-summary");
  if (!tbody) return;
  try {
    const data = await getJson("/api/federated/network-metrics");
    if (!data.available || !Array.isArray(data.rounds) || data.rounds.length === 0) {
      setText(runContext, "Chưa có phiên chạy thử nào hoàn tất.");
      tbody.innerHTML = '<tr><td colspan="12" class="text-center text-muted">Chưa có dữ liệu từ phiên Flower thật.</td></tr>';
      clientBody.innerHTML = '<tr><td colspan="11" class="text-center text-muted">Chưa có nhật ký từng client.</td></tr>';
      setText(summary, "Kiểm tra cuối trên tập ảnh giữ riêng: chưa thực hiện trong phiên này.");
      return;
    }
    const runShape = [
      Number.isFinite(Number(data.configured_rounds)) ? `${data.configured_rounds} vòng` : null,
      Number.isFinite(Number(data.num_clients_required)) ? `${data.num_clients_required} client` : null,
    ].filter(Boolean).join(", ");
    setText(runContext, data.historical_run
      ? `Phiên chạy thử gần nhất đã kiểm: ${String(data.run_dir || "").split(/[\\/]/).pop()}${runShape ? ` · ${runShape}` : ""}; không phải phiên đang chạy.`
      : "Phiên Flower hiện tại · ảnh nhiễu ngẫu nhiên, chỉ để kiểm chứng hệ thống, không dùng làm kết quả khoa học.");
    const ms = (v, d = 2) => (v !== undefined && Number(v) >= 0 ? `${fmtNum(v, d)} ms` : "—");
    tbody.innerHTML = data.rounds.map(r => `
      <tr>
        <td>${esc(r.round)}</td><td>${esc(r.accepted_clients ?? "—")}</td><td>${esc(r.suspicious_clients ?? "—")}</td><td>${esc(r.rejected_clients ?? "—")}</td>
        <td>${formatDec(r.mean_received_update_norm)}</td>
        <td>${r.server_val_accuracy !== undefined ? `${fmtNum(Number(r.server_val_accuracy) * 100, 2)}%` : "—"}</td>
        <td>${formatDec(r.server_val_loss)}</td><td>${formatDec(r.server_val_macro_f1)}</td>
        <td>${ms(r.mean_probe_rtt_ms)}</td><td>${ms(r.max_probe_rtt_ms)}</td>
        <td>${ms(r.round_fit_elapsed_ms, 1)}</td><td>${ms(r.estimated_comm_overhead_ms, 1)}</td>
      </tr>`).join("");
    const tm = data.final_test_metrics;
    // /api/federated/network-metrics gán scientific_valid=false cho mọi phiên smoke;
    // chỉ được gọi số này là "tập ảnh giữ riêng" khi payload xác nhận phiên hợp lệ về mặt khoa học.
    const isScientific = data.scientific_valid === true;
    setText(summary, tm
      ? (isScientific
        ? `Kiểm tra cuối trên tập ảnh giữ riêng — độ chính xác ${fmtNum(Number(tm.accuracy) * 100, 2)}% · loss ${formatDec(tm.loss)} · macro-F1 ${formatDec(tm.macro_f1)}`
        : `Chạy thử trên ảnh nhiễu ngẫu nhiên (1 ảnh mỗi nhóm) — không phải kết quả khoa học: độ chính xác ${fmtNum(Number(tm.accuracy) * 100, 2)}%${Number(tm.accuracy) <= 0.05 ? " ≈ mức đoán ngẫu nhiên 1/38, đúng như kỳ vọng" : ""} · loss ${formatDec(tm.loss)} · macro-F1 ${formatDec(tm.macro_f1)}`)
      : "Kiểm tra cuối trên tập ảnh giữ riêng: chưa thực hiện (cần cấu hình thư mục ảnh đánh giá).");
    const clientRows = (data.audit_rounds || []).flatMap(round => (round.clients || []).map(client => ({ round: round.round, ...client })));
    clientBody.innerHTML = clientRows.length ? clientRows.map(row => `
      <tr>
        <td>${esc(row.round)}</td><td>Client ${esc(row.client_id)}</td>
        <td>${ms(row.probe_rtt_ms)}</td>
        <td>${row.local_train_time_s !== undefined ? `${fmtNum(row.local_train_time_s, 3)} s` : "—"}</td>
        <td>${row.data_mode === "synthetic" ? "ảnh nhiễu ngẫu nhiên" : esc(row.data_mode || "—")}</td><td>${row.training_seed !== undefined ? esc(row.training_seed) : "—"}</td>
        <td title="${esc(row.partition_config_sha256 || "")}">${esc(row.partition_id || "—")}</td>
        <td>${formatDec(row.received_update_norm)}</td>
        <td>${row.received_learnable_update_norm === undefined || row.received_learnable_update_norm === null ? "—" : `${fmtSig(row.received_learnable_update_norm)} (${fmtSig(Number(row.learnable_update_relative) * 100)}%)`}</td>
        <td>${esc(row.status || "—")}</td><td>${esc(row.validation_reason || "—")}</td>
      </tr>`).join("") : '<tr><td colspan="11" class="text-center text-muted">Chưa có nhật ký từng client.</td></tr>';
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">Không đọc được số liệu phiên chạy: ${esc(error.message)}</td></tr>`;
    clientBody.innerHTML = `<tr><td colspan="11" class="text-center text-muted">${esc(error.message)}</td></tr>`;
    setText(summary, `Kiểm tra cuối trên tập ảnh giữ riêng: không đọc được số liệu (${error.message})`);
  }
}

// ───────── Weight Inspector ─────────
function renderLayerStats(stats) {
  const tbody = byId("layer-stats-body");
  if (!stats) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">Chưa có dữ liệu trọng số layer.</td></tr>';
    return;
  }
  const update = stats.update_norm !== null && stats.update_norm !== undefined
    ? esc(stats.update_norm)
    : '<span class="text-muted" title="Cần checkpoint vòng trước để đo khoảng cách delta">— (chưa có reference)</span>';
  tbody.innerHTML = `
    <tr>
      <td class="fw-600">${esc(stats.layer_name)}</td>
      <td>${esc(JSON.stringify(stats.shape))}</td>
      <td>${fmtNum(stats.param_count, 0)}</td>
      <td class="text-sky">${esc(stats.l2_norm)}</td>
      <td class="text-emerald">${update}</td>
      <td>${esc(stats.mean)} ± ${esc(stats.std)}</td>
      <td class="text-muted code">[${stats.sample_values ? esc(stats.sample_values.join(", ")) : ""}]</td>
    </tr>`;
}

async function loadLayerList() {
  const tbody = byId("layer-stats-body");
  try {
    const data = await getJson("/api/weights/inspect");
    const selector = byId("layer-selector");
    if (!data.available) {
      selector.innerHTML = `<option value="">— ${esc(data.message)} —</option>`;
      tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">${esc(data.message)}</td></tr>`;
      state.layerLoaded = false;
      return;
    }
    selector.replaceChildren();
    for (const name of data.available_layers) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      selector.appendChild(option);
    }
    renderLayerStats(data.selected_layer_stats);
    state.layerLoaded = true;
  } catch (error) {
    state.layerLoaded = false;
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Không đọc được thống kê trọng số: ${esc(error.message)}</td></tr>`;
  }
}

async function loadLayerStats() {
  const selector = byId("layer-selector");
  if (!selector.value) return;
  try {
    const data = await getJson(`/api/weights/inspect?layer=${encodeURIComponent(selector.value)}`);
    renderLayerStats(data.selected_layer_stats);
  } catch (error) {
    notify("error", "Không tải được thống kê trọng số", `Lớp ${selector.value}`, { detail: error.message });
  }
}

// ───────── Form ─────────
function syncEngine() {
  const network = byId("fl-engine").value === "network";
  const set = (id, props) => { const el = byId(id); if (el) Object.assign(el, props); };
  toggleHidden(byId("fl-network-fields"), !network);
  toggleHidden(byId("train-fed-chips"), !network); // các chip cố định chỉ đúng với engine Flower
  byId("fl-alpha").disabled = network;
  if (network) {
    byId("fl-alpha").value = "0.1"; // máy chủ chỉ nhận α=0,1 cho engine network
    set("fl-rounds", { value: "2", min: "2", max: "2", disabled: true });
    set("fl-epochs", { value: "1", min: "1", max: "1", disabled: true });
    setText(byId("fl-engine-help"), "Server và các client chạy trên cùng một máy; ảnh gốc không gửi lên server."
      + (policy.requiresAdmin ? " Phiên Flower chạy năm tiến trình client cùng lúc nên tốn tài nguyên; chỉ chạy khi cần trình diễn." : ""));
  } else {
    set("fl-rounds", { min: "1", max: String(Math.min(FORM_MAX.rounds, policy.limits.max_rounds || FORM_MAX.rounds)), disabled: false });
    set("fl-epochs", { min: "1", max: String(Math.min(FORM_MAX.localEpochs, policy.limits.max_local_epochs || FORM_MAX.localEpochs)), disabled: false });
    setText(byId("fl-engine-help"), "Simulation chạy các client trong cùng process; không dùng để chứng minh độ trễ mạng thật.");
  }
  if (byId("train-progress-head")) refreshIdleView();
}

// ───────── Quyền thao tác và trần tham số ─────────
function renderGate() {
  toggleHidden(byId("train-gate"), !(policy.requiresAdmin && !readCredential()));
  renderStartIdle();
}

/** Nút Bắt đầu khi không có tác vụ chạy: bị khóa quyền thì đổi nhãn/kiểu nhưng KHÔNG vô hiệu hóa (bấm vẫn dẫn tới khung đăng nhập). */
function renderStartIdle() {
  const start = byId("btn-train-start");
  if (!start || state.starting || currentTask()?.status === "running") return;
  const locked = policy.requiresAdmin && !readCredential();
  setDisabled(start, false);
  setClass(start, locked ? "btn btn-secondary" : "btn btn-primary");
  setHtml(start, locked ? `${LOCK} Đăng nhập để bắt đầu` : `${PLAY} Bắt đầu`);
}

function applyPolicy(training) {
  const before = JSON.stringify(policy);
  policy.requiresAdmin = !!training.requires_admin;
  policy.limits = training.limits || {};
  if (JSON.stringify(policy) !== before) {
    syncEngine(); // áp trần số vòng/epoch lên ô nhập
    const cent = byId("cent-epochs");
    if (cent) {
      const cap = Math.min(FORM_MAX.centralizedEpochs, policy.limits.max_centralized_epochs || FORM_MAX.centralizedEpochs);
      cent.max = String(cap);
      if (Number(cent.value) > cap) cent.value = String(cap);
    }
    for (const [id, cap] of [["fl-rounds", policy.limits.max_rounds], ["fl-epochs", policy.limits.max_local_epochs]]) {
      const input = byId(id);
      if (input && !input.disabled && cap && Number(input.value) > cap) input.value = String(cap);
    }
  }
  renderGate();
}

/** true nếu được phép gửi lệnh; ngược lại nhắc đăng nhập quản trị (máy chủ vẫn là nơi kiểm quyền thật). */
function ensureAdmin() {
  if (!policy.requiresAdmin || readCredential()) return true;
  renderGate();
  byId("train-gate")?.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  notify("warning", "Cần quyền quản trị", "Hãy đăng nhập quản trị (liên kết ngay phía trên nút Bắt đầu) rồi quay lại bấm Bắt đầu.");
  return false;
}

function handleAuthFailure(error) {
  if (error?.status !== 401) return false;
  clearCredential(); // mật khẩu đã lưu không còn đúng
  renderGate();
  notify("error", "Cần đăng nhập quản trị", "Phiên quản trị không còn hiệu lực. Hãy đăng nhập lại ở trang quản trị.");
  return true;
}

function setTrainMode(mode) {
  state.mode = mode;
  $$("[data-train-mode]").forEach(button => {
    const on = button.dataset.trainMode === mode;
    button.classList.toggle("is-active", on);
    button.setAttribute("aria-checked", String(on));
  });
  toggleHidden(byId("train-fed-box"), mode !== "federated");
  toggleHidden(byId("train-cent-box"), mode !== "centralized");
  // tùy chọn của FedAvg không áp dụng cho Centralized
  const advanced = byId("fl-engine").closest("details");
  if (advanced) toggleHidden(advanced, mode !== "federated");
  refreshIdleView();
}

/** Đổi chế độ/engine khi chưa có tác vụ chạy: đặt lại tiêu đề bảng và KPI theo lựa chọn, không đợi chu kỳ poll sau. */
function refreshIdleView() {
  const kind = modeKind();
  const head = byId("train-progress-head");
  if (head && head.dataset.kind !== kind && !(currentTask()?.status === "running")) {
    head.innerHTML = HEAD[kind];
    head.dataset.kind = kind;
    const body = byId("train-progress-body");
    body.innerHTML = emptyRow(kind, EMPTY_TEXT[kind]);
    delete body.dataset.sig;
  }
  renderViews(currentTask(), last.network);
}

// ───────── Kiểm tra biểu mẫu trước khi gửi ─────────
// Ràng buộc min/max/step đã có trên input; ở đây chỉ đọc `validity` để báo lỗi tiếng Việt ngay tại trường
// thay vì để máy chủ trả 422 tiếng Anh sau một vòng gửi.
const TRAIN_FIELDS = {
  federated: ["fl-rounds", "fl-epochs", "fl-seed", "fl-max-update-norm"],
  centralized: ["cent-epochs", "cent-seed"],
};

function fieldName(input) {
  const label = document.querySelector(`label[for="${input.id}"]`);
  return label ? label.textContent.replace(/\s*\(.*\)\s*$/, "").trim() : "Giá trị";
}

function validationMessage(input) {
  const name = fieldName(input);
  const { validity } = input;
  if (validity.valueMissing) return `${name}: hãy nhập một số.`;
  if (validity.badInput) return `${name}: chỉ nhập số.`;
  if (validity.rangeUnderflow) return `${name}: tối thiểu là ${input.min.replace(".", ",")}.`;
  if (validity.rangeOverflow) return `${name}: tối đa là ${input.max.replace(".", ",")}.`;
  if (validity.stepMismatch) return `${name}: phải là số nguyên.`;
  return `${name}: giá trị không hợp lệ.`;
}

function clearFieldError(input) {
  byId(`${input.id}-error`)?.remove();
  input.removeAttribute("aria-invalid");
  input.removeAttribute("aria-describedby");
}

function showFieldError(input, message) {
  let note = byId(`${input.id}-error`);
  if (!note) {
    note = document.createElement("p");
    note.id = `${input.id}-error`;
    note.className = "field-error";
    input.insertAdjacentElement("afterend", note);
  }
  note.textContent = message;
  input.setAttribute("aria-invalid", "true");
  input.setAttribute("aria-describedby", note.id);
}

/** true nếu mọi trường đang áp dụng đều hợp lệ; ngược lại hiện lỗi tại trường và đưa focus tới trường lỗi đầu tiên. */
function validateTrainForm() {
  let first = null;
  for (const id of TRAIN_FIELDS[state.mode] || []) {
    const input = byId(id);
    // trường bị vô hiệu hóa (engine network cố định 2 vòng) hoặc nằm trong khối đang ẩn thì không áp dụng
    if (!input || input.disabled || input.closest(".hidden")) continue;
    clearFieldError(input);
    if (input.checkValidity()) continue;
    showFieldError(input, validationMessage(input));
    first = first || input;
  }
  if (first) {
    first.closest("details")?.setAttribute("open", ""); // trường lỗi nằm trong mục "Tùy chọn nâng cao" đang đóng
    first.focus();
  }
  return first === null;
}

function resetAfterFailedStart() {
  state.starting = false;
  renderStartIdle();
  setDisplay(byId("btn-train-stop"), "none");
  toggleHidden(byId("train-progress-box"), true);
}

async function start() {
  const button = byId("btn-train-start");
  const stop = byId("btn-train-stop");
  const federated = state.mode === "federated";
  if (!ensureAdmin()) return;
  if (!validateTrainForm()) return;
  state.starting = true; // một lần poll chen giữa lúc đang POST không được bật lại nút
  button.disabled = true;
  setHtml(button, `${SPINNER} ${federated ? "Đang điều phối FedAvg…" : "Đang huấn luyện…"}`);
  // Nút Dừng chỉ dùng được khi máy chủ đã nhận tác vụ: bấm sớm hơn sẽ bị báo "không có tác vụ để dừng"
  // trong khi tác vụ vẫn khởi chạy ngay sau đó.
  setDisplay(stop, "inline-flex");
  stop.disabled = true;
  setHtml(stop, `${SPINNER} Đang khởi tạo…`);
  toggleHidden(byId("train-progress-box"), false);
  setBar(byId("train-progress-fill"), 5);
  setText(byId("train-progress-label"), "Tiến trình: 5%");
  setText(byId("train-progress-detail"), federated ? "Đang khởi tạo 5 trạm client…" : "Đang khởi tạo mô hình…");
  // Huy hiệu của phiên trước ("✓ Hoàn tất") không được còn nguyên khi phiên mới vừa bấm Bắt đầu.
  setClass(byId("train-live-badge"), "status-badge running");
  setHtml(byId("train-live-badge"), '<span class="live-dot pulse"></span> Đang khởi động…');
  const body = byId("train-progress-body");
  const tableKindNow = modeKind();
  delete body.dataset.sig;
  body.innerHTML = emptyRow(tableKindNow, federated ? "Đang phân phối trọng số khởi tạo toàn cục tới 5 client…" : "Đang khởi tạo kiến trúc mạng và phân bổ batch…");
  byId("train-progress-head").innerHTML = HEAD[tableKindNow];
  byId("train-progress-head").dataset.kind = tableKindNow;

  const num = id => parseInt(byId(id).value, 10);
  try {
    let data;
    if (federated) {
      const normRaw = byId("fl-max-update-norm").value.trim();
      data = await postJson("/api/federated/train", {
        rounds: num("fl-rounds"),
        local_epochs: num("fl-epochs"),
        seed: num("fl-seed"),
        alpha: parseFloat(byId("fl-alpha").value),
        smoke: byId("fl-smoke-val").value === "true",
        engine: byId("fl-engine").value,
        num_clients: num("fl-num-clients"),
        server_address: byId("fl-server-address").value || "127.0.0.1:8089",
        require_real_clients: byId("fl-require-real-clients").checked,
        dataset_root: byId("fl-eval-dataset-root").value.trim() || null,
        require_real_evaluation: byId("fl-require-real-evaluation").checked,
        reject_suspicious_updates: byId("fl-validator-policy").value === "reject",
        max_update_norm: normRaw === "" ? null : parseFloat(normRaw),
      }, { headers: storedAuthHeader() });
    } else {
      data = await postJson("/api/centralized/train", {
        epochs: num("cent-epochs"),
        seed: num("cent-seed"),
        smoke: byId("cent-smoke-val").value === "true",
      }, { headers: storedAuthHeader() });
    }
    markTaskStarted();
    setDisabled(stop, false);
    setHtml(stop, `${STOP} Dừng tiến trình`);
    notify("info", `Đã bắt đầu huấn luyện ${federated ? "FedAvg" : "Centralized"}`, data.message || "");
    await pollStatus();
    state.starting = false;
  } catch (error) {
    if (!handleAuthFailure(error)) {
      notify("error", `Không bắt đầu được huấn luyện ${federated ? "FedAvg" : "Centralized"}`, error.message);
    }
    resetAfterFailedStart();
  }
}

async function stopTraining() {
  const stop = byId("btn-train-stop");
  if (!ensureAdmin()) return;
  stop.disabled = true;
  setHtml(stop, `${SPINNER} Đang dừng…`);
  try {
    const data = await postJson("/api/train/stop", {}, { headers: storedAuthHeader() });
    if (data.stopped === false) {
      notify("warning", "Không có tác vụ để dừng", data.message || "Hiện không có tiến trình huấn luyện nào đang chạy.");
    } else {
      notify("info", "Đã gửi lệnh dừng", data.message || "Tiến trình sẽ dừng ở điểm kiểm tra gần nhất.");
    }
    await pollStatus();
  } catch (error) {
    stop.disabled = false;
    setHtml(stop, `${STOP} Dừng tiến trình`);
    if (!handleAuthFailure(error)) notify("error", "Không dừng được huấn luyện", formatApiError(error.body, error.message));
  }
}

export async function mount() {
  byId("topology").innerHTML = diagramHtml({ variant: "live", serverLabel: "Máy chủ", serverSub: "chưa chạy" });
  applyStyleData(byId("topology"));
  $$("[data-train-mode]").forEach(button => button.addEventListener("click", () => setTrainMode(button.dataset.trainMode)));
  byId("fl-engine").addEventListener("change", syncEngine);
  byId("btn-train-start").addEventListener("click", start);
  byId("btn-train-stop").addEventListener("click", stopTraining);
  byId("btn-refresh-network").addEventListener("click", loadNetworkMetrics);
  byId("layer-selector").addEventListener("change", loadLayerStats);
  byId("weight-inspector").addEventListener("toggle", event => {
    if (event.target.open && !state.layerLoaded) loadLayerList();
  });
  for (const id of Object.values(TRAIN_FIELDS).flat()) {
    byId(id)?.addEventListener("input", event => clearFieldError(event.currentTarget));
  }
  syncEngine();
  setTrainMode("federated");
  // đăng nhập ở tab khác (trang /quan-tri với "Ghi nhớ trên máy này") thì mở khóa ngay, không cần tải lại
  window.addEventListener("storage", renderGate);
  onStatus(applyStatus);
  pollStatus(); // đồng bộ ngay trạng thái hiện tại (không đợi chu kỳ 4 giây)
  // Khối "Tham gia từ máy của bạn" tự tải trạng thái của nó; lỗi ở đó không được làm hỏng phần chạy thử Flower.
  mountCommunity().catch(error => notify("error", "Không khởi tạo được phần tham gia từ máy của bạn", formatApiError(error.body, error.message)));
  await loadNetworkMetrics();
}
