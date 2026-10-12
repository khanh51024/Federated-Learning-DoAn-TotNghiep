// Bước 5 · Huấn luyện thử: chạy FedAvg qua Flower (5 client localhost) hoặc Centralized, theo dõi trực tiếp.
import { getJson, postJson } from "../core/api.js";
import { currentTask, markTaskStarted, onStatus, pollStatus, setPollInterval } from "../core/status.js";
import { beat, diagramHtml, pulse } from "../core/flnet.js";
import {
  $$, byId, esc, fmtNum, formatApiError, notify, setBar, setClass, setDisabled, setDisplay, setHtml, setText, toggleHidden,
} from "../core/dom.js";

const SPINNER = '<svg class="svg-icon icon-sm spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>';
const PLAY = '<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
const STOP = '<svg class="svg-icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>';

const state = { mode: "federated", layerLoaded: false };
const topo = { lastRound: null }; // số vòng đã hoàn tất ở lần poll trước (null: chưa biết, không chạy nhịp khi mới mở trang)

const HEAD = {
  federated: "<tr><th>Vòng</th><th>Độ chính xác (mô hình chung)</th><th>Sai số (loss)</th><th>Macro-F1</th><th>Thời gian gộp</th><th>Thời gian cả vòng</th><th>Trạng thái</th></tr>",
  centralized: "<tr><th>Epoch</th><th>Chính xác (học)</th><th>Loss (học)</th><th>Chính xác (kiểm tra)</th><th>Loss (kiểm tra)</th><th>Macro-F1</th><th>Thời gian</th><th>Trạng thái</th></tr>",
};

const EMPTY_ROW = {
  federated: '<tr><td colspan="7" class="text-center text-muted">Chưa có phiên đang theo dõi. Bấm "Bắt đầu" để chạy; nhật ký chi tiết từng client nằm ở mục bên dưới.</td></tr>',
  centralized: '<tr><td colspan="8" class="text-center text-muted">Chưa có tiến trình huấn luyện. Bấm "Bắt đầu" để theo dõi.</td></tr>',
};

function formatAcc(val) {
  if (val === undefined || val === null) return "—";
  const num = Number(val);
  if (Number.isNaN(num)) return "—";
  return `${fmtNum(num <= 1.0 && num > 0 ? num * 100 : num, 2)}%`;
}
const formatDec = (val, digits = 4) => (val === undefined || val === null || Number.isNaN(Number(val)) ? "—" : fmtNum(Number(val), digits));

// ───────── Bảng tiến trình ─────────
function kindOfTask(task) {
  if (!task) return null;
  if (task.type === "centralized" || task.experiment === "Centralized") return "centralized";
  if (task.type === "federated" || task.type === "federated-network" || task.experiment === "Federated") return "federated";
  return null;
}

function signature(history, isLive, task) {
  const first = task?.live_progress?.[0];
  return JSON.stringify({ history, isLive, status: task?.status ?? null, total: first ? (first.total_epochs ?? first.total_rounds ?? null) : null });
}

function renderProgress(kind, history, isLive, task) {
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
    const total = (kind === "centralized" ? first?.total_epochs : first?.total_rounds) || "?";
    setClass(badge, "status-badge running");
    setHtml(badge, `<span class="live-dot pulse"></span> ${kind === "centralized" ? `Đang huấn luyện (${history.length}/${esc(total)} epoch)` : `Đang giao tiếp (vòng ${history.length}/${esc(total)})`}`);
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
      ? `<tr><td colspan="8" class="text-center text-muted">Đang chạy — chờ kết quả ${kind === "centralized" ? "epoch" : "vòng"} đầu tiên…</td></tr>`
      : EMPTY_ROW[kind];
    return;
  }
  const frag = document.createDocumentFragment();
  history.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const isLast = idx === history.length - 1;
    const done = item.status === "completed" || !isLive || !isLast;
    const cancelled = isLast && task?.status === "cancelled";
    let label = kind === "federated" ? "Đã tổng hợp" : "Đạt";
    let cls = "text-emerald";
    if (cancelled) { label = "Đã dừng"; cls = "text-rose"; }
    else if (!done) { label = kind === "federated" ? "Đang tổng hợp" : "Đang chạy"; cls = "text-amber"; }
    if (kind === "federated") {
      tr.innerHTML = `
        <td class="fw-600">Vòng ${esc(item.round)}</td>
        <td class="text-emerald fw-600">${formatAcc(item.global_accuracy)}</td>
        <td class="text-sky">${formatDec(item.global_loss)}</td>
        <td>${formatDec(item.global_macro_f1)}</td>
        <td class="text-purple">${item.aggregation_time_ms !== undefined ? `${fmtNum(item.aggregation_time_ms, 1)} ms` : "—"}</td>
        <td class="text-muted">${item.round_duration_s !== undefined ? `${fmtNum(item.round_duration_s, 1)} s` : "—"}</td>
        <td class="${cls}">● ${label}</td>`;
    } else {
      tr.innerHTML = `
        <td class="fw-600">Epoch ${esc(item.epoch)}</td>
        <td class="text-sky">${formatAcc(item.train_accuracy)}</td>
        <td>${formatDec(item.train_loss)}</td>
        <td class="text-emerald fw-600">${formatAcc(item.val_accuracy)}</td>
        <td>${formatDec(item.val_loss)}</td>
        <td>${formatDec(item.val_macro_f1)}</td>
        <td class="text-muted">${item.epoch_time_s !== undefined ? `${fmtNum(item.epoch_time_s, 1)} s` : "—"}</td>
        <td class="${cls}">● ${label}</td>`;
    }
    frag.appendChild(tr);
  });
  body.replaceChildren(frag);
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
  setPollInterval(running && visible ? 1500 : 4000);
}

// ───────── Cập nhật từ poll trạng thái ─────────
function applyStatus(_data, { task, network, finished, stale }) {
  if (stale) return; // poll đầu tiên sau khi bấm Bắt đầu có thể còn trả tác vụ cũ: không vẽ lại trạng thái cũ
  const kind = kindOfTask(task);
  const running = task.status === "running";
  const start = byId("btn-train-start");
  const stop = byId("btn-train-stop");

  setText(byId("dash-clients"), `${clientView(task, network).connected} / 5`);
  setText(byId("dash-rtt"), Number.isFinite(network.mean_probe_rtt_ms) ? `${fmtNum(network.mean_probe_rtt_ms, 2)} ms (localhost)` : "—");
  renderTopology(task, network);

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
    if (kind && task.live_progress) renderProgress(kind, task.live_progress, true, task);
    byId("train-progress-fill").classList.toggle("fed-fill", kind !== "centralized");
    return;
  }

  if (start.disabled) {
    setDisabled(start, false);
    setHtml(start, `${PLAY} Bắt đầu`);
  }
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
  if (finished) {
    loadNetworkMetrics();
    if (kind && task.live_progress) renderProgress(kind, task.live_progress, false, task);
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
      clientBody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">Chưa có nhật ký từng client.</td></tr>';
      setText(summary, "Kiểm tra cuối trên tập ảnh giữ riêng: chưa thực hiện trong phiên này.");
      return;
    }
    setText(runContext, data.historical_run
      ? `Phiên chạy thử gần nhất đã kiểm: ${String(data.run_dir || "").split(/[\\/]/).pop()} · 2 vòng, 5 client; không phải phiên đang chạy.`
      : "Phiên Flower hiện tại · dữ liệu tổng hợp, không dùng làm kết quả khoa học.");
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
    setText(summary, tm
      ? `Kiểm tra cuối trên tập ảnh giữ riêng — độ chính xác ${fmtNum(Number(tm.accuracy) * 100, 2)}% · loss ${formatDec(tm.loss)} · macro-F1 ${formatDec(tm.macro_f1)}`
      : "Kiểm tra cuối trên tập ảnh giữ riêng: chưa thực hiện (cần cấu hình thư mục ảnh đánh giá).");
    const clientRows = (data.audit_rounds || []).flatMap(round => (round.clients || []).map(client => ({ round: round.round, ...client })));
    clientBody.innerHTML = clientRows.length ? clientRows.map(row => `
      <tr>
        <td>${esc(row.round)}</td><td>Client ${esc(row.client_id)}</td>
        <td>${ms(row.probe_rtt_ms)}</td>
        <td>${row.local_train_time_s !== undefined ? `${fmtNum(row.local_train_time_s, 3)} s` : "—"}</td>
        <td>${esc(row.data_mode || "—")}</td><td>${row.training_seed !== undefined ? esc(row.training_seed) : "—"}</td>
        <td title="${esc(row.partition_config_sha256 || "")}">${esc(row.partition_id || "—")}</td>
        <td>${formatDec(row.received_update_norm)}</td>
        <td>${esc(row.status || "—")}</td><td>${esc(row.validation_reason || "—")}</td>
      </tr>`).join("") : '<tr><td colspan="10" class="text-center text-muted">Chưa có nhật ký từng client.</td></tr>';
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">Không đọc được số liệu phiên chạy: ${esc(error.message)}</td></tr>`;
    clientBody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">${esc(error.message)}</td></tr>`;
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
    setText(byId("fl-engine-help"), "Một server và năm tiến trình client localhost, ảnh tổng hợp, cố định hai vòng và một epoch cục bộ. RTT đo trên cùng máy.");
  } else {
    set("fl-rounds", { min: "1", max: "50", disabled: false });
    set("fl-epochs", { min: "1", max: "10", disabled: false });
    setText(byId("fl-engine-help"), "Simulation chạy các client trong cùng process; không dùng để chứng minh độ trễ mạng thật.");
  }
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
  const kind = mode;
  const head = byId("train-progress-head");
  if (head.dataset.kind !== kind && !(currentTask()?.status === "running")) {
    head.innerHTML = HEAD[kind];
    head.dataset.kind = kind;
    byId("train-progress-body").innerHTML = EMPTY_ROW[kind];
    delete byId("train-progress-body").dataset.sig;
  }
}

function resetAfterFailedStart() {
  const start = byId("btn-train-start");
  setDisabled(start, false);
  setHtml(start, `${PLAY} Bắt đầu`);
  setDisplay(byId("btn-train-stop"), "none");
  toggleHidden(byId("train-progress-box"), true);
}

async function start() {
  const button = byId("btn-train-start");
  const stop = byId("btn-train-stop");
  const federated = state.mode === "federated";
  button.disabled = true;
  setHtml(button, `${SPINNER} ${federated ? "Đang điều phối FedAvg…" : "Đang huấn luyện…"}`);
  setDisplay(stop, "inline-flex");
  stop.disabled = false;
  setHtml(stop, `${STOP} Dừng tiến trình`);
  toggleHidden(byId("train-progress-box"), false);
  setBar(byId("train-progress-fill"), 5);
  setText(byId("train-progress-label"), "Tiến trình: 5%");
  setText(byId("train-progress-detail"), federated ? "Đang khởi tạo 5 trạm client…" : "Đang khởi tạo mô hình…");
  const body = byId("train-progress-body");
  delete body.dataset.sig;
  body.innerHTML = `<tr><td colspan="8" class="text-center text-muted">${federated ? "Đang phân phối trọng số khởi tạo toàn cục tới 5 client…" : "Đang khởi tạo kiến trúc mạng và phân bổ batch…"}</td></tr>`;
  byId("train-progress-head").innerHTML = HEAD[state.mode];
  byId("train-progress-head").dataset.kind = state.mode;

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
      });
    } else {
      data = await postJson("/api/centralized/train", {
        epochs: num("cent-epochs"),
        seed: num("cent-seed"),
        smoke: byId("cent-smoke-val").value === "true",
      });
    }
    markTaskStarted();
    notify("info", `Đã bắt đầu huấn luyện ${federated ? "FedAvg" : "Centralized"}`, data.message || "");
    await pollStatus();
  } catch (error) {
    notify("error", `Không bắt đầu được huấn luyện ${federated ? "FedAvg" : "Centralized"}`, error.message);
    resetAfterFailedStart();
  }
}

async function stopTraining() {
  const stop = byId("btn-train-stop");
  stop.disabled = true;
  setHtml(stop, `${SPINNER} Đang dừng…`);
  try {
    const data = await postJson("/api/train/stop", {});
    if (data.stopped === false) {
      notify("warning", "Không có tác vụ để dừng", data.message || "Hiện không có tiến trình huấn luyện nào đang chạy.");
    } else {
      notify("info", "Đã gửi lệnh dừng", data.message || "Tiến trình sẽ dừng ở điểm kiểm tra gần nhất.");
    }
    await pollStatus();
  } catch (error) {
    stop.disabled = false;
    setHtml(stop, `${STOP} Dừng tiến trình`);
    notify("error", "Không dừng được huấn luyện", formatApiError(error.body, error.message));
  }
}

export async function mount() {
  byId("topology").innerHTML = diagramHtml({ variant: "live", serverLabel: "Máy chủ", serverSub: "chưa chạy" });
  $$("[data-train-mode]").forEach(button => button.addEventListener("click", () => setTrainMode(button.dataset.trainMode)));
  byId("fl-engine").addEventListener("change", syncEngine);
  byId("btn-train-start").addEventListener("click", start);
  byId("btn-train-stop").addEventListener("click", stopTraining);
  byId("btn-refresh-network").addEventListener("click", loadNetworkMetrics);
  byId("layer-selector").addEventListener("change", loadLayerStats);
  byId("weight-inspector").addEventListener("toggle", event => {
    if (event.target.open && !state.layerLoaded) loadLayerList();
  });
  syncEngine();
  setTrainMode("federated");
  onStatus(applyStatus);
  pollStatus(); // đồng bộ ngay trạng thái hiện tại (không đợi chu kỳ 4 giây)
  await loadNetworkMetrics();
}
