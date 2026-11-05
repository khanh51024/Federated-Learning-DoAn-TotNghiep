// Demo · Flower: lần đo thật đang chạy (hoặc vừa chạy). Chạy bằng terminal và .bat; trang này chỉ đọc /api/flower/live.
// Máy chủ tính sẵn mọi số trong src/federated/run_view.py, đúng hàm mà scripts/demo_show.py dùng, nên web và terminal luôn trùng nhau.
import { ADMIN_PAGE, clearCredential, readCredential, storedAuthHeader } from "../core/admin-auth.js";
import { getJson, postJson } from "../core/api.js";
import { makeChart, pillarColors } from "../core/charts.js";
import { renderRunsCompare, runLabel } from "../core/demo-compare.js";
import { renderFlowDiagram } from "../core/flow-diagram.js";
import { byId, copyButton, copyFromButton, esc, fmtInt, fmtNum, notify, setBar } from "../core/dom.js";

const ACTIVE_POLL_MS = 3000;
const IDLE_POLL_MS = 15000;   // đã xong hoặc chưa có lần đo: vẫn hỏi chậm để tự thấy lần đo mới
const TIMEOUT_MS = 12000;
const DETAIL_DEBOUNCE_MS = 150;
const FOLLOW_MS = 90000;
const CLIENT_LINK = "cropfed-demo:start";     // CAI_NUT_WEB.bat đăng ký; lệnh trên máy không nhận gì từ địa chỉ này
const LIGHT_CLIENT_LINK = "cropfed-demo-nhe:start";   // CAI_NUT_WEB.bat: mở 5 client bằng bộ ảnh nhẹ
const MAX_PER_MACHINE = 5;    // máy chủ và client_agent.py cũng chặn đúng số này
const LIGHT_IMAGE_SETS = 10;  // bộ ảnh nhẹ chỉ có ảnh của 10 client
const PRESETS = [
  { rounds: 10, light: true, label: "Mỗi máy 5 client, 10 vòng, bộ ảnh nhẹ" },   // số client tính theo số máy đã chọn
  { clients: 7, rounds: 3, label: "7 client, 3 vòng (2 laptop)" },
  { clients: 5, rounds: 1, label: "5 client, 1 vòng (chỉ laptop 1)" },
];
const ROUND_BOUND = /^(client:|server$)/;   // các mục này đổi theo vòng đang xem
const AGENT_STATES = { ready: "sẵn sàng", starting: "đang mở client", running: "đang chạy client", done: "đã chạy xong client", failed: "có client lỗi, xem cửa sổ client trên máy đó", offline: "mất kết nối" };

let timer = 0;
let loading = false;
let view = null;
let chosenRun = "";           // rỗng = luôn theo lần đo mới nhất
let chosenRound = null;       // null = luôn theo vòng mới nhất
let built = false;
let lastSignature = "";
let selected = null;          // mục đang mở, ví dụ "client:7"; null = không mở
let detailKey = "";
let detailToken = 0;
let detailTimer = 0;
let demo = { can_start: false, running: false };
let clientPackage = null;     // gói client trên máy chủ cho nút tải bộ cài; null = chưa biết hoặc không có
let demoBusy = false;         // đang chờ máy chủ trả lời nút Bắt đầu hoặc Dừng
let followUntil = 0;          // vừa bấm Bắt đầu: hỏi nhanh để thấy ngay lần đo mới
let lastControl = null;
let chosenClient = null;      // client đang xem trong khung các bước kiểm; null = client đầu tiên
let runsKey = "";             // biểu đồ so các lần đo chỉ hỏi lại khi có lần đo mới hoặc xong thêm vòng
let agents = [];              // máy đã cài bộ client và đang báo danh
let pickedMachines = new Set();    // máy quản trị viên đã đánh dấu. Không chọn sẵn: ai cũng báo danh được, máy lạ được chọn sẵn sẽ làm lần đo chờ mãi
let chosenPreset = 0;

const num = (value, digits = 2) => (value === null || value === undefined ? "—" : fmtNum(value, digits));
const isActive = state => state === "waiting" || state === "running" || state === "stale";
const megabytes = bytes => fmtNum(bytes / (1024 * 1024), 0);
const when = iso => (iso ? new Date(iso).toLocaleString("vi-VN") : "—");

function bannerHtml(data) {
  const total = data.total_rounds ?? "?";
  const done = data.completed_rounds;
  const replies = (data.round_replies || []).length;
  const states = {
    waiting: { tone: "is-unknown", title: `Đang chờ client: ${data.connected_clients ?? 0}/${data.num_clients}`, meta: `Máy chủ chỉ bắt đầu khi đủ ${data.min_clients} client, nên không client nào vào trễ.` },
    running: { tone: "is-correct", title: `Đang học vòng ${data.round_in_progress}/${total}`, meta: `${replies}/${data.round_participants ?? data.num_clients} client đã gửi trọng số về ở vòng này.` },
    finished: { tone: "is-correct", title: `Đã xong ${done}/${total} vòng`, meta: `${data.num_clients} client từ ${data.checks.machines} máy${data.checks.host_check_available && data.checks.clients_on_server_host === 0 ? ", máy chủ tách riêng, không client nào chung máy" : ""}. Kết thúc lúc ${when(data.finished_at_utc)}.` },
    failed: data.error?.startsWith("Đã dừng bằng nút")
      ? { tone: "is-unknown", title: "Lần đo đã dừng", meta: `${data.error} Cổng đo đã đóng.` }
      : { tone: "is-wrong", title: `Lần đo dừng vì lỗi${data.failed_round ? ` ở vòng ${data.failed_round}` : ""}`, meta: data.error || "Xem nhật ký máy chủ." },
    stale: { tone: "is-wrong", title: "Mất liên lạc với máy chủ đo", meta: `${data.seconds_since_update ?? "?"} giây chưa có cập nhật. Máy chủ có thể đã dừng; số dưới đây là tới lần cập nhật cuối.` },
  };
  const state = states[data.state] || states.running;
  const percent = total === "?" ? 0 : Math.min(100, Math.round((done / total) * 100));
  return `<div class="diag-headline ${state.tone}" role="status" aria-live="polite">
      <p class="flower-verdict">${esc(state.title)}</p>
      <p class="headline-meta">${esc(state.meta)}</p>
      <div class="progress-track mt-3" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}" aria-label="Số vòng đã xong">
        <div class="progress-fill" id="live-progress-fill" data-percent="${percent}"></div>
      </div>
    </div>`;
}

function detailAttrs(item, panel) {
  const open = selected === item;
  return `data-detail="${esc(item)}" data-selected="${open}" aria-expanded="${open}" aria-controls="${panel}"`;
}

function chipState(data, clientId, index) {
  if (data.state === "waiting") return index < (data.connected_clients ?? 0) ? "is-connected" : "is-idle";
  if (data.state === "finished") return "is-done";
  if (data.state === "failed" || data.state === "stale") return "is-idle";
  return (data.round_replies || []).some(reply => reply.client_id === clientId) ? "is-done" : "is-learning";
}

function chipHtml(data, clientId, index) {
  const reply = (data.round_replies || []).find(item => item.client_id === clientId);
  const detail = data.state === "running" && reply ? `${fmtNum(reply.response_ms / 1000, 1)} s` : "";
  return `<button type="button" class="live-chip ${chipState(data, clientId, index)}" ${detailAttrs(`client:${clientId}`, "live-stage-detail")} aria-label="Xem chi tiết client ${esc(clientId)}"><b>${esc(String(clientId).padStart(2, "0"))}</b>${detail ? `<small>${esc(detail)}</small>` : ""}</button>`;
}

function groupHtml(data, title, sub, ids, offset, item = null) {
  const head = `<span class="live-group-title">${esc(title)}</span>${sub ? `<span class="live-group-sub">${esc(sub)}</span>` : ""}`;
  return `<div class="live-group">
      ${item ? `<button type="button" class="live-group-head" ${detailAttrs(item, "live-stage-detail")}>${head}</button>` : `<div class="live-group-head">${head}</div>`}
      <div class="live-chips">${ids.map((id, i) => chipHtml(data, id, offset + i)).join("")}</div>
    </div>`;
}

// Máy đã nhận phần việc của lần đo đang chờ. Chỉ biết máy nào được giao, chưa biết từng client đã nối chưa, nên ô client chưa tô; số đã nối nằm ở dòng trên cùng.
function assignedAgents(data) {
  return data.state === "waiting" && !chosenRun ? agents.filter(agent => agent.clients.length) : [];
}

function stageHtml(data) {
  // Trước vòng đầu chưa biết client nào ở máy nào: hiện chung một nhóm, đếm số đã nối.
  const machines = data.machines || [];
  const assigned = assignedAgents(data);
  const groups = machines.length
    ? machines.map(machine => groupHtml(data, machine.label,
        machine.same_host_as_server ? "CÙNG MÁY VỚI MÁY CHỦ" : (machine.rtt_ms === null ? "" : `độ trễ ${fmtNum(machine.rtt_ms, 1)} ms`), machine.client_ids, 0, `machine:${machine.label.split(" ").pop()}`))
    : assigned.length
      ? assigned.map(agent => groupHtml(data, agent.label, `mã ${agent.code}, ${AGENT_STATES[agent.state]}`, agent.clients, data.num_clients))
      : [groupHtml(data, "Client", "chưa biết máy nào, sẽ rõ sau vòng đầu", Array.from({ length: data.num_clients || 0 }, (_unused, i) => i), 0)];
  const flowing = data.state === "running" ? " is-flowing" : "";
  const middle = Math.ceil(groups.length / 2);
  return `<div class="live-stage">
      <div class="live-side">${groups.slice(0, middle).join("")}</div>
      <div class="live-link${flowing}" aria-hidden="true"></div>
      <button type="button" class="live-server${flowing}" ${detailAttrs("server", "live-stage-detail")}>
        <span class="live-group-title">Máy chủ</span>
        <span class="live-group-sub">gộp trọng số theo số ảnh, tự kiểm độ chính xác, tự đo độ trễ</span>
        ${data.state === "running" && data.round_in_progress ? `<span class="live-server-round">Vòng ${esc(data.round_in_progress)}</span>` : ""}
      </button>
      ${groups.length > 1 ? `<div class="live-link${flowing}" aria-hidden="true"></div><div class="live-side">${groups.slice(middle).join("")}</div>` : ""}
    </div>
    <p class="field-hint mt-2"><span class="live-key is-learning"></span> đang học <span class="live-key is-done"></span> đã gửi về <span class="live-key is-connected"></span> đã nối <span class="live-key is-idle"></span> chưa nối hoặc đã dừng</p>
    <p class="field-hint">Bấm vào một client, một máy hoặc máy chủ để xem cách làm và dòng lệnh.</p>`;
}

function tile(item, label, value, note, ok = null) {
  const mark = ok === null ? "" : ok ? " is-ok" : " is-bad";
  return `<button type="button" class="kpi-mini live-tile${mark}" ${detailAttrs(item, "live-proofs-detail")}><span class="kpi-mini-label">${esc(label)}</span><span class="kpi-mini-value">${esc(value)}</span><span class="live-tile-note">${esc(note)}</span></button>`;
}

function proofsHtml(data) {
  const c = data.checks;
  const last = data.rounds[data.rounds.length - 1];
  const accuracy = data.final_test?.accuracy ?? last?.accuracy;
  const onServer = c.host_check_available ? String(c.clients_on_server_host) : "chưa kiểm";
  return `<div class="live-tiles">
      ${tile("hash", "Trọng số gửi đúng", `${fmtInt(c.hash_ok)}/${fmtInt(c.hash_total)}`, c.chain_ok ? "mã băm khớp, chuỗi mô hình liền" : "lượt mã băm khớp", c.hash_total ? c.hash_ok === c.hash_total : null)}
      ${tile("norm", "Độ biến thiên khớp", `${fmtInt(c.norm_ok)}/${fmtInt(c.norm_total)}`, "client khai so với máy chủ tự tính", c.norm_total ? c.norm_ok === c.norm_total : null)}
      ${tile("rounds", "Vòng nhận đủ client", `${fmtInt(c.full_rounds ?? 0)}/${fmtInt(c.rounds_total)}`, `đủ ${data.num_clients} client mỗi vòng`, c.rounds_total ? c.full_rounds === c.rounds_total : null)}
      ${tile("validator", "Trình kiểm lần đo", data.validation ? `${fmtInt(data.validation.passed)}/${fmtInt(data.validation.total)}` : "chưa có", "đọc lại tệp kết quả và tệp mô hình", data.validation ? data.validation.ok : null)}
      ${data.tamper_check ? tile("tamper", "Thử cố tình sai", "Bị loại", `client ${data.tamper_check.client_id}: ${data.tamper_check.client_hash.slice(0, 8)} khác ${data.tamper_check.server_hash.slice(0, 8)}`, true) : ""}
      ${tile("host", "Client chung máy chủ", onServer, `client đến từ ${fmtInt(c.machines)} máy`, c.host_check_available ? c.clients_on_server_host === 0 : null)}
      ${tile("bytes", "Đã truyền", `${megabytes(c.bytes_up + c.bytes_down)} MiB`, `lên ${megabytes(c.bytes_up)}, xuống ${megabytes(c.bytes_down)}`)}
      ${tile("accuracy", data.final_test ? "Độ chính xác cuối, tập test" : "Độ chính xác, tập val", accuracy === null || accuracy === undefined ? "—" : `${fmtNum(accuracy * 100, 2)}%`, data.final_test ? "máy chủ tự tính, số đo trên tập nhỏ" : `máy chủ tự tính sau vòng ${last?.round ?? "—"}`)}
    </div>`;
}

function accuracyConfig() {
  const colors = pillarColors();
  const rows = view.rounds;
  return {
    type: "line",
    data: {
      labels: rows.map(r => r.round),
      datasets: [
        { label: "Độ chính xác máy chủ tính (%)", data: rows.map(r => (r.accuracy === null ? null : r.accuracy * 100)), borderColor: colors.fedavg, backgroundColor: colors.fedavg, tension: 0.2, pointRadius: 3, yAxisID: "y" },
        { label: "Loss", data: rows.map(r => r.loss), borderColor: colors.danger, backgroundColor: colors.danger, tension: 0.2, pointRadius: 2, yAxisID: "loss" },
      ],
    },
    options: {
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Vòng" } },
        y: { min: 0, max: 100, title: { display: true, text: "Độ chính xác (%)" } },
        loss: { position: "right", min: 0, grid: { drawOnChartArea: false }, title: { display: true, text: "Loss" } },
      },
    },
  };
}

function timeConfig() {
  const colors = pillarColors();
  const rows = view.rounds;
  const series = (label, key, color) => ({ label, data: rows.map(r => r[key]), backgroundColor: color, stack: "round" });
  return {
    type: "bar",
    data: {
      labels: rows.map(r => r.round),
      datasets: [
        series("Học và chờ client chậm nhất", "learn_and_wait_s", colors.fedavg),
        series("Truyền (ước lượng)", "transfer_s", colors.centralized),
        series("Máy chủ kiểm, gộp, đánh giá", "server_s", colors.local),
      ],
    },
    options: {
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: { x: { stacked: true, title: { display: true, text: "Vòng" } }, y: { stacked: true, min: 0, title: { display: true, text: "Giây" } } },
    },
  };
}

function normConfig() {
  const colors = pillarColors();
  const rows = view.rounds;
  const line = (label, key, color) => ({ label, data: rows.map(r => r[key]), borderColor: color, backgroundColor: color, tension: 0.2, pointRadius: 3 });
  return {
    type: "line",
    data: {
      labels: rows.map(r => r.round),
      datasets: [
        line("Số máy chủ dùng để kiểm (có cả thống kê BatchNorm)", "mean_update_norm", colors.fedavg),
        line("Riêng phần trọng số học được", "mean_learnable_update_norm", colors.centralized),
      ],
    },
    options: {
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: { x: { title: { display: true, text: "Vòng" } }, y: { min: 0, title: { display: true, text: "Độ biến thiên trung bình" } } },
    },
  };
}

const NORM_LINES = [
  { key: "update_norm_server", label: "Có cả thống kê BatchNorm", color: "fedavg" },
  { key: "learnable_update_norm", label: "Riêng trọng số học được", color: "centralized" },
];

// Giống đường trung bình: client bị loại ở vòng nào thì vòng đó không có điểm.
function acceptedNorm(row, clientId, key) {
  const client = row.clients.find(item => item.client_id === clientId);
  return client?.status === "ok" ? client[key] : null;
}

// Cùng một thang dọc cho mọi client thì mới so được hai biểu đồ với nhau.
function sharedNormMax() {
  const values = view.rounds.flatMap(row => row.clients.filter(client => client.status === "ok").flatMap(client => NORM_LINES.map(line => client[line.key]))).filter(value => value !== null);
  const top = Math.max(0, ...values);
  if (!top) return undefined;
  const step = 10 ** Math.floor(Math.log10(top)) / 2;
  return Math.ceil(top / step) * step;
}

function clientNormConfig(clientId) {
  const colors = pillarColors();
  const rows = view.rounds;
  return {
    type: "line",
    data: {
      labels: rows.map(r => r.round),
      datasets: NORM_LINES.map(line => ({ label: line.label, data: rows.map(r => acceptedNorm(r, clientId, line.key)), borderColor: colors[line.color], backgroundColor: colors[line.color], borderWidth: 2, tension: 0.2, pointRadius: 2 })),
    },
    options: {
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },     // chú thích chung nằm trên lưới biểu đồ
        tooltip: { callbacks: { title: items => `Client ${clientId}, vòng ${items[0].label}` } },
      },
      scales: {
        x: { title: { display: true, text: "Vòng" }, ticks: { maxRotation: 0, autoSkipPadding: 8 } },
        y: { min: 0, max: sharedNormMax(), ticks: { maxTicksLimit: 5 } },
      },
    },
  };
}

function clientNormNote(clientId) {
  const owner = view.rounds.flatMap(row => row.clients).find(client => client.client_id === clientId);
  const rejected = view.rounds.filter(row => row.clients.some(client => client.client_id === clientId && client.status !== "ok")).map(row => row.round);
  return [owner?.machine, rejected.length ? `bị loại vòng ${rejected.join(", ")}` : ""].filter(Boolean).join(", ");
}

function renderClientNorms() {
  const host = byId("live-client-norms");
  if (!host) return;
  const ids = [...new Set(view.rounds.flatMap(row => row.clients.map(client => client.client_id)))].sort((a, b) => a - b);
  const cards = ids.map(id => ({ id, note: clientNormNote(id) }));
  const shape = JSON.stringify(cards);
  if (host.dataset.shape !== shape) {       // chỉ dựng lại khung khi đổi client hoặc đổi dòng ghi chú, còn lại chỉ thay số
    host.dataset.shape = shape;
    host.innerHTML = cards.length
      ? cards.map(card => `<figure class="live-norm-card">
          <figcaption class="live-norm-head"><b>Client ${esc(card.id)}</b>${card.note ? `<span class="field-hint">${esc(card.note)}</span>` : ""}</figcaption>
          <div class="live-norm-chart"><canvas id="live-chart-norm-${esc(card.id)}" aria-label="Độ biến thiên của client ${esc(card.id)} theo vòng"></canvas></div>
        </figure>`).join("")
      : '<p class="field-hint">Chưa xong vòng nào.</p>';
  }
  ids.forEach(id => refreshChart(`live-chart-norm-${id}`, () => clientNormConfig(id)));
}

function refreshChart(id, factory) {
  const canvas = byId(id);
  if (!canvas) return;
  const chart = window.Chart?.getChart?.(canvas);
  if (!chart) { makeChart(canvas, factory); return; }
  const next = factory();
  chart.data.labels = next.data.labels;
  next.data.datasets.forEach((dataset, index) => { chart.data.datasets[index].data = dataset.data; });
  chart.options.scales.y.max = next.options.scales.y.max;   // thang chung của các client có thể đổi khi có vòng mới
  chart.update("none");
}

function shownRound() {
  const rows = view.rounds;
  if (!rows.length) return null;
  return rows.find(row => row.round === chosenRound) || rows[rows.length - 1];
}

function clientRow(client) {
  const norm = client.update_norm_reported === null
    ? "—"
    : `${num(client.update_norm_reported, 3)} / ${num(client.update_norm_server, 3)} ${client.update_norm_matches === true ? "✓" : "✕"}`;
  const bad = client.status && client.status !== "ok" ? ' class="is-bad-row"' : "";
  return `<tr${bad}>
      <td>${esc(client.client_id)}</td><td class="live-nowrap">${esc(client.machine)}${client.same_host_as_server ? " (cùng máy chủ)" : ""}</td><td>${esc(fmtInt(client.num_examples))}</td>
      <td>${num(client.rtt_ms, 1)}</td><td>${num(client.up_mb_s)}</td><td>${num(client.down_mb_s)}</td>
      <td>${num(client.response_s, 1)}</td><td>${num(client.train_s, 1)}</td><td>${norm}</td>
      <td class="live-nowrap"><span class="mono" title="client ${esc(client.client_hash)} / máy chủ ${esc(client.server_hash)}">${esc(client.client_hash.slice(0, 8))} / ${esc(client.server_hash.slice(0, 8))}</span> ${client.hash_ok ? "✓" : "✕"}</td>
    </tr>`;
}

function stepRow(text, ok = null) {
  const mark = ok === null ? "" : ok ? '<span class="live-step-ok">✓ khớp</span>' : '<span class="live-step-bad">✕ lệch</span>';
  return `<li>${text} ${mark}</li>`;
}

const shortHash = value => `<span class="mono">${esc(String(value ?? "").slice(0, 8))}</span>`;

function stepsHtml(row) {
  const client = row.clients.find(item => item.client_id === chosenClient) || row.clients[0];
  if (!client) return "";
  const next = view.rounds.find(item => item.round === row.round + 1);
  const total = row.clients.reduce((sum, item) => sum + (item.num_examples || 0), 0);
  const share = total ? fmtNum((client.num_examples / total) * 100, 1) : "—";
  const merge = row.aggregation_max_diff === null || row.aggregation_max_diff === undefined
    ? stepRow(`Máy chủ gộp theo số ảnh, client này chiếm ${share}%. Kiểm lại phép gộp: lần đo này chưa ghi.`)
    : stepRow(`Máy chủ gộp theo số ảnh, client này chiếm ${share}%. Tính lại phép gộp độc lập, lệch lớn nhất ${esc(row.aggregation_max_diff.toExponential(1))}.`, row.aggregation_max_diff <= 1e-6);
  const end = next
    ? stepRow(`Mô hình mới ${shortHash(row.global_hash)}, vòng ${esc(next.round)} bắt đầu từ ${shortHash(next.parent_hash)}.`, row.global_hash === next.parent_hash)
    : stepRow(`Mô hình mới ${shortHash(row.global_hash)}${view.state === "finished" ? ", là mô hình cuối của lần đo" : ""}.`, view.state === "finished" && view.final_model_hash ? view.final_model_hash.startsWith(row.global_hash) : null);
  return `<div class="live-steps-box">
      <div class="live-control-row">
        <h4 class="live-detail-title">Một vòng đi qua những bước kiểm nào</h4>
        <label class="field-hint" for="live-step-client">Xem client</label>
        <select id="live-step-client" class="form-control-sm">${row.clients.map(item => `<option value="${esc(item.client_id)}"${item.client_id === client.client_id ? " selected" : ""}>${esc(item.client_id)}</option>`).join("")}</select>
      </div>
      <ol class="live-steps">
        ${stepRow(`Máy chủ gửi mô hình ${shortHash(row.parent_hash)} xuống client.`)}
        ${client.reported_parent ? stepRow(`Client báo đã nhận ${shortHash(client.reported_parent)}.`, client.parent_hash_ok === true) : stepRow("Client báo mô hình đã nhận: lần đo này chưa ghi mã.")}
        ${stepRow(`Client học ${esc(fmtInt(client.num_examples))} ảnh. Ảnh không rời khỏi máy.`)}
        ${stepRow(`Client gửi trọng số, kèm mã băm ${shortHash(client.client_hash)}.`)}
        ${stepRow(`Máy chủ băm lại trọng số nhận được, ra ${shortHash(client.server_hash)}.`, client.hash_ok)}
        ${stepRow(`Độ biến thiên: client khai ${num(client.update_norm_reported, 3)}, máy chủ tự tính ${num(client.update_norm_server, 3)}.`, client.update_norm_matches === true)}
        ${merge}
        ${end}
      </ol>
      <p class="field-hint">Máy chủ còn so số vòng, seed, số epoch, danh sách ảnh của client và ảnh thật. Sai một mục là client đó bị loại.</p>
    </div>`;
}

function roundTableHtml() {
  const row = shownRound();
  if (!row) return '<p class="field-hint">Chưa xong vòng nào.</p>';
  const count = view.rounds.length;
  const accuracy = row.accuracy === null ? "" : ` · máy chủ tính độ chính xác trên tập val ${fmtNum(row.accuracy * 100, 2)}%`;
  return `<div class="live-round-pick">
      <label for="live-round">Xem vòng <b id="live-round-value">${esc(row.round)}</b>/${esc(count)}</label>
      <input type="range" id="live-round" min="${esc(view.rounds[0].round)}" max="${esc(view.rounds[count - 1].round)}" value="${esc(row.round)}" step="1">
      <span class="field-hint">cả vòng ${num(row.elapsed_s, 1)} s${accuracy}</span>
    </div>
    ${stepsHtml(row)}
    <div class="table-responsive mt-3" tabindex="0" role="region" aria-label="Bảng từng client trong vòng đang xem"><table class="data-table wrap-head">
      <thead><tr><th>Client</th><th>Máy</th><th>Ảnh</th><th>RTT (ms)</th><th>Lên (MiB/s)</th><th>Xuống (MiB/s)</th><th>Phản hồi (s)</th><th>Học, ước lượng (s)</th><th>Độ biến thiên: client khai / máy chủ tính</th><th>Mã băm: client gửi / máy chủ tính</th></tr></thead>
      <tbody>${row.clients.map(clientRow).join("")}</tbody>
    </table></div>
    <p class="field-hint mt-2">Mã băm giống dấu vân tay của bộ trọng số: chỉ cần đổi một con số là mã đổi hẳn. Client tính mã trước khi gửi, máy chủ tính lại sau khi nhận. Hai mã giống nhau nghĩa là trọng số đến nơi nguyên vẹn. Mỗi vòng mỗi client học ra trọng số khác nhau nên mã giữa các hàng khác nhau là đúng.</p>`;
}

function machineTableHtml() {
  if (!view.machines.length) return "";
  return `<div class="table-responsive mt-2" tabindex="0" role="region" aria-label="Bảng theo từng máy"><table class="data-table wrap-head">
      <thead><tr><th>Máy</th><th>Client</th><th>RTT (ms)</th><th>Lên (MiB/s)</th><th>Xuống (MiB/s)</th><th>Học, ước lượng (s)</th></tr></thead>
      <tbody>${view.machines.map(m => `<tr><td>${esc(m.label)}${m.same_host_as_server ? " (cùng máy chủ)" : ""}</td><td>${esc(m.client_ids.join(", "))}</td><td>${num(m.rtt_ms, 1)}</td><td>${num(m.up_mb_s)}</td><td>${num(m.down_mb_s)}</td><td>${num(m.train_s, 1)}</td></tr>`).join("")}</tbody>
    </table></div><p class="field-hint mt-2">Trung vị qua các vòng. Mọi thời gian và tốc độ do máy chủ tự đo; thời gian học là ước lượng của máy chủ vì client không gửi thời gian.</p>`;
}

function pickerHtml(data) {
  const runs = data.runs || [];
  if (runs.length < 2) return "";
  return `<label class="field-hint">Lần đo <select id="live-run" class="form-control-sm">
      <option value="">Mới nhất</option>
      ${runs.map(id => `<option value="${esc(id)}"${id === chosenRun ? " selected" : ""}>${esc(runLabel(id))}</option>`).join("")}
    </select></label>`;
}

function emptyHtml() {
  return `<div class="callout callout-warn" role="note">
      <b>Chưa có lần đo thật nào trên máy chủ này.</b> Khi chạy lần đo bằng terminal (<code class="code">scripts/remote/run_network_server.sh</code> trên máy chủ, <code class="code">CHAY_CLIENT.bat</code> trên các laptop), trang này tự cập nhật theo từng vòng, không cần tải lại.
    </div>`;
}

function detailPanelId(item) {
  return /^(client:|machine:|server$)/.test(item) ? "live-stage-detail" : "live-proofs-detail";
}

function terminalText(lines) {
  return lines.map(line => esc(line)).join("\n");
}

function terminalHtml(data) {
  const title = "Số của lần đo này, chạy lại được trên VM";
  return `<div class="live-term">
      <div class="live-term-bar"><span>${title}</span>${data.command ? copyButton(data.command, "lệnh") : ""}</div>
      <pre class="live-term-body mono" id="live-term-main" tabindex="0" aria-label="${title}">${data.command ? `<span class="live-term-cmd">$ ${esc(data.command)}</span>\n` : ""}${terminalText(data.lines)}</pre>
    </div>`;
}

function logTerminalHtml(log) {
  return `<div class="live-term">
      <div class="live-term-bar"><span>${esc(log.title)}</span></div>
      <pre class="live-term-body mono" id="live-term-log" tabindex="0" aria-label="${esc(log.title)}">${terminalText(log.lines)}</pre>
    </div>`;
}

function detailHtml(data) {
  const roundText = data.round_bound ? ` · ${data.round === null ? "vòng mới nhất" : `vòng ${esc(data.round)}`}, đổi bằng thanh kéo ở bảng từng client` : "";
  return `<div class="live-detail-head">
      <h4 class="live-detail-title">${esc(data.title)}<small>${roundText}</small></h4>
      <button type="button" class="btn btn-sm" data-detail-close>Đóng</button>
    </div>
    <div class="live-detail-grid">
      <div>
        <p class="live-detail-label">Cách làm</p>
        <ol class="live-detail-steps">${data.how.map(step => `<li>${esc(step)}</li>`).join("")}</ol>
        ${data.sent ? `<p class="live-detail-label">Client gửi</p><p class="live-detail-list">${esc(data.sent.join(", "))}</p><p class="live-detail-label">Client không gửi</p><p class="live-detail-list">${esc(data.not_sent.join(", "))}</p>` : ""}
        ${data.note ? `<p class="field-hint">${esc(data.note)}</p>` : ""}
      </div>
      <div class="live-detail-terms">
        ${terminalHtml(data)}
        ${data.log ? logTerminalHtml(data.log) : ""}
        <p class="field-hint">${esc(data.source)}${data.log ? " Nhật ký đã che IP, mã kết nối và đường dẫn." : ""}</p>
      </div>
    </div>`;
}

function clearDetailPanels() {
  for (const id of ["live-stage-detail", "live-proofs-detail"]) {
    const panel = byId(id);
    if (panel) { panel.innerHTML = ""; panel.classList.add("hidden"); }
  }
}

function markSelected() {
  byId("flower-live-root")?.querySelectorAll("[data-detail]").forEach(button => {
    const open = button.dataset.detail === selected;
    button.dataset.selected = String(open);
    button.setAttribute("aria-expanded", String(open));
  });
}

// Giữ vị trí đang đọc khi dữ liệu mới về; đang ở cuối nhật ký thì đi theo dòng mới.
function scrollSnapshot() {
  const main = byId("live-term-main");
  const log = byId("live-term-log");
  return {
    main: main?.scrollTop ?? 0,
    log: log ? { top: log.scrollTop, atEnd: log.scrollTop + log.clientHeight >= log.scrollHeight - 4 } : null,
  };
}

function restoreScroll(saved) {
  const main = byId("live-term-main");
  const log = byId("live-term-log");
  if (main) main.scrollTop = saved.main;
  if (log) log.scrollTop = !saved.log || saved.log.atEnd ? log.scrollHeight : saved.log.top;
}

function detailQuery() {
  const params = new URLSearchParams({ item: selected });
  if (chosenRun) params.set("run", chosenRun);
  if (chosenRound !== null && ROUND_BOUND.test(selected)) params.set("round", String(chosenRound));
  return params.toString();
}

async function loadDetail({ reveal = false } = {}) {
  if (!selected) return;
  const token = ++detailToken;
  const item = selected;
  const panel = byId(detailPanelId(item));
  if (!panel) return;
  try {
    const data = await getJson(`/api/flower/detail?${detailQuery()}`, { timeoutMs: TIMEOUT_MS });
    if (token !== detailToken || selected !== item) return;     // đã bấm mục khác trong lúc chờ
    const saved = scrollSnapshot();
    panel.innerHTML = detailHtml(data);
    panel.classList.remove("hidden");
    restoreScroll(saved);
  } catch (error) {
    if (token !== detailToken || selected !== item) return;
    panel.innerHTML = `<div class="callout callout-error" role="alert"><b>Không đọc được chi tiết.</b> ${esc(error.message)}</div>`;
    panel.classList.remove("hidden");
  }
  if (reveal) panel.scrollIntoView({ block: "nearest" });
}

// Chỉ hỏi lại máy chủ khi có gì mới với mục đang mở, không hỏi theo nhịp 3 giây.
function currentDetailKey() {
  if (!selected || !view) return "";
  const round = ROUND_BOUND.test(selected) ? chosenRound : null;
  return JSON.stringify([selected, chosenRun, round, view.state, view.completed_rounds, (view.round_replies || []).length, view.connected_clients]);
}

function syncDetail() {
  const key = currentDetailKey();
  if (key === detailKey) return;
  detailKey = key;
  if (key) loadDetail();
}

function closeDetail({ refocus = false } = {}) {
  const previous = selected;
  selected = null;
  detailKey = "";
  detailToken += 1;
  clearDetailPanels();
  markSelected();
  if (refocus && previous) byId("flower-live-root")?.querySelector(`[data-detail="${previous}"]`)?.focus();
}

function openDetail(item) {
  if (selected === item) { closeDetail(); return; }
  clearDetailPanels();
  selected = item;
  detailKey = currentDetailKey();
  markSelected();
  loadDetail({ reveal: true });
}

function installHtml() {
  if (!clientPackage?.available) return "";
  return `<div class="live-control-row mt-3">
      <a class="btn btn-secondary" href="/api/flower/client-package/install.bat" download>Tải bộ cài cho máy này</a>
      <span class="field-hint">Chỉ cần một lần, máy Windows. Mở file vừa tải, nó tự tải thêm ${fmtInt(Math.round(clientPackage.bytes / 1000000))} MB và tự cài, không cần quyền quản trị. Cài xong máy tự nối với trang mỗi lần bật, hiện ở mục Máy đã cài bộ client, và quản trị viên có thể cho máy tự mở client để tham gia lần đo; muốn tắt thì chạy GO_NUT_WEB.bat. Windows cảnh báo thì chọn "More info" rồi "Run anyway".</span>
    </div>`;
}

function agentRow(agent) {
  const ram = agent.ram_gb === null ? "" : `, trống ${fmtNum(agent.ram_free_gb, 1)}/${fmtNum(agent.ram_gb, 1)} GB RAM`;
  const clients = agent.clients.length ? `, client ${agent.clients[0]} đến ${agent.clients[agent.clients.length - 1]}` : "";
  return `<li class="live-agent is-${esc(agent.state)}">
      <span class="live-agent-dot" aria-hidden="true"></span>
      <b>${esc(agent.label)}</b>
      <span class="mono">${esc(agent.code)}</span>
      <span>${esc(AGENT_STATES[agent.state] ?? agent.state)}${esc(clients)}</span>
      <span class="field-hint">${agent.cores === null ? "" : `${esc(agent.cores)} nhân`}${esc(ram)}</span>
    </li>`;
}

function agentsHtml() {
  return `<h4 class="live-detail-title">Máy đã cài bộ client</h4>
    ${agents.length
      ? `<ul class="live-agents">${agents.map(agentRow).join("")}</ul>
         <p class="field-hint mt-2">Máy cài xong tự nối với trang này và tự nối lại mỗi lần bật máy. Mã để biết máy nào là máy nào, trình cài in mã đó khi cài xong.</p>`
      : '<p class="field-hint">Chưa có máy nào đang nối. Mở trang này trên máy cần tham gia, bấm Tải bộ cài, cài xong máy tự hiện ở đây sau vài giây.</p>'}`;
}

function renderAgents() {
  const host = byId("live-agents");
  if (host) host.innerHTML = agentsHtml();
}

// Chỉ dựng theo mã và tên máy, không theo số RAM đang đổi, để không mất lựa chọn đang mở (xem renderControl).
function lightClients(machineCount) {
  return Math.min(LIGHT_IMAGE_SETS, MAX_PER_MACHINE * Math.max(1, machineCount));
}

// Cùng cách chia với split() ở web/services/client_agents.py: máy trước nhận phần dư.
function splitText(clients, machineCount) {
  const size = Math.floor(clients / machineCount);
  const extra = clients % machineCount;
  return Array.from({ length: machineCount }, (_unused, index) => size + (index < extra ? 1 : 0)).join(", ");
}

function machinePickHtml(preset) {
  const online = agents.filter(agent => agent.state !== "offline");
  if (!online.length) return "";
  if (!preset.light) return `<p class="field-hint mt-2">Lựa chọn này phải bấm Mở client trên từng laptop, vì máy cài chỉ có ảnh của client 5 và 6. Muốn các máy đã cài tự mở client, chọn lần đo có bộ ảnh nhẹ.</p>`;
  const picked = online.filter(agent => pickedMachines.has(agent.id)).length;
  return `<fieldset class="live-machine-pick mt-3">
      <legend class="field-hint">Máy tự mở client</legend>
      ${online.map(agent => `<label><input type="checkbox" data-machine="${esc(agent.id)}"${pickedMachines.has(agent.id) ? " checked" : ""}> ${esc(agent.label)} <span class="mono">${esc(agent.code)}</span></label>`).join("")}
      <p class="field-hint">${picked ? `${esc(lightClients(picked))} client cho ${esc(picked)} máy, mỗi máy ${esc(splitText(lightClients(picked), picked))} client, không cần bấm gì trên các máy đó. Mỗi máy tối đa ${MAX_PER_MACHINE} client${picked > 2 ? `, và bộ ảnh nhẹ chỉ có ảnh của ${LIGHT_IMAGE_SETS} client` : ""}.` : `Đánh dấu các máy tham gia, đối chiếu mã với mã trình cài đã in. Không chọn máy nào thì lần đo có ${MAX_PER_MACHINE} client, bấm Mở client trên một laptop như trước.`}</p>
    </fieldset>`;
}

function controlHtml() {
  const active = view && (view.state === "waiting" || view.state === "running");
  if (active) {
    const light = /_nhe$/.test(view.run_id || "");
    const auto = agents.some(agent => agent.clients.length);    // đã giao cho máy: bấm tay nữa sẽ mở trùng client
    return `<section class="panel live-control" aria-label="Điều khiển lần đo">
        <div class="live-control-row">
          ${auto ? "" : `<a class="btn btn-primary" href="${light ? LIGHT_CLIENT_LINK : CLIENT_LINK}">Mở client trên máy này</a>`}
          ${demo.running ? `<button type="button" class="btn" id="live-demo-stop"${demoBusy ? " disabled" : ""}>Dừng lần đo</button>` : ""}
        </div>
        <p class="field-hint mt-2">${auto ? "Các máy được chọn tự mở client, không cần bấm gì trên máy đó." : `${light ? `Chỉ bấm trên một laptop, máy đó mở ${MAX_PER_MACHINE} client` : "Bấm trên từng laptop"}, trình duyệt hỏi thì chọn Mở. Bấm mà không có gì xảy ra nghĩa là máy đó chưa cài.`}</p>
        ${installHtml()}
      </section>`;
  }
  if (!demo.can_start) return installHtml();
  if (!readCredential()) return `<p class="field-hint">Muốn chạy lần đo mới từ trang này, <a href="${ADMIN_PAGE}">đăng nhập quản trị</a> rồi quay lại.</p>${installHtml()}`;
  return `<section class="panel live-control" aria-label="Điều khiển lần đo">
      <div class="live-control-row">
        <label class="field-hint" for="live-demo-preset">Lần đo mới</label>
        <select id="live-demo-preset" class="form-control-sm">${PRESETS.map((preset, index) => `<option value="${index}"${index === chosenPreset ? " selected" : ""}>${esc(preset.label)}</option>`).join("")}</select>
        <button type="button" class="btn btn-primary" id="live-demo-start"${demoBusy ? " disabled" : ""}>${demoBusy ? "Đang mở máy chủ…" : "Bắt đầu lần đo"}</button>
      </div>
      ${machinePickHtml(PRESETS[chosenPreset])}
      <p class="field-hint mt-2">Máy chủ mở trên VM, cổng đo mở cho mọi IP trong lúc đo và tự đóng khi xong, khi bấm dừng, hoặc sau 1 giờ.</p>
    </section>`;
}

function renderControl() {
  const host = byId("live-control");
  if (!host) return;
  const html = controlHtml();
  if (html === lastControl) return;       // không dựng lại nếu không đổi, để không mất lựa chọn đang mở
  lastControl = html;
  host.innerHTML = html;
}

function demoFailed(error, title) {
  if (error?.status === 401) {
    clearCredential();
    notify("warning", "Cần đăng nhập quản trị", "Hãy đăng nhập ở trang quản trị rồi quay lại bấm nút này.");
    setTimeout(() => location.assign(ADMIN_PAGE), 1200);
  } else {
    notify("error", title, error.message);
  }
}

async function startDemo() {
  const preset = PRESETS[chosenPreset];
  const machines = preset.light ? agents.filter(agent => agent.state !== "offline" && pickedMachines.has(agent.id)).map(agent => agent.id) : [];
  demoBusy = true;
  renderControl();
  try {
    await postJson("/api/flower/demo/start", { clients: preset.light ? lightClients(machines.length) : preset.clients, rounds: preset.rounds, light: Boolean(preset.light), machines }, { headers: storedAuthHeader(), timeoutMs: 30000 });
    notify("info", "Đã mở máy chủ", machines.length ? `${machines.length} máy sẽ tự mở client.` : preset.light ? "Bấm Mở client trên laptop 1." : "Bấm Mở client trên từng laptop.");
    chosenRun = "";
    chosenRound = null;
    built = false;
    lastSignature = "";
    lastControl = null;
    followUntil = Date.now() + FOLLOW_MS;
    closeDetail();
    selected = "server";      // hiện luôn terminal của máy chủ
    detailKey = "";
  } catch (error) {
    demoFailed(error, "Không mở được máy chủ");
  } finally {
    demoBusy = false;
    renderControl();
    refresh();
  }
}

async function stopDemo() {
  demoBusy = true;
  renderControl();
  try {
    await postJson("/api/flower/demo/stop", {}, { headers: storedAuthHeader(), timeoutMs: 40000 });
    notify("info", "Đã dừng lần đo", "Cổng đo đã đóng.");
  } catch (error) {
    demoFailed(error, "Không dừng được lần đo");
  } finally {
    demoBusy = false;
    lastSignature = "";
    renderControl();
    refresh();
  }
}

function layoutHtml() {
  return `
    <div id="live-banner"></div>
    <div id="live-control" class="mt-4"></div>
    <section class="panel mt-4" aria-labelledby="live-stage-title">
      <div class="panel-header-flex">
        <h3 class="panel-title panel-title-sm" id="live-stage-title">Client và máy chủ lúc này</h3>
        <span id="live-picker"></span>
      </div>
      <div id="live-stage" class="mt-3"></div>
      <div id="live-stage-detail" class="live-detail hidden" role="region" aria-label="Chi tiết mục đang chọn"></div>
      <div id="live-agents" class="live-detail"></div>
    </section>
    <div id="live-proofs" class="mt-4"></div>
    <p class="field-hint mt-2">Bấm vào một ô để xem số đó làm ra thế nào.</p>
    <section id="live-proofs-detail" class="live-detail panel mt-3 hidden" aria-label="Chi tiết ô đang chọn"></section>
    <div id="live-caveats" class="mt-4"></div>
    <section class="panel mt-4" id="live-flow" aria-label="Sơ đồ truyền nhận dữ liệu trong một vòng"></section>
    <div class="live-charts mt-4">
      <section class="panel"><h3 class="panel-title panel-title-sm">Máy chủ kiểm mô hình chung sau mỗi vòng, tập val</h3>
        <div class="chart-container mt-3"><canvas id="live-chart-accuracy" aria-label="Độ chính xác và loss theo vòng"></canvas></div></section>
      <section class="panel"><h3 class="panel-title panel-title-sm">Thời gian mỗi vòng dùng vào đâu</h3>
        <div class="chart-container mt-3"><canvas id="live-chart-time" aria-label="Thời gian học, truyền và xử lý của máy chủ theo vòng"></canvas></div></section>
      <section class="panel live-chart-wide"><h3 class="panel-title panel-title-sm">Độ biến thiên trung bình của client theo vòng</h3>
        <div class="chart-container mt-3"><canvas id="live-chart-norm" aria-label="Độ biến thiên trung bình của các client được nhận, theo vòng"></canvas></div>
        <p class="field-hint mt-2">Độ biến thiên là mức trọng số của client thay đổi sau khi học. Đường giảm nghĩa là mô hình thay đổi ít dần, nhưng tự nó không chứng minh mô hình tốt lên, nên cần đọc cùng biểu đồ độ chính xác.</p>
        <h4 class="live-detail-title mt-4">Từng client</h4>
        <p class="live-norm-legend"><span><i class="live-norm-key is-total"></i>${esc(NORM_LINES[0].label)}</span><span><i class="live-norm-key is-learnable"></i>${esc(NORM_LINES[1].label)}</span></p>
        <div id="live-client-norms" class="live-norm-grid"></div>
        <p class="field-hint mt-2">Các biểu đồ dùng chung một thang dọc nên so được với nhau. Số của từng vòng nằm ở bảng "Từng client, từng vòng" bên dưới.</p></section>
    </div>
    <section class="panel mt-4" id="live-runs-compare" aria-label="So các lần đo"></section>
    <section class="panel mt-4"><h3 class="panel-title panel-title-sm">Từng client, từng vòng</h3><div id="live-round-table" class="mt-3"></div></section>
    <section class="panel mt-4"><h3 class="panel-title panel-title-sm">Theo từng máy</h3><div id="live-machines"></div></section>
    <p class="field-hint mt-3" id="live-note"></p>`;
}

function syncFlowDiagram() {
  const row = shownRound();
  const client = row ? row.clients.find(item => item.client_id === chosenClient) || row.clients[0] || null : null;
  renderFlowDiagram(byId("live-flow"), { row, client, tamper: view.tamper_check || null, numClients: view.num_clients });
}

function syncRunsCompare() {
  const host = byId("live-runs-compare");
  if (!host) return;
  const key = JSON.stringify([view.runs, view.state, view.completed_rounds]);
  if (key === runsKey && host.childElementCount) return;
  runsKey = key;
  renderRunsCompare(host).catch(error => console.warn("Không dựng được biểu đồ so các lần đo:", error));   // các phần khác của trang vẫn dùng được
}

function render(data) {
  const root = byId("flower-live-root");
  if (!root) return;
  if (!data.available) {
    view = null;
    built = false;
    lastSignature = "";
    root.innerHTML = emptyHtml();
    return;
  }
  view = data;
  const signature = JSON.stringify({ ...data, seconds_since_update: undefined, assigned: assignedAgents(data) });
  if (signature === lastSignature && built) return;   // nhịp ghi của máy chủ không đổi gì: không dựng lại để khỏi mất chỗ đang xem
  lastSignature = signature;
  if (!built) {
    root.innerHTML = layoutHtml();
    built = true;
    lastControl = null;
  }
  const focused = document.activeElement?.dataset?.detail;   // dựng lại khung thì giữ nguyên chỗ đang đứng khi dùng bàn phím
  byId("live-banner").innerHTML = bannerHtml(data);
  byId("live-stage").innerHTML = stageHtml(data);
  byId("live-picker").innerHTML = pickerHtml(data);
  byId("live-proofs").innerHTML = proofsHtml(data);
  byId("live-machines").innerHTML = machineTableHtml();
  byId("live-round-table").innerHTML = roundTableHtml();
  const caveats = [data.shared_machine_note, data.note, data.caveat].filter(Boolean);
  byId("live-caveats").innerHTML = caveats.length
    ? `<div class="callout callout-warn" role="note"><b>Cần đọc kèm các giới hạn:</b><ul>${caveats.map(text => `<li>${esc(text)}</li>`).join("")}</ul></div>`
    : "";
  byId("live-note").textContent = "Terminal in đúng các số này bằng lệnh: python scripts/demo_show.py flower";
  const fill = byId("live-progress-fill");
  setBar(fill, fill?.dataset.percent);
  refreshChart("live-chart-accuracy", accuracyConfig);
  refreshChart("live-chart-time", timeConfig);
  refreshChart("live-chart-norm", normConfig);
  renderClientNorms();
  syncRunsCompare();
  syncFlowDiagram();
  if (focused) root.querySelector(`[data-detail="${focused}"]`)?.focus();
  syncDetail();
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const query = chosenRun ? `?run=${encodeURIComponent(chosenRun)}` : "";
    const [live, status, machines] = await Promise.all([getJson(`/api/flower/live${query}`, { timeoutMs: TIMEOUT_MS }), getJson("/api/flower/demo/status", { timeoutMs: TIMEOUT_MS }), getJson("/api/flower/agents", { timeoutMs: TIMEOUT_MS })]);
    demo = status;
    agents = machines.agents;
    render(live);
    renderAgents();
    renderControl();
    if (view && isActive(view.state)) {
      followUntil = 0;
      if (selected === "server") loadDetail();      // nhật ký máy chủ chạy tiếp cả khi chưa có client mới xong
    }
    byId("flower-live-root")?.querySelector("[data-live-error]")?.remove();
  } catch (error) {
    const root = byId("flower-live-root");
    if (root && !root.querySelector("[data-live-error]")) {
      root.insertAdjacentHTML("afterbegin", `<div class="callout callout-error" role="alert" data-live-error><b>Không đọc được lần đo.</b> ${esc(error.message)} Trang sẽ tự thử lại.</div>`);
    }
  } finally {
    loading = false;
    schedule();
  }
}

function schedule() {
  clearTimeout(timer);
  if (location.hash !== "#flower") return;   // rời trang: dừng hỏi
  timer = setTimeout(() => {
    if (document.hidden) { schedule(); return; }
    refresh();
  }, (view && isActive(view.state)) || Date.now() < followUntil ? ACTIVE_POLL_MS : IDLE_POLL_MS);
}

function bind(root) {
  root.addEventListener("input", event => {
    if (event.target.id !== "live-round" || !view) return;
    const value = Number(event.target.value);
    chosenRound = value === view.rounds[view.rounds.length - 1]?.round ? null : value;   // kéo về cuối là lại theo vòng mới nhất
    const body = byId("live-round-table");
    body.innerHTML = roundTableHtml();
    syncFlowDiagram();
    const slider = byId("live-round");
    slider.focus();
    clearTimeout(detailTimer);
    detailTimer = setTimeout(syncDetail, DETAIL_DEBOUNCE_MS);
  });
  root.addEventListener("click", event => {
    const copy = event.target.closest(".copy-btn");
    if (copy) { copyFromButton(copy); return; }
    if (event.target.closest("#live-demo-start")) { startDemo(); return; }
    if (event.target.closest("#live-demo-stop")) { stopDemo(); return; }
    if (event.target.closest("[data-detail-close]")) { closeDetail({ refocus: true }); return; }
    const button = event.target.closest("[data-detail]");
    if (button) openDetail(button.dataset.detail);
  });
  root.addEventListener("keydown", event => {
    if (event.key === "Escape" && selected) closeDetail({ refocus: true });
  });
  root.addEventListener("change", event => {
    if (event.target.id === "live-demo-preset") {
      chosenPreset = Number(event.target.value);
      renderControl();
      byId("live-demo-preset")?.focus();
      return;
    }
    if (event.target.dataset?.machine) {
      const id = event.target.dataset.machine;
      if (event.target.checked) pickedMachines.add(id);
      else pickedMachines.delete(id);
      renderControl();
      byId("live-control")?.querySelector(`[data-machine="${id}"]`)?.focus();
      return;
    }
    if (event.target.id === "live-step-client") {
      chosenClient = Number(event.target.value);
      byId("live-round-table").innerHTML = roundTableHtml();
      syncFlowDiagram();
      byId("live-step-client")?.focus();
      return;
    }
    if (event.target.id !== "live-run") return;
    chosenRun = event.target.value;
    chosenRound = null;
    closeDetail();
    built = false;
    lastSignature = "";
    refresh();
  });
}

async function loadClientPackage() {
  try {
    clientPackage = await getJson("/api/flower/client-package/status", { timeoutMs: TIMEOUT_MS });
    renderControl();
  } catch (error) {
    console.warn("Không hỏi được gói client, ẩn nút tải bộ cài:", error);   // trang đo vẫn dùng được, chỉ thiếu nút này
  }
}

export async function mountLive() {
  const root = byId("flower-live-root");
  if (!root) {
    notify("error", "Không dựng được trang", "Thiếu vùng hiển thị lần đo.");
    return;
  }
  bind(root);
  loadClientPackage();
  await refresh();
}

export async function showLive() {
  await refresh();
}
