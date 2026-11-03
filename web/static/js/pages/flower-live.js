// Demo · Flower: lần đo thật đang chạy (hoặc vừa chạy). Chạy bằng terminal và .bat; trang này chỉ đọc /api/flower/live.
// Máy chủ tính sẵn mọi số trong src/federated/run_view.py, đúng hàm mà scripts/demo_show.py dùng, nên web và terminal luôn trùng nhau.
import { getJson } from "../core/api.js";
import { makeChart, pillarColors } from "../core/charts.js";
import { byId, esc, fmtInt, fmtNum, notify, setBar } from "../core/dom.js";

const ACTIVE_POLL_MS = 3000;
const IDLE_POLL_MS = 15000;   // đã xong hoặc chưa có lần đo: vẫn hỏi chậm để tự thấy lần đo mới
const TIMEOUT_MS = 12000;

let timer = 0;
let loading = false;
let view = null;
let chosenRun = "";           // rỗng = luôn theo lần đo mới nhất
let chosenRound = null;       // null = luôn theo vòng mới nhất
let built = false;
let lastSignature = "";

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
    failed: { tone: "is-wrong", title: `Lần đo dừng vì lỗi${data.failed_round ? ` ở vòng ${data.failed_round}` : ""}`, meta: data.error || "Xem nhật ký máy chủ." },
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

function chipState(data, clientId, index) {
  if (data.state === "waiting") return index < (data.connected_clients ?? 0) ? "is-connected" : "is-idle";
  if (data.state === "finished") return "is-done";
  if (data.state === "failed" || data.state === "stale") return "is-idle";
  return (data.round_replies || []).some(reply => reply.client_id === clientId) ? "is-done" : "is-learning";
}

function chipHtml(data, clientId, index) {
  const reply = (data.round_replies || []).find(item => item.client_id === clientId);
  const detail = data.state === "running" && reply ? `${fmtNum(reply.response_ms / 1000, 1)} s` : "";
  return `<span class="live-chip ${chipState(data, clientId, index)}"><b>${esc(String(clientId).padStart(2, "0"))}</b>${detail ? `<small>${esc(detail)}</small>` : ""}</span>`;
}

function groupHtml(data, title, sub, ids, offset) {
  return `<div class="live-group">
      <p class="live-group-title">${esc(title)}</p>
      ${sub ? `<p class="live-group-sub">${esc(sub)}</p>` : ""}
      <div class="live-chips">${ids.map((id, i) => chipHtml(data, id, offset + i)).join("")}</div>
    </div>`;
}

function stageHtml(data) {
  // Trước vòng đầu chưa biết client nào ở máy nào: hiện chung một nhóm, đếm số đã nối.
  const machines = data.machines || [];
  const groups = machines.length
    ? machines.map(machine => groupHtml(data, machine.label,
        machine.same_host_as_server ? "CÙNG MÁY VỚI MÁY CHỦ" : (machine.rtt_ms === null ? "" : `độ trễ ${fmtNum(machine.rtt_ms, 1)} ms`), machine.client_ids, 0))
    : [groupHtml(data, "Client", "chưa biết máy nào, sẽ rõ sau vòng đầu", Array.from({ length: data.num_clients || 0 }, (_unused, i) => i), 0)];
  const flowing = data.state === "running" ? " is-flowing" : "";
  const middle = Math.ceil(groups.length / 2);
  return `<div class="live-stage">
      <div class="live-side">${groups.slice(0, middle).join("")}</div>
      <div class="live-link${flowing}" aria-hidden="true"></div>
      <div class="live-server${flowing}">
        <p class="live-group-title">Máy chủ</p>
        <p class="live-group-sub">gộp trọng số theo số ảnh, tự kiểm độ chính xác, tự đo độ trễ</p>
        ${data.state === "running" && data.round_in_progress ? `<p class="live-server-round">Vòng ${esc(data.round_in_progress)}</p>` : ""}
      </div>
      ${groups.length > 1 ? `<div class="live-link${flowing}" aria-hidden="true"></div><div class="live-side">${groups.slice(middle).join("")}</div>` : ""}
    </div>
    <p class="field-hint mt-2"><span class="live-key is-learning"></span> đang học <span class="live-key is-done"></span> đã gửi về <span class="live-key is-connected"></span> đã nối <span class="live-key is-idle"></span> chưa nối hoặc đã dừng</p>`;
}

function tile(label, value, note, ok = null) {
  const mark = ok === null ? "" : ok ? " is-ok" : " is-bad";
  return `<div class="kpi-mini live-tile${mark}"><span class="kpi-mini-label">${esc(label)}</span><span class="kpi-mini-value">${esc(value)}</span><span class="live-tile-note">${esc(note)}</span></div>`;
}

function proofsHtml(data) {
  const c = data.checks;
  const last = data.rounds[data.rounds.length - 1];
  const accuracy = data.final_test?.accuracy ?? last?.accuracy;
  const onServer = c.host_check_available ? String(c.clients_on_server_host) : "chưa kiểm";
  return `<div class="live-tiles">
      ${tile("Trọng số gửi đúng", `${fmtInt(c.hash_ok)}/${fmtInt(c.hash_total)}`, c.chain_ok ? "mã băm khớp, chuỗi mô hình liền" : "lượt mã băm khớp", c.hash_total ? c.hash_ok === c.hash_total : null)}
      ${tile("Độ biến thiên khớp", `${fmtInt(c.norm_ok)}/${fmtInt(c.norm_total)}`, "client khai so với máy chủ tự tính", c.norm_total ? c.norm_ok === c.norm_total : null)}
      ${tile("Vòng nhận đủ client", `${fmtInt(c.full_rounds ?? 0)}/${fmtInt(c.rounds_total)}`, `đủ ${data.num_clients} client mỗi vòng`, c.rounds_total ? c.full_rounds === c.rounds_total : null)}
      ${tile("Client chung máy chủ", onServer, `client đến từ ${fmtInt(c.machines)} máy`, c.host_check_available ? c.clients_on_server_host === 0 : null)}
      ${tile("Đã truyền", `${megabytes(c.bytes_up + c.bytes_down)} MiB`, `lên ${megabytes(c.bytes_up)}, xuống ${megabytes(c.bytes_down)}`)}
      ${tile("Độ chính xác máy chủ tính", accuracy === null || accuracy === undefined ? "—" : `${fmtNum(accuracy * 100, 2)}%`, data.final_test ? "tập test giữ riêng của máy chủ" : `sau vòng ${last?.round ?? "—"}`)}
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

function refreshChart(id, factory) {
  const canvas = byId(id);
  if (!canvas) return;
  const chart = window.Chart?.getChart?.(canvas);
  if (!chart) { makeChart(canvas, factory); return; }
  const next = factory().data;
  chart.data.labels = next.labels;
  next.datasets.forEach((dataset, index) => { chart.data.datasets[index].data = dataset.data; });
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
      <td>${esc(client.client_id)}</td><td>${esc(client.machine)}${client.same_host_as_server ? " (cùng máy chủ)" : ""}</td><td>${esc(fmtInt(client.num_examples))}</td>
      <td>${num(client.rtt_ms, 1)}</td><td>${num(client.up_mb_s)}</td><td>${num(client.down_mb_s)}</td>
      <td>${num(client.response_s, 1)}</td><td>${num(client.train_s, 1)}</td><td>${norm}</td>
      <td><span class="mono" title="client ${esc(client.client_hash)} / máy chủ ${esc(client.server_hash)}">${esc(client.server_hash.slice(0, 8))}</span> ${client.hash_ok ? "✓" : "✕"}</td>
    </tr>`;
}

function roundTableHtml() {
  const row = shownRound();
  if (!row) return '<p class="field-hint">Chưa xong vòng nào.</p>';
  const count = view.rounds.length;
  const accuracy = row.accuracy === null ? "" : ` · máy chủ tính độ chính xác ${fmtNum(row.accuracy * 100, 2)}%`;
  return `<div class="live-round-pick">
      <label for="live-round">Xem vòng <b id="live-round-value">${esc(row.round)}</b>/${esc(count)}</label>
      <input type="range" id="live-round" min="${esc(view.rounds[0].round)}" max="${esc(view.rounds[count - 1].round)}" value="${esc(row.round)}" step="1">
      <span class="field-hint">cả vòng ${num(row.elapsed_s, 1)} s${accuracy}</span>
    </div>
    <div class="table-responsive mt-2" tabindex="0" role="region" aria-label="Bảng từng client trong vòng đang xem"><table class="data-table wrap-head">
      <thead><tr><th>Client</th><th>Máy</th><th>Ảnh</th><th>RTT (ms)</th><th>Lên (MiB/s)</th><th>Xuống (MiB/s)</th><th>Phản hồi (s)</th><th>Học, ước lượng (s)</th><th>Độ biến thiên: client khai / máy chủ tính</th><th>Mã băm</th></tr></thead>
      <tbody>${row.clients.map(clientRow).join("")}</tbody>
    </table></div>`;
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
      ${runs.map(id => `<option value="${esc(id)}"${id === chosenRun ? " selected" : ""}>${esc(id)}</option>`).join("")}
    </select></label>`;
}

function emptyHtml() {
  return `<div class="callout callout-warn" role="note">
      <b>Chưa có lần đo thật nào trên máy chủ này.</b> Khi chạy lần đo bằng terminal (<code class="code">scripts/remote/run_network_server.sh</code> trên máy chủ, <code class="code">CHAY_CLIENT.bat</code> trên các laptop), trang này tự cập nhật theo từng vòng, không cần tải lại.
    </div>`;
}

function layoutHtml() {
  return `
    <div id="live-banner"></div>
    <section class="panel mt-4" aria-labelledby="live-stage-title">
      <div class="panel-header-flex">
        <h3 class="panel-title panel-title-sm" id="live-stage-title">Client và máy chủ lúc này</h3>
        <span id="live-picker"></span>
      </div>
      <div id="live-stage" class="mt-3"></div>
    </section>
    <div id="live-caveats" class="mt-4"></div>
    <div id="live-proofs" class="mt-4"></div>
    <div class="live-charts mt-4">
      <section class="panel"><h3 class="panel-title panel-title-sm">Máy chủ kiểm mô hình chung sau mỗi vòng</h3>
        <div class="chart-container mt-3"><canvas id="live-chart-accuracy" aria-label="Độ chính xác và loss theo vòng"></canvas></div></section>
      <section class="panel"><h3 class="panel-title panel-title-sm">Thời gian mỗi vòng dùng vào đâu</h3>
        <div class="chart-container mt-3"><canvas id="live-chart-time" aria-label="Thời gian học, truyền và xử lý của máy chủ theo vòng"></canvas></div></section>
    </div>
    <section class="panel mt-4"><h3 class="panel-title panel-title-sm">Từng client, từng vòng</h3><div id="live-round-table" class="mt-3"></div></section>
    <section class="panel mt-4"><h3 class="panel-title panel-title-sm">Theo từng máy</h3><div id="live-machines"></div></section>
    <p class="field-hint mt-3" id="live-note"></p>`;
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
  const signature = JSON.stringify({ ...data, seconds_since_update: undefined });
  if (signature === lastSignature && built) return;   // nhịp ghi của máy chủ không đổi gì: không dựng lại để khỏi mất chỗ đang xem
  lastSignature = signature;
  if (!built) {
    root.innerHTML = layoutHtml();
    built = true;
  }
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
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const query = chosenRun ? `?run=${encodeURIComponent(chosenRun)}` : "";
    render(await getJson(`/api/flower/live${query}`, { timeoutMs: TIMEOUT_MS }));
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
  }, view && isActive(view.state) ? ACTIVE_POLL_MS : IDLE_POLL_MS);
}

function bind(root) {
  root.addEventListener("input", event => {
    if (event.target.id !== "live-round" || !view) return;
    const value = Number(event.target.value);
    chosenRound = value === view.rounds[view.rounds.length - 1]?.round ? null : value;   // kéo về cuối là lại theo vòng mới nhất
    const body = byId("live-round-table");
    body.innerHTML = roundTableHtml();
    const slider = byId("live-round");
    slider.focus();
  });
  root.addEventListener("change", event => {
    if (event.target.id !== "live-run") return;
    chosenRun = event.target.value;
    chosenRound = null;
    built = false;
    lastSignature = "";
    refresh();
  });
}

export async function mountLive() {
  const root = byId("flower-live-root");
  if (!root) {
    notify("error", "Không dựng được trang", "Thiếu vùng hiển thị lần đo.");
    return;
  }
  bind(root);
  await refresh();
}

export async function showLive() {
  await refresh();
}
