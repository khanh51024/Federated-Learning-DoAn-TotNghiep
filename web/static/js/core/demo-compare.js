// Nối ba trang demo: dải đầu trang, bảng và biểu đồ so sánh, cùng biểu đồ riêng của từng trang. Số kết quả đều đọc từ API.
// Không vẽ độ chính xác của ba demo trên cùng một trục: Chẩn đoán và Cộng đồng chấm trên cùng tập kiểm tra, còn lần đo qua mạng chấm trên tập nhỏ của chính nó.
import { getJson } from "./api.js";
import { makeChart, pillarColors } from "./charts.js";
import { byId, esc, fmtInt, fmtNum, isInt8, MODEL_LABELS, MODEL_ORDER, sourceLabel } from "./dom.js";
import { getEvidence } from "./store.js";

const TIMEOUT_MS = 15000;
const FACTS_TTL_MS = 15000;
const COMMUNITY_HEAD_BYTES = 155800;   // 38.950 tham số float32; khi gửi còn mã hóa base64 nên đường truyền tốn khoảng 208 KB
const SOURCES = ["plantvillage", "plantdoc"];   // hai nguồn có ở cả mô hình của đề tài lẫn tập giám sát cộng đồng
const DEMOS = [
  { id: "chan-doan", label: "Chẩn đoán ảnh", question: "Học chung có hơn học riêng không?" },
  { id: "hoc-lien-ket", label: "Cộng đồng tham gia", question: "Người dùng thường góp được không?" },
  { id: "flower", label: "Đo qua mạng", question: "Trọng số có đi qua mạng thật không?" },
];
const compareRows = imageCount => [
  ["Client là ai", "Không có lúc dùng; mô hình đã học sẵn", "Người dùng, trên trình duyệt", "Laptop chạy chương trình client"],
  ["Học phần nào", "Không học, chỉ nhận diện", "Chỉ lớp cuối của FedAvg", "Cả mô hình, khởi tạo từ ImageNet"],
  ["Client gửi gì", "Không gửi trọng số", "Trọng số lớp cuối, số ảnh", "Trọng số cả mô hình, số ảnh, độ biến thiên"],
  ["Ảnh có rời máy không", "Có, ảnh được gửi lên máy chủ để nhận diện", "Không", "Không"],
  ["Chấm trên ảnh nào", imageCount ? `${fmtInt(imageCount)} ảnh của tập kiểm tra` : "Tập kiểm tra của đề tài", "Cùng tập kiểm tra đó", "Tập nhỏ riêng của từng lần đo"],
  ["Kết quả đi đâu", "Hiện ngay trên trang", "Thành phiên bản mới, dùng ở thẻ Cộng đồng của trang Chẩn đoán", "Chỉ làm bằng chứng hệ thống, không đưa lên web"],
];
const COMMUNITY_METRICS = [
  { key: "accuracy", label: "Độ chính xác" },
  { key: "macro_recall", label: "Macro-recall" },
  { key: "plantvillage_accuracy", label: sourceLabel("plantvillage") },
  { key: "plantdoc_accuracy", label: sourceLabel("plantdoc") },
];

const percent = value => (typeof value === "number" ? value * 100 : null);
const axisPercent = text => ({ min: 0, max: 100, title: { display: true, text } });

// Tên thư mục lần đo là giờ UTC và số client, ví dụ 20261104T075312Z_10client. Tên lạ thì hiện nguyên.
export function runLabel(id) {
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z_(\d+)client(_nhe)?$/.exec(id);
  if (!match) return id;
  const [, year, month, day, hour, minute, second, clients, light] = match;
  const time = new Date(Date.UTC(+year, +month - 1, +day, +hour, +minute, +second));
  const pad = value => String(value).padStart(2, "0");
  return `${pad(time.getDate())}/${pad(time.getMonth() + 1)} ${pad(time.getHours())}:${pad(time.getMinutes())}, ${clients} client${light ? ", bộ ảnh nhẹ" : ""}`;
}

/** Tạo biểu đồ, hoặc chỉ thay số nếu đã có, để không nháy mỗi lần trang hỏi lại máy chủ. */
function drawChart(canvas, factory) {
  if (!canvas) return;
  const chart = window.Chart?.getChart?.(canvas);
  if (!chart) { makeChart(canvas, factory); return; }
  const next = factory().data;
  chart.data.labels = next.labels;
  next.datasets.forEach((dataset, index) => {
    Object.assign(chart.data.datasets[index], { label: dataset.label, data: dataset.data, backgroundColor: dataset.backgroundColor });   // tên có số phiên bản nên cũng đổi
  });
  chart.update("none");
}

function ensurePanel(host, title, canvasHtml) {
  if (host.querySelector("canvas")) return;
  host.innerHTML = `<h3 class="panel-title panel-title-sm">${esc(title)}</h3>
    <div class="chart-container mt-3">${canvasHtml}</div>
    <p class="field-hint mt-2" data-compare-note></p>`;
}

function setNote(host, text) {
  const note = host.querySelector("[data-compare-note]");
  if (note) note.textContent = text;
}

function unavailable(host, title, text) {
  host.innerHTML = `<h3 class="panel-title panel-title-sm">${esc(title)}</h3><p class="field-hint mt-2">${esc(text)}</p>`;
}

/* ───────── Trang Chẩn đoán: các mô hình trên cùng một tập ảnh ───────── */

function modelRows(evidence, community) {
  const byModel = new Map((evidence.models || []).map(model => [model.model_id, model.metrics]));
  // bản nén int8 không vẽ vào biểu đồ so sánh; học riêng trước, rồi tới FedAvg và Centralized
  const order = MODEL_ORDER.filter(id => !isInt8(id)).sort((a, b) => Number(/^local/.test(b)) - Number(/^local/.test(a)));
  const rows = order.filter(id => byModel.has(id)).map(id => ({
    label: MODEL_LABELS[id] || id,
    plantvillage: percent(byModel.get(id).plantvillage?.top1_accuracy),
    plantdoc: percent(byModel.get(id).plantdoc?.top1_accuracy),
  }));
  if (community?.metrics) {
    rows.push({ label: `Cộng đồng v${community.version}`, plantvillage: percent(community.metrics.plantvillage_accuracy), plantdoc: percent(community.metrics.plantdoc_accuracy) });
  }
  return rows;
}

function modelConfig(rows) {
  const colors = pillarColors();
  const series = (source, color) => ({ label: sourceLabel(source), data: rows.map(row => row[source]), backgroundColor: color, borderRadius: 4, categoryPercentage: 0.7, barPercentage: 0.9 });
  return {
    type: "bar",
    data: { labels: rows.map(row => row.label), datasets: [series(SOURCES[0], colors.centralized), series(SOURCES[1], colors.local)] },
    options: { animation: false, interaction: { mode: "index", intersect: false }, scales: { y: axisPercent("Đoán đúng nhóm bệnh (%)") } },
  };
}

export async function renderModelCompare(host) {
  if (!host) return;
  const title = "Các mô hình trên cùng một tập ảnh";
  const [evidence, community] = await Promise.allSettled([getEvidence(), getJson("/api/community/status", { timeoutMs: TIMEOUT_MS })]);
  if (evidence.status !== "fulfilled" || !evidence.value?.available) {
    unavailable(host, title, "Chưa đọc được số liệu của các mô hình.");
    return;
  }
  const status = community.status === "fulfilled" && community.value?.enabled !== false ? community.value : null;
  const rows = modelRows(evidence.value, status);
  ensurePanel(host, title, '<canvas id="demo-chart-models" aria-label="Độ chính xác của từng mô hình trên ảnh PlantVillage và PlantDoc"></canvas>');
  const counts = Object.fromEntries((evidence.value.source_groups || []).map(group => [group.id, group.val_count]));
  const communityNote = !status ? " Chưa đọc được bản Cộng đồng nên không có cột này."
    : status.version === 0 ? " Cộng đồng chưa gộp vòng nào nên cột Cộng đồng bằng đúng FedAvg."
      : " Cột Cộng đồng là phiên bản hiện hành, cùng tập ảnh.";
  setNote(host, `Cùng ${fmtInt(counts.plantvillage)} ảnh PlantVillage và ${fmtInt(counts.plantdoc)} ảnh PlantDoc nên so trực tiếp được. Khác biểu đồ này với kết quả ở trên: trên là một ảnh, đây là cả tập.${communityNote} Mô hình tham chiếu PlantVillage không vẽ ở đây vì chưa đối chiếu được là đúng tập ảnh này.`);
  drawChart(byId("demo-chart-models"), () => modelConfig(rows));
}

/* ───────── Trang Cộng đồng: bản hiện hành so với FedAvg gốc ───────── */

function communityConfig(status) {
  const colors = pillarColors();
  const series = (label, metrics, color) => ({ label, data: COMMUNITY_METRICS.map(metric => percent(metrics?.[metric.key])), backgroundColor: color, borderRadius: 4, categoryPercentage: 0.6, barPercentage: 0.9 });
  return {
    type: "bar",
    data: {
      labels: COMMUNITY_METRICS.map(metric => metric.label),
      datasets: [series("FedAvg gốc (v0)", status.base_metrics, colors.fedavg), series(`Phiên bản hiện hành (v${status.version})`, status.metrics, colors.violet)],
    },
    options: { animation: false, interaction: { mode: "index", intersect: false }, scales: { y: axisPercent("%") } },
  };
}

export function renderCommunityCompare(host, status) {
  if (!host) return;
  const title = "Phiên bản hiện hành so với FedAvg gốc";
  if (!status?.metrics || !status?.base_metrics) {
    unavailable(host, title, "Chưa đọc được số đo của phiên bản hiện hành.");
    return;
  }
  ensurePanel(host, title, '<canvas id="demo-chart-community" aria-label="Bốn chỉ số của FedAvg gốc và của phiên bản cộng đồng hiện hành"></canvas>');
  setNote(host, `${status.version === 0 ? "Chưa gộp vòng nào nên hai cột bằng nhau, đây không phải kết quả học." : `Phiên bản ${status.version} được so với điểm xuất phát.`} Cả hai chấm trên ${fmtInt(status.metrics.count)} ảnh của cùng tập giám sát. Dùng thử phiên bản này ở thẻ Cộng đồng của trang Chẩn đoán ảnh.`);
  drawChart(byId("demo-chart-community"), () => communityConfig(status));
}

/* ───────── Trang Đo qua mạng: so các lần đo ───────── */

function runRowLabel(run) {
  const time = runLabel(run.run_id).split(",")[0];
  const ending = run.state === "finished" ? "" : run.state === "failed" || run.state === "stale" ? ", dừng giữa chừng" : ", đang chạy";
  return `${time} · ${run.num_clients} client · ${run.completed_rounds} vòng${ending}`;
}

function runConfig(runs, key, scale, axis, digits, xMax) {
  const colors = pillarColors();
  return {
    type: "bar",
    data: {
      labels: runs.map(runRowLabel),
      datasets: [{ label: axis, data: runs.map(run => (run[key] === null ? null : run[key] * scale)), backgroundColor: runs.map(run => (run.light ? colors.local : colors.fedavg)), borderRadius: 4, maxBarThickness: 18 }],
    },
    options: {
      animation: false,
      indexAxis: "y",
      interaction: { mode: "nearest", axis: "y", intersect: false },
      plugins: {
        legend: { display: false },     // màu giải thích ở chú thích chung phía trên hai biểu đồ
        tooltip: {
          callbacks: {
            label: item => `${axis}: ${fmtNum(item.parsed.x, digits)}`,
            afterLabel: item => {
              const run = runs[item.dataIndex];
              return `${run.light ? "Bộ ảnh nhẹ" : "Bộ ảnh đầy đủ"}, ${run.machines ?? "?"} máy${run.eval_images ? `, tập val ${fmtInt(run.eval_images)} ảnh` : ""}`;
            },
          },
        },
      },
      scales: { x: { min: 0, max: xMax, title: { display: true, text: axis } }, y: { ticks: { autoSkip: false } } },
    },
  };
}

export async function renderRunsCompare(host) {
  if (!host) return;
  const title = "So các lần đo";
  let runs;
  try {
    runs = (await getJson("/api/flower/runs", { timeoutMs: TIMEOUT_MS })).runs || [];
  } catch (error) {
    if (!byId("demo-chart-runs-time")) unavailable(host, title, `Chưa đọc được danh sách lần đo. ${error.message}`);
    return;
  }
  const shown = runs.filter(run => run.median_round_s !== null);
  if (shown.length < 2) {
    unavailable(host, title, "Cần ít nhất hai lần đo đã xong một vòng để so sánh.");
    return;
  }
  if (!byId("demo-chart-runs-time")) {
    host.innerHTML = `<h3 class="panel-title panel-title-sm">${esc(title)}</h3>
      <p class="demo-legend mt-2"><span><i class="demo-key is-full"></i>Bộ ảnh đầy đủ, 38 loại</span><span><i class="demo-key is-light"></i>Bộ ảnh nhẹ, 10 loại</span></p>
      <div class="live-charts mt-3">
        <div><h4 class="subhead">Thời gian một vòng, trung vị</h4><div class="chart-container demo-chart-runs mt-2"><canvas id="demo-chart-runs-time" aria-label="Thời gian một vòng của từng lần đo"></canvas></div></div>
        <div><h4 class="subhead">Độ chính xác cuối, tập test</h4><div class="chart-container demo-chart-runs mt-2"><canvas id="demo-chart-runs-accuracy" aria-label="Độ chính xác cuối của từng lần đo"></canvas></div></div>
      </div>
      <p class="field-hint mt-2">Hai màu khác loại bệnh và khác tập test nên chỉ so độ chính xác trong cùng một màu, và lần ít vòng đương nhiên thấp hơn lần nhiều vòng. Lần đang chạy hoặc dừng giữa chừng không có số cuối. Rê chuột vào cột để xem số máy và cỡ tập val.</p>`;
  }
  drawChart(byId("demo-chart-runs-time"), () => runConfig(shown, "median_round_s", 1, "Giây", 1));
  drawChart(byId("demo-chart-runs-accuracy"), () => runConfig(shown, "final_accuracy", 100, "Độ chính xác (%)", 2, 100));
}

/* ───────── Dải nối đầu trang và phần so sánh chung ───────── */

let facts = null;
let factsAt = 0;

async function loadFacts() {
  if (facts && Date.now() - factsAt < FACTS_TTL_MS) return facts;
  const [evidence, status, history, runs] = await Promise.allSettled([
    getEvidence(),
    getJson("/api/community/status", { timeoutMs: TIMEOUT_MS }),
    getJson("/api/community/history?limit=1", { timeoutMs: TIMEOUT_MS }),
    getJson("/api/flower/runs", { timeoutMs: TIMEOUT_MS }),
  ]);
  const value = result => (result.status === "fulfilled" ? result.value : null);
  facts = { evidence: value(evidence), status: value(status), history: value(history), runs: value(runs)?.runs || null };
  factsAt = Date.now();
  return facts;
}

function factText(id, data) {
  if (id === "chan-doan") {
    const count = data.evidence?.available ? (data.evidence.models || []).filter(model => !isInt8(model.model_id)).length : 0;
    return count ? `${count} mô hình của đề tài, cùng một tập kiểm tra` : "chưa đọc được số";
  }
  if (id === "hoc-lien-ket") {
    if (!data.status || data.status.enabled === false) return "chưa đọc được số";
    const accepted = data.history?.totals?.accepted;
    return `đang ở phiên bản ${data.status.version}${accepted === undefined ? "" : `, ${accepted} vòng đã nhận`}`;
  }
  if (!data.runs) return "chưa đọc được số";
  const latest = data.runs[0];
  return latest ? `lần mới nhất ${runLabel(latest.run_id).split(",")[0]}, ${latest.num_clients} client, ${latest.completed_rounds} vòng` : "chưa có lần đo";
}

function uploadConfig(flowerBytes) {
  const colors = pillarColors();
  const rows = [["Cộng đồng, chỉ lớp cuối", COMMUNITY_HEAD_BYTES, colors.violet], ["Đo qua mạng, cả mô hình", flowerBytes, colors.fedavg]];
  return {
    type: "bar",
    data: { labels: rows.map(row => row[0]), datasets: [{ label: "Trọng số một client gửi mỗi vòng (MB)", data: rows.map(row => row[1] / 1000000), backgroundColor: rows.map(row => row[2]), borderRadius: 4, maxBarThickness: 22 }] },
    options: {
      animation: false,
      indexAxis: "y",
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: item => `${fmtNum(item.parsed.x, 2)} MB` } } },
      scales: { x: { min: 0, title: { display: true, text: "MB mỗi client, mỗi vòng" } } },
    },
  };
}

function compareHtml(routeId, flowerBytes, imageCount) {
  const heads = DEMOS.map(demo => `<th scope="col"${demo.id === routeId ? ' aria-current="true"' : ""}>${esc(demo.label)}</th>`).join("");
  const rows = compareRows(imageCount).map(([name, ...cells]) => `<tr><th scope="row">${esc(name)}</th>${cells.map(cell => `<td>${esc(cell)}</td>`).join("")}</tr>`).join("");
  return `<div class="table-responsive mt-2" tabindex="0" role="region" aria-label="Bảng so sánh ba demo"><table class="data-table wrap-head"><thead><tr><th scope="col">Điểm so</th>${heads}</tr></thead><tbody>${rows}</tbody></table></div>
    ${flowerBytes ? `<h4 class="subhead mt-3">Trọng số một client gửi mỗi vòng</h4>
    <div class="chart-container demo-chart-upload mt-2"><canvas id="demo-chart-upload-${esc(routeId)}" aria-label="Dung lượng trọng số một client gửi mỗi vòng ở Cộng đồng và Đo qua mạng"></canvas></div>
    <p class="field-hint mt-2">Đo qua mạng: ${fmtNum(flowerBytes / 1000000, 2)} MB, số máy chủ ghi lại. Cộng đồng: ${fmtNum(COMMUNITY_HEAD_BYTES / 1000, 1)} KB, tính từ 38.950 tham số; khi gửi được mã hóa base64 nên đường truyền tốn khoảng 208 KB. Trang Chẩn đoán không gửi trọng số nên không có cột.</p>` : ""}
    <p class="field-hint mt-2">Không vẽ độ chính xác của ba demo trên cùng một trục vì chúng chấm trên các tập ảnh khác nhau. Số chính thức so phân tán với tập trung nằm ở <a href="#ket-qua">chương 4</a>, tab Phân tán và tập trung.</p>`;
}

const hosts = new Set();

async function renderTrail(routeId) {
  const host = byId(`demo-trail-${routeId}`);
  if (!host) return;
  const data = await loadFacts();
  const steps = DEMOS.map((demo, index) => {
    const current = demo.id === routeId;
    return `<a class="demo-trail-step${current ? " is-current" : ""}" href="#${esc(demo.id)}"${current ? ' aria-current="page"' : ""}>
        <span class="demo-trail-title"><span class="step-no">${index + 1}</span>${esc(demo.label)}</span>
        <span class="demo-trail-question">${esc(demo.question)}</span>
        <span class="field-hint">${esc(factText(demo.id, data))}</span>
      </a>`;
  }).join("");
  const flowerBytes = (data.runs || []).map(run => run.client_upload_bytes).find(Boolean) || null;
  const openBefore = host.querySelector("details")?.open;
  host.innerHTML = `<nav class="demo-trail" aria-label="Ba trang demo">${steps}</nav>
    <p class="field-hint mt-2">Cả ba cùng xoay quanh FedAvg. Trang Chẩn đoán dùng mô hình FedAvg đã học sẵn, trang Cộng đồng học tiếp lớp cuối của chính mô hình đó, còn trang Đo qua mạng chạy cùng thuật toán trên một lần học riêng để xem trọng số đi qua mạng thật. Mô hình của các lần đo không được đưa sang trang Chẩn đoán.</p>
    <details class="fold mt-2"${openBefore ? " open" : ""}><summary>So sánh ba demo</summary><div class="fold-body">${compareHtml(routeId, flowerBytes, data.status?.metrics?.count)}</div></details>`;
  if (flowerBytes) {
    const details = host.querySelector("details");
    const draw = () => { if (details.open) drawChart(byId(`demo-chart-upload-${routeId}`), () => uploadConfig(flowerBytes)); };
    details.addEventListener("toggle", draw);    // canvas trong mục đang đóng rộng 0, chỉ vẽ khi mở
    draw();
  }
}

function logTrailError(routeId, error) {
  console.warn(`Không dựng được dải nối ba demo ở trang ${routeId}:`, error);    // trang vẫn dùng được, chỉ thiếu dải này
}

/** Gọi khi dựng trang. Không chặn trang: mỗi lần quay lại trang thì cập nhật số, nguồn nào lỗi thì ô đó ghi "chưa đọc được". */
export function mountDemoTrail(routeId) {
  if (!hosts.size) {
    window.addEventListener("hashchange", () => {
      const id = location.hash.slice(1);
      if (hosts.has(id)) renderTrail(id).catch(error => logTrailError(id, error));
    });
  }
  hosts.add(routeId);
  renderTrail(routeId).catch(error => logTrailError(routeId, error));
}
