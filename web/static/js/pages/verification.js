// Chương 4 · tab "Phân tán và tập trung": đọc /api/evidence/verification. Số do src/federated/verification_view.py tính, cùng hàm với mục 9 và 1 của DEMO_TUNG_MUC.bat.
import { getJson } from "../core/api.js";
import { barRise, makeChartWhenVisible, pillarColors } from "../core/charts.js";
import { byId, esc, fmtInt, fmtNum, sectionHead } from "../core/dom.js";

const num = (value, digits = 2) => (value === null || value === undefined ? "—" : fmtNum(value, digits));
const SCHEME_NAMES = { iid: "chia đều", label_alpha_100_0: "α=100", label_alpha_10_0: "α=10", label_alpha_1_0: "α=1", label_alpha_0_5: "α=0,5", label_alpha_0_1: "α=0,1" };
const CHECK_NAMES = {
  A_fedsgd: "Phân tán 1 bước mỗi vòng, so với tập trung",
  B_local_steps: "Nhiều bước học cục bộ mỗi vòng",
  C_batchnorm_live: "Bật BatchNorm khi học",
};

function missing(section) {
  return `<div class="callout callout-warn" role="note"><b>Phần này chưa có dữ liệu.</b> ${esc(section?.reason || "Báo cáo tương ứng chưa có trên máy chủ.")}</div>`;
}

function localNote(sweep) {
  const gap = sweep.by_alpha.local[sweep.by_alpha.local.length - 1]?.gap;
  return gap === null || gap === undefined ? "" : `<p class="field-hint mt-2">Mỗi máy tự học một mình ở α=0,1 thấp hơn tập trung ${esc(num(gap))} điểm, quá xa nên không vẽ vào biểu đồ; xem ở bảng dưới.</p>`;
}

function sweepHtml(sweep) {
  if (!sweep?.available) return missing(sweep);
  const rows = sweep.rows.map(row => `<tr>
      <td>${esc(row.label)}</td><td>${num(row.top1)} ± ${num(row.top1_std)}</td><td>${num(row.macro_f1)}</td>
      <td>${row.gap === null ? "—" : `${num(row.gap)} điểm`}</td>
      ${sweep.sources.map(name => `<td>${num(row.by_source[name])}</td>`).join("")}
    </tr>`).join("");
  const pending = sweep.not_measured.length
    ? `<div class="callout callout-warn mt-3" role="note"><b>Chưa đo:</b> ${esc(sweep.not_measured.join("; "))}. Các ô này cần thuê máy GPU để train.</div>`
    : "";
  return `
    <p class="panel-desc">Trung bình và độ lệch chuẩn của ${esc(sweep.seeds)} lần chạy trên ${esc(fmtInt(sweep.images))} ảnh thử, mỗi run phân tán học 40 vòng. Tập trung đạt ${esc(num(sweep.centralized_top1))}%. Biểu đồ cho thấy phân tán thấp hơn bao nhiêu điểm: dữ liệu càng lệch (α càng nhỏ), khoảng cách càng lớn.</p>
    <div class="chart-container mt-3"><canvas id="chart-sweep-alpha" aria-label="Độ chính xác thấp hơn tập trung bao nhiêu điểm, theo mức lệch dữ liệu giữa các client"></canvas></div>
    ${localNote(sweep)}
    <div class="table-responsive mt-3" tabindex="0" role="region" aria-label="Bảng độ chính xác theo cách học"><table class="data-table wrap-head">
      <thead><tr><th>Cách học</th><th>Độ chính xác (%)</th><th>Macro-F1 (%)</th><th>Thấp hơn tập trung</th>${sweep.sources.map(name => `<th>${esc(name)} (%)</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>${pending}`;
}

function speedHtml(speed) {
  if (!speed?.available) return missing(speed);
  const rows = speed.runs.flatMap(run => run.machines.map((machine, index) => `<tr${machine.on_server_host ? ' class="is-muted-row"' : ""}>
      ${index === 0 ? `<td rowspan="${run.machines.length}">${esc(run.title)}<br><span class="field-hint">${esc(run.date.slice(6, 8))}/${esc(run.date.slice(4, 6))}, ${esc(run.clients)} client, ${esc(run.rounds)} vòng${run.via_ssh_tunnel ? ", qua đường hầm SSH" : ""}</span></td>` : ""}
      <td>${esc(machine.name)}${machine.on_server_host ? " (cùng máy với máy chủ)" : ""}</td>
      <td>${num(machine.rtt_ms, 1)}</td><td>${num(machine.up_mb_s)}</td><td>${num(machine.down_mb_s)}</td><td>${num(machine.train_s, 1)}</td>
    </tr>`)).join("");
  return `
    <p class="panel-desc">Trung vị qua các vòng. Dòng mờ là client nằm cùng máy với máy chủ: số mạng ở đó rất lớn và không phải đường truyền thật. Các lần đo qua đường hầm SSH có độ trễ cao hơn mạng thường. Lần đo 10 client với máy chủ tách riêng xem ở <a href="#flower">trang Flower</a>.</p>
    <div class="table-responsive mt-3" tabindex="0" role="region" aria-label="Bảng các lần đo tốc độ"><table class="data-table wrap-head">
      <thead><tr><th>Lần đo</th><th>Máy</th><th>RTT (ms)</th><th>Lên (MiB/s)</th><th>Xuống (MiB/s)</th><th>Học 1 vòng (s)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function weightsHtml(weights) {
  if (!weights?.available) return missing(weights);
  const t = weights.transport;
  const distance = weights.distance.map(row => `<tr>
      <td>${esc(row.runtime === "flower" ? "Flower" : "Tự cài")}, ${esc(SCHEME_NAMES[row.scheme] || row.scheme)}</td><td>${esc(row.runs)}</td>
      <td>${num(row.relative_distance * 100, 1)}%</td><td>${num(row.cosine, 4)}</td><td>${num(row.prediction_agreement * 100, 2)}%</td>
    </tr>`).join("");
  const checks = (weights.fedsgd?.rows || []).map(row => `<tr>
      <td>${esc(CHECK_NAMES[row.check] || row.check)}</td><td>${esc(SCHEME_NAMES[row.scheme] || row.scheme)}</td><td>${esc(row.dtype)}</td><td>${esc(row.local_steps)}</td>
      <td class="mono">${esc(Number(row.l2).toExponential(2))}</td><td>${num(row.prediction_agreement * 100, 2)}%</td>
    </tr>`).join("");
  return `
    <div class="live-tiles">
      <div class="kpi-mini live-tile is-ok"><span class="kpi-mini-label">Gửi trọng số đúng từng bit</span><span class="kpi-mini-value">${esc(fmtInt(t.comparisons))} lượt</span><span class="live-tile-note">${esc(t.passed)}/${esc(t.runs)} run trên Flower, ${esc(fmtInt(t.rounds))} vòng, cả hai chiều</span></div>
      <div class="kpi-mini live-tile"><span class="kpi-mini-label">Sai khác lớn nhất khi tự tính lại phép gộp</span><span class="kpi-mini-value mono">${esc(t.max_aggregation_diff === null ? "—" : Number(t.max_aggregation_diff).toExponential(1))}</span><span class="live-tile-note">${esc(t.runs_audited_passed)}/${esc(t.runs_audited)} run qua kiểm; cỡ sai số làm tròn float32</span></div>
      <div class="kpi-mini live-tile is-ok"><span class="kpi-mini-label">Cùng trọng số khởi đầu W0</span><span class="kpi-mini-value">${esc(weights.w0.match)}/${esc(weights.w0.recorded)}</span><span class="live-tile-note">run có ghi W0 khớp W0 dựng lại từ hạt giống; ${esc(weights.w0.total_runs - weights.w0.recorded)} run cũ không ghi</span></div>
      <div class="kpi-mini live-tile is-ok"><span class="kpi-mini-label">Phân tán 1 bước mỗi vòng so với tập trung</span><span class="kpi-mini-value mono">${esc(weights.fedsgd?.max_l2_float64 === null || weights.fedsgd?.max_l2_float64 === undefined ? "—" : Number(weights.fedsgd.max_l2_float64).toExponential(1))}</span><span class="live-tile-note">khoảng cách L2, float64, ${esc(weights.fedsgd?.schemes_float64 ?? 0)} cách chia: trùng nhau</span></div>
    </div>
    <h4 class="subhead mt-4">Mô hình phân tán khác mô hình tập trung bao nhiêu</h4>
    <p class="panel-desc">Cùng hạt giống, sau khi học xong. Chỉ tính tham số học được. Dự đoán trùng tính trên ${esc(fmtInt(weights.distance[0]?.agreement_images))} ảnh thử.</p>
    <div class="table-responsive mt-2" tabindex="0" role="region" aria-label="Bảng khoảng cách giữa mô hình phân tán và tập trung"><table class="data-table wrap-head">
      <thead><tr><th>Cách chạy</th><th>Số run</th><th>Khoảng cách tương đối</th><th>Cosine</th><th>Dự đoán trùng</th></tr></thead>
      <tbody>${distance}</tbody>
    </table></div>
    <h4 class="subhead mt-4">Công thức gộp có tạo sai lệch không</h4>
    <p class="panel-desc">Phép kiểm công thức trên ảnh thật, số ảnh nhỏ và ít bước, không phải huấn luyện đầy đủ. Hai dòng float64 đầu đạt ngưỡng nên phép gộp không tạo sai lệch. Dòng float32 lệch vì sai số làm tròn: tập trung chạy hai lần chỉ đổi thứ tự ảnh cũng lệch cùng cỡ. Khi học nhiều bước mỗi vòng hoặc bật BatchNorm, hai cách học khác nhau thật, nên có khoảng cách.</p>
    <div class="table-responsive mt-2" tabindex="0" role="region" aria-label="Bảng phép kiểm công thức gộp"><table class="data-table wrap-head">
      <thead><tr><th>Phép kiểm</th><th>Cách chia</th><th>Kiểu số</th><th>Bước cục bộ</th><th>L2</th><th>Dự đoán trùng</th></tr></thead>
      <tbody>${checks}</tbody>
    </table></div>`;
}

function drawSweep(sweep) {
  makeChartWhenVisible(byId("chart-sweep-alpha"), () => {
    const colors = pillarColors();
    const { labels, fedavg, flower } = sweep.by_alpha;
    const pick = series => series.map(point => point.gap);
    return {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Phân tán tự cài", data: pick(fedavg), backgroundColor: colors.fedavg },
          { label: "Phân tán trên Flower", data: pick(flower), backgroundColor: colors.violet },
        ],
      },
      options: {
        animation: barRise(),
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display: true, text: "Mức lệch dữ liệu giữa các client: phải là lệch nhiều hơn" } },
          y: { min: 0, title: { display: true, text: "Thấp hơn tập trung (điểm %)" } },
        },
      },
    };
  });
}

export async function buildVerification() {
  const root = byId("result-verification");
  const data = await getJson("/api/evidence/verification", { timeoutMs: 15000 });
  root.innerHTML = `
    ${sectionHead("4.7", "Phân tán và tập trung", { lead: "Học phân tán đạt gần mức học tập trung tới đâu, trọng số có được gửi đúng không, và mạng thật nhanh chậm thế nào." })}
    <section class="panel mt-3"><h4 class="panel-title panel-title-sm">Độ chính xác theo mức lệch dữ liệu</h4>${sweepHtml(data.sweep)}</section>
    <section class="panel mt-4"><h4 class="panel-title panel-title-sm">Kiểm trọng số</h4><div class="mt-3">${weightsHtml(data.weights)}</div></section>
    <section class="panel mt-4"><h4 class="panel-title panel-title-sm">Các lần đo tốc độ giữa các máy</h4>${speedHtml(data.speed)}</section>`;
  if (data.sweep?.available) drawSweep(data.sweep);
}
