// Trọng số và độ trễ của học phân tán cộng đồng, dùng chung cho trang "Học phân tán" và trang quản trị.
// Nhận phản hồi của GET /api/community/history và trả về HTML (không DOM, không gọi mạng) để hai trang dựng giống nhau.
// Mọi số đều đến từ máy chủ: trọng số gộp là min(n, n_cap) / tổng đúng như phép gộp FedAvg; độ trễ do client báo, chỉ để theo dõi.
import { esc, fmtNum } from "./dom.js";

const KIND_LABELS = { base: "Gốc (FedAvg v7)", aggregate: "Gộp từ vòng", rollback: "Khôi phục" };
const SERIES = [
  { key: "accuracy", label: "Độ chính xác", cls: "cw-acc" },
  { key: "macro_recall", label: "Macro-recall", cls: "cw-macro" },
  { key: "plantdoc_accuracy", label: "Ảnh ngoài đồng (PlantDoc)", cls: "cw-pd" },
];

const pct = value => (value === null || value === undefined ? "—" : `${fmtNum(Number(value) * 100, 2)}%`);
const sec = ms => (ms === null || ms === undefined ? "—" : `${fmtNum(Number(ms) / 1000, 2)} s`);
const num = (value, digits = 0) => (value === null || value === undefined ? "—" : fmtNum(Number(value), digits));

export function median(values) {
  const list = values.filter(v => typeof v === "number" && Number.isFinite(v)).sort((a, b) => a - b);
  if (!list.length) return null;
  const mid = list.length >> 1;
  return list.length % 2 ? list[mid] : (list[mid - 1] + list[mid]) / 2;
}

function contributorsOf(rounds) {
  return rounds.flatMap(round => round.contributors || []);
}

/** Sáu thẻ độ trễ: trung vị trên các vòng đang hiển thị. */
export function latencyHtml(rounds) {
  const people = contributorsOf(rounds);
  const cards = [
    ["Học trên máy người tham gia", sec(median(people.map(c => c.train_ms))), "thời gian huấn luyện lớp cuối"],
    ["Trích đặc trưng ảnh", sec(median(people.map(c => c.extract_ms))), "tổng cho các ảnh của một người"],
    ["Gọi máy chủ", `${num(median(people.map(c => c.rtt_ms)))} ms`, "một lượt /round đo ở trình duyệt"],
    ["Gửi sau khi vòng mở", sec(median(people.map(c => (c.wait_s === null || c.wait_s === undefined ? null : c.wait_s * 1000)))), "từ lúc mở vòng đến lúc nhận"],
    ["Thời lượng một vòng", sec(median(rounds.map(r => (r.duration_s === null || r.duration_s === undefined ? null : r.duration_s * 1000)))), "từ lúc mở đến lúc đóng"],
    ["Gộp và đo trên máy chủ", `${num(median(rounds.map(r => (r.agg_ms ?? null) === null ? null : r.agg_ms + (r.eval_ms || 0))), 1)} ms`, "gộp trọng số rồi chấm tập giám sát"],
  ];
  return `<div class="cw-cards" role="list">${cards.map(([title, value, hint]) =>
    `<div class="cw-card" role="listitem"><span class="cw-card-title">${esc(title)}</span><b class="cw-card-value mono">${esc(value)}</b><span class="cw-card-hint">${esc(hint)}</span></div>`).join("")}</div>
    <p class="field-hint mt-2">Độ trễ do trình duyệt người tham gia báo về, chỉ để theo dõi; không ảnh hưởng việc gộp. Trung vị trên ${rounds.length} vòng gần nhất${people.length ? `, ${people.length} lượt đóng góp` : ""}.</p>`;
}

/** Thay đổi độ chính xác, macro-recall và PlantDoc so với phiên bản đầu (điểm phần trăm), theo phiên bản (SVG, không cần thư viện biểu đồ).
 *  Vẽ độ lệch so với v0 thay vì giá trị tuyệt đối để thấy mô hình chung đi lên hay đi xuống; giá trị tuyệt đối nằm ở chú thích và bảng. */
export function chartHtml(versions) {
  const points = versions.filter(v => v.metrics);
  if (points.length < 2) {
    return '<p class="field-hint">Cần ít nhất hai phiên bản để vẽ biểu đồ. Hiện mới có mô hình gốc.</p>';
  }
  const base = points[0].metrics;
  const delta = (v, key) => (typeof v.metrics[key] === "number" && typeof base[key] === "number" ? (v.metrics[key] - base[key]) * 100 : null);
  const W = 640; const H = 250; const L = 52; const R = 14; const T = 14; const B = 36;
  const all = points.flatMap(v => SERIES.map(s => delta(v, s.key))).filter(x => x !== null);
  const low = Math.min(-0.5, Math.floor((Math.min(...all) - 0.2) * 2) / 2);
  const high = Math.max(0.5, Math.ceil((Math.max(...all) + 0.2) * 2) / 2);
  const x = i => L + (points.length === 1 ? 0 : (i * (W - L - R)) / (points.length - 1));
  const y = value => T + ((high - value) * (H - T - B)) / (high - low || 1);
  const sign = value => `${value > 0 ? "+" : value < 0 ? "−" : ""}${fmtNum(Math.abs(value), 1)}`;
  const ticks = 4;
  const grid = Array.from({ length: ticks + 1 }, (_, k) => {
    const value = low + ((high - low) * k) / ticks;
    return `<line class="cw-grid" x1="${L}" x2="${W - R}" y1="${y(value).toFixed(1)}" y2="${y(value).toFixed(1)}"/>
      <text class="cw-tick" x="${L - 6}" y="${(y(value) + 4).toFixed(1)}" text-anchor="end">${sign(value)}</text>`;
  }).join("");
  const zero = `<line class="cw-zero" x1="${L}" x2="${W - R}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}"/>`;
  const step = Math.max(1, Math.ceil(points.length / 10));
  const labels = points.map((v, i) => (i % step === 0 || i === points.length - 1
    ? `<text class="cw-tick" x="${x(i).toFixed(1)}" y="${H - 12}" text-anchor="middle">v${esc(v.version)}</text>` : "")).join("");
  const lines = SERIES.map(series => {
    const pts = points.map((v, i) => (delta(v, series.key) !== null ? `${x(i).toFixed(1)},${y(delta(v, series.key)).toFixed(1)}` : null)).filter(Boolean);
    const dots = points.map((v, i) => (delta(v, series.key) !== null
      ? `<circle class="cw-dot ${series.cls}" cx="${x(i).toFixed(1)}" cy="${y(delta(v, series.key)).toFixed(1)}" r="3"><title>v${esc(v.version)} · ${esc(series.label)} ${esc(pct(v.metrics[series.key]))} (${esc(sign(delta(v, series.key)))} điểm so với v${esc(points[0].version)})</title></circle>` : "")).join("");
    return `<polyline class="cw-line ${series.cls}" fill="none" points="${pts.join(" ")}"/>${dots}`;
  }).join("");
  const last = points[points.length - 1];
  const summary = `Độ chính xác đổi ${sign(delta(last, "accuracy") ?? 0)} điểm phần trăm từ v${points[0].version} (${pct(base.accuracy)}) đến v${last.version} (${pct(last.metrics.accuracy)})`;
  return `<figure class="cw-chart">
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(summary)}" class="cw-svg">${grid}${zero}${lines}${labels}</svg>
      <figcaption class="cw-legend">${SERIES.map(s => `<span class="cw-key"><i class="cw-swatch ${s.cls}" aria-hidden="true"></i>${esc(s.label)}</span>`).join("")}
        <span class="cw-note">Điểm phần trăm so với v${esc(points[0].version)}; ${esc(summary)}.</span></figcaption>
    </figure>`;
}

/** Bảng các phiên bản: số đo giám sát và liên kết tải tệp trọng số (.npz) của từng phiên bản còn lưu. */
export function versionsHtml(versions) {
  if (!versions.length) return '<p class="field-hint">Chưa có phiên bản nào.</p>';
  const rows = versions.slice().reverse().map(v => `<tr>
      <th scope="row" class="fw-600">v${esc(v.version)}</th>
      <td>${esc(KIND_LABELS[v.kind] || v.kind)}</td>
      <td class="mono">${esc(v.clients)}</td><td class="mono">${esc(v.examples)}</td>
      <td class="mono">${esc(pct(v.metrics?.accuracy))}</td><td class="mono">${esc(pct(v.metrics?.macro_recall))}</td>
      <td class="mono">${esc(pct(v.metrics?.plantdoc_accuracy))}</td>
      <td>${v.downloadable
        ? `<a class="btn btn-ghost btn-sm" href="/api/community/heads/${esc(v.version)}" download>Tải .npz</a><span class="row-sub mono" title="Mã băm của trọng số">${esc(String(v.sha || "").slice(0, 10))}</span>`
        : '<span class="text-muted">đã dọn</span>'}</td></tr>`).join("");
  return `<div tabindex="0" role="group" aria-label="Bảng các phiên bản mô hình và tệp trọng số, cuộn được bằng phím mũi tên" class="table-responsive mt-2">
      <table class="data-table wrap-head"><thead><tr><th scope="col">Phiên bản</th><th scope="col">Loại</th><th scope="col">Người</th><th scope="col">Ảnh</th>
        <th scope="col">Chính xác</th><th scope="col">Macro-recall</th><th scope="col">PlantDoc</th><th scope="col">Trọng số</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="field-hint mt-2">Tệp .npz gồm <code>weight</code> (38 × 1024) và <code>bias</code> (38) của lớp cuối <code>classifier.3</code>; ghép với phần đóng băng của FedAvg v7 để chạy lại mô hình. Chỉ giữ 200 phiên bản mới nhất và bản gốc.</p>`;
}

function contributorRows(round) {
  return (round.contributors || []).map((c, i) => `<tr>
      <th scope="row">Người ${i + 1}</th>
      <td class="mono">${esc(c.n)}${c.n_used !== c.n ? `<span class="row-sub">tính ${esc(c.n_used)}</span>` : ""}</td>
      <td><meter class="cw-meter" min="0" max="1" value="${Number(c.weight) || 0}" aria-label="Trọng số gộp">${esc(pct(c.weight))}</meter> <b class="mono">${esc(pct(c.weight))}</b></td>
      <td class="mono">${esc(num(c.norm, 3))}${c.clipped ? `<span class="row-sub">cắt còn ${esc(num(c.norm_used, 3))}</span>` : ""}</td>
      <td class="mono">${esc(sec(c.train_ms))}</td><td class="mono">${esc(sec(c.extract_ms))}</td>
      <td class="mono">${esc(c.rtt_ms === null || c.rtt_ms === undefined ? "—" : `${num(c.rtt_ms)} ms`)}</td>
      <td class="mono">${esc(c.wait_s === null || c.wait_s === undefined ? "—" : `${num(c.wait_s, 1)} s`)}</td></tr>`).join("");
}

/** Từng vòng: số người, độ lớn thay đổi của mô hình chung và bảng trọng số gộp từng người. */
export function roundsHtml(rounds) {
  if (!rounds.length) return '<p class="field-hint">Chưa có vòng nào hoàn tất.</p>';
  return `<div class="cw-rounds">${rounds.map(round => {
    let result = "Đã hủy";
    let cls = "is-warn";
    if (round.status === "aggregated") { result = `Đã nhận → v${round.version}`; cls = "is-ok"; }
    else if (round.status === "rejected") { result = "Bị từ chối"; cls = "is-bad"; }
    const people = round.contributors || [];
    const total = people.reduce((s, c) => s + (Number(c.weight) || 0), 0);
    return `<details class="cw-round fold" data-round="${esc(round.round_id)}">
      <summary><b>Vòng #${esc(round.round_id)}</b> · từ v${esc(round.parent_version)} · <span class="row-state ${cls}">${esc(result)}</span>
        <span class="cw-sum">${esc(round.clients)} người · ${esc(round.examples)} ảnh · Δ trọng số ${esc(num(round.delta_norm, 4))} · ${esc(sec(round.duration_s === null || round.duration_s === undefined ? null : round.duration_s * 1000))}</span></summary>
      <div class="fold-body">
        ${round.status === "rejected" ? `<p class="field-hint">${esc((round.messages || []).join("; ") || "Không đạt cổng chất lượng.")}</p>` : ""}
        ${people.length ? `<div tabindex="0" role="group" aria-label="Trọng số gộp của từng người trong vòng ${esc(round.round_id)}, cuộn được bằng phím mũi tên" class="table-responsive mt-2">
          <table class="data-table wrap-head"><thead><tr><th scope="col">Đóng góp</th><th scope="col">Ảnh</th><th scope="col">Trọng số gộp</th><th scope="col">Độ lớn cập nhật</th>
            <th scope="col">Học</th><th scope="col">Trích đặc trưng</th><th scope="col">Gọi /round</th><th scope="col">Gửi sau</th></tr></thead><tbody>${contributorRows(round)}</tbody></table></div>
          <p class="field-hint mt-2">Trọng số = min(số ảnh, ${esc(round.n_cap ?? "trần")}) chia tổng, cộng lại ${esc(pct(total))}. Độ lớn cập nhật là chuẩn L2 của (trọng số người gửi trừ trọng số mô hình cha) trước khi cắt. Δ trọng số là độ lớn thay đổi của mô hình chung sau khi gộp. Gộp ${esc(num(round.agg_ms, 1))} ms, chấm giám sát ${esc(num(round.eval_ms, 1))} ms.</p>`
    : '<p class="field-hint">Vòng này không có dữ liệu đóng góp (vòng cũ hoặc đã hủy).</p>'}
      </div></details>`;
  }).join("")}</div>`;
}

/** Vẽ khối vào một phần tử, giữ các vòng người dùng đang mở khi trang làm mới định kỳ. */
export function renderWeightsInto(container, history) {
  if (!container) return;
  const open = new Set(Array.from(container.querySelectorAll("details.cw-round[open]")).map(el => el.dataset.round));
  container.innerHTML = weightsHtml(history);
  container.querySelectorAll("details.cw-round").forEach(el => { if (open.has(el.dataset.round)) el.open = true; });
}

/** Toàn bộ khối "Trọng số và độ trễ" từ một phản hồi /history. */
export function weightsHtml(history) {
  const rounds = history?.rounds || [];
  const versions = history?.versions || [];
  return `<div class="cw-block"><h5 class="cw-h">Độ trễ</h5>${latencyHtml(rounds)}</div>
    <div class="cw-block"><h5 class="cw-h">Thay đổi độ chính xác theo phiên bản</h5>${chartHtml(versions)}</div>
    <div class="cw-block"><h5 class="cw-h">Phiên bản và tệp trọng số</h5>${versionsHtml(versions)}</div>
    <div class="cw-block"><h5 class="cw-h">Trọng số gộp theo vòng</h5>${roundsHtml(rounds)}</div>`;
}
