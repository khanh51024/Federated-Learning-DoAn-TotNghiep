// Demo · Flower: Flower có chạy thật không? Chỉ đọc /api/flower/health (bằng chứng của một phiên đã chạy), không khởi động gì.
// "Server online" chưa đủ: phải thấy flwr được cài cho đúng Python của máy chủ, 5 client kết nối, mọi vòng nhận đủ cập nhật hợp lệ
// và chuỗi mã băm khớp đến checkpoint. Khởi động một phiên mới ở Phụ lục kỹ thuật mục D (cần đăng nhập quản trị).
import { getJson } from "../core/api.js";
import { byId, esc, fmtNum, setHtml } from "../core/dom.js";

const REFRESH_MS = 3000;
const VERDICT = {
  working: { tone: "is-correct", title: "Flower đang hoạt động", mark: "✓" },
  not_verified: { tone: "is-unknown", title: "Chưa kiểm chứng", mark: "•" },
  broken: { tone: "is-wrong", title: "Flower chưa chạy được", mark: "✕" },
};
const MARK = { true: "✓", false: "✕", null: "•" };

let timer = 0;
let loading = false;

const sec = ms => (ms === null || ms === undefined ? "—" : `${fmtNum(Number(ms) / 1000, 1)} s`);
const when = iso => (iso ? new Date(iso).toLocaleString("vi-VN") : "—");

const STATE = { true: "ok", false: "bad", null: "na" };

function itemsHtml(items) {
  return `<ul class="flower-checks" role="list">${items.map(item => `
    <li class="flower-check is-${STATE[String(item.ok)]}" role="listitem">
      <span class="flower-mark" aria-hidden="true">${MARK[String(item.ok)]}</span>
      <div><b>${esc(item.label)}</b><span class="flower-detail">${esc(item.detail)}</span>
      <span class="sr-only">${item.ok === true ? "đạt" : item.ok === false ? "không đạt" : "chưa có dữ liệu"}</span></div>
    </li>`).join("")}</ul>`;
}

function roundsHtml(evidence) {
  if (!evidence?.rounds?.length) return "";
  const rows = evidence.rounds.map(r => `<tr>
      <th scope="row">Vòng ${esc(r.round)}</th>
      <td class="mono"><b>${esc(r.accepted)}/5</b><span class="row-sub">đáng ngờ ${esc(r.suspicious ?? 0)} · bị loại ${esc(r.rejected ?? 0)}</span></td>
      <td class="mono">${esc(sec(r.fit_ms))}</td>
      <td class="mono">${r.mean_rtt_ms === null || r.mean_rtt_ms === undefined ? "—" : `${fmtNum(r.mean_rtt_ms, 1)} ms`}</td>
      <td class="mono" title="${esc(r.parent_hash || "")}">${esc(String(r.parent_hash || "—").slice(0, 12))}</td>
      <td class="mono" title="${esc(r.global_hash || "")}">${esc(String(r.global_hash || "—").slice(0, 12))}</td></tr>`).join("");
  const ckpt = evidence.checkpoint;
  return `<h4 class="subhead mt-4">Các vòng của phiên ${esc(evidence.name)}</h4>
    <div tabindex="0" role="group" aria-label="Bảng các vòng Flower, cuộn được bằng phím mũi tên" class="table-responsive mt-2">
      <table class="data-table wrap-head"><thead><tr><th scope="col">Vòng</th><th scope="col">Cập nhật nhận được</th><th scope="col">Thời gian học</th>
        <th scope="col">RTT trung bình</th><th scope="col">Mã băm đầu vào</th><th scope="col">Mã băm sau gộp</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="field-hint mt-2">Mã băm sau gộp của vòng r phải bằng mã băm đầu vào của vòng r+1, và vòng cuối phải bằng mã băm ghi trong checkpoint.
      Checkpoint <code>global.pt</code>: ${ckpt?.present ? `${esc(ckpt.bytes)} byte, SHA-256 <span class="mono">${esc(String(ckpt.sha256 || "").slice(0, 32))}…</span>` : "chưa có"}.
      Bắt đầu ${esc(when(evidence.started_at_utc))}, kết thúc ${esc(when(evidence.finished_at_utc))}.</p>
    <p class="field-hint">${esc(evidence.note || "")}</p>`;
}

function render(data) {
  const verdict = VERDICT[data.verdict] || VERDICT.not_verified;
  const live = data.live?.active
    ? `<p class="headline-meta"><span class="verdict ok">● Đang chạy</span> ${esc(data.live.client_processes_running ?? 0)} client · ${esc(data.live.connected_clients ?? 0)}/5 đã kết nối${data.live.round_in_progress ? ` · vòng ${esc(data.live.round_in_progress)}` : ""}</p>` : "";
  const attempt = data.latest_attempt;
  const failure = attempt && attempt.status === "failed"
    ? `<div class="callout callout-error mt-3" role="alert"><b>Phiên gần nhất bị lỗi.</b> ${esc(attempt.error || attempt.log_tail || "không rõ nguyên nhân")}${attempt.log_tail && attempt.log_tail !== attempt.error ? `<br><span class="mono">${esc(attempt.log_tail)}</span>` : ""}</div>` : "";
  setHtml(byId("flower-root"), `
    <div class="diag-headline ${verdict.tone}" role="status" aria-live="polite">
      <p class="summary-line">Kết luận lúc ${esc(when(data.checked_at))}</p>
      <p class="flower-verdict"><span aria-hidden="true">${verdict.mark}</span> ${esc(verdict.title)}</p>
      <p class="headline-meta">${esc(data.summary)}</p>${live}
    </div>
    ${failure}
    <h4 class="subhead mt-4">Các điều đã kiểm</h4>${itemsHtml(data.items || [])}
    ${roundsHtml(data.evidence)}
    <div class="callout mt-4" role="note"><b>Muốn có bằng chứng mới?</b> Mở <a href="#phu-luc">Phụ lục kỹ thuật, mục D</a>, đăng nhập quản trị rồi bấm Bắt đầu (FedAvg qua Flower, 5 client).
      Trang này tự cập nhật khi phiên chạy. Phiên kiểm tra dùng ảnh nhiễu nên chỉ chứng minh hệ thống chạy thật, không đo độ chính xác của mô hình.</div>`);
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    render(await getJson("/api/flower/health", { timeoutMs: 15000 }));
  } catch (error) {
    setHtml(byId("flower-root"), `<div class="callout callout-error" role="alert"><b>Không đọc được trạng thái Flower.</b> ${esc(error.message)}</div>`);
  } finally {
    loading = false;
  }
}

function schedule() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (location.hash !== "#flower") { clearInterval(timer); return; } // rời trang: dừng poll
    if (!document.hidden) refresh();
  }, REFRESH_MS);
}

export async function mount() {
  await refresh();
  schedule();
}

export async function onShow() {
  await refresh();
  schedule();
}
