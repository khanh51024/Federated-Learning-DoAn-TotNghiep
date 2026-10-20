// Demo · Flower: Flower có chạy thật không? Đọc /api/flower/health cho kết luận, /api/flower/evidence/{id} cho bằng chứng thô.
// "Server online" chưa đủ: mỗi mục kiểm vì thế có một nút mở ra đúng lệnh máy chủ đã chạy + output nguyên bản (log server/client,
// session/history/progress, mã băm tính lại từ tệp). Trang không khởi động gì — muốn có phiên mới thì mở Phụ lục kỹ thuật mục D.
import { getJson } from "../core/api.js";
import { byId, esc, fmtInt, fmtNum, notify, setHtml, skeleton } from "../core/dom.js";

const REFRESH_MS = 3000;
const EVIDENCE_TIMEOUT_MS = 30000; // một bảng bằng chứng có thể spawn vài tiến trình kiểm tra rồi mới trả lời
const HEALTH_TIMEOUT_MS = 15000; // /health chỉ đọc tệp, nhưng lúc đang chạy Flower thì máy rất bận
const LIVE_REFRESH_MS = 10000; // nhịp đọc lại bảng "đang chạy": máy chủ đã cache 2 giây, lấy nhanh hơn nữa chỉ tổ hết hạn mức
const VERDICT = {
  working: { tone: "is-correct", title: "Flower đang hoạt động", mark: "✓" },
  not_verified: { tone: "is-unknown", title: "Chưa kiểm chứng", mark: "•" },
  broken: { tone: "is-wrong", title: "Flower chưa chạy được", mark: "✕" },
};
const MARK = { true: "✓", false: "✕", null: "•" };
const STATE = { true: "ok", false: "bad", null: "na" };
const OS_LABEL = { linux: "Linux", darwin: "macOS", win32: "Windows" };

let timer = 0;
let loading = false;
let lastSignature = ""; // không dựng lại DOM khi dữ liệu đổi mỗi 3 giây một chút: đang mở bảng mà dựng lại thì mất chỗ đang cuộn
let lastRunKey = "";    // đổi phiên là mất hiệu lực mọi bảng đã tải, không được cho người xem thấy bằng chứng của phiên cũ
const OPEN = new Set();
const EVIDENCE = new Map();

const sec = ms => (ms === null || ms === undefined ? "—" : `${fmtNum(Number(ms) / 1000, 1)} s`);
const when = iso => (iso ? new Date(iso).toLocaleString("vi-VN") : "—");

function copyButton(value, what) {
  return `<button type="button" class="copy-btn" data-copy="${esc(value)}" data-what="${esc(what)}" aria-label="Sao chép ${esc(what)}">sao chép</button>`;
}

// ---------------------------------------------------------------------------
// từng loại khối bằng chứng
// ---------------------------------------------------------------------------
function cellText(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "✓ khớp" : "✕ lệch";
  if (typeof value === "number") return Number.isInteger(value) ? fmtInt(value) : fmtNum(value, 3);
  return esc(value);
}

function commandHtml(block) {
  const exit = block.exit_code;
  const exitText = exit === null || exit === undefined
    ? '<span class="ev-exit is-bad">không chạy được</span>'
    : exit === 0 ? '<span class="ev-exit is-ok">mã thoát 0</span>' : `<span class="ev-exit is-bad">mã thoát ${esc(exit)}</span>`;
  return `<div class="ev-block">
    <h5 class="ev-title">${esc(block.title)}</h5>
    <div class="ev-labeled"><span class="ev-tag">lệnh máy chủ đã thực thi</span>${copyButton(block.command || "", "lệnh")}</div>
    <pre class="ev-pre ev-cmd mono">${esc(block.command || "")}</pre>
    ${block.note ? `<p class="field-hint">${esc(block.note)}</p>` : ""}
    <div class="ev-labeled"><span class="ev-tag">kết quả trả về</span>${copyButton(block.output || "", "kết quả")}</div>
    <pre class="ev-pre ev-out mono">${esc(block.output || "(không có output)")}</pre>
    <p class="ev-meta">${exitText} · chạy tại thư mục <code class="code">${esc(block.cwd || ".")}</code>${block.platform ? ` · máy chủ ${esc(OS_LABEL[block.platform] || block.platform)}` : ""}${block.truncated ? " · output dài nên đã cắt" : ""}</p>
  </div>`;
}

function fileHtml(block) {
  return `<div class="ev-block">
    <h5 class="ev-title">${esc(block.title)}</h5>
    <p class="ev-meta"><code class="code">${esc(block.path || "")}</code>${block.line_count ? ` · ${esc(fmtInt(block.line_count))} dòng` : ""}${block.source ? ` · ${esc(block.source)}` : ""}${block.truncated ? " · chỉ hiển thị phần cuối" : ""}</p>
    ${block.note ? `<p class="field-hint">${esc(block.note)}</p>` : ""}
    <pre class="ev-pre ev-out mono">${esc(block.content || "(trống)")}</pre>
  </div>`;
}

function recordsHtml(block) {
  if (!block.rows?.length) return `<div class="ev-block"><h5 class="ev-title">${esc(block.title)}</h5><p class="field-hint">Chưa có bản ghi nào.</p></div>`;
  const head = (block.columns || []).map(c => `<th scope="col">${esc(c.label)}</th>`).join("");
  const body = block.rows.map(row => `<tr>${(block.columns || []).map(c => {
    const isHash = /hash|sha/i.test(c.key) && typeof row[c.key] === "string" && row[c.key].length > 20;
    return `<td class="${isHash ? "ev-hash mono" : "mono"}" title="${esc(row[c.key] ?? "")}">${cellText(row[c.key])}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="ev-block">
    <h5 class="ev-title">${esc(block.title)}</h5>
    ${block.caption ? `<p class="field-hint">${esc(block.caption)}</p>` : ""}
    <div class="table-responsive" tabindex="0" role="group" aria-label="Bảng ${esc(block.title)}, cuộn được bằng phím mũi tên">
      <table class="data-table wrap-head ev-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
  </div>`;
}

function blockHtml(block) {
  if (block.kind === "command") return commandHtml(block);
  if (block.kind === "file") return fileHtml(block);
  if (block.kind === "records") return recordsHtml(block);
  if (block.kind === "note") return `<p class="ev-note">${esc(block.text || "")}</p>`;
  return "";
}

function evidenceBody(data) {
  return `<p class="ev-assert"><b>Cách kiểm:</b> ${esc(data.assertion || "")}${data.run ? ` · phiên <code class="code">${esc(data.run)}</code>` : ""}</p>
    ${(data.blocks || []).map(blockHtml).join("")}`;
}

// ---------------------------------------------------------------------------
// mở / đóng bảng bằng chứng
// ---------------------------------------------------------------------------
let lastLiveAt = 0;

// Trang poll 3 giây/lần, còn API bằng chứng chỉ cho 20 lượt/phút: bảng "đang chạy" vì thế chỉ tự đọc lại
// khi người xem đang mở nó và không quá dày, nếu không một phiên chạy 1 phút sẽ làm cả trang hết hạn mức.
function refreshLive(data) {
  if (!data.live?.active || !OPEN.has("live") || Date.now() - lastLiveAt < LIVE_REFRESH_MS) return;
  lastLiveAt = Date.now();
  loadEvidence("live", { force: true });
}

async function loadEvidence(id, { force = false } = {}) {
  const host = byId(`flower-ev-${id}`);
  if (!host) return;
  const cached = EVIDENCE.get(id);
  if (cached && !force) { setHtml(host, evidenceBody(cached)); return; }
  // Làm mới nền (đã có dữ liệu): giữ nguyên bảng đang xem, chỉ thay khi có kết quả mới → không nháy, không mất chỗ cuộn.
  if (cached) { if (!host.children.length) setHtml(host, evidenceBody(cached)); }
  else setHtml(host, `<div class="ev-block"><h5 class="ev-title">Đang đọc bằng chứng từ máy chủ…</h5>${skeleton(4)}</div>`);
  try {
    const data = await getJson(`/api/flower/evidence/${encodeURIComponent(id)}`, { timeoutMs: EVIDENCE_TIMEOUT_MS });
    EVIDENCE.set(id, data);
    if (OPEN.has(id)) setHtml(host, evidenceBody(data)); // có thể đã bị đóng trong lúc chờ
  } catch (error) {
    if (OPEN.has(id) && !cached) { // làm mới nền thất bại thì giữ bảng cũ, lần tự đọc sau sẽ thử lại
      setHtml(host, `<div class="callout callout-error" role="alert"><b>Không đọc được bằng chứng.</b> ${esc(error.message)}
        <br>Bấm đóng rồi mở lại để thử lại.</div>`);
    }
  }
}

function foldHtml(id) {
  const open = OPEN.has(id);
  const cached = EVIDENCE.get(id);
  return `<details class="fold flower-ev" data-evidence="${esc(id)}"${open ? " open" : ""}>
    <summary>Xem lệnh đã chạy + kết quả thô</summary>
    <div class="fold-body flower-ev-body" id="flower-ev-${esc(id)}">${open ? (cached ? evidenceBody(cached) : "") : ""}</div>
  </details>`;
}

function itemsHtml(items) {
  return `<ul class="flower-checks" role="list">${items.map(item => `
    <li class="flower-check is-${STATE[String(item.ok)]}" role="listitem">
      <span class="flower-mark" aria-hidden="true">${MARK[String(item.ok)]}</span>
      <div><b>${esc(item.label)}</b><span class="flower-detail">${esc(item.detail)}</span>
      <span class="sr-only">${item.ok === true ? "đạt" : item.ok === false ? "không đạt" : "chưa có dữ liệu"}</span>
      ${foldHtml(item.id)}</div>
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
      Bấm "Xem lệnh đã chạy + kết quả thô" ở mục <b>Chuỗi mã băm khớp đến checkpoint</b> để thấy cả 64 ký tự và phép băm tính lại từ tệp.
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

  const runKey = `${attempt?.name || ""}|${data.evidence?.name || ""}`;
  if (runKey !== lastRunKey) { EVIDENCE.clear(); lastRunKey = runKey; } // bằng chứng cũ không còn mô tả phiên hiện tại
  const signature = JSON.stringify({ ...data, checked_at: undefined });
  if (signature === lastSignature) { // chỉ cập nhật dòng thời gian: dựng lại sẽ gập các bảng đang mở và-reset vị trí cuộn
    const stamp = byId("flower-checked-at");
    if (stamp) stamp.textContent = `Kết luận lúc ${when(data.checked_at)}`;
    refreshLive(data);
    return;
  }
  lastSignature = signature;

  setHtml(byId("flower-root"), `
    <div class="diag-headline ${verdict.tone}" role="status" aria-live="polite">
      <p class="summary-line" id="flower-checked-at">Kết luận lúc ${esc(when(data.checked_at))}</p>
      <p class="flower-verdict"><span aria-hidden="true">${verdict.mark}</span> ${esc(verdict.title)}</p>
      <p class="headline-meta">${esc(data.summary)}</p>${live}
    </div>
    ${failure}
    <h4 class="subhead mt-4">Các điều đã kiểm</h4>
    <p class="panel-desc">Mỗi mục mở được để xem đúng lệnh máy chủ đã chạy và kết quả nguyên bản — không phải lời khẳng định "đạt" suông.</p>
    ${itemsHtml(data.items || [])}
    ${roundsHtml(data.evidence)}
    <div class="callout mt-4" role="note"><b>Muốn có bằng chứng mới?</b> Mở <a href="#phu-luc">Phụ lục kỹ thuật, mục D</a>, đăng nhập quản trị rồi bấm Bắt đầu (FedAvg qua Flower, 5 client).
      Trang này tự cập nhật khi phiên chạy. Phiên kiểm tra dùng ảnh nhiễu nên chỉ chứng minh hệ thống chạy thật, không đo độ chính xác của mô hình.</div>`);

  for (const id of OPEN) if (!EVIDENCE.has(id)) loadEvidence(id); // mục đang mở sau khi dựng lại: tải tiếp phần còn thiếu
  refreshLive(data);
}

// Một lượt đọc chậm lúc máy chủ đang chạy 6 tiến trình Flower không được xóa sạch những gì người xem đang xem:
// đã dựng rồi thì chỉ dán thêm dải cảnh báo, các mục và bảng bằng chứng đang mở vẫn còn nguyên.
function showError(message) {
  const root = byId("flower-root");
  if (!root) return;
  const html = `<b>Không đọc được trạng thái Flower.</b> ${esc(message)} Trang sẽ tự thử lại ở lần cập nhật sau.`;
  if (!root.children.length) { setHtml(root, `<div class="callout callout-error" role="alert">${html}</div>`); return; }
  if (!root.querySelector("[data-flower-error]")) {
    root.insertAdjacentHTML("afterbegin", `<div class="callout callout-error" role="alert" data-flower-error>${html}</div>`);
  }
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const data = await getJson("/api/flower/health", { timeoutMs: HEALTH_TIMEOUT_MS });
    byId("flower-root")?.querySelector("[data-flower-error]")?.remove();
    render(data);
  } catch (error) {
    showError(error.message);
  } finally {
    loading = false;
  }
}

function bind(root) {
  root.addEventListener("toggle", event => {
    const fold = event.target.closest?.("[data-evidence]");
    if (!fold) return;
    const id = fold.dataset.evidence;
    if (!fold.open) { OPEN.delete(id); return; }
    OPEN.add(id);
    // bảng "đang chạy" mà mở ra rồi đưa số cache cũ thì vô nghĩa: người xem bấm là đọc lại, và nhịp tự động tính từ lúc này
    const refetch = id === "live" && EVIDENCE.has(id);
    if (refetch) lastLiveAt = Date.now();
    loadEvidence(id, { force: refetch });
  }, true); // sự kiện toggle không nổi bóng: phải nghe ở giai đoạn capture
  root.addEventListener("click", async event => {
    const button = event.target.closest(".copy-btn");
    if (!button) return;
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      notify("success", "Đã sao chép", button.dataset.what);
    } catch (_error) {
      notify("warning", "Không sao chép được", "Trình duyệt chặn clipboard; hãy bôi đen và sao chép thủ công.");
    }
  });
}

function schedule() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (location.hash !== "#flower") { clearInterval(timer); return; } // rời trang: dừng poll
    if (!document.hidden) refresh();
  }, REFRESH_MS);
}

export async function mount() {
  bind(byId("flower-root"));
  await refresh();
  schedule();
}

export async function onShow() {
  await refresh();
  schedule();
}
