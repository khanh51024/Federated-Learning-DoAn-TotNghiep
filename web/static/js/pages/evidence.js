// Bước 6 · Bằng chứng & hạn chế: những gì đã kiểm, những gì chưa làm được, nói thẳng.
import { getEvidence, getSummary } from "../core/store.js";
import { MODEL_LABELS, byId, esc, fmtInt, fmtNum, modelPill, notify } from "../core/dom.js";

const LIMITS = [
  ["Một kịch bản non-IID, một seed", "Chỉ có Dirichlet α=0,1 và seed 42. Chưa đủ để kết luận về độ ổn định hay đường suy giảm theo nhiều mức α."],
  ["Validation nội bộ, không phải ảnh ngoài thực địa", "PlantVillage là ảnh chụp trong phòng; PlantDoc là ảnh ngoài đồng nhưng chỉ phủ 13/38 lớp. Cả hai đều là validation của cùng release."],
  ["Chỉ nhận ảnh một lá đã cắt", "Bộ phân loại không tự tìm lá trên ảnh toàn cây; ảnh ngoài miền sẽ cho kết quả kém dù độ tin cậy vẫn cao."],
  ["Xác suất chưa hiệu chuẩn", "Độ tin cậy softmax hiển thị nguyên bản. Trên PlantDoc các mô hình tự tin quá mức (xem bước 3, tab Độ tin cậy xác suất)."],
  ["Huấn luyện thử trên web chỉ là kỹ thuật", "Phiên Flower dùng ảnh tổng hợp, hai vòng, trên localhost: chứng minh luồng FedAvg và đo RTT cùng máy, không phải kết quả khoa học hay độ trễ mạng liên vùng."],
  ["Chưa có thí nghiệm client lỗi hay tấn công", "Bộ kiểm tra update (observe/reject, ngưỡng L2 norm) có trong mã và chạy được ở bước 5, nhưng chưa có thí nghiệm độ chính xác cùng protocol nên chưa có số liệu để kết luận."],
  ["Shard là mô phỏng", "Năm client là năm phân vùng của một tập dữ liệu, không phải năm nông hộ vật lý."],
  ["Quyền riêng tư không tuyệt đối", "FL chỉ giữ ảnh thô không rời client; trọng số vẫn có thể rò thông tin. Cần differential privacy hoặc secure aggregation nếu muốn bảo đảm mạnh hơn."],
];

function shortHash(value, head = 12) {
  return value ? `${value.slice(0, head)}…${value.slice(-6)}` : "—";
}

function copyButton(value, what) {
  return `<button type="button" class="copy-btn" data-copy="${esc(value)}" data-what="${esc(what)}" aria-label="Sao chép ${esc(what)}">sao chép</button>`;
}

function checklistHtml(cl) {
  if (!cl?.available) {
    return `<div class="callout callout-warn" role="note"><b>Chưa có báo cáo trạng thái.</b> ${esc(cl?.reason || "Chạy scripts/update_status.py.")}</div>`;
  }
  const blocked = cl.items.filter(i => i.status !== "XONG");
  const done = cl.items.filter(i => i.status === "XONG");
  const blockedHtml = blocked.map(item => `
    <article class="todo-card">
      <header><span class="status-badge error">${esc(item.status === "BLOCKED" ? "Chưa làm được" : item.status)}</span><b>${esc(item.title)}</b></header>
      <p>${esc(item.detail)}</p>
      ${(item.unlock || []).length ? `<div class="unlock"><span>Để mở khóa:</span>${item.unlock.map(cmd => `<code>${esc(cmd)}</code>`).join("")}</div>` : ""}
    </article>`).join("");
  return `
    <div class="panel-header-flex">
      <div>
        <h3 class="panel-title">Tiến độ nghiệm thu: ${cl.done}/${cl.total} mục xong</h3>
        <p class="panel-desc">${esc(cl.readiness || "")}</p>
      </div>
    </div>
    <div class="progress-track mt-3" role="img" aria-label="${cl.done} trên ${cl.total} mục xong"><div class="progress-fill fed-fill" style="transform:scaleX(${cl.total ? (cl.done / cl.total).toFixed(3) : 0})"></div></div>
    ${blocked.length ? `<h4 class="subhead mt-4">Còn thiếu (${blocked.length})</h4><div class="todo-grid mt-2">${blockedHtml}</div>` : ""}
    <details class="fold mt-4">
      <summary>Đã xong (${done.length} mục)</summary>
      <ul class="done-list">${done.map(i => `<li><span class="verdict ok">✓</span><div><b>${esc(i.title)}</b><span>${esc(i.detail)}</span></div></li>`).join("")}</ul>
    </details>`;
}

function identityHtml(identity) {
  if (!identity) return "";
  const rows = identity.models.map(m => `
    <tr>
      <th scope="row">${modelPill(m.model_id, m.name)}</th>
      <td>${esc(m.phase)}</td>
      <td class="mono">${esc(m.branch)} @ ${esc(String(m.commit).slice(0, 8))}</td>
      <td class="mono" title="${esc(m.checkpoint_sha256)}">${esc(shortHash(m.checkpoint_sha256))} ${copyButton(m.checkpoint_sha256, `SHA-256 của ${m.name}`)}</td>
    </tr>`).join("");
  return `
    <section class="panel mt-4">
      <h3 class="panel-title">Danh tính artifact</h3>
      <p class="panel-desc">Mọi số trên web chỉ hiện khi manifest và checkpoint khớp các hash dưới đây; sai một byte là web ẩn toàn bộ số liệu (không có số dự phòng).</p>
      <dl class="fact-grid mt-3">
        <div><dt>Protocol</dt><dd class="mono">${esc(identity.protocol_id)}</dd></div>
        <div><dt>Release manifest SHA-256</dt><dd class="mono" title="${esc(identity.release_manifest_sha256)}">${esc(shortHash(identity.release_manifest_sha256, 16))} ${copyButton(identity.release_manifest_sha256, "release manifest SHA-256")}</dd></div>
        <div><dt>Validation manifest SHA-256</dt><dd class="mono" title="${esc(identity.validation_manifest_sha256)}">${esc(shortHash(identity.validation_manifest_sha256, 16))} ${copyButton(identity.validation_manifest_sha256, "validation manifest SHA-256")}</dd></div>
        <div><dt>Seed · tiền xử lý</dt><dd class="mono">${esc(identity.seed)} · ${esc(identity.transform)}</dd></div>
      </dl>
      <div class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">Giai đoạn</th><th scope="col">Nhánh @ commit</th><th scope="col">Checkpoint SHA-256</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>`;
}

function verificationHtml(summary) {
  const re = summary?.reeval;
  const lk = summary?.leakage;
  let reLines = "";
  if (re?.available) {
  const nonExact = (re.models || [])
    .filter(m => Object.values(m.vs_published || {}).some(v => v.status !== "exact"))
    .map(m => m.name);
    reLines = [
      `Centralized và FedAvg: ${re.main_models_exact ? "khớp tuyệt đối trên cả hai nguồn" : "CÓ lệch so với số công bố"}.`,
      nonExact.length
        ? `${re.all_within_tolerance ? "Lệch trong dung sai (≤ 1 ảnh đúng)" : "Lệch VƯỢT dung sai"}: ${nonExact.map(esc).join(", ")}; nguyên nhân chưa chứng minh (có ảnh sát ngưỡng quyết định).`
        : "Không có phép so khớp nào lệch.",
      `Dung sai của báo cáo: ${fmtInt(re.tolerance_rule?.correct_images)} ảnh đúng, ${fmtInt(re.tolerance_rule?.high_confidence_wrong_images)} ảnh sai-tự-tin, ${fmtNum(re.tolerance_rule?.metric, 3)} metric.`,
    ].map(line => `<li>${line}</li>`).join("");
  }
  const reCard = re?.available ? `
      <article class="verify-card ok">
        <h4>Tái đánh giá trên máy này</h4>
        <p class="verify-big mono">${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)}</p>
        <p>phép so khớp <b>tuyệt đối</b> với số công bố (${fmtInt(re.validation_count)} ảnh validation, chạy CPU).</p>
        <ul>${reLines}</ul>
      </article>` : `
      <article class="verify-card warn"><h4>Tái đánh giá trên máy này</h4><p>${esc(re?.reason || "Chưa có báo cáo.")}</p></article>`;
  const lkCard = lk?.available ? `
      <article class="verify-card ${lk.passed ? "ok" : "bad"}">
        <h4>Audit rò rỉ dữ liệu (release ${esc(lk.release_name)})</h4>
        <p class="verify-big mono">${fmtInt(lk.overlap_total)}</p>
        <p>phần tử trùng giữa train / val / calibration trên ${lk.keys.length} khóa (${esc(lk.keys.join(", "))}).</p>
        <ul>
          <li>${fmtInt(lk.images_verified)} ảnh được băm lại byte-by-byte; ${fmtInt(lk.hash_mismatch)} sai hash, ${fmtInt(lk.missing_files)} thiếu file.</li>
          <li>${lk.partitions_clean}/${lk.partitions_total} partition là tập con rời nhau của train, không nhóm ảnh nào bị chia cho nhiều client.</li>
          <li>Train ${fmtInt(lk.counts.train)} · val ${fmtInt(lk.counts.val)} · calibration ${fmtInt(lk.counts.calibration)}.</li>
        </ul>
        <p class="field-hint">${esc(lk.scope || "")}</p>
      </article>` : `
      <article class="verify-card warn"><h4>Audit rò rỉ dữ liệu</h4><p>${esc(lk?.reason || "Chưa có báo cáo.")}</p></article>`;
  return `<section class="verify-grid mt-4">${reCard}${lkCard}</section>`;
}

export async function mount() {
  const root = byId("evidence-root");
  const [evidence, summary] = await Promise.all([getEvidence().catch(() => null), getSummary().catch(() => null)]);
  root.innerHTML = `
    <section class="panel">${checklistHtml(summary?.checklist)}</section>
    ${summary?.available ? identityHtml(summary.identity) : `
      <div class="callout callout-error mt-4" role="alert"><b>Chưa có bộ số liệu đã kiểm hash.</b> ${esc(summary?.reason || evidence?.reason || "Không đọc được /api/evidence/summary.")}</div>`}
    ${summary?.available ? verificationHtml(summary) : ""}
    <section class="panel mt-4">
      <h3 class="panel-title">Hạn chế cần nói rõ</h3>
      <ol class="limit-list mt-3">
        ${LIMITS.map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}
      </ol>
    </section>`;

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
