// Bước 6 · Bằng chứng & hạn chế: những gì đã kiểm, những gì chưa làm được, nói thẳng.
import { getEvidence, getSummary } from "../core/store.js";
import { MODEL_LABELS, byId, esc, fmtInt, fmtNum, fmtPct, modelPill, notify } from "../core/dom.js";

const BASE_LIMITS = [
  ["Một kịch bản non-IID, một seed", "Chỉ có Dirichlet α=0,1 và seed 42. Chưa đủ để kết luận về độ ổn định hay đường suy giảm theo nhiều mức α (bước 2 cho xem 12 cách chia, nhưng chỉ α=0,1 có mô hình)."],
  ["Chưa phải ảnh ngoài thực địa", "PlantVillage là ảnh chụp trong phòng; PlantDoc là ảnh ngoài đồng nhưng chỉ phủ 13/38 lớp, không có lớp lá khỏe. Cả validation lẫn test giữ riêng đều lấy từ cùng release."],
  ["Chỉ nhận ảnh một lá đã cắt", "Bộ phân loại không tự tìm lá trên ảnh toàn cây; ảnh ngoài miền sẽ cho kết quả kém dù độ tin cậy vẫn cao."],
  ["Hiệu chuẩn chỉ sửa được một phần", "Bước 4 hiện cả xác suất thô và xác suất đã hiệu chuẩn (temperature scaling). T học trên tập calibration chủ yếu là PlantVillage nên trên PlantDoc các mô hình vẫn tự tin quá mức (bước 3, tab Độ tin cậy xác suất)."],
  ["Huấn luyện thử trên web chỉ là kỹ thuật", "Phiên Flower dùng ảnh tổng hợp, hai vòng, trên localhost: chứng minh luồng FedAvg và đo RTT cùng máy, không phải kết quả khoa học hay độ trễ mạng liên vùng."],
  ["Chưa có thí nghiệm client lỗi hay tấn công", "Bộ kiểm tra update (observe/reject, ngưỡng L2 norm) có trong mã và chạy được ở bước 5, nhưng chưa có thí nghiệm độ chính xác cùng protocol nên chưa có số liệu để kết luận."],
  ["Shard là mô phỏng", "Năm client là năm phân vùng của một tập dữ liệu, không phải năm nông hộ vật lý."],
  ["Quyền riêng tư không tuyệt đối", "FL chỉ giữ ảnh thô không rời client; trọng số vẫn có thể rò thông tin. Cần differential privacy hoặc secure aggregation nếu muốn bảo đảm mạnh hơn."],
];

/** Hạn chế về phép đánh giá: số liệu lấy từ báo cáo thống kê đã kiểm danh tính, không viết cứng. */
function evaluationLimits(summary) {
  const st = summary?.statistics;
  const out = [];
  const sel = st?.selection_effect;
  if (sel?.available) {
    const central = (summary?.holdout?.models || []).find(m => m.model_id === "centralized");
    const pdTest = central?.plantdoc?.top1;
    const pdVal = sel.chosen_plantdoc_accuracy;
    const observed = pdTest === undefined || pdTest === null ? ""
      : pdTest >= pdVal
        ? ` Trên test, PlantDoc Centralized đạt ${fmtPct(pdTest)} (val ${fmtPct(pdVal)}): chưa thấy số val bị đẩy lên, nhưng test PlantDoc chỉ ${fmtInt(summary.holdout.counts_by_source?.plantdoc)} ảnh.`
        : ` Trên test, PlantDoc Centralized chỉ đạt ${fmtPct(pdTest)} (val ${fmtPct(pdVal)}) — phù hợp với thiên lệch do chọn.`;
    out.push(["Checkpoint được chọn trên chính tập validation",
      `Centralized lấy epoch ${fmtInt(sel.chosen_epoch)}/${fmtInt(sel.epochs)} vì có PlantDoc ${sel.chosen_rank_among_epochs === 1 ? "tốt nhất" : `hạng ${fmtInt(sel.chosen_rank_among_epochs)}`} trên validation (${fmtPct(pdVal)}; epoch cuối ${fmtPct(sel.last_plantdoc_accuracy)}), nên số validation có thể bị đẩy lên. Tập "legacy diagnostic test" của release (bước 3) không dùng để chọn nên không chịu thiên lệch này.${observed}`]);
  }
  const test = st?.splits?.test?.available ? st.splits.test.by_source : null;
  const pd = test?.plantdoc;
  if (pd?.models?.fedavg?.low !== undefined && pd?.models?.fedavg?.low !== null) {
    const f = pd.models.fedavg;
    out.push(["Khoảng tin cậy PlantDoc rộng",
      `Test PlantDoc chỉ có ${fmtInt(pd.count)} ảnh từ ${fmtInt(pd.clusters)} cảnh chụp: FedAvg ${fmtPct(f.estimate, 1)} với khoảng tin cậy 95% từ ${fmtPct(f.low, 1)} đến ${fmtPct(f.high, 1)}. Chênh lệch vài điểm giữa hai mô hình trên PlantDoc thường chưa có ý nghĩa thống kê.`]);
  }
  const leaf = test?.plantvillage?.same_leaf;
  if (leaf) {
    const c = leaf.models?.centralized || {};
    const f = leaf.models?.fedavg || {};
    const holds = [c, f].every(m => m.without_suspected_reshoots !== null && m.without_suspected_reshoots !== undefined
      && m.all !== null && m.all !== undefined && m.without_suspected_reshoots >= m.all - 0.005);
    out.push(["Một phần PlantVillage chưa gom nhóm theo lá",
      `${fmtInt(leaf.ungrouped_images)} ảnh test PlantVillage không có nhóm lá trong release; ${fmtInt(leaf.suspected_reshoots)} ảnh có ảnh train cùng lớp mang số liền kề (nghi chụp lại cùng lá). Bỏ các ảnh đó: Centralized ${fmtPct(c.all)} → ${fmtPct(c.without_suspected_reshoots)}, FedAvg ${fmtPct(f.all)} → ${fmtPct(f.without_suspected_reshoots)} — ${holds ? "chưa thấy dấu hiệu bị thổi phồng" : "accuracy giảm, số có thể bị thổi phồng một phần"}; muốn loại hẳn rủi ro cần gom lại nhóm lá và dựng release mới.`]);
  }
  out.push(["So sánh Centralized – FedAvg chưa hoàn toàn công bằng",
    "Centralized dùng AdamW 83 epoch; FedAvg dùng SGD không momentum và dừng ở khoảng 32 vòng khi F1 PlantVillage còn đang tăng. Khoảng cách giữa hai mô hình vì vậy là cận trên; muốn kết luận \"FedAvg ≈ Centralized\" cần train lại cùng optimizer, cùng ngân sách, nhiều seed (docs/EXPERIMENT_PROTOCOL.md mục 7)."]);
  return out;
}

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
      ${(item.unlock || []).length ? `<details class="fold tech-only unlock-fold"><summary>Lệnh để mở khóa</summary><div class="unlock">${item.unlock.map(cmd => `<code>${esc(cmd)}</code>`).join("")}</div></details>` : ""}
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
  const ho = summary?.holdout;
  let hoCard = "";
  if (ho?.available) {
    const byId_ = Object.fromEntries((ho.models || []).map(m => [m.model_id, m]));
    const line = id => {
      const m = byId_[id];
      return m ? `<li>${esc(MODEL_LABELS[id])}: PlantVillage <b>${fmtPct(m.plantvillage?.top1)}</b> · PlantDoc <b>${fmtPct(m.plantdoc?.top1)}</b></li>` : "";
    };
    const recorded = byId_.centralized?.vs_recorded || {};
    const reproduced = Object.values(recorded).filter(v => v.available);
    const pair = summary?.statistics?.splits?.test?.available ? summary.statistics.splits.test.by_source.plantdoc?.pairs?.centralized_vs_fedavg : null;
    const d = pair?.difference;
    hoCard = `
      <article class="verify-card ok">
        <h4>Test giữ riêng (legacy diagnostic test, không dùng để chọn checkpoint)</h4>
        <p class="verify-big mono">${fmtInt(ho.count)}</p>
        <p>ảnh (${fmtInt(ho.counts_by_source?.plantvillage)} PlantVillage, ${fmtInt(ho.counts_by_source?.plantdoc)} PlantDoc), cả 7 mô hình được chấm trên máy này.</p>
        <ul>
          ${line("centralized")}${line("fedavg")}
          ${reproduced.length ? `<li>Số Centralized do nhóm huấn luyện tự ghi: ${reproduced.every(v => v.reproduced) ? "tái lập khớp" : "CÓ lệch"} trên ${reproduced.length} nguồn.</li>` : ""}
          ${d && d.low !== null && d.low !== undefined ? `<li>PlantDoc, Centralized − FedAvg: ${fmtNum(d.estimate * 100, 1)} điểm, khoảng tin cậy 95% [${fmtNum(d.low * 100, 1)}; ${fmtNum(d.high * 100, 1)}], McNemar p = ${fmtNum(pair.p_value, 3)}.</li>` : ""}
        </ul>
      </article>`;
  } else if (ho) {
    hoCard = `<article class="verify-card warn"><h4>Test giữ riêng</h4><p>${esc(ho.reason || "Chưa có báo cáo.")}</p></article>`;
  }
  return `<section class="verify-grid mt-4">${reCard}${hoCard}${lkCard}</section>`;
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
        ${[...(summary?.available ? evaluationLimits(summary) : []), ...BASE_LIMITS].map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}
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
