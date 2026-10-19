// Phụ lục kỹ thuật: kiểm chứng số liệu, nguồn gốc mô hình, hạng mục kỹ thuật; phần kiểm thử hệ thống trên máy chủ nằm trong
// index.html và do training.js điều khiển. Dành cho người chấm và người vận hành, không phải nội dung báo cáo.
import { getEvidence, getSummary } from "../core/store.js";
import { MODEL_LABELS, byId, esc, fmtInt, fmtNum, fmtPct, modelPill, notify, prefersReducedMotion } from "../core/dom.js";
import { mount as mountTraining } from "./training.js";

// Mã nội bộ của API → câu tiếng Việt (giữ mã làm chú thích); mã lạ thì hiện nguyên văn.
const SOURCE_TEXT = {
  "published validation_comparison.csv; no local image release":
    "bảng validation_comparison.csv đã công bố; chưa có ảnh gốc trên máy chủ để chấm lại trực tiếp",
};
const EVIDENCE_LEVEL_TEXT = {
  published_artifact_hash_verified_not_locally_reevaluated: "tệp công bố đã kiểm mã băm, chưa chấm lại cục bộ",
  unavailable: "chưa có",
};

function reproductionCell(model) {
  const sources = model?.vs_published;
  if (!sources) return '<span class="repro repro-none" title="Chưa có báo cáo chấm lại">—</span>';
  const statuses = ["plantdoc", "plantvillage"].map(s => sources[s]?.status || "unavailable");
  if (statuses.every(s => s === "exact")) {
    return '<span class="repro repro-exact" title="Kết quả chấm lại trên máy này trùng tuyệt đối với số công bố">✓ khớp tuyệt đối</span>';
  }
  if (statuses.every(s => s === "exact" || s === "within_tolerance")) {
    const diffs = ["plantdoc", "plantvillage"].map(s => {
      const d = sources[s]?.correct_diff_images;
      return `${s === "plantdoc" ? "ảnh ngoài đồng" : "ảnh trong phòng"}: ${d === null || d === undefined ? "?" : (d > 0 ? "+" : "") + d} ảnh`;
    }).join(" · ");
    return `<span class="repro repro-approx" title="Lệch ${esc(diffs)} so với số công bố (≤ 1 ảnh theo quy tắc của báo cáo)">≈ lệch ≤ 1 ảnh</span>`;
  }
  return '<span class="repro repro-diff" title="Không khớp số công bố trong dung sai">✕ lệch</span>';
}

/** Bốn ô tóm tắt mức kiểm chứng (chuyển từ trang Tổng quan cũ). */
function trustChips(summary) {
  const chips = [];
  const prov = (summary?.checklist?.items || []).find(i => i.id === "provenance");
  const provOpen = prov && prov.status !== "XONG";
  chips.push(`<div class="trust-chip ok"><b>Mô hình đúng bản công bố</b><span>mã băm SHA-256 của mô hình và dữ liệu đã được đối chiếu${provOpen ? " · nguồn huấn luyện gốc chưa kiểm" : ""}</span></div>`);
  const re = summary?.reeval;
  if (re?.available) {
    const main = re.main_models_exact ? "Centralized và FedAvg khớp tuyệt đối" : "có lệch ở mô hình chính";
    chips.push(`<div class="trust-chip ${re.all_within_tolerance ? "ok" : "warn"}"><b>Đã chấm lại ${fmtInt(re.validation_count)} ảnh trên máy này</b><span>${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)} số liệu khớp tuyệt đối · ${main}</span></div>`);
  } else {
    chips.push(`<div class="trust-chip warn"><b>Chưa chấm lại trên máy này</b><span>${esc(re?.reason || "thiếu báo cáo")}</span></div>`);
  }
  const lk = summary?.leakage;
  if (lk?.available) {
    chips.push(`<div class="trust-chip ${lk.passed ? "ok" : "bad"}"><b>Kiểm tra trùng ảnh: ${lk.passed ? "đạt" : "KHÔNG đạt"}</b><span>${fmtInt(lk.overlap_total)} ảnh trùng giữa tập học và tập kiểm tra · đã kiểm ${fmtInt(lk.images_verified)} ảnh</span></div>`);
  } else {
    chips.push(`<div class="trust-chip warn"><b>Chưa kiểm tra trùng ảnh</b><span>${esc(lk?.reason || "thiếu báo cáo")}</span></div>`);
  }
  const ho = summary?.holdout;
  if (ho?.available) {
    const fed = ho.models.find(x => x.model_id === "fedavg");
    chips.push(`<a class="trust-chip warn" href="#ket-qua"><b>Kiểm tra lại trên ${fmtInt(ho.count)} ảnh giữ riêng</b><span>bộ chẩn đoán có sẵn của bộ dữ liệu (legacy), không dùng để chọn mô hình nhưng chưa nằm trong kiểm tra trùng ảnh · FedAvg đạt ${fmtPct(fed?.plantvillage?.top1)} (ảnh trong phòng) và ${fmtPct(fed?.plantdoc?.top1)} (ảnh ngoài đồng)</span></a>`);
  }
  return chips.join("");
}

const KEY_LABELS = {
  byte_sha256: "mã băm ảnh", group_id: "nhóm lá", image_id: "mã ảnh", parent_scene_id: "cảnh chụp", sample_id: "mã mẫu",
};

function shortHash(value, head = 12) {
  return value ? `${value.slice(0, head)}…${value.slice(-6)}` : "—";
}

function copyButton(value, what) {
  return `<button type="button" class="copy-btn" data-copy="${esc(value)}" data-what="${esc(what)}" aria-label="Sao chép ${esc(what)}">sao chép</button>`;
}

// Các hạng mục còn dở, nói bằng lời thường (mã lỗi và lệnh nằm trong phần gập "chi tiết kỹ thuật").
function remainingCopy(seed) {
  return {
  provenance: {
    title: "Kiểm chứng nguồn gốc file mô hình gốc",
    why: "Cần file mô hình gốc trên Kaggle để kiểm chứng nguồn huấn luyện.",
  },
  multiseed: {
    title: "Chạy lặp lại nhiều lần (3 seed)",
    why: `Mới có seed ${seed ?? "—"}; cần chạy lại nhiều seed trên GPU để đánh giá độ ổn định.`,
  },
  reports_synced: {
    title: "Cập nhật báo cáo sau lần chạy mới",
    why: "Cập nhật sau khi có kết quả chạy lặp lại.",
  },
  };
}

function checklistHtml(cl, seed) {
  const REMAINING = remainingCopy(seed);
  if (!cl?.available) {
    return `<div class="callout callout-warn" role="note"><b>Chưa có báo cáo trạng thái.</b> ${esc(cl?.reason || "Chạy scripts/update_status.py.")}</div>`;
  }
  // Hai loại việc khác nhau: BLOCKED = việc nghiên cứu cần file gốc hoặc GPU; còn lại chưa XONG = hạng mục kỹ thuật cần xử lý.
  const research = cl.items.filter(i => i.status === "BLOCKED");
  const technicalOpen = cl.items.filter(i => i.status !== "XONG" && i.status !== "BLOCKED");
  const done = cl.items.filter(i => i.status === "XONG");
  const technicalTotal = done.length + technicalOpen.length;
  const card = item => {
    const friendly = REMAINING[item.id];
    const blocked = item.status === "BLOCKED";
    return `
    <article class="todo-card">
      <header><span class="status-badge ${blocked ? "is-neutral" : "error"}">${blocked ? "Chưa làm được" : "Cần xử lý"}</span><b>${esc(friendly?.title || item.title)}</b></header>
      <p>${esc(blocked && friendly?.why ? friendly.why : item.detail)}</p>
      <details class="fold unlock-fold">
        <summary>Dành cho người vận hành (mã lỗi và lệnh)</summary>
        <p class="field-hint mt-2">${esc(item.detail)}</p>
        ${(item.unlock || []).length ? `<div class="unlock">${item.unlock.map(cmd => `<code>${esc(cmd)}</code>`).join("")}</div>` : ""}
      </details>
    </article>`;
  };
  const intro = !research.length && !technicalOpen.length
    ? "Mọi hạng mục trong danh mục kiểm tra dự án đã hoàn tất."
    : `${research.length} việc nghiên cứu còn lại cần file mô hình gốc hoặc GPU` +
      (technicalOpen.length ? `; ${technicalOpen.length} hạng mục kỹ thuật cần xử lý.` : ".");
  return `
    <div class="panel-header-flex">
      <div>
        <h3 class="panel-title">Việc nghiên cứu còn lại: ${research.length}</h3>
        <p class="panel-desc">${esc(intro)}</p>
      </div>
    </div>
    ${research.length ? `<div class="todo-grid mt-4">${research.map(card).join("")}</div>` : ""}
    ${technicalOpen.length ? `
    <h3 class="panel-title mt-4">Hạng mục kỹ thuật cần xử lý: ${technicalOpen.length}</h3>
    <div class="todo-grid mt-3">${technicalOpen.map(card).join("")}</div>` : ""}
    <h3 class="panel-title mt-4">Hạng mục kỹ thuật đã xong: ${done.length}/${technicalTotal}</h3>
    <div class="progress-track mt-3" role="img" aria-label="${done.length} trên ${technicalTotal} hạng mục kỹ thuật đã xong"><div class="progress-fill fed-fill" data-style="transform:scaleX(${technicalTotal ? (done.length / technicalTotal).toFixed(3) : 0})"></div></div>
    <details class="fold mt-4">
      <summary>Các hạng mục kỹ thuật đã xong (${done.length})</summary>
      <ul class="done-list">${done.map(i => `<li><span class="verdict ok">✓</span><div><b>${esc(i.title)}</b><span>${esc(i.detail)}</span></div></li>`).join("")}</ul>
    </details>`;
}

function identityHtml(identity, evidence, summary) {
  if (!identity) return "";
  const reeval = summary?.reeval?.available ? Object.fromEntries(summary.reeval.models.map(m => [m.model_id, m])) : {};
  const rows = identity.models.map(m => `
    <tr>
      <th scope="row">${modelPill(m.model_id, m.name)}</th>
      <td>${esc(m.phase)}</td>
      <td class="code">${esc(m.branch)} @ ${esc(String(m.commit).slice(0, 8))}</td>
      <td class="code" title="${esc(m.checkpoint_sha256)}">${esc(shortHash(m.checkpoint_sha256))} ${copyButton(m.checkpoint_sha256, `SHA-256 của ${m.name}`)}</td>
      <td>${reproductionCell(reeval[m.model_id])}</td>
    </tr>`).join("");
  const re = summary?.reeval;
  let reNote;
  if (re?.available) {
    const nonExact = (re.models || [])
      .filter(m => Object.values(m.vs_published || {}).some(v => v.status !== "exact"))
      .map(m => m.name);
    const main = re.main_models_exact ? "Centralized và FedAvg khớp số công bố tuyệt đối" : "Centralized/FedAvg CÓ lệch so với số công bố";
    const rest = nonExact.length
      ? `; lệch: ${nonExact.map(esc).join(", ")} (${re.all_within_tolerance ? "trong dung sai, ≤ 1 ảnh đúng; nguyên nhân chưa chứng minh, có ảnh sát ngưỡng quyết định" : "VƯỢT dung sai của báo cáo"})`
      : "";
    reNote = `Đã chấm lại trên máy này (${fmtInt(re.validation_count)} ảnh kiểm tra): ${main}; ${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)} số liệu khớp tuyệt đối${rest}.`;
  } else {
    reNote = `Số liệu là số công bố đã kiểm mã băm SHA-256. ${esc(re?.reason || "Chưa có báo cáo chấm lại cục bộ.")}`;
  }
  return `
    <div class="mt-3">
      <p class="panel-desc">Mọi số trên web chỉ hiện khi file mô hình và dữ liệu khớp các mã băm SHA-256 dưới đây; sai một byte là web ẩn toàn bộ số liệu (không có số dự phòng).</p>
      <dl class="fact-grid mt-3">
        <div><dt>Nguồn số</dt><dd>${esc(SOURCE_TEXT[evidence?.metrics_source] || evidence?.metrics_source || "—")}</dd></div>
        <div><dt>Mức bằng chứng</dt><dd>${esc(EVIDENCE_LEVEL_TEXT[evidence?.evidence_level] || evidence?.evidence_level || "—")} <span class="code">(${esc(evidence?.evidence_level)})</span></dd></div>
        <div><dt>Protocol</dt><dd class="code">${esc(identity.protocol_id)}</dd></div>
        <div><dt>Mã băm manifest của release</dt><dd class="code" title="${esc(identity.release_manifest_sha256)}">${esc(shortHash(identity.release_manifest_sha256, 16))} ${copyButton(identity.release_manifest_sha256, "release manifest SHA-256")}</dd></div>
        <div><dt>Mã băm manifest tập kiểm tra</dt><dd class="code" title="${esc(identity.validation_manifest_sha256)}">${esc(shortHash(identity.validation_manifest_sha256, 16))} ${copyButton(identity.validation_manifest_sha256, "validation manifest SHA-256")}</dd></div>
        <div><dt>Seed · tiền xử lý</dt><dd class="code">${esc(identity.seed)} · ${esc(identity.transform)}</dd></div>
      </dl>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">Giai đoạn</th><th scope="col">Nhánh @ commit</th><th scope="col">Checkpoint SHA-256</th><th scope="col">Chấm lại</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="field-hint mt-3">${reNote}</p>
    </div>`;
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
      `Centralized và FedAvg: ${re.main_models_exact ? "khớp tuyệt đối trên cả hai loại ảnh" : "CÓ lệch so với số công bố"}.`,
      nonExact.length
        ? `${re.all_within_tolerance ? "Lệch rất nhỏ (không quá 1 ảnh)" : "Lệch VƯỢT dung sai"}: ${nonExact.map(esc).join(", ")}; chưa rõ nguyên nhân (có ảnh nằm sát ranh giới quyết định).`
        : "Không có phép so khớp nào lệch.",
    ].map(line => `<li>${line}</li>`).join("")
      + `<li>Dung sai của báo cáo: ${fmtInt(re.tolerance_rule?.correct_images)} ảnh đúng, ${fmtInt(re.tolerance_rule?.high_confidence_wrong_images)} ảnh sai-tự-tin, ${fmtNum(re.tolerance_rule?.metric, 3)} metric.</li>`;
  }
  const reCard = re?.available ? `
      <article class="verify-card ${re.main_models_exact ? "ok" : "warn"}">
        <h4>Đối chiếu kết quả suy luận</h4>
        <p class="verify-big">${re.main_models_exact ? "Khớp" : "Có lệch"}</p>
        <p>Centralized và FedAvg ${re.main_models_exact ? "khớp số công bố" : "có lệch so với số công bố"} trên ${fmtInt(re.validation_count)} ảnh kiểm tra.</p>
        <details class="fold mt-3"><summary>Xem phạm vi đối chiếu</summary><p class="field-hint mt-2">${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)} phép so khớp tuyệt đối; chấm bằng CPU.</p><ul>${reLines}</ul></details>
      </article>` : `
      <article class="verify-card warn"><h4>Đối chiếu kết quả suy luận</h4><p>${esc(re?.reason || "Chưa có báo cáo.")}</p></article>`;
  const lkCard = lk?.available ? `
      <article class="verify-card ${lk.passed ? "ok" : "bad"}">
        <h4>Kiểm tra trùng ảnh</h4>
        <p class="verify-big mono">${fmtInt(lk.overlap_total)}</p>
        <p>phần tử trùng trong tập học, kiểm tra và hiệu chuẩn. Tập giữ riêng chưa nằm trong phép kiểm này.</p>
        <details class="fold mt-3"><summary>Xem cách kiểm tra</summary>
        <p class="field-hint mt-2">Đối chiếu theo ${lk.keys.length} tiêu chí: ${esc(lk.keys.map(k => KEY_LABELS[k] || k).join(", "))}.</p>
        <ul>
          <li>${fmtInt(lk.images_verified)} ảnh được băm lại từng ảnh một; ${fmtInt(lk.hash_mismatch)} ảnh sai mã băm, ${fmtInt(lk.missing_files)} ảnh thiếu file.</li>
          <li>${lk.partitions_clean}/${lk.partitions_total} cách chia cho client đều là tập con của tập học, không nhóm ảnh nào bị chia cho nhiều client.</li>
          <li>Ảnh học ${fmtInt(lk.counts.train)} · ảnh kiểm tra ${fmtInt(lk.counts.val)} · ảnh hiệu chuẩn ${fmtInt(lk.counts.calibration)}.</li>
        </ul>
        <p class="field-hint">Phạm vi: kiểm tra việc chia tập của bộ dữ liệu (tập học, tập kiểm tra, tập hiệu chuẩn). Việc từng mô hình đã học đúng tập này dựa vào mã băm ghi trong file mô hình, không phải chạy lại huấn luyện. Tập giữ riêng chưa nằm trong phép kiểm tra này.</p>
        <p class="field-hint">${esc(lk.scope || "")}</p>
        </details>
      </article>` : `
      <article class="verify-card warn"><h4>Kiểm tra trùng ảnh</h4><p>${esc(lk?.reason || "Chưa có báo cáo.")}</p></article>`;
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
        <h4>Đánh giá trên tập PlantDoc giữ riêng</h4>
        <p class="verify-big mono">${fmtInt(ho.counts_by_source?.plantdoc)}</p>
        <p>ảnh PlantDoc không dùng để chọn mô hình; mẫu nhỏ nên kết quả còn dao động rộng.</p>
        <details class="fold mt-3"><summary>Xem toàn bộ tập giữ riêng</summary>
        <p class="field-hint mt-2">Tổng ${fmtInt(ho.count)} ảnh (${fmtInt(ho.counts_by_source?.plantvillage)} PlantVillage, ${fmtInt(ho.counts_by_source?.plantdoc)} PlantDoc), không dùng để chọn mô hình; cả 7 mô hình được chấm lại trên máy này. Đây là bộ chẩn đoán có sẵn của bộ dữ liệu (legacy), chưa phải bài kiểm tra cuối được khóa trước, và chưa nằm trong phép kiểm tra trùng ảnh bên cạnh.</p>
        <ul>
          ${line("centralized")}${line("fedavg")}
          ${reproduced.length ? `<li>Số Centralized do nhóm huấn luyện tự ghi: ${reproduced.every(v => v.reproduced) ? "tái lập khớp" : "CÓ lệch"} trên ${reproduced.length} nguồn.</li>` : ""}
          ${d && d.low !== null && d.low !== undefined ? `<li>PlantDoc, Centralized − FedAvg: ${fmtNum(d.estimate * 100, 1)} điểm, khoảng tin cậy 95% [${fmtNum(d.low * 100, 1)}; ${fmtNum(d.high * 100, 1)}], McNemar p = ${fmtNum(pair.p_value, 3)}.</li>` : ""}
        </ul>
        </details>
      </article>`;
  } else if (ho) {
    hoCard = `<article class="verify-card warn"><h4>Kiểm tra trên tập giữ riêng</h4><p>${esc(ho.reason || "Chưa có báo cáo.")}</p></article>`;
  }
  return `<section class="verify-grid mt-4">${reCard}${hoCard}${lkCard}</section>`;
}

function section(id, letter, title, intro, body) {
  return `
    <section class="panel appendix-section mt-4" id="${id}" aria-labelledby="${id}-title">
      <p class="eyebrow">${letter}</p>
      <h3 class="panel-title" id="${id}-title">${title}</h3>
      <p class="panel-desc">${intro}</p>
      ${body}
    </section>`;
}

export async function mount() {
  const root = byId("appendix-root");
  const [evidence, summary] = await Promise.all([getEvidence().catch(() => null), getSummary().catch(() => null)]);
  const seed = evidence?.seed ?? summary?.identity?.seed;
  const missing = `<div class="callout callout-error mt-3" role="alert"><b>Chưa có bộ số liệu đã kiểm chứng.</b> ${esc(summary?.reason || evidence?.reason || "Không đọc được /api/evidence/summary.")}</div>`;
  root.innerHTML = `
    <p class="panel-desc">Dành cho người chấm và người vận hành. Nội dung báo cáo nằm ở chương 1 đến 5.</p>
    <nav class="chip-row appendix-toc mt-3" aria-label="Mục lục phụ lục">
      <button type="button" class="chip" data-scroll="appendix-verify">A · Kiểm chứng số liệu</button>
      <button type="button" class="chip" data-scroll="appendix-origin">B · Nguồn gốc mô hình</button>
      <button type="button" class="chip" data-scroll="appendix-todo">C · Hạng mục kỹ thuật</button>
      <button type="button" class="chip" data-scroll="appendix-system">D · Kiểm thử hệ thống trên máy chủ</button>
      <button type="button" class="chip" data-scroll="appendix-weights">E · Trọng số và cách chạy</button>
    </nav>
    ${section("appendix-verify", "A", "Kiểm chứng số liệu", "Ba phương pháp kiểm tra độc lập với nhau, mỗi phương pháp có phạm vi riêng.",
      summary?.available ? `<div class="trust-grid mt-3">${trustChips(summary)}</div>${verificationHtml(summary)}` : missing)}
    ${section("appendix-origin", "B", "Nguồn gốc mô hình", "Mã băm SHA-256 của mô hình và dữ liệu để đối chiếu với bản công bố.",
      summary?.available ? identityHtml(summary.identity, evidence, summary) : missing)}
    ${section("appendix-todo", "C", "Hạng mục kỹ thuật", "Việc nghiên cứu còn dở và các hạng mục kỹ thuật của dự án, kèm lệnh cho người vận hành.",
      checklistHtml(summary?.checklist, seed))}`;

  root.addEventListener("click", async event => {
    const jump = event.target.closest("[data-scroll]");
    if (jump) {
      const target = byId(jump.dataset.scroll);
      if (!target) return;
      target.setAttribute("tabindex", "-1"); // bàn phím: đưa focus tới mục vừa cuộn tới
      target.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
      target.focus({ preventScroll: true });
      return;
    }
    const button = event.target.closest(".copy-btn");
    if (!button) return;
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      notify("success", "Đã sao chép", button.dataset.what);
    } catch (_error) {
      notify("warning", "Không sao chép được", "Trình duyệt chặn clipboard; hãy bôi đen và sao chép thủ công.");
    }
  });

  // Phần D và E là markup tĩnh trong index.html; lỗi ở đó không được làm hỏng ba phần kiểm chứng ở trên.
  try {
    await mountTraining();
  } catch (error) {
    notify("error", "Không khởi tạo được phần kiểm thử trên máy chủ", error.message);
  }
}
