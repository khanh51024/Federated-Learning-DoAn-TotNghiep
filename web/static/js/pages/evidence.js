// Bước 6 · Bằng chứng & hạn chế: những gì đã kiểm, những gì chưa làm được, nói thẳng.
import { getEvidence, getSummary } from "../core/store.js";
import { MODEL_LABELS, byId, esc, fmtInt, fmtNum, fmtPct, modelPill, notify } from "../core/dom.js";

function baseLimits(seed) {
  return [
    ["Mới có một kịch bản và một lần chạy", `Mới thử mức lệch dữ liệu α = 0,1 với seed ${seed ?? "—"}, chưa đủ để nói kết quả ổn định khi điều kiện thay đổi (bước 2 cho xem 12 cách chia, nhưng chỉ α = 0,1 có mô hình).`],
    ["Chưa có kiểm thử thực địa độc lập", "PlantVillage là ảnh chụp trong phòng; PlantDoc là ảnh ngoài đồng nhưng chỉ phủ một phần các nhóm và không có lá khỏe. Cả tập kiểm tra lẫn tập giữ riêng đều lấy từ cùng bộ dữ liệu."],
    ["Chỉ nhận ảnh một chiếc lá đã cắt", "Mô hình không tự tìm lá trong ảnh toàn cây; ảnh khác kiểu sẽ cho kết quả kém dù độ tự tin hiển thị vẫn cao."],
    ["Độ tự tin chỉ được chỉnh một phần", "Web hiện cả xác suất gốc và xác suất đã hiệu chuẩn. Việc hiệu chuẩn học trên ảnh chủ yếu là PlantVillage nên với ảnh ngoài đồng mô hình vẫn tự tin quá mức (xem bước 3, tab \"Độ tin cậy xác suất\")."],
    ["Phần huấn luyện thử chỉ để minh họa", "Phiên Flower dùng ảnh tổng hợp, hai vòng, trên cùng một máy: chứng minh quy trình FedAvg chạy được, không phải kết quả khoa học hay độ trễ mạng thật."],
    ["Chưa thử nghiệm client lỗi hay bị tấn công", "Cơ chế kiểm tra bản cập nhật có sẵn và chạy được ở bước 5, nhưng chưa có thí nghiệm đo độ chính xác nên chưa kết luận được."],
    ["Năm client của phiên Flower là giả lập", "Năm client là năm phần của cùng một bộ dữ liệu, không phải năm nông hộ thật."],
    ["Học liên kết cộng đồng chỉ là thử nghiệm", "Người tham gia chỉ học lớp cuối trên đặc trưng của backbone đóng băng, với ảnh và nhãn tự gán (chưa có chuyên gia kiểm). Máy chủ chấm bằng tập đã dùng để chọn mô hình gốc nên con số chỉ là tín hiệu theo dõi; khi mỗi người chỉ có vài nhóm thì mô hình dễ kém đi và máy chủ sẽ từ chối vòng đó. Mô hình cộng đồng không nằm trong bảng bảy mô hình đã kiểm hash."],
    ["Riêng tư chưa tuyệt đối", "Học liên kết chỉ giữ ảnh gốc không rời client; trọng số vẫn có thể làm lộ thông tin. Muốn bảo đảm mạnh hơn cần thêm kỹ thuật như differential privacy hoặc secure aggregation."],
  ];
}

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
        ? ` Khi chấm lại trên tập giữ riêng, Centralized đạt ${fmtPct(pdTest)} trên ảnh ngoài đồng (tập kiểm tra: ${fmtPct(pdVal)}) — chưa thấy con số bị đẩy lên, nhưng tập giữ riêng chỉ có ${fmtInt(summary.holdout.counts_by_source?.plantdoc)} ảnh ngoài đồng.`
        : ` Khi chấm lại trên tập giữ riêng, Centralized chỉ đạt ${fmtPct(pdTest)} trên ảnh ngoài đồng (tập kiểm tra: ${fmtPct(pdVal)}) — phù hợp với thiên lệch do cách chọn.`;
    out.push(["Mô hình được chọn dựa trên chính tập kiểm tra",
      `Phiên bản Centralized được chọn ở epoch ${fmtInt(sel.chosen_epoch)}/${fmtInt(sel.epochs)} vì có điểm PlantDoc ${sel.chosen_rank_among_epochs === 1 ? "cao nhất" : `hạng ${fmtInt(sel.chosen_rank_among_epochs)}`} trên tập kiểm tra (${fmtPct(pdVal)}; epoch cuối ${fmtPct(sel.last_plantdoc_accuracy)}), nên số ở tập này có thể hơi đẹp hơn thực tế. Tập giữ riêng (bước 3, tập chẩn đoán "legacy" của bộ dữ liệu) không dùng để chọn nên không bị thiên lệch đó.${observed}`]);
  }
  const test = st?.splits?.test?.available ? st.splits.test.by_source : null;
  const pd = test?.plantdoc;
  if (pd?.models?.fedavg?.low !== undefined && pd?.models?.fedavg?.low !== null) {
    const f = pd.models.fedavg;
    out.push(["Ảnh ngoài đồng ít nên con số dao động rộng",
      `Tập giữ riêng chỉ có ${fmtInt(pd.count)} ảnh ngoài đồng từ ${fmtInt(pd.clusters)} cảnh chụp: FedAvg đạt ${fmtPct(f.estimate, 1)}, nhưng con số thật có thể nằm trong khoảng từ ${fmtPct(f.low, 1)} đến ${fmtPct(f.high, 1)} (khoảng tin cậy 95%). Chênh vài điểm giữa hai mô hình ở phần này thường chưa có ý nghĩa thống kê.`]);
  }
  const leaf = test?.plantvillage?.same_leaf;
  if (leaf) {
    const c = leaf.models?.centralized || {};
    const f = leaf.models?.fedavg || {};
    const holds = [c, f].every(m => m.without_suspected_reshoots !== null && m.without_suspected_reshoots !== undefined
      && m.all !== null && m.all !== undefined && m.without_suspected_reshoots >= m.all - 0.005);
    out.push(["Một phần ảnh PlantVillage có thể là ảnh chụp lại cùng một lá",
      `${fmtInt(leaf.ungrouped_images)} ảnh PlantVillage trong tập giữ riêng không được gắn nhóm theo lá; ${fmtInt(leaf.suspected_reshoots)} ảnh trong đó có ảnh học cùng loại với số thứ tự liền kề (nghi là chụp lại cùng lá). Bỏ các ảnh đó: Centralized ${fmtPct(c.all)} → ${fmtPct(c.without_suspected_reshoots)}, FedAvg ${fmtPct(f.all)} → ${fmtPct(f.without_suspected_reshoots)} — ${holds ? "chưa thấy dấu hiệu con số bị thổi phồng" : "độ chính xác giảm, con số có thể bị thổi phồng một phần"}; muốn loại hẳn rủi ro cần gom lại nhóm lá và dựng lại bộ dữ liệu.`]);
  }
  const ff = summary?.fullframe;
  if (ff?.available) {
    const pick = (split, mode, id) => ff.splits?.[split]?.[mode]?.[id];
    const pair = (id, mode) => ["val", "test"].map(split => pick(split, mode, id));
    const ok = ["fedavg", "centralized"].every(id => [...pair(id, "annotator_crop"), ...pair(id, "full_frame")].every(v => v !== null && v !== undefined));
    if (ok) {
      const [fCropV, fCropT] = pair("fedavg", "annotator_crop");
      const [fFullV, fFullT] = pair("fedavg", "full_frame");
      const [cCropV, cCropT] = pair("centralized", "annotator_crop");
      const [cFullV, cFullT] = pair("centralized", "full_frame");
      // đặt đầu danh sách: đây là hạn chế quan trọng nhất của việc dùng ngoài đời, không được rơi vào phần gập
      out.unshift(["Ảnh cả cây hoặc nhiều lá làm mô hình kém đi rõ rệt",
        `Với cùng các ảnh PlantDoc, đưa cả khung ảnh vào thay vì ảnh đã cắt theo khung lá: FedAvg giảm từ ${fmtPct(fCropV, 1)} xuống ${fmtPct(fFullV, 1)} trên tập kiểm tra và từ ${fmtPct(fCropT, 1)} xuống ${fmtPct(fFullT, 1)} trên tập giữ riêng; Centralized từ ${fmtPct(cCropV, 1)} xuống ${fmtPct(cFullV, 1)} và từ ${fmtPct(cCropT, 1)} xuống ${fmtPct(cFullT, 1)}. Phép đo dùng ${fmtInt(ff.splits.val.scenes)} + ${fmtInt(ff.splits.test.scenes)} cảnh (loại ${fmtInt(ff.excluded_scenes)} cảnh không còn ở kho công khai) nên khoảng dao động còn rộng. Web chưa có bước tự tìm và cắt lá: hãy tải ảnh chụp một chiếc lá.`]);
    }
  }
  out.push(["So sánh Centralized và FedAvg chưa hoàn toàn công bằng",
    "Các cách học hiện khác bộ tối ưu và số bước huấn luyện; FedAvg dừng sớm. Vì vậy chưa tách được ảnh hưởng của cách học khỏi ảnh hưởng của thiết lập. Cần huấn luyện lại cùng điều kiện, nhiều seed trước khi kết luận hai cách tương đương (docs/EXPERIMENT_PROTOCOL.md, mục 7)."]);
  return out;
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
  const open = cl.items.filter(i => i.status !== "XONG");
  const done = cl.items.filter(i => i.status === "XONG");
  const cards = open.map(item => {
    const friendly = REMAINING[item.id];
    return `
    <article class="todo-card">
      <header><span class="status-badge ${item.status === "BLOCKED" ? "is-neutral" : "error"}">${item.status === "BLOCKED" ? "Chưa làm được" : "Cần xử lý"}</span><b>${esc(friendly?.title || item.title)}</b></header>
      <p>${esc(item.status === "BLOCKED" && friendly?.why ? friendly.why : item.detail)}</p>
      <details class="fold unlock-fold">
        <summary>Chi tiết kỹ thuật</summary>
        <p class="field-hint mt-2">${esc(item.detail)}</p>
        ${(item.unlock || []).length ? `<div class="unlock">${item.unlock.map(cmd => `<code>${esc(cmd)}</code>`).join("")}</div>` : ""}
      </details>
    </article>`;
  }).join("");
  const onlyBlocked = open.every(i => i.status === "BLOCKED");
  const intro = !open.length
    ? "Các hạng mục trong checklist dự án đã hoàn tất."
    : onlyBlocked
      ? `${cl.done}/${cl.total} hạng mục trong checklist dự án đã hoàn thành; các việc còn lại cần file gốc hoặc GPU.`
      : `Còn ${open.length} hạng mục chưa đạt, trong đó có hạng mục kỹ thuật cần xử lý — xem chi tiết bên dưới.`;
  return `
    <div class="panel-header-flex">
      <div>
        <h3 class="panel-title">Việc còn lại: ${open.length} trên ${cl.total} hạng mục</h3>
        <p class="panel-desc">${esc(intro)}</p>
      </div>
    </div>
    <div class="progress-track mt-3" role="img" aria-label="${cl.done} trên ${cl.total} hạng mục đã xong"><div class="progress-fill fed-fill" style="transform:scaleX(${cl.total ? (cl.done / cl.total).toFixed(3) : 0})"></div></div>
    ${open.length ? `<div class="todo-grid mt-4">${cards}</div>` : ""}
    <details class="fold mt-4">
      <summary>Các hạng mục đã xong (${done.length})</summary>
      <ul class="done-list">${done.map(i => `<li><span class="verdict ok">✓</span><div><b>${esc(i.title)}</b><span>${esc(i.detail)}</span></div></li>`).join("")}</ul>
    </details>`;
}

function identityHtml(identity) {
  if (!identity) return "";
  const rows = identity.models.map(m => `
    <tr>
      <th scope="row">${modelPill(m.model_id, m.name)}</th>
      <td>${esc(m.phase)}</td>
      <td class="code">${esc(m.branch)} @ ${esc(String(m.commit).slice(0, 8))}</td>
      <td class="code" title="${esc(m.checkpoint_sha256)}">${esc(shortHash(m.checkpoint_sha256))} ${copyButton(m.checkpoint_sha256, `SHA-256 của ${m.name}`)}</td>
    </tr>`).join("");
  return `
    <div class="mt-3">
      <p class="panel-desc">Mọi số trên web chỉ hiện khi file mô hình và dữ liệu khớp các mã băm SHA-256 dưới đây; sai một byte là web ẩn toàn bộ số liệu (không có số dự phòng).</p>
      <dl class="fact-grid mt-3">
        <div><dt>Protocol</dt><dd class="code">${esc(identity.protocol_id)}</dd></div>
        <div><dt>Mã băm manifest của release</dt><dd class="code" title="${esc(identity.release_manifest_sha256)}">${esc(shortHash(identity.release_manifest_sha256, 16))} ${copyButton(identity.release_manifest_sha256, "release manifest SHA-256")}</dd></div>
        <div><dt>Mã băm manifest tập kiểm tra</dt><dd class="code" title="${esc(identity.validation_manifest_sha256)}">${esc(shortHash(identity.validation_manifest_sha256, 16))} ${copyButton(identity.validation_manifest_sha256, "validation manifest SHA-256")}</dd></div>
        <div><dt>Seed · tiền xử lý</dt><dd class="code">${esc(identity.seed)} · ${esc(identity.transform)}</dd></div>
      </dl>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">Giai đoạn</th><th scope="col">Nhánh @ commit</th><th scope="col">Checkpoint SHA-256</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
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
        <h4>Chấm lại trên máy này</h4>
        <p class="verify-big">${re.main_models_exact ? "Khớp" : "Có lệch"}</p>
        <p>Centralized và FedAvg ${re.main_models_exact ? "khớp số công bố" : "có lệch so với số công bố"} trên ${fmtInt(re.validation_count)} ảnh kiểm tra.</p>
        <details class="fold mt-3"><summary>Xem phạm vi đối chiếu</summary><p class="field-hint mt-2">${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)} phép so khớp tuyệt đối; chấm bằng CPU.</p><ul>${reLines}</ul></details>
      </article>` : `
      <article class="verify-card warn"><h4>Chấm lại trên máy này</h4><p>${esc(re?.reason || "Chưa có báo cáo.")}</p></article>`;
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
      <article class="verify-card warn"><h4>Kiểm tra ảnh bị trùng</h4><p>${esc(lk?.reason || "Chưa có báo cáo.")}</p></article>`;
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
        <h4>Ảnh ngoài đồng giữ riêng</h4>
        <p class="verify-big mono">${fmtInt(ho.counts_by_source?.plantdoc)}</p>
        <p>ảnh PlantDoc không dùng để chọn mô hình; mẫu nhỏ nên kết quả còn dao động rộng.</p>
        <details class="fold mt-3"><summary>Xem toàn bộ tập giữ riêng</summary><p class="field-hint mt-2">Tổng ${fmtInt(ho.count)} ảnh:</p>
        <p>ảnh (${fmtInt(ho.counts_by_source?.plantvillage)} PlantVillage, ${fmtInt(ho.counts_by_source?.plantdoc)} PlantDoc) không dùng để chọn mô hình; cả 7 mô hình được chấm lại trên máy này. Đây là tập chẩn đoán "legacy" của bộ dữ liệu — chưa phải bài kiểm tra cuối được khóa trước, và chưa nằm trong phép kiểm tra trùng ảnh bên cạnh.</p>
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

export async function mount() {
  const root = byId("evidence-root");
  const [evidence, summary] = await Promise.all([getEvidence().catch(() => null), getSummary().catch(() => null)]);
  const seed = evidence?.seed ?? summary?.identity?.seed;
  const base = baseLimits(seed);
  // thứ tự: phạm vi của kết quả (2 mục đầu) -> các hạn chế của phép đánh giá (theo số liệu) -> còn lại
  const limits = [...base.slice(0, 2), ...(summary?.available ? evaluationLimits(summary) : []), ...base.slice(2)];
  root.innerHTML = `
    ${summary?.available ? `
    <section>
      <h3 class="panel-title">Những gì đã kiểm chứng</h3>
      <p class="panel-desc">Ba phép kiểm, mỗi phép có phạm vi riêng.</p>
      ${verificationHtml(summary)}
      <p class="field-hint mt-3">Xem số liệu gốc ở <a href="#ket-qua">bước 3 · Kết quả &amp; so sánh</a>: tab “Tổng hợp” (bảy mô hình), “Tập giữ riêng &amp; thống kê” (khoảng tin cậy), “Độ tin cậy xác suất” (hiệu chuẩn). Mã băm của từng mô hình nằm trong mục gập ở cuối trang này.</p>
    </section>` : `
      <div class="callout callout-error" role="alert"><b>Chưa có bộ số liệu đã kiểm chứng.</b> ${esc(summary?.reason || evidence?.reason || "Không đọc được /api/evidence/summary.")}</div>`}

    <section class="panel mt-4">
      <h3 class="panel-title">Ba giới hạn chính</h3>
      <div class="viewer-limits mt-3">
        <article><h4>Chưa chứng minh độ ổn định</h4><p>Một kịch bản, một seed; các client là phân vùng giả lập.</p></article>
        <article><h4>Chưa kiểm thử thực địa độc lập</h4><p>Ảnh ngoài đồng hiện có chỉ phủ một phần nhóm lá; kết quả chưa đại diện cho triển khai thực tế.</p></article>
        <article><h4>Cần ảnh một lá đã cắt</h4><p>Web chưa tự tìm lá trên ảnh toàn cây. Độ tự tin cao vẫn có thể sai.</p></article>
      </div>
      <p class="viewer-scope mt-3">Tập kiểm tra cũng dùng để chọn mô hình; tập giữ riêng chưa được đối soát trùng ảnh. Điều kiện huấn luyện giữa các cách học còn khác nhau.</p>
      <details class="fold mt-3"><summary>Xem đầy đủ ${limits.length} hạn chế và số liệu liên quan</summary>
        <ol class="limit-list mt-3">${limits.map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}</ol>
      </details>
    </section>

    <section class="panel mt-4">${checklistHtml(summary?.checklist, seed)}</section>

    ${summary?.available ? `
    <details class="panel panel-fold mt-4" id="hash-details">
      <summary>Mã băm SHA-256 của mô hình và dữ liệu (để đối chiếu)</summary>
      ${identityHtml(summary.identity)}
    </details>` : ""}`;

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
