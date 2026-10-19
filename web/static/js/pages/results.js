// Bước 3 · Kết quả chi tiết: tổng hợp 7 mô hình, PlantDoc theo lớp, độ tin cậy của xác suất.
import { getDetails, getEvidence, getSummary, modelsById } from "../core/store.js";
import { attachCrosshair } from "../core/crosshair.js";
import { lineDraw, makeChartWhenVisible, pillarColors } from "../core/charts.js";
import { play } from "../core/motion.js";
import {
  applyStyleData, openOnWide, $$, MODEL_LABELS, MODEL_ORDER, byId, diseaseOf, cropOf, esc, fmtInt, fmtNum, fmtPct, fmtPctValue,
  formatDiseaseName, modelPill, notify, pillarOf, withAlpha,
} from "../core/dom.js";

const built = new Set();
let ctx = null; // { evidence, summary }

// Mã nội bộ của API → câu tiếng Việt (giữ mã làm chú thích); mã lạ thì hiện nguyên văn.
const SOURCE_TEXT = {
  "published validation_comparison.csv; no local image release":
    "bảng validation_comparison.csv đã công bố; chưa có ảnh gốc trên máy chủ để chấm lại trực tiếp",
};
const EVIDENCE_LEVEL_TEXT = {
  published_artifact_hash_verified_not_locally_reevaluated: "tệp công bố đã kiểm mã băm, chưa chấm lại cục bộ",
  unavailable: "chưa có",
};

function notReady(reason) {
  return `<div class="callout callout-warn" role="note"><b>Phần này chưa có dữ liệu.</b> ${esc(reason || "")}</div>`;
}

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

// ───────── Tab 1: Tổng hợp ─────────
function buildSummary() {
  const { evidence, summary } = ctx;
  const root = byId("result-summary");
  const byModel = modelsById(evidence);
  const reeval = summary?.reeval?.available ? Object.fromEntries(summary.reeval.models.map(m => [m.model_id, m])) : {};
  const best = { pvTop1: 0, pdTop1: 0 };
  for (const id of MODEL_ORDER) {
    best.pvTop1 = Math.max(best.pvTop1, byModel[id].metrics.plantvillage.top1_accuracy);
    best.pdTop1 = Math.max(best.pdTop1, byModel[id].metrics.plantdoc.top1_accuracy);
  }
  const rows = MODEL_ORDER.map(id => {
    const mm = byModel[id].metrics;
    const bold = (value, top) => (value === top ? ' class="is-best"' : "");
    return `<tr class="row-${pillarOf(id)}">
      <th scope="row">${modelPill(id)}</th>
      <td>${esc(byModel[id].phase)}</td>
      <td${bold(mm.plantvillage.top1_accuracy, best.pvTop1)}>${fmtPct(mm.plantvillage.top1_accuracy)}</td>
      <td>${fmtNum(mm.plantvillage.macro_f1_supported, 4)}</td>
      <td${bold(mm.plantdoc.top1_accuracy, best.pdTop1)}>${fmtPct(mm.plantdoc.top1_accuracy)}</td>
      <td>${fmtNum(mm.plantdoc.macro_f1_supported, 4)}</td>
      <td>${fmtPct(mm.plantdoc.crop_accuracy, 1)}</td>
      <td>${reproductionCell(reeval[id])}</td>
    </tr>`;
  }).join("");

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

  root.innerHTML = `
    <section class="panel">
      <div class="panel-header-flex"><div><h3 class="panel-title">So sánh ba phương pháp huấn luyện</h3><p class="panel-desc">1 mô hình gom ảnh · 1 mô hình học liên kết · 5 mô hình học riêng</p></div>
        <div class="segmented" role="radiogroup" aria-label="Nguồn ảnh so sánh">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-result-source="plantvillage">PlantVillage</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-result-source="plantdoc">PlantDoc · ngoài đồng</button>
        </div>
      </div>
      <p class="field-hint mt-2" id="result-comparison-source"></p>
      <details class="fold mt-2" id="result-provenance">
        <summary>Nguồn dữ liệu và phương pháp đánh giá</summary>
        <div class="callout mt-2">
          <ul>
            <li><b>Tập đánh giá:</b> tập kiểm tra (validation) của release đã ghim — PlantVillage ${fmtInt(evidence.validation_counts.plantvillage)} ảnh, PlantDoc ${fmtInt(evidence.validation_counts.plantdoc)} ảnh. Tập này cũng được dùng để chọn mô hình nên không thay cho kiểm thử độc lập.</li>
            <li><b>Giao thức:</b> <span class="code">${esc(evidence.protocol_id)}</span> · seed ${esc(evidence.seed)} · tiền xử lý <span class="code">${esc(evidence.transform)}</span>.</li>
            <li><b>Nguồn số:</b> ${esc(SOURCE_TEXT[evidence.metrics_source] || evidence.metrics_source)}. <b>Mức bằng chứng:</b> ${esc(EVIDENCE_LEVEL_TEXT[evidence.evidence_level] || evidence.evidence_level)} <span class="code">(${esc(evidence.evidence_level)})</span>.</li>
            <li><b>Checkpoint:</b> Centralized v6 <span class="code">${esc(String(byModel.centralized.checkpoint_sha256).slice(0, 12))}…</span> · FedAvg v7 <span class="code">${esc(String(byModel.fedavg.checkpoint_sha256).slice(0, 12))}…</span> (SHA-256 đã kiểm trước khi nạp).</li>
            <li><b>Cách đọc:</b> Top-1 là tỉ lệ ảnh mà lựa chọn đầu tiên của mô hình đúng nhóm bệnh; chưa phải độ chính xác ngoài thực địa.</li>
          </ul>
        </div>
      </details>
      <div class="viewer-comparison mt-3" id="result-key-comparison"></div>
      <p class="viewer-takeaway mt-3" id="result-takeaway" aria-live="polite"></p>
      <p class="viewer-scope mt-2"><b class="scope-tag">Phạm vi</b> Một kịch bản, một lần chạy; tập này cũng dùng để chọn mô hình, nên chưa phải độ chính xác ngoài thực địa. <a href="#bang-chung">Xem giới hạn ở bước 6</a></p>
    </section>

    <details class="overview-detail mt-4" id="result-full-table">
      <summary><span><strong>Bảng đầy đủ của 7 mô hình</strong><small>Độ chính xác, Macro-F1 và kết quả chấm lại</small></span></summary>
      <div class="overview-detail-body">
      <p class="panel-desc">PlantVillage ${fmtInt(evidence.validation_counts.plantvillage)} ảnh (cùng kiểu với ảnh học) và PlantDoc ${fmtInt(evidence.validation_counts.plantdoc)} ảnh (ảnh ngoài đồng). Số đậm là cao nhất mỗi cột.</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table result-table">
          <thead>
            <tr>
              <th scope="col">Mô hình</th><th scope="col">Giai đoạn</th>
              <th scope="col">Ảnh phòng (PlantVillage): đúng</th><th scope="col">Macro-F1</th>
              <th scope="col">Ảnh ngoài đồng (PlantDoc): đúng</th><th scope="col">Macro-F1</th>
              <th scope="col">Đúng loại cây (ngoài đồng)</th><th scope="col">Chấm lại</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="field-hint mt-3">${reNote}</p>
      <p class="field-hint mt-2"><b>Cách đọc:</b> "đúng" là tỉ lệ ảnh mà mô hình đoán đúng bệnh (Top-1). Macro-F1 là điểm trung bình công bằng giữa các nhóm (không bị nhóm nhiều ảnh lấn át), tính trên các nhóm có ảnh kiểm tra — PlantDoc chỉ phủ 13/38 loại. Mới có một kịch bản non-IID (α=0,1) và một lần chạy, nên chưa nói được độ ổn định.</p>
      </div>
    </details>

    <p class="field-hint mt-3">Mở tab “Tập giữ riêng và thống kê” để xem kết quả trên ảnh không dùng để chọn mô hình và khoảng tin cậy.</p>

    <details class="overview-detail mt-4">
      <summary><span><strong>Xem lịch sử huấn luyện Centralized</strong><small>Đường độ chính xác và sai số theo epoch</small></span></summary>
      <div class="overview-detail-body">
      <h3 class="panel-title">Quá trình huấn luyện Centralized v6</h3>
      <p class="panel-desc">${fmtInt(evidence.centralized_history.length)} epoch trong dữ liệu công bố. Lịch sử từng vòng của FedAvg chưa đủ trong bộ dữ liệu giao diện để vẽ đường so sánh tương ứng.</p>
      <div class="chart-container mt-4"><canvas id="chart-central-history" aria-label="Độ chính xác và sai số trên tập kiểm tra của Centralized qua các epoch"></canvas></div>
      </div>
    </details>`;

  const showSource = source => {
    const locals = MODEL_ORDER.filter(id => id.startsWith("local_"));
    const values = locals.map(id => byModel[id].metrics[source].top1_accuracy);
    const fed = byModel.fedavg.metrics[source].top1_accuracy;
    const central = byModel.centralized.metrics[source].top1_accuracy;
    const gain = (fed - Math.max(...values)) * 100;
    const name = source === "plantvillage" ? "PlantVillage" : "PlantDoc";
    byId("result-comparison-source").textContent = `${fmtInt(evidence.validation_counts[source])} ảnh ${name} trong tập kiểm tra · tỉ lệ dự đoán đúng (Top-1) · seed ${evidence.seed}`;
    byId("result-key-comparison").innerHTML = [
      ["local", "Local-only", `${fmtPct(Math.min(...values), 1)}–${fmtPct(Math.max(...values), 1)}`, "Mỗi client học riêng"],
      ["fedavg", "FedAvg", fmtPct(fed), "Học chung, giữ ảnh tại client"],
      ["centralized", "Centralized", fmtPct(central), "Gom ảnh về máy chủ"],
    ].map(([kind, label, value, desc]) => `<article class="viewer-model ${kind}"><span>${label}</span><b>${value}</b><small>${desc}</small></article>`).join("");
    byId("result-takeaway").textContent = `Trong lần chạy này, FedAvg ${gain >= 0 ? "cao hơn" : "thấp hơn"} mô hình học riêng tốt nhất ${fmtNum(Math.abs(gain), 1)} điểm phần trăm trên ${name}.`;
    $$("[data-result-source]", root).forEach(button => {
      const on = button.dataset.resultSource === source;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-checked", String(on));
      button.setAttribute("tabindex", on ? "0" : "-1");
    });
  };
  $$("[data-result-source]", root).forEach(button => button.addEventListener("click", () => showSource(button.dataset.resultSource)));
  showSource("plantvillage");

  const history = evidence.centralized_history || [];
  makeChartWhenVisible(byId("chart-central-history"), () => {
    const colors = pillarColors();
    return {
    type: "line",
    data: {
      labels: history.map(h => h.epoch),
      datasets: [
        { label: "Độ chính xác trên tập kiểm tra (%)", data: history.map(h => +(h.val_accuracy * 100).toFixed(2)), borderColor: colors.centralized, backgroundColor: withAlpha(colors.centralized, 0.12), fill: true, tension: 0.25, pointRadius: 0, yAxisID: "y" },
        { label: "Sai số (loss) trên tập kiểm tra", data: history.map(h => +h.val_loss.toFixed(4)), borderColor: colors.danger, tension: 0.25, pointRadius: 0, yAxisID: "y1" },
      ],
    },
    options: {
      animation: lineDraw(),
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Epoch" } },
        y: { min: 0, max: 100, title: { display: true, text: "Độ chính xác (%)" } },
        y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Sai số (loss)" } },
      },
    },
    };
  });
}

// ───────── Tab 2: PlantDoc theo lớp ─────────
function recallBar(value, pillar) {
  const pct = Math.max(0, Math.min(100, (value || 0) * 100));
  return `<div class="recall-bar" title="${fmtPct(value, 1)}"><span class="recall-track"><span class="recall-fill ${pillar}" data-style="--s:${(pct / 100).toFixed(3)}"></span></span><span class="recall-num mono">${fmtNum(pct, 0)}</span></div>`;
}

function breakdownBar(label, pillar, block) {
  const total = block.count || 1;
  const e = block.error_breakdown || {};
  const seg = (n, cls, name) => `<span class="seg ${cls}" data-style="--w:${((n / total) * 100).toFixed(2)}%" title="${esc(name)}: ${fmtInt(n)} ảnh"><i>${n >= total * 0.08 ? fmtInt(n) : ""}</i></span>`;
  return `
    <div class="breakdown">
      <div class="breakdown-head">${modelPill(pillar === "centralized" ? "centralized" : "fedavg", label)}
        <span class="field-hint">${fmtInt(total)} ảnh PlantDoc · đoán đúng ${fmtPct(block.top1, 1)} · ${fmtInt(block.high_confidence_wrong?.count)} ảnh sai với xác suất ≥ ${fmtNum((block.high_confidence_wrong?.threshold || 0.9) * 100, 0)}%</span></div>
      <div class="stacked" role="img" aria-label="${esc(label)}: đúng ${e.correct}, sai loại cây ${e.wrong_crop}, đúng cây nhưng sai bệnh ${e.right_crop_wrong_disease}">
        ${seg(e.correct, "seg-ok", "Đúng")}${seg(e.right_crop_wrong_disease, "seg-mid", "Đúng loại cây, sai bệnh")}${seg(e.wrong_crop, "seg-bad", "Sai cả loại cây")}
      </div>
    </div>`;
}

function buildPlantdoc() {
  const { summary } = ctx;
  const root = byId("result-plantdoc");
  const pd = summary?.plantdoc;
  if (!pd?.available) {
    root.innerHTML = notReady(pd?.reason || "Chạy scripts/report_per_class.py.");
    return;
  }
  const gap = summary?.domain_gap?.available ? summary.domain_gap.models : null;
  const confusions = summary?.reeval?.available ? summary.reeval.top_confusions?.fedavg?.plantdoc || [] : [];
  const rows = pd.table.map(r => {
    const weak = r.fedavg_recall < (pd.weak_threshold ?? 0.5);
    return `<tr${weak ? ' class="row-weak"' : ""}>
      <th scope="row">${esc(cropOf(r.class))} — ${esc(diseaseOf(r.class))}${r.low_support ? ' <span class="pill-flag" title="Ít ảnh kiểm tra nên tỉ lệ nhận đúng dao động mạnh">ít ảnh</span>' : ""}</th>
      <td class="mono">${fmtInt(r.support)}</td>
      <td>${recallBar(r.fedavg_recall, "fedavg")}</td>
      <td>${recallBar(r.centralized_recall, "centralized")}</td>
      <td class="mono">${fmtPct(r.fedavg_pv_recall, 0)}</td>
      <td class="mono ${r.domain_gap_pp >= 50 ? "text-rose" : ""}">${fmtNum(r.domain_gap_pp, 1)}</td>
    </tr>`;
  }).join("");

  root.innerHTML = `
    <section class="stat-tiles">
      <article class="stat-tile"><b>${esc(pd.classes_covered)}</b><span>nhóm có ảnh PlantDoc trong tập kiểm tra</span></article>
      <article class="stat-tile warn"><b>${fmtInt(pd.weak_images)}</b><span>ảnh thuộc ${pd.weak_classes.length} nhóm mà FedAvg nhận đúng dưới ${fmtNum((pd.weak_threshold || 0.5) * 100, 0)}%</span></article>
      <article class="stat-tile"><b>${pd.fedavg_worse_in}/${pd.classes_compared}</b><span>nhóm FedAvg kém Centralized (tốt hơn ở ${pd.fedavg_better_in} nhóm)</span></article>
      ${gap ? `<article class="stat-tile"><b>${fmtNum(gap["FedAvg v7"].gap_pp, 1)} điểm</b><span>FedAvg giảm từ PlantVillage xuống PlantDoc (Centralized: ${fmtNum(gap["Centralized v6"].gap_pp, 1)})</span></article>` : ""}
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Phân loại lỗi: sai loại cây hoặc sai bệnh</h3>
      <p class="panel-desc">Chia ${fmtInt(pd.models["FedAvg v7"].count)} ảnh PlantDoc theo kết quả dự đoán.</p>
      <div class="legend-row mt-3"><span><i class="swatch seg-ok"></i>Đúng</span><span><i class="swatch seg-mid"></i>Đúng loại cây, sai bệnh</span><span><i class="swatch seg-bad"></i>Sai cả loại cây</span></div>
      <div class="mt-3">${breakdownBar("FedAvg v7", "fedavg", pd.models["FedAvg v7"])}${breakdownBar("Centralized v6", "centralized", pd.models["Centralized v6"])}</div>
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Tỉ lệ nhận đúng theo từng nhóm (ảnh ngoài đồng PlantDoc)</h3>
      <p class="panel-desc">Chỉ các nhóm có ảnh PlantDoc trong tập kiểm tra. Hàng tô nền là nhóm FedAvg nhận đúng dưới ${fmtNum((pd.weak_threshold || 0.5) * 100, 0)}%. Cột “Chênh lệch miền” = tỉ lệ nhận đúng ở ảnh PlantVillage trừ tỉ lệ ở ảnh PlantDoc của FedAvg (điểm phần trăm).</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Nhóm</th><th scope="col">Số ảnh</th><th scope="col">FedAvg (%)</th><th scope="col">Centralized (%)</th><th scope="col">FedAvg trên ảnh trong phòng</th><th scope="col">Chênh lệch miền (PlantVillage − PlantDoc)</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>

    ${confusions.length ? `
    <section class="panel mt-4">
      <h3 class="panel-title">Nhầm lẫn nhiều nhất của FedAvg trên PlantDoc</h3>
      <ol class="confusion-list mt-3">
        ${confusions.map(c => `<li><span>${esc(formatDiseaseName(c.target))}</span><span class="arrow">→</span><span>${esc(formatDiseaseName(c.predicted))}</span><b class="mono">${fmtInt(c.count)} ảnh</b></li>`).join("")}
      </ol>
    </section>` : ""}`;
  applyStyleData(root); // riseBars đọc --s ngay bên dưới nên không đợi MutationObserver
  openOnWide(byId("result-full-table"));
  riseBars(root);
}

/** Thanh tỉ lệ mọc lần lượt, thanh xếp chồng hiện bằng một "rèm" kéo ra: chỉ transform, chạy trên compositor. */
function riseBars(root) {
  root.querySelectorAll(".recall-fill").forEach((bar, i) => {
    const scale = bar.style.getPropertyValue("--s").trim() || "1";
    play(bar, [{ transform: "scaleX(0)" }, { transform: `scaleX(${scale})` }], { delay: Math.min(i * 15, 300), duration: 360 });
  });
  root.querySelectorAll(".stacked").forEach((stack, i) => {
    play(stack, [{ transform: "scaleX(1)" }, { transform: "scaleX(0)" }], { pseudoElement: "::after", delay: i * 100, duration: 400 });
  });
}

// ───────── Tab 3: Độ tin cậy xác suất ─────────
function buildCalibration() {
  const { summary } = ctx;
  const root = byId("result-calibration");
  const re = summary?.reeval;
  if (!re?.available) {
    root.innerHTML = notReady(re?.reason || "Chạy scripts/evaluate_published_models.py trên máy có dataset.");
    return;
  }
  const rows = re.models.map(m => {
    const pv = m.calibration.plantvillage;
    const pd = m.calibration.plantdoc;
    return `<tr>
      <th scope="row">${modelPill(m.model_id)}</th>
      <td class="mono">${fmtNum(m.temperature, 2)}</td>
      <td class="mono">${fmtNum(pv.ece_raw, 3)} → ${fmtNum(pv.ece_temperature_scaled, 3)}</td>
      <td class="mono ${pd.ece_raw >= 0.25 ? "text-rose" : ""}">${fmtNum(pd.ece_raw, 3)} → ${fmtNum(pd.ece_temperature_scaled, 3)}</td>
      <td class="mono">${fmtPct(pd.mean_confidence, 0)} / ${fmtPct(pd.accuracy, 0)}</td>
      <td class="mono">${fmtInt(m.plantdoc.high_confidence_wrong)}</td>
    </tr>`;
  }).join("");

  root.innerHTML = `
    <section class="callout callout-warn">
      <b>Xác suất softmax thô của các mô hình chưa được hiệu chuẩn.</b> Bước 4 hiện cả số thô lẫn số đã hiệu chuẩn bằng temperature scaling (T ở bảng dưới).
      Trên PlantVillage độ tin cậy khá sát độ chính xác thực, nhưng trên PlantDoc các mô hình <b>gán xác suất cao hơn nhiều so với tỉ lệ đúng thực tế</b>:
      FedAvg báo trung bình ${fmtPct(re.models.find(m => m.model_id === "fedavg").calibration.plantdoc.mean_confidence, 0)} nhưng chỉ đúng
      ${fmtPct(re.models.find(m => m.model_id === "fedavg").calibration.plantdoc.accuracy, 0)}.
      Việc hiệu chuẩn (temperature scaling) học trên tập ảnh chủ yếu là PlantVillage nên chỉ sửa được một phần sai lệch trên PlantDoc (ECE vẫn cao).
      Vì vậy một ảnh có độ tin cậy cao không có nghĩa là chắc đúng.
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">ECE: độ tin cậy lệch bao nhiêu so với thực tế</h3>
      <p class="panel-desc">ECE (Expected Calibration Error, ${re.bins} khoảng) càng thấp càng tốt. "Thô → chỉnh T" là trước và sau khi chia logit cho nhiệt độ T học trên tập hiệu chuẩn riêng ${fmtInt(re.calibration_count)} ảnh; temperature scaling không đổi dự đoán, chỉ đổi xác suất.</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">T</th><th scope="col">ECE PlantVillage</th><th scope="col">ECE PlantDoc</th><th scope="col">Ngoài đồng: xác suất TB / tỉ lệ đúng thực</th><th scope="col">Sai dù tự tin ≥ ${fmtNum(re.high_confidence_threshold * 100, 0)}% (ngoài đồng)</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>

    <section class="panel mt-4">
      <div class="panel-header-flex">
        <div>
          <h3 class="panel-title">Biểu đồ độ tin cậy</h3>
          <p class="panel-desc">Mỗi điểm là một khoảng độ tin cậy: trục ngang là độ tin cậy trung bình, trục dọc là tỉ lệ đúng thực tế. Đường nét đứt là hiệu chuẩn hoàn hảo; điểm dưới đường là tự tin quá mức.</p>
        </div>
        <div class="segmented" role="radiogroup" aria-label="Nguồn ảnh kiểm tra">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-cal="plantdoc">PlantDoc</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-cal="plantvillage">PlantVillage</button>
        </div>
      </div>
      <div class="chart-container chart-tall mt-4"><canvas id="chart-reliability" aria-label="Biểu đồ độ tin cậy của FedAvg và Centralized"></canvas></div>
    </section>`;

  const draw = source => {
    const points = id => (re.reliability?.[id]?.[source] || []).filter(b => b.n > 0)
      .map(b => ({ x: +(b.confidence * 100).toFixed(1), y: +(b.accuracy * 100).toFixed(1), n: b.n }));
    makeChartWhenVisible(byId("chart-reliability"), () => {
      const colors = pillarColors();
      return {
      type: "scatter",
      data: {
        datasets: [
          { label: "Hiệu chuẩn hoàn hảo", data: [{ x: 0, y: 0 }, { x: 100, y: 100 }], showLine: true, borderDash: [6, 6], borderColor: colors.neutral, pointRadius: 0, borderWidth: 1.5 },
          { label: "FedAvg v7", data: points("fedavg"), showLine: true, borderColor: colors.fedavg, backgroundColor: colors.fedavg, tension: 0.2 },
          { label: "Centralized v6", data: points("centralized"), showLine: true, borderColor: colors.centralized, backgroundColor: colors.centralized, tension: 0.2 },
        ],
      },
      options: {
        interaction: { mode: "nearest", intersect: true },
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: "Độ tin cậy trung bình (%)" } },
          y: { min: 0, max: 100, title: { display: true, text: "Tỉ lệ đúng thực tế (%)" } },
        },
        plugins: {
          tooltip: { callbacks: { label: c => `${c.dataset.label}: tin cậy ${fmtNum(c.parsed.x, 0)}% → đúng ${fmtNum(c.parsed.y, 0)}%${c.raw.n ? ` (${fmtInt(c.raw.n)} ảnh)` : ""}` } },
        },
      },
      };
    });
  };
  draw("plantdoc");
  root.querySelectorAll("[data-cal]").forEach(button => {
    button.addEventListener("click", () => {
      root.querySelectorAll("[data-cal]").forEach(b => {
        b.classList.toggle("is-active", b === button);
        b.setAttribute("aria-checked", String(b === button));
      });
      draw(button.dataset.cal);
    });
  });
}

// ───────── Tab: Test giữ riêng & thống kê ─────────
const pct1 = value => (value === null || value === undefined ? "—" : `${fmtNum(value * 100, 1)}`);

function withCi(estimate, interval) {
  if (estimate === null || estimate === undefined) return "—";
  const ci = interval && interval.low !== null && interval.low !== undefined
    ? `<span class="ci">[${pct1(interval.low)}; ${pct1(interval.high)}]</span>` : "";
  return `<span class="mono">${fmtPct(estimate)}</span>${ci}`;
}

function pairLine(label, pair) {
  if (!pair) return "";
  const d = pair.difference || {};
  const clear = d.low !== null && d.low !== undefined && (d.low > 0 || d.high < 0);
  const sign = v => `${v >= 0 ? "+" : "−"}${fmtNum(Math.abs(v) * 100, 1)}`;
  return `<li>
      <span class="pair-label">${label}</span>
      <span class="mono">${d.estimate === null || d.estimate === undefined ? "—" : sign(d.estimate)} điểm</span>
      <span class="ci">${d.low === null || d.low === undefined ? "" : `[${sign(d.low)}; ${sign(d.high)}]`}</span>
      <span class="mono">McNemar ${pair.p_value === null || pair.p_value === undefined ? "p = —" : pair.p_value < 0.001 ? "p &lt; 0,001" : `p = ${fmtNum(pair.p_value, 3)}`}</span>
      <span class="verdict ${clear ? "ok" : "mid"}">${clear ? "khác biệt rõ" : "chưa đủ chắc chắn"}</span>
    </li>`;
}

function buildHoldout() {
  const { evidence, summary } = ctx;
  const root = byId("result-holdout");
  const ho = summary?.holdout;
  const st = summary?.statistics;
  if (!ho?.available && !st?.available) {
    root.innerHTML = notReady(ho?.reason || st?.reason || "Chạy scripts/evaluate_published_models.py --holdout và scripts/report_statistics.py.");
    return;
  }
  const byModel = modelsById(evidence);
  const holdoutById = Object.fromEntries((ho?.models || []).map(m => [m.model_id, m]));
  const split = name => (st?.splits?.[name]?.available ? st.splits[name].by_source : null);
  const val = split("val");
  const test = split("test");
  const sel = st?.selection_effect;

  const renderRows = ids => ids.map(id => {
    const pub = byModel[id].metrics;
    const h = holdoutById[id];
    const cell = (source, block, value) => withCi(value, block?.[source]?.models?.[id]);
    return `<tr class="row-${pillarOf(id)}">
      <th scope="row">${modelPill(id)}</th>
      <td>${cell("plantvillage", val, pub.plantvillage.top1_accuracy)}</td>
      <td>${h ? cell("plantvillage", test, h.plantvillage.top1) : "—"}</td>
      <td>${cell("plantdoc", val, pub.plantdoc.top1_accuracy)}</td>
      <td>${h ? cell("plantdoc", test, h.plantdoc.top1) : "—"}</td>
    </tr>`;
  }).join("");
  const rows = renderRows(["centralized", "fedavg"]);
  const localRows = renderRows(MODEL_ORDER.filter(id => id.startsWith("local_")));

  const bestLocal = st?.best_local_chosen_on_val || {};
  // so val với test của Centralized trên PlantDoc: câu nhận xét đi theo số liệu, không viết cứng
  const pdVal = byModel.centralized?.metrics?.plantdoc?.top1_accuracy;
  const pdTest = holdoutById.centralized?.plantdoc?.top1;
  const valVsTest = pdVal === undefined || pdTest === undefined || pdTest === null ? ""
    : pdTest >= pdVal
      ? `Trên dữ liệu này con số ở tập giữ riêng không thấp hơn: PlantDoc Centralized đạt ${fmtPct(pdTest)} (tập kiểm tra: ${fmtPct(pdVal)}); tập giữ riêng chỉ có ${fmtInt(ho?.counts_by_source?.plantdoc)} ảnh PlantDoc nên khoảng tin cậy rộng.`
      : `Con số ở tập giữ riêng thấp hơn (PlantDoc Centralized ${fmtPct(pdTest)} so với ${fmtPct(pdVal)} ở tập kiểm tra) — phù hợp với thiên lệch do chọn mô hình trên tập kiểm tra.`;
  const pairs = (block, source) => block?.[source]?.pairs || {};
  const pairBlock = (title, block) => (block ? `
      <div class="pair-block">
        <h4 class="subhead">${title}</h4>
        <ul class="pair-list">
          ${pairLine("PlantDoc · Centralized − FedAvg", pairs(block, "plantdoc").centralized_vs_fedavg)}
          ${pairLine(`PlantDoc · FedAvg − ${esc(MODEL_LABELS[bestLocal.plantdoc] || "Local tốt nhất")}`, pairs(block, "plantdoc").fedavg_vs_best_local)}
          ${pairLine("PlantVillage · Centralized − FedAvg", pairs(block, "plantvillage").centralized_vs_fedavg)}
          ${pairLine(`PlantVillage · FedAvg − ${esc(MODEL_LABELS[bestLocal.plantvillage] || "Local tốt nhất")}`, pairs(block, "plantvillage").fedavg_vs_best_local)}
        </ul>
      </div>` : "");

  const leaf = test?.plantvillage?.same_leaf;
  const leafRows = leaf ? ["centralized", "fedavg", bestLocal.plantvillage].filter(Boolean).map(id => {
    const m = leaf.models[id] || {};
    return `<tr><th scope="row">${modelPill(id)}</th><td class="mono">${fmtPct(m.all)}</td><td class="mono">${fmtPct(m.grouped_only)}</td><td class="mono">${fmtPct(m.without_suspected_reshoots)}</td></tr>`;
  }).join("") : "";

  const recorded = holdoutById.centralized?.vs_recorded || {};
  const recordedText = Object.entries(recorded).filter(([, v]) => v.available)
    .map(([src, v]) => `${src === "plantdoc" ? "PlantDoc" : "PlantVillage"} ${v.reproduced ? "khớp" : `lệch ${fmtInt(v.correct_diff_images)} ảnh`}`).join(", ");

  root.innerHTML = `
    <p class="viewer-scope">Ảnh giữ riêng không dùng để chọn mô hình; đây là bộ chẩn đoán có sẵn của bộ dữ liệu (legacy), chưa phải kiểm thử cuối được khóa trước và chưa nằm trong kiểm tra trùng ảnh.</p>


    <section class="panel mt-4">
      <h3 class="panel-title">Tập kiểm tra và tập giữ riêng, kèm khoảng tin cậy 95%</h3>
      <p class="panel-desc">Tỉ lệ dự đoán đúng, kèm khoảng tin cậy 95%. Khoảng càng rộng, kết quả càng dao động. Tập giữ riêng: PlantVillage ${fmtInt(ho?.counts_by_source?.plantvillage)} ảnh, PlantDoc ${fmtInt(ho?.counts_by_source?.plantdoc)} ảnh.</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table result-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">PlantVillage · tập kiểm tra</th><th scope="col">PlantVillage · tập giữ riêng</th><th scope="col">PlantDoc · tập kiểm tra</th><th scope="col">PlantDoc · tập giữ riêng</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <details class="fold mt-3"><summary>Xem thêm 5 mô hình học riêng</summary>
        <div class="table-responsive mt-3" tabindex="0" role="group" aria-label="Kết quả các mô hình học riêng">
          <table class="data-table result-table"><thead><tr><th scope="col">Mô hình</th><th scope="col">PlantVillage · tập kiểm tra</th><th scope="col">PlantVillage · tập giữ riêng</th><th scope="col">PlantDoc · tập kiểm tra</th><th scope="col">PlantDoc · tập giữ riêng</th></tr></thead><tbody>${localRows}</tbody></table>
        </div>
      </details>
      ${recordedText ? `<p class="field-hint mt-2">Đối chiếu: số Centralized trên tập giữ riêng do nhóm huấn luyện tự ghi (summary.json) — ${recordedText} (số gốc làm tròn 4 chữ số).</p>` : ""}
    </section>

    <details class="overview-detail mt-4">
      <summary><span><strong>Cách chọn mô hình và phương pháp thống kê</strong><small>Phạm vi tập giữ riêng, bootstrap theo cụm</small></span></summary>
      <div class="overview-detail-body">
    ${sel?.available ? `
    <section class="callout callout-warn">
      <b>Vì sao cần tập giữ riêng:</b>
      <ul>
        <li>Phiên bản mô hình được chọn theo điểm Macro-F1 PlantDoc (chỉ xét epoch đã đạt ngưỡng F1 PlantVillage) ngay trên <b>chính tập kiểm tra dùng để báo cáo</b>, nên con số ở đó có thể hơi đẹp hơn thực tế.
          Với Centralized, epoch được chọn (${fmtInt(sel.chosen_epoch)}/${fmtInt(sel.epochs)}) có độ chính xác PlantDoc ${fmtPct(sel.chosen_plantdoc_accuracy)}
          — ${sel.chosen_rank_among_epochs === 1 ? "cao nhất trong mọi epoch" : `hạng ${fmtInt(sel.chosen_rank_among_epochs)}/${fmtInt(sel.epochs)}`} (epoch cuối: ${fmtPct(sel.last_plantdoc_accuracy)}).</li>
        <li>Vì vậy đồ án chấm lại thêm trên ${fmtInt(ho?.count)} ảnh <b>giữ riêng</b> — bộ chẩn đoán có sẵn của bộ dữ liệu (legacy), không dùng để chọn mô hình nên không bị thiên lệch này. ${valVsTest} Với FedAvg: web chưa có đủ lịch sử từng vòng để đo hiệu ứng chọn mô hình.</li>
        <li><b>Lưu ý:</b> đây chưa phải bài kiểm tra cuối được khóa trước, và chưa nằm trong phép kiểm tra trùng ảnh ở bước 6;
          một phần ảnh PlantVillage trong tập này có thể là ảnh chụp lại cùng lá với ảnh học (xem bảng cuối trang).</li>
      </ul>
    </section>` : ""}
      <p class="field-hint mt-3">Khoảng tin cậy tính bằng bootstrap theo cụm: PlantDoc theo cảnh chụp, PlantVillage theo nhóm lá; nhiều ảnh cùng cảnh/lá không độc lập nhau.</p>
      </div>
    </details>

    ${val || test ? `
    <details class="overview-detail mt-4"><summary>So sánh theo cặp</summary><div class="overview-detail-body">
      <h3 class="panel-title">So sánh theo cặp trên cùng ảnh</h3>
      <p class="panel-desc">Mỗi dòng là hiệu số độ chính xác (điểm phần trăm) giữa hai mô hình trên đúng cùng tập ảnh.</p>
      <ul class="mt-2">
        <li>“Khác biệt rõ” nghĩa là khoảng tin cậy không chứa 0; ưu tiên đọc khoảng tin cậy hơn giá trị p.</li>
        <li>Chưa hiệu chỉnh cho 8 phép so sánh; theo Bonferroni, ngưỡng p là 0,05/8 ≈ 0,006.</li>
        <li>“Local tốt nhất” được chọn theo tập kiểm tra, không theo tập giữ riêng.</li>
      </ul>
      <p class="field-hint mt-2"><b>McNemar:</b> kiểm định hai mô hình trên cùng ảnh (giả định ảnh độc lập). <b>Bootstrap theo cụm:</b> ước lượng khoảng dao động khi nhiều ảnh cùng cảnh hoặc cùng lá.</p>
      <div class="pair-grid mt-3">${pairBlock("Trên tập kiểm tra", val)}${pairBlock("Trên tập giữ riêng", test)}</div>
    </div></details>` : ""}

    ${leaf ? `
    <details class="overview-detail mt-4"><summary>Độ nhạy với ảnh nghi chụp lại cùng một lá</summary><div class="overview-detail-body">
      <h3 class="panel-title">Độ nhạy với ảnh chụp lại cùng một lá (PlantVillage, tập giữ riêng)</h3>
      <p class="panel-desc">${fmtInt(leaf.ungrouped_images)} ảnh PlantVillage trong tập giữ riêng không được gắn nhóm theo lá; ${fmtInt(leaf.suspected_reshoots)} ảnh trong đó có ảnh học cùng nhóm mang số thứ tự liền kề (ước lượng ảnh chụp lại cùng lá — có thể làm số đẹp hơn thật). Nếu bỏ chúng mà độ chính xác không giảm thì chưa thấy bằng chứng con số bị phóng đại.</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">Tất cả</th><th scope="col">Chỉ ảnh đã gom nhóm lá</th><th scope="col">Bỏ ảnh nghi chụp lại</th></tr></thead>
          <tbody>${leafRows}</tbody>
        </table>
      </div>
    </div></details>` : ""}`;
}

// ───────── Tab: Ma trận nhầm lẫn ─────────
function confusionHtml(matrix, classes) {
  const rowTotals = matrix.map(row => row.reduce((a, b) => a + b, 0));
  const keep = new Set();
  matrix.forEach((row, i) => {
    if (!rowTotals[i]) return;
    keep.add(i);
    row.forEach((n, j) => { if (n > 0) keep.add(j); });
  });
  const idx = [...keep].sort((a, b) => a - b);
  const head = idx.map(j => `<th scope="col" title="${esc(formatDiseaseName(classes[j]))}">${j + 1}</th>`).join("");
  const rows = idx.filter(i => rowTotals[i] > 0).map(i => {
    const cells = idx.map(j => {
      const n = matrix[i][j];
      const share = rowTotals[i] ? n / rowTotals[i] : 0;
      const cls = i === j ? "cm-diag" : n > 0 ? "cm-off" : "cm-zero";
      const title = `Thật: ${formatDiseaseName(classes[i])} → Đoán: ${formatDiseaseName(classes[j])}: ${fmtInt(n)} ảnh (${fmtNum(share * 100, 1)}%)`;
      return `<td class="cm-cell ${cls}" data-style="--p:${Math.sqrt(share).toFixed(3)}" title="${esc(title)}">${n > 0 ? fmtInt(n) : ""}</td>`;
    }).join("");
    return `<tr><th scope="row" class="cm-row"><span class="cm-n">${i + 1}</span> ${esc(formatDiseaseName(classes[i]))}</th>${cells}</tr>`;
  }).join("");
  return `<table class="cm-table"><thead><tr><th scope="col" class="cm-corner">Thật ↓ · Đoán →</th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function topConfusions(matrix, classes, limit = 8) {
  const out = [];
  matrix.forEach((row, i) => row.forEach((n, j) => { if (i !== j && n > 0) out.push({ i, j, n }); }));
  return out.sort((a, b) => b.n - a.n).slice(0, limit)
    .map(c => `<li><span>${esc(formatDiseaseName(classes[c.i]))}</span><span class="arrow">→</span><span>${esc(formatDiseaseName(classes[c.j]))}</span><b class="mono">${fmtInt(c.n)} ảnh</b></li>`).join("");
}

async function buildConfusion() {
  const root = byId("result-confusion");
  root.innerHTML = '<div class="skeleton skeleton-block"></div>';
  const details = await getDetails();
  if (!details.available || !details.confusion?.val) {
    root.innerHTML = notReady(details.reason || "Chưa có ma trận nhầm lẫn (cần chạy chấm lại cục bộ).");
    return;
  }
  const state = { model: "fedavg", source: "plantdoc", split: "val" };
  const splits = Object.keys(details.confusion);
  root.innerHTML = `
    <section class="panel">
      <div class="panel-header-flex">
        <div>
          <h3 class="panel-title">Ma trận nhầm lẫn</h3>
          <p class="panel-desc">Mỗi hàng là nhóm thật, mỗi cột là nhóm mô hình đoán. Ô xanh trên đường chéo là đoán đúng; ô đỏ là nhầm, càng đậm thì tỉ lệ (trong hàng) càng lớn. Chỉ hiện các nhóm có ảnh trong nguồn đã chọn; rê chuột lên ô để xem tên bệnh và số ảnh.</p>
        </div>
      </div>
      <div class="cm-controls mt-3">
        <div class="segmented" role="radiogroup" aria-label="Mô hình">
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-cm-model="centralized">Centralized</button>
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-cm-model="fedavg">FedAvg</button>
        </div>
        <div class="segmented" role="radiogroup" aria-label="Nguồn">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-cm-source="plantdoc">PlantDoc</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-cm-source="plantvillage">PlantVillage</button>
        </div>
        ${splits.length > 1 ? `<div class="segmented" role="radiogroup" aria-label="Tập đánh giá">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-cm-split="val">Tập kiểm tra</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-cm-split="test">Tập giữ riêng</button>
        </div>` : ""}
      </div>
      <p class="field-hint mt-3" id="cm-summary"></p>
      <div tabindex="0" role="region" aria-label="Ma trận nhầm lẫn, cuộn được bằng phím mũi tên" class="table-responsive cm-wrap mt-2" id="cm-wrap"></div>
    </section>
    <section class="panel mt-4">
      <h3 class="panel-title">Nhầm lẫn nhiều nhất</h3>
      <ol class="confusion-list mt-3" id="cm-top"></ol>
    </section>`;
  const render = () => {
    const matrix = details.confusion[state.split]?.[state.model]?.[state.source];
    if (!matrix) {
      byId("cm-summary").textContent = "";
      byId("cm-wrap").innerHTML = '<p class="field-hint">Không có ma trận cho lựa chọn này.</p>';
      byId("cm-top").innerHTML = "";
      return;
    }
    const total = matrix.reduce((a, row) => a + row.reduce((x, y) => x + y, 0), 0);
    const correct = matrix.reduce((a, row, i) => a + row[i], 0);
    byId("cm-summary").textContent = `${MODEL_LABELS[state.model]} · ${state.source === "plantdoc" ? "PlantDoc" : "PlantVillage"} · ${state.split === "val" ? "tập kiểm tra" : "tập giữ riêng"}: ${fmtInt(correct)}/${fmtInt(total)} ảnh đúng (${fmtPct(total ? correct / total : 0)}).`;
    byId("cm-wrap").innerHTML = confusionHtml(matrix, details.classes);
    attachCrosshair(byId("cm-wrap"));
    byId("cm-top").innerHTML = topConfusions(matrix, details.classes) || "<li>Không có nhầm lẫn.</li>";
  };
  const bind = (attr, key) => root.querySelectorAll(`[${attr}]`).forEach(button => button.addEventListener("click", () => {
    root.querySelectorAll(`[${attr}]`).forEach(b => {
      b.classList.toggle("is-active", b === button);
      b.setAttribute("aria-checked", String(b === button));
    });
    state[key] = button.getAttribute(attr);
    render();
  }));
  bind("data-cm-model", "model");
  bind("data-cm-source", "source");
  bind("data-cm-split", "split");
  render();
}

// ───────── Tab: Sai với xác suất cao ─────────
const ERR_PAGE = 20; // số thẻ hiện mỗi trang (UI-01): tab không còn dựng cả 175 ảnh một lần
async function buildErrors() {
  const root = byId("result-errors");
  root.innerHTML = '<div class="skeleton skeleton-block"></div>';
  const details = await getDetails();
  const errors = details.errors;
  if (!details.available || !errors?.available) {
    root.innerHTML = notReady(errors?.reason || details.reason || "Chưa có thư viện ảnh lỗi.");
    return;
  }
  const items = errors.items;
  const floor = Math.round((errors.threshold || 0.9) * 100); // ngưỡng mà thư viện ảnh được dựng theo
  const levels = [...new Set([floor, 95, 99].filter(value => value >= floor))];
  const state = { model: "all", source: "all", min: floor, shown: ERR_PAGE };

  // Tóm tắt trên TOÀN BỘ thư viện (không đổi theo bộ lọc): giúp thấy ngay lỗi nào đáng chú ý trước khi cuộn qua từng ảnh.
  const fieldCount = items.filter(item => item.source === "plantdoc").length;
  const pairs = new Map();
  for (const item of items) {
    const wrong = item.wrong_by.fedavg;
    if (!wrong) continue;
    const key = `${item.target}\u0000${wrong.predicted}`;
    pairs.set(key, { target: item.target, predicted: wrong.predicted, count: (pairs.get(key)?.count || 0) + 1 });
  }
  const topPairs = [...pairs.values()].sort((a, b) => b.count - a.count).slice(0, 3);
  const fedavgWrong = items.filter(item => item.wrong_by.fedavg).length;

  root.innerHTML = `
    <section class="panel">
      <h3 class="panel-title">Ảnh kiểm tra bị dự đoán sai với xác suất ≥ ${fmtNum(floor, 0)}%</h3>
      <p class="panel-desc">Đây là các dự đoán sai có xác suất cao nhất. Phần lớn là ảnh PlantDoc chụp ngoài đồng, nơi xác suất dự đoán lệch xa nhất so với độ chính xác thực.</p>
      <div class="stat-tiles mt-3">
        <article class="stat-tile"><b>${fmtInt(items.length)}</b><span>ảnh sai với xác suất ≥ ${fmtNum(floor, 0)}% (ít nhất một trong hai mô hình)</span></article>
        <article class="stat-tile warn"><b>${fmtPct(fieldCount / Math.max(items.length, 1), 0)}</b><span>là ảnh PlantDoc chụp ngoài đồng (${fmtInt(fieldCount)} ảnh)</span></article>
        <article class="stat-tile"><b>${fmtInt(fedavgWrong)}</b><span>ảnh trong đó FedAvg sai</span></article>
      </div>
      ${topPairs.length ? `
      <h4 class="subhead mt-4">FedAvg nhầm nhiều nhất</h4>
      <ol class="confusion-list mt-2">
        ${topPairs.map(c => `<li><span>${esc(formatDiseaseName(c.target))}</span><span class="arrow">→</span><span>${esc(formatDiseaseName(c.predicted))}</span><b class="mono">${fmtInt(c.count)} ảnh</b></li>`).join("")}
      </ol>` : ""}
      <div class="cm-controls mt-4">
        <div class="segmented" role="radiogroup" aria-label="Mô hình">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-err-model="all">Tất cả</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-err-model="fedavg">FedAvg sai</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-err-model="centralized">Centralized sai</button>
        </div>
        <div class="segmented" role="radiogroup" aria-label="Nguồn">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-err-source="all">Mọi nguồn</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-err-source="plantdoc">PlantDoc</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-err-source="plantvillage">PlantVillage</button>
        </div>
        <div class="segmented" role="radiogroup" aria-label="Xác suất dự đoán tối thiểu">
          ${levels.map((level, index) => `<button type="button" class="segmented-option${index === 0 ? " is-active" : ""}" role="radio" aria-checked="${index === 0}" data-err-min="${level}">Xác suất ≥ ${level}%</button>`).join("")}
        </div>
      </div>
      <p class="field-hint mt-3" id="err-count" role="status" aria-live="polite"></p>
      <div class="error-grid mt-2" id="err-grid"></div>
      <div class="mt-3"><button type="button" class="btn btn-secondary hidden" id="err-more"></button></div>
      <p class="field-hint mt-3">${esc(errors.attribution)}</p>
    </section>`;

  const matches = item => (state.source === "all" || item.source === state.source)
    && Object.entries(item.wrong_by).some(([model, v]) => (state.model === "all" || model === state.model) && v.confidence * 100 >= state.min - 1e-9);
  const render = () => {
    const matched = items.filter(matches);
    const visible = matched.slice(0, state.shown); // chỉ dựng một trang: DOM không tăng theo số lần lọc
    const rest = matched.length - visible.length;
    byId("err-count").textContent = `${fmtInt(matched.length)} ảnh khớp bộ lọc · đang hiện ${fmtInt(visible.length)}`;
    byId("err-grid").innerHTML = visible.map(item => `
      <figure class="error-card">
        <img src="/static/samples/errors/${encodeURIComponent(item.thumb)}" alt="" loading="lazy" width="160" height="160">
        <figcaption>
          <span class="error-src">${item.source === "plantdoc" ? "PlantDoc" : "PlantVillage"}</span>
          <b>${esc(formatDiseaseName(item.target))}</b>
          ${Object.entries(item.wrong_by).map(([model, v]) => `<span class="error-pred">${modelPill(model)} đoán <b>${esc(formatDiseaseName(v.predicted))}</b> · ${fmtPct(v.confidence, 0)}</span>`).join("")}
        </figcaption>
      </figure>`).join("") || '<p class="field-hint">Không có ảnh khớp bộ lọc.</p>';
    const more = byId("err-more");
    more.classList.toggle("hidden", rest <= 0);
    more.textContent = `Xem thêm ${Math.min(ERR_PAGE, rest)} ảnh (còn ${fmtInt(rest)})`;
  };
  const bind = (attr, key, parse = value => value) => root.querySelectorAll(`[${attr}]`).forEach(button => button.addEventListener("click", () => {
    root.querySelectorAll(`[${attr}]`).forEach(b => {
      b.classList.toggle("is-active", b === button);
      b.setAttribute("aria-checked", String(b === button));
    });
    state[key] = parse(button.getAttribute(attr));
    state.shown = ERR_PAGE; // đổi bộ lọc thì về trang đầu
    render();
  }));
  bind("data-err-model", "model");
  bind("data-err-source", "source");
  bind("data-err-min", "min", Number);
  byId("err-more").addEventListener("click", () => {
    state.shown += ERR_PAGE;
    render();
  });
  render();
}

const BUILDERS = {
  summary: buildSummary, holdout: buildHoldout, plantdoc: buildPlantdoc,
  confusion: buildConfusion, errors: buildErrors, calibration: buildCalibration,
};
const building = new Map();

async function activate(name) {
  $$(".subtab").forEach(tab => {
    const on = tab.dataset.subtab === name;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
  });
  $$(".subpanel").forEach(panel => panel.classList.toggle("is-active", panel.id === `result-${name}`));
  if (!ctx) return; // bằng chứng không khả dụng: panel đã có thông báo lý do, không có gì để dựng
  if (built.has(name) || building.has(name)) return;
  // chờ một khung: nút tab/panel đổi trạng thái được vẽ trước, việc dựng nặng không làm click có cảm giác trễ
  const task = new Promise(resolve => requestAnimationFrame(() => resolve())).then(() => BUILDERS[name]());
  building.set(name, task);
  try {
    await task;
    built.add(name); // chỉ đánh dấu đã dựng khi thành công; lỗi thì lần bấm sau thử lại
  } catch (error) {
    console.error(`[Kết quả] Không dựng được tab ${name}:`, error);
    const panel = byId(`result-${name}`);
    if (panel) panel.innerHTML = `<div class="callout callout-error" role="alert"><b>Không dựng được tab này.</b> ${esc(error.message)}</div>`;
    notify("error", "Không dựng được tab", name, { detail: error.message });
  } finally {
    building.delete(name);
  }
}

export async function mount() {
  const [evidence, summary] = await Promise.all([
    getEvidence().catch(error => ({ available: false, reason: error.message })),
    getSummary().catch(() => null),
  ]);
  // Gắn handler trước khi xét `available`: tab không được "chết"; khi thiếu bằng chứng, mọi tab đổi panel và nói rõ lý do.
  $$(".subtab").forEach(tab => tab.addEventListener("click", () => activate(tab.dataset.subtab)));
  if (!evidence.available) {
    const callout = role => `<div class="callout callout-error"${role}><b>Chưa có bộ số liệu đã kiểm hash.</b> ${esc(evidence.reason || "")}</div>`;
    $$(".subpanel").forEach(panel => { panel.innerHTML = callout(panel.id === "result-summary" ? ' role="alert"' : ""); });
    return;
  }
  ctx = { evidence, summary };
  activate("summary");
}
