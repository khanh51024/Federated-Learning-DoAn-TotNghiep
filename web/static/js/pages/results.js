// Bước 3 · Kết quả chi tiết: tổng hợp 7 mô hình, PlantDoc theo lớp, độ tin cậy của xác suất.
import { getEvidence, getSummary, modelsById } from "../core/store.js";
import { makeChart, pillarColors } from "../core/charts.js";
import {
  $$, MODEL_LABELS, MODEL_ORDER, byId, diseaseOf, cropOf, esc, fmtInt, fmtNum, fmtPct, fmtPctValue,
  formatDiseaseName, modelPill, notify, pillarOf, withAlpha,
} from "../core/dom.js";

const built = new Set();
let ctx = null; // { evidence, summary }

function notReady(reason) {
  return `<div class="callout callout-warn" role="note"><b>Phần này chưa có dữ liệu.</b> ${esc(reason || "")}</div>`;
}

function reproductionCell(model) {
  const sources = model?.vs_published;
  if (!sources) return '<span class="repro repro-none" title="Chưa có báo cáo tái đánh giá">—</span>';
  const statuses = ["plantdoc", "plantvillage"].map(s => sources[s]?.status || "unavailable");
  if (statuses.every(s => s === "exact")) {
    return '<span class="repro repro-exact" title="Số tái đánh giá trên máy này trùng tuyệt đối số công bố">✓ khớp tuyệt đối</span>';
  }
  if (statuses.every(s => s === "exact" || s === "within_tolerance")) {
    const diffs = ["plantdoc", "plantvillage"].map(s => {
      const d = sources[s]?.correct_diff_images;
      return `${s === "plantdoc" ? "PD" : "PV"}: ${d === null || d === undefined ? "?" : (d > 0 ? "+" : "") + d} ảnh`;
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
    reNote = `Đã tái đánh giá trên máy này (${fmtInt(re.validation_count)} ảnh validation): ${main}; ${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)} phép so khớp tuyệt đối${rest}.`;
  } else {
    reNote = `Số liệu là số công bố đã kiểm SHA-256. ${esc(re?.reason || "Chưa có báo cáo tái đánh giá cục bộ.")}`;
  }

  root.innerHTML = `
    <section class="panel">
      <div class="panel-header-flex">
        <div>
          <h3 class="panel-title">Bảy mô hình trên cùng validation</h3>
          <p class="panel-desc">PlantVillage ${fmtInt(evidence.validation_counts.plantvillage)} ảnh (cùng miền với train) và PlantDoc ${fmtInt(evidence.validation_counts.plantdoc)} ảnh (ảnh ngoài đồng). Số đậm là cao nhất mỗi cột.</p>
        </div>
      </div>
      <div class="table-responsive mt-3">
        <table class="data-table result-table">
          <thead>
            <tr>
              <th scope="col">Mô hình</th><th scope="col">Giai đoạn</th>
              <th scope="col">PlantVillage Top-1</th><th scope="col">Macro-F1</th>
              <th scope="col">PlantDoc Top-1</th><th scope="col">Macro-F1</th>
              <th scope="col">Đúng loại cây (PD)</th><th scope="col">Tái đánh giá</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="field-hint mt-3">${reNote}</p>
      <p class="field-hint mt-2">Macro-F1 tính trên các lớp có ảnh validation (PlantDoc chỉ phủ 13/38 lớp). Chỉ có một kịch bản non-IID (α=0,1) và một seed; chưa đủ để nói về độ ổn định.</p>
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Quá trình huấn luyện Centralized v6</h3>
      <p class="panel-desc">${fmtInt(evidence.centralized_history.length)} epoch. FedAvg v7 không công bố log từng vòng nên không vẽ đường hội tụ tương ứng — web không tự dựng đường từ một điểm cuối.</p>
      <div class="chart-container mt-4"><canvas id="chart-central-history" aria-label="Val accuracy và val loss của Centralized qua các epoch"></canvas></div>
    </section>`;

  const colors = pillarColors();
  const history = evidence.centralized_history || [];
  makeChart(byId("chart-central-history"), {
    type: "line",
    data: {
      labels: history.map(h => h.epoch),
      datasets: [
        { label: "Val accuracy (%)", data: history.map(h => +(h.val_accuracy * 100).toFixed(2)), borderColor: colors.centralized, backgroundColor: withAlpha(colors.centralized, 0.12), fill: true, tension: 0.25, pointRadius: 0, yAxisID: "y" },
        { label: "Val loss", data: history.map(h => +h.val_loss.toFixed(4)), borderColor: colors.danger, tension: 0.25, pointRadius: 0, yAxisID: "y1" },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Epoch" } },
        y: { min: 0, max: 100, title: { display: true, text: "Accuracy (%)" } },
        y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Loss" } },
      },
    },
  });
}

// ───────── Tab 2: PlantDoc theo lớp ─────────
function recallBar(value, pillar) {
  const pct = Math.max(0, Math.min(100, (value || 0) * 100));
  return `<div class="recall-bar" title="${fmtPct(value, 1)}"><span class="recall-track"><span class="recall-fill ${pillar}" style="--w:${pct.toFixed(1)}%"></span></span><span class="recall-num mono">${fmtNum(pct, 0)}</span></div>`;
}

function breakdownBar(label, pillar, block) {
  const total = block.count || 1;
  const e = block.error_breakdown || {};
  const seg = (n, cls, name) => `<span class="seg ${cls}" style="--w:${((n / total) * 100).toFixed(2)}%" title="${esc(name)}: ${fmtInt(n)} ảnh"><i>${n >= total * 0.08 ? fmtInt(n) : ""}</i></span>`;
  return `
    <div class="breakdown">
      <div class="breakdown-head">${modelPill(pillar === "centralized" ? "centralized" : "fedavg", label)}
        <span class="field-hint">${fmtInt(total)} ảnh PlantDoc · Top-1: ${fmtPct(block.top1, 1)} · ${fmtInt(block.high_confidence_wrong?.count)} ảnh sai nhưng tự tin ≥ ${fmtNum((block.high_confidence_wrong?.threshold || 0.9) * 100, 0)}%</span></div>
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
      <th scope="row">${esc(cropOf(r.class))} — ${esc(diseaseOf(r.class))}${r.low_support ? ' <span class="pill-flag" title="Ít ảnh validation: recall dao động mạnh">ít ảnh</span>' : ""}</th>
      <td class="mono">${fmtInt(r.support)}</td>
      <td>${recallBar(r.fedavg_recall, "fedavg")}</td>
      <td>${recallBar(r.centralized_recall, "centralized")}</td>
      <td class="mono">${fmtPct(r.fedavg_pv_recall, 0)}</td>
      <td class="mono ${r.domain_gap_pp >= 50 ? "text-rose" : ""}">${fmtNum(r.domain_gap_pp, 1)}</td>
    </tr>`;
  }).join("");

  root.innerHTML = `
    <section class="stat-tiles">
      <article class="stat-tile"><b>${esc(pd.classes_covered)}</b><span>lớp có ảnh PlantDoc trong validation</span></article>
      <article class="stat-tile warn"><b>${fmtInt(pd.weak_images)}</b><span>ảnh thuộc ${pd.weak_classes.length} lớp mà FedAvg nhận đúng dưới ${fmtNum((pd.weak_threshold || 0.5) * 100, 0)}%</span></article>
      <article class="stat-tile"><b>${pd.fedavg_worse_in}/${pd.classes_compared}</b><span>lớp FedAvg kém Centralized (tốt hơn ở ${pd.fedavg_better_in} lớp)</span></article>
      ${gap ? `<article class="stat-tile"><b>${fmtNum(gap["FedAvg v7"].gap_pp, 1)} điểm</b><span>FedAvg giảm từ PlantVillage xuống PlantDoc (Centralized: ${fmtNum(gap["Centralized v6"].gap_pp, 1)})</span></article>` : ""}
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Sai ở đâu: nhầm cây hay nhầm bệnh?</h3>
      <p class="panel-desc">Chia ${fmtInt(pd.models["FedAvg v7"].count)} ảnh PlantDoc theo kết quả dự đoán.</p>
      <div class="legend-row mt-3"><span><i class="swatch seg-ok"></i>Đúng</span><span><i class="swatch seg-mid"></i>Đúng loại cây, sai bệnh</span><span><i class="swatch seg-bad"></i>Sai cả loại cây</span></div>
      <div class="mt-3">${breakdownBar("FedAvg v7", "fedavg", pd.models["FedAvg v7"])}${breakdownBar("Centralized v6", "centralized", pd.models["Centralized v6"])}</div>
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Recall theo lớp trên PlantDoc</h3>
      <p class="panel-desc">Chỉ các lớp có ảnh PlantDoc trong validation. Hàng tô nền là lớp FedAvg nhận đúng dưới ${fmtNum((pd.weak_threshold || 0.5) * 100, 0)}%. Cột "Chênh miền" = recall PlantVillage − recall PlantDoc của FedAvg (điểm phần trăm).</p>
      <div class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Lớp</th><th scope="col">Ảnh PD</th><th scope="col">FedAvg · PD (%)</th><th scope="col">Centralized · PD (%)</th><th scope="col">FedAvg · PV</th><th scope="col">Chênh miền</th></tr></thead>
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
      <b>Xác suất softmax hiển thị trên web chưa được hiệu chuẩn.</b>
      Trên PlantVillage độ tin cậy khá sát độ chính xác thực, nhưng trên PlantDoc các mô hình <b>tự tin hơn nhiều so với thực tế</b>:
      FedAvg báo trung bình ${fmtPct(re.models.find(m => m.model_id === "fedavg").calibration.plantdoc.mean_confidence, 0)} nhưng chỉ đúng
      ${fmtPct(re.models.find(m => m.model_id === "fedavg").calibration.plantdoc.accuracy, 0)}.
      Vì vậy một ảnh có độ tin cậy cao không có nghĩa là chắc đúng.
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">ECE: độ tin cậy lệch bao nhiêu so với thực tế</h3>
      <p class="panel-desc">ECE (Expected Calibration Error, ${re.bins} khoảng) càng thấp càng tốt. "Thô → chỉnh T" là trước và sau khi chia logit cho nhiệt độ T học trên tập calibration ${fmtInt(re.calibration_count)} ảnh riêng; temperature scaling không đổi dự đoán, chỉ đổi xác suất.</p>
      <div class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">T</th><th scope="col">ECE PlantVillage</th><th scope="col">ECE PlantDoc</th><th scope="col">PlantDoc: tin cậy TB / đúng</th><th scope="col">Sai mà tin cậy ≥ ${fmtNum(re.high_confidence_threshold * 100, 0)}% (PD)</th></tr></thead>
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
        <div class="segmented" role="radiogroup" aria-label="Nguồn validation">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-cal="plantdoc">PlantDoc</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-cal="plantvillage">PlantVillage</button>
        </div>
      </div>
      <div class="chart-container chart-tall mt-4"><canvas id="chart-reliability" aria-label="Biểu đồ độ tin cậy của FedAvg và Centralized"></canvas></div>
    </section>`;

  const colors = pillarColors();
  const draw = source => {
    const points = id => (re.reliability?.[id]?.[source] || []).filter(b => b.n > 0)
      .map(b => ({ x: +(b.confidence * 100).toFixed(1), y: +(b.accuracy * 100).toFixed(1), n: b.n }));
    makeChart(byId("chart-reliability"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "Hiệu chuẩn hoàn hảo", data: [{ x: 0, y: 0 }, { x: 100, y: 100 }], showLine: true, borderDash: [6, 6], borderColor: "#6b7280", pointRadius: 0, borderWidth: 1.5 },
          { label: "FedAvg v7", data: points("fedavg"), showLine: true, borderColor: colors.fedavg, backgroundColor: colors.fedavg, tension: 0.2 },
          { label: "Centralized v6", data: points("centralized"), showLine: true, borderColor: colors.centralized, backgroundColor: colors.centralized, tension: 0.2 },
        ],
      },
      options: {
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: "Độ tin cậy trung bình (%)" } },
          y: { min: 0, max: 100, title: { display: true, text: "Tỉ lệ đúng thực tế (%)" } },
        },
        plugins: {
          tooltip: { callbacks: { label: c => `${c.dataset.label}: tin cậy ${fmtNum(c.parsed.x, 0)}% → đúng ${fmtNum(c.parsed.y, 0)}%${c.raw.n ? ` (${fmtInt(c.raw.n)} ảnh)` : ""}` } },
        },
      },
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

const BUILDERS = { summary: buildSummary, plantdoc: buildPlantdoc, calibration: buildCalibration };

function activate(name) {
  $$(".subtab").forEach(tab => {
    const on = tab.dataset.subtab === name;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
  });
  $$(".subpanel").forEach(panel => panel.classList.toggle("is-active", panel.id === `result-${name}`));
  if (!built.has(name)) {
    try {
      BUILDERS[name]();
      built.add(name); // chỉ đánh dấu đã dựng khi thành công; lỗi thì lần bấm sau thử lại
    } catch (error) {
      console.error(`[Kết quả] Không dựng được tab ${name}:`, error);
      const panel = byId(`result-${name}`);
      if (panel) panel.innerHTML = `<div class="callout callout-error" role="alert"><b>Không dựng được tab này.</b> ${esc(error.message)}</div>`;
      notify("error", "Không dựng được tab", name, { detail: error.message });
    }
  }
}

export async function mount() {
  const [evidence, summary] = await Promise.all([getEvidence(), getSummary().catch(() => null)]);
  if (!evidence.available) {
    byId("result-summary").innerHTML = `<div class="callout callout-error" role="alert"><b>Chưa có bộ số liệu đã kiểm hash.</b> ${esc(evidence.reason || "")}</div>`;
    return;
  }
  ctx = { evidence, summary };
  $$(".subtab").forEach(tab => tab.addEventListener("click", () => activate(tab.dataset.subtab)));
  activate("summary");
}
