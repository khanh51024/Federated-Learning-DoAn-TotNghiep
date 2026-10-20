// Chương 4 · Kết quả: tổng hợp 7 mô hình, tập giữ riêng và thống kê, phân tích lỗi, độ tin cậy của xác suất.
import { getDetails, getEvidence, getSummary, modelsById } from "../core/store.js";
import { attachCrosshair } from "../core/crosshair.js";
import { barRise, lineDraw, makeChartWhenVisible, pillarColors } from "../core/charts.js";
import { play } from "../core/motion.js";
import { chartHtml } from "../core/comm-weights.js";
import {
  applyStyleData, $$, MODEL_LABELS, MODEL_ORDER, byId, diseaseOf, cropOf, esc, fmtInt, fmtNum, fmtPct,
  formatDiseaseName, modelPill, notify, pillarOf, sectionHead, withAlpha,
} from "../core/dom.js";

const built = new Set();
let ctx = null; // { evidence, summary }

function notReady() {
  return '<div class="callout callout-warn" role="note"><b>Phần này chưa có dữ liệu.</b> Báo cáo tương ứng chưa được sinh trên máy chủ.</div>';
}

// ───────── Tab 1: Tổng hợp ─────────
function buildSummary() {
  const { evidence, summary } = ctx;
  const root = byId("result-summary");
  const byModel = modelsById(evidence);
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
      <td${bold(mm.plantvillage.top1_accuracy, best.pvTop1)}>${fmtPct(mm.plantvillage.top1_accuracy)}</td>
      <td>${fmtNum(mm.plantvillage.macro_f1_supported, 4)}</td>
      <td${bold(mm.plantdoc.top1_accuracy, best.pdTop1)}>${fmtPct(mm.plantdoc.top1_accuracy)}</td>
      <td>${fmtNum(mm.plantdoc.macro_f1_supported, 4)}</td>
      <td>${fmtPct(mm.plantdoc.crop_accuracy, 1)}</td>
    </tr>`;
  }).join("");

  const covered = summary?.plantdoc?.available ? summary.plantdoc.classes_covered : null;
  root.innerHTML = `
    <section class="panel">
      <div class="panel-header-flex"><div>${sectionHead("4.1", "Tổng hợp: so sánh ba phương pháp huấn luyện")}<p class="panel-desc">1 mô hình gom ảnh · 1 mô hình học phân tán · 5 mô hình học riêng</p></div>
        <div class="segmented" role="radiogroup" aria-label="Nguồn ảnh so sánh">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-result-source="plantvillage">PlantVillage</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-result-source="plantdoc">PlantDoc · ngoài đồng</button>
        </div>
      </div>
      <p class="field-hint mt-2" id="result-comparison-source"></p>
      <div class="viewer-comparison mt-3" id="result-key-comparison"></div>
      <p class="viewer-takeaway mt-3" id="result-takeaway" aria-live="polite"></p>
      <p class="viewer-scope mt-2"><b class="scope-tag">Phạm vi</b> Một kịch bản, một lần chạy; tập này cũng dùng để chọn mô hình, nên chưa phải độ chính xác ngoài thực địa. <a href="#ket-luan">Xem hạn chế ở chương 5</a></p>
    </section>

    <section class="panel mt-4" aria-labelledby="result-chart-title">
      <div class="overview-section-heading">
        <div><h4 class="panel-title" id="result-chart-title">Bảy mô hình trên hai nguồn ảnh</h4></div>
        <span class="overview-chart-unit">Tỉ lệ đoán đúng · %</span>
      </div>
      <div class="chart-container chart-tall mt-3"><canvas id="chart-overview" aria-label="So sánh tỉ lệ đoán đúng của bảy mô hình"></canvas></div>
      <p class="field-hint mt-2">Cột đậm: PlantVillage (${fmtInt(evidence.validation_counts.plantvillage)} ảnh). Cột nhạt: PlantDoc (${fmtInt(evidence.validation_counts.plantdoc)} ảnh ngoài đồng).</p>
    </section>

    <section class="panel mt-4" id="result-full-table" aria-labelledby="result-table-title">
      <h4 class="panel-title" id="result-table-title">Bảng đầy đủ của 7 mô hình</h4>
      <p class="panel-desc">PlantVillage ${fmtInt(evidence.validation_counts.plantvillage)} ảnh (cùng kiểu với ảnh học) và PlantDoc ${fmtInt(evidence.validation_counts.plantdoc)} ảnh (ảnh ngoài đồng). Số đậm là cao nhất mỗi cột.</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table result-table">
          <thead>
            <tr>
              <th scope="col">Mô hình</th>
              <th scope="col">Ảnh phòng (PlantVillage): đúng</th><th scope="col">Macro-F1</th>
              <th scope="col">Ảnh ngoài đồng (PlantDoc): đúng</th><th scope="col">Macro-F1</th>
              <th scope="col">Đúng loại cây (ngoài đồng)</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="field-hint mt-3"><b>Cách đọc:</b> “đúng” là tỉ lệ ảnh mà mô hình đoán đúng bệnh (Top-1). Macro-F1 là điểm trung bình công bằng giữa các nhóm (không bị nhóm nhiều ảnh lấn át), tính trên các nhóm có ảnh kiểm tra${covered ? `; PlantDoc chỉ phủ ${esc(covered)} nhóm` : ""}. Cách chia tập và các thuật ngữ nằm ở <a href="#du-lieu">chương 2</a> và <a href="#phuong-phap">chương 3</a>.</p>
    </section>

    <details class="overview-detail mt-4">
      <summary><span><strong>Lịch sử huấn luyện Centralized</strong><small>Đường độ chính xác và sai số theo epoch</small></span></summary>
      <div class="overview-detail-body">
      <h4 class="panel-title">Quá trình huấn luyện Centralized v6</h4>
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

  const val = evidence.validation_counts || {};
  makeChartWhenVisible(byId("chart-overview"), () => {
    const colors = pillarColors();
    const colorOf = id => (id === "centralized" ? colors.centralized : id === "fedavg" ? colors.fedavg : colors.local);
    return {
      type: "bar",
      data: {
        labels: MODEL_ORDER.map(id => MODEL_LABELS[id]),
        datasets: [
          {
            label: "PlantVillage",
            data: MODEL_ORDER.map(id => +(byModel[id].metrics.plantvillage.top1_accuracy * 100).toFixed(2)),
            backgroundColor: MODEL_ORDER.map(colorOf),
            borderRadius: 6,
          },
          {
            label: "PlantDoc",
            data: MODEL_ORDER.map(id => +(byModel[id].metrics.plantdoc.top1_accuracy * 100).toFixed(2)),
            backgroundColor: MODEL_ORDER.map(id => withAlpha(colorOf(id), 0.45)),
            borderColor: MODEL_ORDER.map(colorOf),
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        animation: barRise(),
        scales: {
          x: { grid: { display: false } },
          y: { min: 0, max: 100, ticks: { callback: v => `${v}%` } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: items => {
                const id = MODEL_ORDER[items[0].dataIndex];
                const role = id === "centralized" ? "gom ảnh về một chỗ" : id === "fedavg" ? "học phân tán" : "mỗi nơi tự học";
                return `${MODEL_LABELS[id]} · ${role}`;
              },
              label: ctx => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}%`,
              footer: () => `Tập kiểm tra: ${fmtInt(val.plantvillage)} ảnh PlantVillage · ${fmtInt(val.plantdoc)} ảnh PlantDoc`,
            },
          },
        },
      },
    };
  });

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

// ───────── Phân tích lỗi · 4.3.1: PlantDoc theo nhóm ─────────
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
  const root = byId("errors-plantdoc");
  const pd = summary?.plantdoc;
  if (!pd?.available) {
    root.innerHTML = notReady();
    return;
  }
  const gap = summary?.domain_gap?.available ? summary.domain_gap.models : null;
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
      <h5 class="panel-title">Phân loại lỗi: sai loại cây hoặc sai bệnh</h5>
      <p class="panel-desc">Chia ${fmtInt(pd.models["FedAvg v7"].count)} ảnh PlantDoc theo kết quả dự đoán.</p>
      <div class="legend-row mt-3"><span><i class="swatch seg-ok"></i>Đúng</span><span><i class="swatch seg-mid"></i>Đúng loại cây, sai bệnh</span><span><i class="swatch seg-bad"></i>Sai cả loại cây</span></div>
      <div class="mt-3">${breakdownBar("FedAvg v7", "fedavg", pd.models["FedAvg v7"])}${breakdownBar("Centralized v6", "centralized", pd.models["Centralized v6"])}</div>
    </section>

    <section class="panel mt-4">
      <h5 class="panel-title">Tỉ lệ nhận đúng theo từng nhóm (ảnh ngoài đồng PlantDoc)</h5>
      <p class="panel-desc">Chỉ các nhóm có ảnh PlantDoc trong tập kiểm tra. Hàng tô nền là nhóm FedAvg nhận đúng dưới ${fmtNum((pd.weak_threshold || 0.5) * 100, 0)}%. Cột “Chênh lệch miền” = tỉ lệ nhận đúng ở ảnh PlantVillage trừ tỉ lệ ở ảnh PlantDoc của FedAvg (điểm phần trăm).</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Nhóm</th><th scope="col">Số ảnh</th><th scope="col">FedAvg (%)</th><th scope="col">Centralized (%)</th><th scope="col">FedAvg trên ảnh trong phòng</th><th scope="col">Chênh lệch miền (PlantVillage − PlantDoc)</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>`;
  applyStyleData(root); // riseBars đọc --s ngay bên dưới nên không đợi MutationObserver
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

// ───────── Tab Độ tin cậy xác suất ─────────
function buildCalibration() {
  const { summary } = ctx;
  const root = byId("result-calibration");
  const re = summary?.reeval;
  if (!re?.available) {
    root.innerHTML = notReady();
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
    ${sectionHead("4.4", "Độ tin cậy của xác suất")}
    <section class="callout callout-warn mt-3">
      <b>Xác suất softmax thô của các mô hình chưa được hiệu chuẩn.</b> Trang Chẩn đoán ảnh hiện cả số thô lẫn số đã hiệu chuẩn bằng temperature scaling (T ở bảng dưới).
      Trên PlantVillage độ tin cậy khá sát độ chính xác thực, nhưng trên PlantDoc các mô hình <b>gán xác suất cao hơn nhiều so với tỉ lệ đúng thực tế</b>:
      FedAvg báo trung bình ${fmtPct(re.models.find(m => m.model_id === "fedavg").calibration.plantdoc.mean_confidence, 0)} nhưng chỉ đúng
      ${fmtPct(re.models.find(m => m.model_id === "fedavg").calibration.plantdoc.accuracy, 0)}.
      Việc hiệu chuẩn (temperature scaling) học trên tập ảnh chủ yếu là PlantVillage nên chỉ sửa được một phần sai lệch trên PlantDoc (ECE vẫn cao).
      Vì vậy một ảnh có độ tin cậy cao không có nghĩa là chắc đúng.
    </section>

    <section class="panel mt-4">
      <h4 class="panel-title">ECE: độ tin cậy lệch bao nhiêu so với thực tế</h4>
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
          <h4 class="panel-title">Biểu đồ độ tin cậy</h4>
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

// ───────── Tab Tập giữ riêng và thống kê ─────────
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
    root.innerHTML = notReady();
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
        <h5 class="subhead">${title}</h5>
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
    ${sectionHead("4.2", "Tập giữ riêng và thống kê")}
    <p class="viewer-scope mt-2">Tập giữ riêng không dùng để chọn mô hình, nhưng chưa phải kiểm thử cuối được khóa trước và chưa nằm trong kiểm tra trùng ảnh (định nghĩa ở <a href="#du-lieu">chương 2</a>).</p>

    <section class="panel mt-3">
      <h4 class="panel-title">Tập kiểm tra và tập giữ riêng, kèm khoảng tin cậy 95%</h4>
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
      ${recordedText ? `<p class="field-hint mt-2">Đối chiếu: số Centralized trên tập giữ riêng do nhóm huấn luyện tự ghi — ${recordedText} (số gốc làm tròn 4 chữ số).</p>` : ""}
    </section>

    <details class="overview-detail mt-4">
      <summary><span><strong>Cách chọn mô hình và phương pháp thống kê</strong><small>Phạm vi tập giữ riêng, bootstrap theo cụm</small></span></summary>
      <div class="overview-detail-body">
    ${sel?.available ? `
    <section class="callout callout-warn">
      <b>Vì sao cần tập giữ riêng:</b>
      <ul>
        <li>Phiên bản mô hình được chọn theo điểm Macro-F1 PlantDoc ngay trên <b>chính tập kiểm tra dùng để báo cáo</b>, nên con số ở đó có thể hơi đẹp hơn thực tế (số liệu về epoch được chọn ở <a href="#ket-luan">chương 5</a>).</li>
        <li>Vì vậy đồ án chấm lại thêm trên ${fmtInt(ho?.count)} ảnh <b>giữ riêng</b>, không dùng để chọn mô hình nên không bị thiên lệch này. ${valVsTest} Với FedAvg: web chưa có đủ lịch sử từng vòng để đo hiệu ứng chọn mô hình.</li>
        <li><b>Lưu ý:</b> tập giữ riêng chưa nằm trong phép kiểm tra trùng ảnh; một phần ảnh PlantVillage trong tập này có thể là ảnh chụp lại cùng lá với ảnh học (xem mục “Độ nhạy” bên dưới).</li>
      </ul>
    </section>` : ""}
      <p class="field-hint mt-3">Khoảng tin cậy tính bằng bootstrap theo cụm: PlantDoc theo cảnh chụp, PlantVillage theo nhóm lá; nhiều ảnh cùng cảnh/lá không độc lập nhau.</p>
      </div>
    </details>

    ${val || test ? `
    <details class="overview-detail mt-4"><summary><span><strong>So sánh theo cặp trên cùng ảnh</strong><small>Hiệu số giữa hai mô hình, khoảng tin cậy, McNemar</small></span></summary><div class="overview-detail-body">
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
    <details class="overview-detail mt-4"><summary><span><strong>Độ nhạy với ảnh nghi chụp lại cùng một lá</strong><small>PlantVillage, tập giữ riêng</small></span></summary><div class="overview-detail-body">
      <p class="panel-desc">${fmtInt(leaf.ungrouped_images)} ảnh PlantVillage trong tập giữ riêng không được gắn nhóm theo lá; ${fmtInt(leaf.suspected_reshoots)} ảnh trong đó có ảnh học cùng nhóm mang số thứ tự liền kề (ước lượng ảnh chụp lại cùng lá — có thể làm số đẹp hơn thật). Nếu bỏ chúng mà độ chính xác không giảm thì chưa thấy bằng chứng con số bị phóng đại.</p>
      <div tabindex="0" role="group" aria-label="Bảng số liệu, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
        <table class="data-table">
          <thead><tr><th scope="col">Mô hình</th><th scope="col">Tất cả</th><th scope="col">Chỉ ảnh đã gom nhóm lá</th><th scope="col">Bỏ ảnh nghi chụp lại</th></tr></thead>
          <tbody>${leafRows}</tbody>
        </table>
      </div>
    </div></details>` : ""}`;
}

// ───────── Phân tích lỗi · 4.3.2: ma trận nhầm lẫn ─────────
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
  const root = byId("errors-confusion");
  root.innerHTML = '<div class="skeleton skeleton-block"></div>';
  const details = await getDetails();
  if (!details.available || !details.confusion?.val) {
    root.innerHTML = notReady();
    return;
  }
  const state = { model: "fedavg", source: "plantdoc", split: "val" };
  const splits = Object.keys(details.confusion);
  root.innerHTML = `
    <section class="panel">
      <div class="panel-header-flex">
        <div>
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
      <h5 class="panel-title">Nhầm lẫn nhiều nhất</h5>
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

// ───────── Phân tích lỗi · 4.3.3: dự đoán sai với xác suất cao ─────────
const ERR_PAGE = 20; // số thẻ hiện mỗi trang (UI-01): tab không còn dựng cả 175 ảnh một lần
async function buildErrors() {
  const root = byId("errors-confident");
  root.innerHTML = '<div class="skeleton skeleton-block"></div>';
  const details = await getDetails();
  const errors = details.errors;
  if (!details.available || !errors?.available) {
    root.innerHTML = notReady();
    return;
  }
  const items = errors.items;
  const floor = Math.round((errors.threshold || 0.9) * 100); // ngưỡng mà thư viện ảnh được dựng theo
  const levels = [...new Set([floor, 95, 99].filter(value => value >= floor))];
  const state = { model: "all", source: "all", min: floor, shown: ERR_PAGE };

  // Tóm tắt trên TOÀN BỘ thư viện (không đổi theo bộ lọc): giúp thấy ngay lỗi nào đáng chú ý trước khi cuộn qua từng ảnh.
  const fieldCount = items.filter(item => item.source === "plantdoc").length;
  const fedavgWrong = items.filter(item => item.wrong_by.fedavg).length;

  root.innerHTML = `
    <section class="panel">
      <p class="panel-desc">Ảnh kiểm tra bị dự đoán sai dù mô hình báo xác suất từ ${fmtNum(floor, 0)}% trở lên. Phần lớn là ảnh PlantDoc chụp ngoài đồng, nơi xác suất dự đoán lệch xa nhất so với độ chính xác thực.</p>
      <div class="stat-tiles mt-3">
        <article class="stat-tile"><b>${fmtInt(items.length)}</b><span>ảnh sai với xác suất ≥ ${fmtNum(floor, 0)}% (ít nhất một trong hai mô hình)</span></article>
        <article class="stat-tile warn"><b>${fmtPct(fieldCount / Math.max(items.length, 1), 0)}</b><span>là ảnh PlantDoc chụp ngoài đồng (${fmtInt(fieldCount)} ảnh)</span></article>
        <article class="stat-tile"><b>${fmtInt(fedavgWrong)}</b><span>ảnh trong đó FedAvg sai</span></article>
      </div>
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

/** Dựng một phần của tab "Phân tích lỗi" trong khung của nó; lỗi chỉ hiện ở khung đó, kèm nút thử lại. */
async function buildIsolated(containerId, builder) {
  try {
    await builder();
  } catch (error) {
    console.error(`[Kết quả] Không dựng được ${containerId}:`, error);
    const box = byId(containerId);
    box.innerHTML = `<div class="callout callout-error" role="alert"><b>Không dựng được phần này.</b> ${esc(error.message)} <button type="button" class="btn btn-secondary btn-sm">Thử lại</button></div>`;
    box.querySelector("button").addEventListener("click", () => buildIsolated(containerId, builder));
  }
}

/** Tab "Phân tích lỗi": ba phần dựng độc lập (một phần lỗi không làm hỏng hai phần kia). */
async function buildErrorAnalysis() {
  await Promise.all([
    buildIsolated("errors-plantdoc", buildPlantdoc),
    buildIsolated("errors-confusion", buildConfusion),
    buildIsolated("errors-confident", buildErrors),
  ]);
}

// ───────── Tab 5: Ba cách nhận diện ─────────
const SCENARIO_LABEL = {
  iid: "Ảnh bất kỳ (IID)", three_groups: "Mỗi người 3 nhóm bệnh", one_group: "Mỗi người 1 nhóm bệnh", plantdoc_only: "Chỉ ảnh ngoài đồng (PlantDoc)",
};
const signed = (value, digits = 2) => (value === null || value === undefined ? "—" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${fmtNum(Math.abs(value), digits)}`);

function pvSplitRow(label, block) {
  if (!block) return "";
  const ci = block.top1_ci95;
  return `<tr><th scope="row">${esc(label)}</th><td class="mono">${fmtInt(block.count)}</td>
    <td class="mono"><b>${fmtPct(block.top1_accuracy, 1)}</b>${ci && ci.low !== null ? `<span class="row-sub">${fmtPct(ci.low, 1)} đến ${fmtPct(ci.high, 1)}</span>` : ""}</td>
    <td class="mono">${fmtNum(block.macro_f1_supported, 3)}</td><td class="mono">${fmtPct(block.crop_accuracy, 1)}</td></tr>`;
}

function pvSection(pv) {
  if (!pv?.available) return `<section class="report-sub mt-5">${sectionHead("4.5.2", "PlantVillage", { tag: "h4" })}${notReady()}</section>`;
  const h = pv.holdout.by_source;
  const v = pv.validation.by_source;
  const cmp = pv.vs_fedavg_on_holdout || {};
  const pre = pv.preprocessing_comparison_on_val || {};
  const line = (src, name) => {
    const c = cmp[src];
    if (!c) return "";
    return `<tr><th scope="row">${name}</th><td class="mono">${fmtInt(c.count)}</td><td class="mono">${fmtPct(c.plantvillage_model_top1, 1)}</td>
      <td class="mono">${fmtPct(c.fedavg_top1, 1)}</td><td class="mono">${c.mcnemar.a_right_b_wrong} / ${c.mcnemar.a_wrong_b_right}</td>
      <td class="mono">${c.mcnemar.p_value < 0.001 ? "&lt; 0,001" : fmtNum(c.mcnemar.p_value, 3)}</td></tr>`;
  };
  return `<section class="report-sub mt-5" aria-labelledby="three-pv">
    ${sectionHead("4.5.2", "PlantVillage: mô hình ngoài, đánh giá lại, không huấn luyện lại", { id: "three-pv", tag: "h4" })}
    <p class="section-lead">Mô hình là <b>${esc(pv.model.origin)}</b> (${esc(pv.model.architecture)}), học trên ${esc(pv.model.training_data)}. Đồ án không huấn luyện lại mà chấm nó trên đúng các tập kiểm tra đã kiểm hash của release (nhiệt độ học trên tập hiệu chuẩn).</p>
    <div tabindex="0" role="group" aria-label="Bảng đánh giá mô hình PlantVillage, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
      <table class="data-table wrap-head"><thead><tr><th scope="col">Tập</th><th scope="col">Số ảnh</th><th scope="col">Top-1 (khoảng tin cậy 95%)</th><th scope="col">Macro-F1</th><th scope="col">Đúng loại cây</th></tr></thead>
      <tbody>${pvSplitRow("PlantDoc, test giữ riêng", h.plantdoc)}${pvSplitRow("PlantVillage, test giữ riêng", h.plantvillage)}${pvSplitRow("PlantDoc, validation", v.plantdoc)}${pvSplitRow("PlantVillage, validation", v.plantvillage)}</tbody></table></div>
    <div class="callout callout-warn mt-3" role="note"><b>Con số PlantVillage không dùng để so sánh.</b> ${esc(pv.leakage?.note || "")}
      Ở PlantDoc, nơi so sánh được, mô hình này kém xa FedAvg v7.</div>
    <h5 class="subhead mt-4">So với FedAvg v7 trên cùng ảnh test giữ riêng (kiểm định McNemar chính xác)</h5>
    <div tabindex="0" role="group" aria-label="Bảng so sánh với FedAvg" class="table-responsive mt-2">
      <table class="data-table wrap-head"><thead><tr><th scope="col">Nguồn ảnh</th><th scope="col">Số ảnh</th><th scope="col">PlantVillage</th><th scope="col">FedAvg v7</th><th scope="col">Chỉ PV đúng / chỉ FedAvg đúng</th><th scope="col">p</th></tr></thead>
      <tbody>${line("plantdoc", "PlantDoc (ảnh ngoài đồng)")}${line("plantvillage", "PlantVillage (ảnh trong phòng)")}</tbody></table></div>
    <p class="field-hint mt-2">Tiền xử lý đúng của tác giả (mean = std = 0,5) cho Top-1 validation ${fmtPct(pre.hf_half_256_224?.val_top1, 1)}; chuẩn hóa ImageNet chỉ ${fmtPct(pre.imagenet_256_224?.val_top1, 1)}, nên web dùng cách của tác giả.
      ECE trên PlantDoc ${fmtNum(pv.holdout.ece?.plantdoc, 3)}: mô hình quá tự tin khi ảnh ngoài đồng. Nhiệt độ T = ${fmtNum(pv.holdout.temperature, 2)} làm ECE chung xấu đi (${fmtNum(pv.holdout.ece?.overall, 3)} khi dùng xác suất thô), nên web hiện xác suất thô.</p>
  </section>`;
}

function simTable(sim) {
  const rows = sim.runs.filter(r => r.gate).sort((a, b) => a.scenario.localeCompare(b.scenario) || a.k - b.k);
  const order = ["iid", "three_groups", "one_group", "plantdoc_only"];
  const body = order.flatMap(sc => rows.filter(r => r.scenario === sc)).map(r => `<tr>
      <th scope="row">${esc(SCENARIO_LABEL[r.scenario])}</th><td class="mono">${r.k}</td>
      <td class="mono"><b>${signed(r.final_delta_mean_points.accuracy)}</b><span class="row-sub">±${fmtNum(r.final_std.accuracy * 100, 2)}</span></td>
      <td class="mono">${signed(r.final_delta_mean_points.macro_recall)}</td>
      <td class="mono">${signed(r.final_delta_mean_points.plantdoc_accuracy)}</td>
      <td class="mono">${fmtPct(r.accepted_round_share, 0)}</td><td class="mono">${fmtPct(r.clipped_share, 0)}</td></tr>`).join("");
  return `<div tabindex="0" role="group" aria-label="Bảng kết quả mô phỏng cộng đồng, cuộn được bằng phím mũi tên" class="table-responsive mt-3">
    <table class="data-table wrap-head"><thead><tr><th scope="col">Người tham gia</th><th scope="col">K</th><th scope="col">Chính xác (điểm %, ±độ lệch giữa seed)</th>
      <th scope="col">Macro-recall</th><th scope="col">PlantDoc</th><th scope="col">Vòng được nhận</th><th scope="col">Cập nhật bị cắt</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

function simSection(sim) {
  if (!sim?.available) return `<section class="report-sub mt-5">${sectionHead("4.5.3", "Cộng đồng", { tag: "h4" })}${notReady()}</section>`;
  const base = sim.base_test;
  const noGate = sim.runs.filter(r => !r.gate);
  const gateOf = r => sim.runs.find(g => g.gate && g.scenario === r.scenario && g.k === r.k);
  const control = noGate.map(r => {
    const g = gateOf(r);
    return `<li>${esc(SCENARIO_LABEL[r.scenario])}, K = ${r.k}: PlantDoc ${signed(r.final_delta_mean_points.plantdoc_accuracy)} điểm khi <b>không</b> có cổng, ${g ? `${signed(g.final_delta_mean_points.plantdoc_accuracy)} khi có cổng` : ""}; chính xác ${signed(r.final_delta_mean_points.accuracy)} và ${g ? signed(g.final_delta_mean_points.accuracy) : "—"}.</li>`;
  }).join("");
  const options = sim.runs.filter(r => r.gate).map(r => `<option value="${esc(r.key)}">${esc(SCENARIO_LABEL[r.scenario])}, K = ${r.k}</option>`).join("");
  return `<section class="report-sub mt-5" aria-labelledby="three-comm">
    ${sectionHead("4.5.3", "Cộng đồng: thí nghiệm mô phỏng học phân tán", { id: "three-comm", tag: "h4" })}
    <p class="section-lead">${esc(sim.pool.images.toLocaleString("vi-VN"))} ảnh của kho hiệu chuẩn (không giao train và validation) chia cho K người mô phỏng, mỗi người ${sim.local_config.images_per_user[0]} đến ${sim.local_config.images_per_user[1]} ảnh,
      học lớp cuối ${sim.local_config.epochs} epoch bằng đúng thuật toán của trình duyệt, máy chủ kiểm, cắt chuẩn, gộp FedAvg và qua cổng chất lượng như bản chạy thật. ${sim.rounds} vòng, ${sim.seeds.length} seed.
      Số đo trên <b>test giữ riêng ${fmtInt(sim.test.images)} ảnh</b>; mô hình gốc FedAvg v7: chính xác ${fmtPct(base.accuracy, 2)}, macro-recall ${fmtPct(base.macro_recall, 2)}, PlantDoc ${fmtPct(base.plantdoc_accuracy, 1)} (${fmtInt(base.plantdoc_count)} ảnh).</p>
    ${simTable(sim)}
    <div class="form-group mt-3"><label for="sim-run">Thay đổi theo vòng cho</label><select id="sim-run" class="input-field">${options}</select></div>
    <div id="sim-chart" class="mt-2"></div>
    <h5 class="subhead mt-4">Cổng chất lượng có tác dụng gì (đối chứng K = 10)</h5>
    <ul class="local-why">${control}</ul>
    <div class="callout mt-3" role="note"><b>Cách đọc.</b> Dữ liệu người tham gia đủ đa dạng thì mô hình chung nhích lên khoảng 1 đến 2 điểm phần trăm; khi mỗi người chỉ có một nhóm bệnh thì cần K lớn hơn;
      với ảnh ngoài đồng (PlantDoc) hầu như không tăng vì phần trích đặc trưng bị đóng băng. ${esc(sim.caveat || "")}</div>
  </section>`;
}

function buildThree() {
  const root = byId("result-three");
  const summary = ctx.summary;
  root.innerHTML = `${sectionHead("4.5", "Ba cách nhận diện ngang hàng", { lead: "Trang Chẩn đoán ảnh chạy cả ba cách trên cùng một ảnh. Đều là cách chính của đồ án, đều dùng bản mô hình mới nhất; phần này ghi số đo của từng cách." })}
    <section class="report-sub mt-4" aria-labelledby="three-local">
      ${sectionHead("4.5.1", "Local only", { id: "three-local", tag: "h4" })}
      <p class="section-lead">Năm client tự huấn luyện, không chia sẻ trọng số; số đo đầy đủ ở mục 4.1 và 4.2. Mỗi client chỉ nhận đúng các nhóm bệnh nó từng thấy.</p>
    </section>
    ${pvSection(summary?.plantvillage_reference)}
    ${simSection(summary?.community_simulation)}`;
  const sim = summary?.community_simulation;
  const select = byId("sim-run");
  if (!sim?.available || !select) return;
  const draw = () => {
    const run = sim.runs.find(r => r.key === select.value);
    const versions = run.mean_curve.accuracy.map((_, i) => ({
      version: i, metrics: { accuracy: run.mean_curve.accuracy[i], macro_recall: run.mean_curve.macro_recall[i], plantdoc_accuracy: run.mean_curve.plantdoc_accuracy[i] },
    }));
    byId("sim-chart").innerHTML = chartHtml(versions);
  };
  select.addEventListener("change", draw);
  draw();
}

const BUILDERS = {
  summary: buildSummary, holdout: buildHoldout, errors: buildErrorAnalysis, calibration: buildCalibration, three: buildThree,
};
const building = new Map();

async function activate(name) {
  $$("#page-ket-qua .subtab").forEach(tab => {
    const on = tab.dataset.subtab === name;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
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
  $$("#page-ket-qua .subtab").forEach(tab => tab.addEventListener("click", () => activate(tab.dataset.subtab)));
  if (!evidence.available) {
    const callout = role => `<div class="callout callout-error"${role}><b>Chưa có số liệu để hiển thị.</b> ${esc(evidence.reason || "")}</div>`;
    $$(".subpanel").forEach(panel => { panel.innerHTML = callout(panel.id === "result-summary" ? ' role="alert"' : ""); });
    return;
  }
  ctx = { evidence, summary };
  activate("summary");
}
