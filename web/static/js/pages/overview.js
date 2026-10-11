// Bước 1 · Tổng quan: câu hỏi, ba trụ cột, kết quả chính, độ tin cậy.
import { getEvidence, getSummary, modelsById } from "../core/store.js";
import { makeChart, pillarColors } from "../core/charts.js";
import {
  MODEL_LABELS, MODEL_ORDER, byId, esc, fmtInt, fmtNum, fmtPct, revealItems, setMetric, withAlpha,
} from "../core/dom.js";

const LOCALS = ["local_00", "local_01", "local_02", "local_03", "local_04"];

function unavailable(reason) {
  return `<div class="callout callout-error" role="alert"><b>Chưa có bộ số liệu đã kiểm hash.</b> ${esc(reason || "Chưa nhập artifact.")}
    Web không hiển thị số khi chưa đối soát được SHA-256 (không có số dự phòng).</div>`;
}

function range(values, digits = 1) {
  const lo = Math.min(...values) * 100;
  const hi = Math.max(...values) * 100;
  return `${fmtNum(lo, digits)}–${fmtNum(hi, digits)}%`;
}

function trustChips(summary) {
  const chips = [];
  chips.push(`<a class="trust-chip ok" href="#bang-chung"><b>SHA-256 đã đối soát</b><span>checkpoint và manifest khớp bundle công bố</span></a>`);
  const re = summary?.reeval;
  if (re?.available) {
    const main = re.main_models_exact ? "Centralized và FedAvg khớp tuyệt đối" : "có lệch ở mô hình chính";
    chips.push(`<a class="trust-chip ${re.all_within_tolerance ? "ok" : "warn"}" href="#bang-chung"><b>Đã tái đánh giá ${fmtInt(re.validation_count)} ảnh</b><span>${re.comparisons_exact}/${re.comparisons_total} phép so khớp tuyệt đối · ${main}</span></a>`);
  } else {
    chips.push(`<a class="trust-chip warn" href="#bang-chung"><b>Chưa tái đánh giá cục bộ</b><span>${esc(re?.reason || "thiếu báo cáo")}</span></a>`);
  }
  const lk = summary?.leakage;
  if (lk?.available) {
    chips.push(`<a class="trust-chip ${lk.passed ? "ok" : "bad"}" href="#bang-chung"><b>Audit rò rỉ: ${lk.passed ? "đạt" : "KHÔNG đạt"}</b><span>${fmtInt(lk.overlap_total)} trùng lặp giữa train/val/calibration · ${fmtInt(lk.images_verified)} ảnh kiểm byte</span></a>`);
  } else {
    chips.push(`<a class="trust-chip warn" href="#bang-chung"><b>Chưa audit rò rỉ</b><span>${esc(lk?.reason || "thiếu báo cáo")}</span></a>`);
  }
  const ho = summary?.holdout;
  if (ho?.available) {
    const fed = ho.models.find(x => x.model_id === "fedavg");
    chips.push(`<a class="trust-chip ok" href="#ket-qua"><b>Test giữ riêng ${fmtInt(ho.count)} ảnh</b><span>chưa từng dùng để chọn checkpoint · FedAvg ${fmtPct(fed?.plantvillage?.top1)} PV / ${fmtPct(fed?.plantdoc?.top1)} PD (bước 3)</span></a>`);
  }
  const cl = summary?.checklist;
  if (cl?.available) {
    chips.push(`<a class="trust-chip ${cl.blocked ? "warn" : "ok"}" href="#bang-chung"><b>${cl.done}/${cl.total} mục nghiệm thu xong</b><span>${cl.blocked} mục còn thiếu dữ liệu/máy huấn luyện (xem bước 6)</span></a>`);
  }
  return chips.join("");
}

function testLine(holdout, ids) {
  if (!holdout?.available) return "";
  const rows = holdout.models.filter(x => ids.includes(x.model_id));
  if (!rows.length) return "";
  const pv = rows.map(x => x.plantvillage.top1);
  const pd = rows.map(x => x.plantdoc.top1);
  const show = values => (values.length === 1 ? fmtPct(values[0]) : range(values));
  return `<p class="pillar-test" title="Tập ${fmtInt(holdout.count)} ảnh chưa từng dùng để chọn checkpoint">Test giữ riêng: PV <b>${show(pv)}</b> · PD <b>${show(pd)}</b></p>`;
}

export async function mount() {
  const root = byId("overview-root");
  const [evidence, summary] = await Promise.all([getEvidence(), getSummary().catch(() => null)]);
  if (!evidence.available) {
    root.innerHTML = unavailable(evidence.reason);
    return;
  }

  const m = modelsById(evidence);
  const central = m.centralized.metrics;
  const fed = m.fedavg.metrics;
  const locals = LOCALS.map(id => m[id].metrics);
  const pvLocals = locals.map(x => x.plantvillage.top1_accuracy);
  const pdLocals = locals.map(x => x.plantdoc.top1_accuracy);
  const bestLocalPv = Math.max(...pvLocals);
  const trainTotal = Object.values(evidence.shard_counts || {}).reduce((a, b) => a + b, 0);
  const val = evidence.validation_counts || {};

  const gapFedCentral = (central.plantvillage.top1_accuracy - fed.plantvillage.top1_accuracy) * 100;
  const fedOverLocal = (fed.plantvillage.top1_accuracy - bestLocalPv) * 100;
  const domainFed = (fed.plantvillage.top1_accuracy - fed.plantdoc.top1_accuracy) * 100;
  const domainCentral = (central.plantvillage.top1_accuracy - central.plantdoc.top1_accuracy) * 100;

  root.innerHTML = `
    <section class="hero">
      <p class="eyebrow">Câu hỏi nghiên cứu</p>
      <h2 class="hero-question">Có thể huấn luyện chung một mô hình chẩn đoán bệnh lá từ năm nơi mà không gửi ảnh về máy chủ không — và nó chính xác tới đâu so với gom hết ảnh về một chỗ?</h2>
      <p class="hero-note">Số lớn là validation (cũng là tập dùng để chọn checkpoint); dòng "Test giữ riêng" là tập legacy diagnostic test của release, không dùng để chọn checkpoint. Một kịch bản non-IID (α=0,1), một seed (42); chưa phải độ chính xác ngoài thực địa.</p>
    </section>

    <section class="pillar-grid" aria-label="Ba cách huấn luyện">
      <article class="pillar pillar-centralized">
        <header><span class="pillar-tag">Trần lý tưởng</span><h3>Centralized</h3></header>
        <p class="pillar-desc">Gom toàn bộ ảnh về một máy rồi huấn luyện. Chính xác nhất, nhưng ảnh phải rời khỏi nơi sinh ra nó.</p>
        <dl class="pillar-metrics">
          <div><dt>PlantVillage</dt><dd class="metric-big" id="m-central-pv">—</dd></div>
          <div><dt>PlantDoc</dt><dd class="metric-big" id="m-central-pd">—</dd></div>
        </dl>
        ${testLine(summary?.holdout, ["centralized"])}
        <p class="pillar-foot">Không bảo vệ riêng tư dữ liệu</p>
      </article>
      <article class="pillar pillar-fedavg">
        <header><span class="pillar-tag">Giải pháp của đồ án</span><h3>FedAvg</h3></header>
        <p class="pillar-desc">Năm nơi huấn luyện trên ảnh của mình, chỉ gửi trọng số; server lấy trung bình có trọng số theo số ảnh.</p>
        <dl class="pillar-metrics">
          <div><dt>PlantVillage</dt><dd class="metric-big" id="m-fed-pv">—</dd></div>
          <div><dt>PlantDoc</dt><dd class="metric-big" id="m-fed-pd">—</dd></div>
        </dl>
        ${testLine(summary?.holdout, ["fedavg"])}
        <p class="pillar-foot">Ảnh thô không rời khỏi client</p>
      </article>
      <article class="pillar pillar-local">
        <header><span class="pillar-tag">Không hợp tác</span><h3>Local-only</h3></header>
        <p class="pillar-desc">Mỗi nơi chỉ học dữ liệu của mình. Năm mô hình riêng biệt — không có "mô hình trung bình".</p>
        <dl class="pillar-metrics">
          <div><dt>PlantVillage</dt><dd class="metric-big">${range(pvLocals)}</dd></div>
          <div><dt>PlantDoc</dt><dd class="metric-big">${range(pdLocals)}</dd></div>
        </dl>
        ${testLine(summary?.holdout, LOCALS)}
        <p class="pillar-foot">Khoảng của 5 mô hình Local 00–04</p>
      </article>
    </section>

    <section class="conclusion-grid mt-4" aria-label="Kết luận chính">
      <article class="conclusion fedavg">
        <b>FedAvg gần Centralized</b>
        <p>Đạt <b>${fmtPct(fed.plantvillage.top1_accuracy)}</b> trên PlantVillage, chỉ kém Centralized <b>${fmtNum(gapFedCentral, 1)} điểm</b> — trong khi không ảnh nào rời khỏi client.</p>
      </article>
      <article class="conclusion local">
        <b>Học riêng thì thất bại</b>
        <p>Local-only chỉ đạt <b>${range(pvLocals)}</b> trên PlantVillage; FedAvg hơn mô hình riêng tốt nhất <b>${fmtNum(fedOverLocal, 1)} điểm</b>. Nguyên nhân nằm ở dữ liệu lệch nhãn (bước 2).</p>
      </article>
      <article class="conclusion warn">
        <b>Ra ngoài miền thì giảm mạnh</b>
        <p>Từ PlantVillage sang PlantDoc, Centralized giảm <b>${fmtNum(domainCentral, 1)} điểm</b> và FedAvg giảm <b>${fmtNum(domainFed, 1)} điểm</b>: đây là khoảng cách miền dữ liệu, không phải lỗi riêng của FL.</p>
      </article>
    </section>

    <section class="panel mt-4">
      <div class="panel-header-flex">
        <div>
          <h3 class="panel-title">Bảy mô hình trên cùng validation</h3>
          <p class="panel-desc">Top-1 accuracy. Cột đậm: PlantVillage (${fmtInt(val.plantvillage)} ảnh, cùng miền với train). Cột nhạt: PlantDoc (${fmtInt(val.plantdoc)} ảnh, ảnh ngoài đồng — khác miền).</p>
        </div>
      </div>
      <div class="chart-container chart-tall mt-4"><canvas id="chart-overview" aria-label="So sánh Top-1 của bảy mô hình"></canvas></div>
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Thiết lập thí nghiệm</h3>
      <dl class="fact-grid mt-3">
        <div><dt>Mô hình</dt><dd>MobileNetV3-Small, 38 lớp</dd></div>
        <div><dt>Client</dt><dd>5 (Flower, gRPC)</dd></div>
        <div><dt>Phân chia dữ liệu</dt><dd>Dirichlet α = 0,1 (lệch nhãn nặng)</dd></div>
        <div><dt>Ảnh huấn luyện</dt><dd>${fmtInt(trainTotal)} ảnh PlantVillage + PlantDoc</dd></div>
        <div><dt>Validation</dt><dd>${fmtInt(val.plantvillage)} + ${fmtInt(val.plantdoc)} ảnh</dd></div>
        <div><dt>Seed</dt><dd>${esc(evidence.seed)} (một seed)</dd></div>
      </dl>
    </section>

    <section class="mt-4">
      <h3 class="panel-title">Độ tin cậy của số liệu</h3>
      <div class="trust-grid mt-3">${trustChips(summary)}</div>
    </section>

    <section class="cta-row mt-4">
      <a class="btn btn-primary" href="#non-iid">Vì sao Local-only thất bại →</a>
      <a class="btn btn-secondary" href="#chan-doan">Thử chẩn đoán một ảnh →</a>
    </section>`;

  setMetric(byId("m-central-pv"), fmtPct(central.plantvillage.top1_accuracy));
  setMetric(byId("m-central-pd"), fmtPct(central.plantdoc.top1_accuracy));
  setMetric(byId("m-fed-pv"), fmtPct(fed.plantvillage.top1_accuracy));
  setMetric(byId("m-fed-pd"), fmtPct(fed.plantdoc.top1_accuracy));
  revealItems(root.querySelectorAll(".pillar, .conclusion"), { stagger: 60 });

  makeChart(byId("chart-overview"), () => {
    const colors = pillarColors();
    const colorOf = id => (id === "centralized" ? colors.centralized : id === "fedavg" ? colors.fedavg : colors.local);
    return {
    type: "bar",
    data: {
      labels: MODEL_ORDER.map(id => MODEL_LABELS[id]),
      datasets: [
        {
          label: "PlantVillage",
          data: MODEL_ORDER.map(id => +(m[id].metrics.plantvillage.top1_accuracy * 100).toFixed(2)),
          backgroundColor: MODEL_ORDER.map(colorOf),
          borderRadius: 6,
        },
        {
          label: "PlantDoc",
          data: MODEL_ORDER.map(id => +(m[id].metrics.plantdoc.top1_accuracy * 100).toFixed(2)),
          backgroundColor: MODEL_ORDER.map(id => withAlpha(colorOf(id), 0.45)),
          borderColor: MODEL_ORDER.map(colorOf),
          borderWidth: 1.5,
          borderRadius: 6,
        },
      ],
    },
    options: {
      scales: {
        x: { grid: { display: false } },
        y: { min: 0, max: 100, ticks: { callback: v => `${v}%` } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}%` } },
      },
    },
    };
  });
}
