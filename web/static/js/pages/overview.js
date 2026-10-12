// Bước 1 · Tổng quan: câu hỏi, ba trụ cột, kết quả chính, độ tin cậy.
import { getEvidence, getSummary, modelsById } from "../core/store.js";
import { makeChart, pillarColors } from "../core/charts.js";
import {
  MODEL_LABELS, MODEL_ORDER, byId, esc, fmtInt, fmtNum, fmtPct, revealItems, setMetric, withAlpha,
} from "../core/dom.js";

const LOCALS = ["local_00", "local_01", "local_02", "local_03", "local_04"];

function unavailable(reason) {
  return `<div class="callout callout-error" role="alert"><b>Chưa có bộ số liệu đã kiểm hash.</b> ${esc(reason || "Chưa nạp bộ số liệu.")}
    Web không hiển thị số khi chưa đối soát được SHA-256 (không có số dự phòng).</div>`;
}

function range(values, digits = 1) {
  const lo = Math.min(...values) * 100;
  const hi = Math.max(...values) * 100;
  return `${fmtNum(lo, digits)}–${fmtNum(hi, digits)}%`;
}

function trustChips(summary) {
  const chips = [];
  const prov = (summary?.checklist?.items || []).find(i => i.id === "provenance");
  const provOpen = prov && prov.status !== "XONG";
  chips.push(`<a class="trust-chip ok" href="#bang-chung"><b>Mô hình đúng bản công bố</b><span>mã băm SHA-256 của mô hình và dữ liệu đã được đối chiếu${provOpen ? " · nguồn huấn luyện gốc chưa kiểm (bước 6)" : ""}</span></a>`);
  const re = summary?.reeval;
  if (re?.available) {
    const main = re.main_models_exact ? "Centralized và FedAvg khớp tuyệt đối" : "có lệch ở mô hình chính";
    chips.push(`<a class="trust-chip ${re.all_within_tolerance ? "ok" : "warn"}" href="#bang-chung"><b>Đã chấm lại ${fmtInt(re.validation_count)} ảnh trên máy này</b><span>${re.comparisons_exact}/${re.comparisons_total} số liệu khớp tuyệt đối · ${main}</span></a>`);
  } else {
    chips.push(`<a class="trust-chip warn" href="#bang-chung"><b>Chưa chấm lại trên máy này</b><span>${esc(re?.reason || "thiếu báo cáo")}</span></a>`);
  }
  const lk = summary?.leakage;
  if (lk?.available) {
    chips.push(`<a class="trust-chip ${lk.passed ? "ok" : "bad"}" href="#bang-chung"><b>Kiểm tra trùng ảnh: ${lk.passed ? "đạt" : "KHÔNG đạt"}</b><span>${fmtInt(lk.overlap_total)} ảnh trùng giữa tập học và tập kiểm tra · đã kiểm ${fmtInt(lk.images_verified)} ảnh</span></a>`);
  } else {
    chips.push(`<a class="trust-chip warn" href="#bang-chung"><b>Chưa kiểm tra trùng ảnh</b><span>${esc(lk?.reason || "thiếu báo cáo")}</span></a>`);
  }
  const ho = summary?.holdout;
  if (ho?.available) {
    const fed = ho.models.find(x => x.model_id === "fedavg");
    chips.push(`<a class="trust-chip ok" href="#ket-qua"><b>Kiểm tra lại trên ${fmtInt(ho.count)} ảnh giữ riêng</b><span>tập ảnh không dùng để chọn mô hình · FedAvg đạt ${fmtPct(fed?.plantvillage?.top1)} (ảnh trong phòng) và ${fmtPct(fed?.plantdoc?.top1)} (ảnh ngoài đồng)</span></a>`);
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
  return `<p class="pillar-test" title="Tập ${fmtInt(holdout.count)} ảnh không dùng để chọn mô hình">Tập giữ riêng: <b>${show(pv)}</b> (trong phòng) · <b>${show(pd)}</b> (ngoài đồng)</p>`;
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
      <p class="eyebrow">Đề tài</p>
      <h2 class="hero-question">Nhiều nơi cùng muốn một mô hình nhận ra bệnh trên lá cây, nhưng không nơi nào muốn gửi ảnh của mình đi. Có thể cùng huấn luyện mà ảnh vẫn ở nguyên chỗ không — và kết quả kém đi bao nhiêu?</h2>
      <p class="hero-note">Đồ án thử trên 38 nhóm (26 bệnh và 12 trường hợp lá khỏe) của 14 loại cây, chia ảnh cho năm nơi tham gia giả lập (mỗi nơi gọi là một client), rồi so sánh ba cách huấn luyện dưới đây.</p>
    </section>

    <section class="flow" aria-labelledby="flow-title">
      <h3 class="flow-title" id="flow-title">Học liên kết (FedAvg) hoạt động thế nào?</h3>
      <ol class="flow-steps" role="list">
        <li><span class="flow-n" aria-hidden="true">1</span><div><b>Mỗi nơi tự học</b><p>Từng nơi huấn luyện mô hình trên ảnh của mình, ảnh không rời máy.</p></div></li>
        <li><span class="flow-n" aria-hidden="true">2</span><div><b>Chỉ gửi "kinh nghiệm"</b><p>Họ gửi về máy chủ các trọng số của mô hình — con số, không phải ảnh.</p></div></li>
        <li><span class="flow-n" aria-hidden="true">3</span><div><b>Gộp lại rồi chia sẻ</b><p>Máy chủ lấy trung bình thành một mô hình chung và gửi lại cho tất cả; lặp nhiều vòng.</p></div></li>
      </ol>
    </section>

    <section class="pillar-grid" aria-label="Ba cách huấn luyện">
      <article class="pillar pillar-centralized">
        <header><span class="pillar-tag">Gom ảnh về một chỗ · mức tham chiếu</span><h3>Centralized</h3></header>
        <p class="pillar-desc">Gom toàn bộ ảnh về một máy rồi huấn luyện. Chính xác nhất, nhưng ảnh phải rời khỏi nơi sinh ra nó.</p>
        <dl class="pillar-metrics">
          <div><dt>Ảnh chụp trong phòng <small>(PlantVillage)</small></dt><dd class="metric-big" id="m-central-pv">—</dd></div>
          <div><dt>Ảnh ngoài đồng <small>(PlantDoc)</small></dt><dd class="metric-big" id="m-central-pd">—</dd></div>
        </dl>
        ${testLine(summary?.holdout, ["centralized"])}
        <p class="pillar-foot">Ảnh phải gửi về một chỗ — không riêng tư</p>
      </article>
      <article class="pillar pillar-fedavg">
        <header><span class="pillar-tag">Học liên kết · giải pháp của đồ án</span><h3>FedAvg</h3></header>
        <p class="pillar-desc">Năm nơi huấn luyện trên ảnh của mình, chỉ gửi trọng số; máy chủ lấy trung bình có trọng số theo số ảnh.</p>
        <dl class="pillar-metrics">
          <div><dt>Ảnh chụp trong phòng <small>(PlantVillage)</small></dt><dd class="metric-big" id="m-fed-pv">—</dd></div>
          <div><dt>Ảnh ngoài đồng <small>(PlantDoc)</small></dt><dd class="metric-big" id="m-fed-pd">—</dd></div>
        </dl>
        ${testLine(summary?.holdout, ["fedavg"])}
        <p class="pillar-foot">Ảnh gốc không rời khỏi nơi sinh ra nó</p>
      </article>
      <article class="pillar pillar-local">
        <header><span class="pillar-tag">Mỗi nơi tự học · không hợp tác</span><h3>Local-only</h3></header>
        <p class="pillar-desc">Mỗi nơi chỉ học ảnh của mình. Có năm mô hình riêng biệt, không chia sẻ gì với nhau.</p>
        <dl class="pillar-metrics">
          <div><dt>Ảnh chụp trong phòng <small>(PlantVillage)</small></dt><dd class="metric-big">${range(pvLocals)}</dd></div>
          <div><dt>Ảnh ngoài đồng <small>(PlantDoc)</small></dt><dd class="metric-big">${range(pdLocals)}</dd></div>
        </dl>
        ${testLine(summary?.holdout, LOCALS)}
        <p class="pillar-foot">Khoảng kết quả của 5 mô hình Local 00–04</p>
      </article>
    </section>

    <p class="field-hint mt-3">Số lớn tính trên tập kiểm tra — cũng là tập dùng để chọn mô hình; dòng "Tập giữ riêng" là bộ ảnh không dùng để chọn
      (bộ chẩn đoán "legacy" của bộ dữ liệu, xem bước 3). Mới có một kịch bản và một lần chạy; chưa phải độ chính xác ngoài thực địa.</p>

    <section class="conclusion-grid mt-4" aria-label="Kết luận chính">
      <article class="conclusion fedavg">
        <b>${gapFedCentral > 0.5 ? `FedAvg kém Centralized ${fmtNum(gapFedCentral, 1)} điểm` : "FedAvg ngang Centralized"}</b>
        <p>Đạt <b>${fmtPct(fed.plantvillage.top1_accuracy)}</b> trên ảnh chụp trong phòng, trong khi không ảnh nào rời khỏi nơi sinh ra nó. Hai cách chưa huấn luyện cùng điều kiện nên mức chênh này chỉ mang tính tham khảo (bước 6).</p>
      </article>
      <article class="conclusion local">
        <b>Học riêng thì thất bại</b>
        <p>Mô hình học riêng chỉ đạt <b>${range(pvLocals)}</b> trên ảnh chụp trong phòng; FedAvg hơn mô hình riêng tốt nhất <b>${fmtNum(fedOverLocal, 1)} điểm</b>. Nguyên nhân: mỗi nơi chỉ có đủ ảnh ở một số nhóm, các nhóm khác rất ít hoặc không có (bước 2).</p>
      </article>
      <article class="conclusion warn">
        <b>Ảnh ngoài đồng khó hơn nhiều</b>
        <p>Từ ảnh chụp trong phòng sang ảnh ngoài đồng, Centralized giảm <b>${fmtNum(domainCentral, 1)} điểm</b> và FedAvg giảm <b>${fmtNum(domainFed, 1)} điểm</b>: chủ yếu do ảnh ngoài đồng khác kiểu và chiếm tỉ lệ nhỏ trong ảnh học; Centralized cũng giảm tương tự nên đây không phải lỗi riêng của học liên kết.</p>
      </article>
    </section>

    <section class="panel mt-4">
      <div class="panel-header-flex">
        <div>
          <h3 class="panel-title">Bảy mô hình, cùng một bộ ảnh kiểm tra</h3>
          <p class="panel-desc">Tỉ lệ đoán đúng bệnh. Cột đậm: ảnh chụp trong phòng (${fmtInt(val.plantvillage)} ảnh, cùng kiểu với ảnh học). Cột nhạt: ảnh ngoài đồng (${fmtInt(val.plantdoc)} ảnh, khác kiểu nên khó hơn).</p>
        </div>
      </div>
      <div class="chart-container chart-tall mt-4"><canvas id="chart-overview" aria-label="So sánh tỉ lệ đoán đúng của bảy mô hình"></canvas></div>
    </section>

    <section class="panel mt-4">
      <h3 class="panel-title">Thí nghiệm được thiết lập thế nào</h3>
      <dl class="fact-grid mt-3">
        <div><dt>Mô hình</dt><dd>MobileNetV3-Small, 38 nhóm</dd></div>
        <div><dt>Số nơi tham gia</dt><dd>5 client</dd></div>
        <div><dt>Cách chia ảnh</dt><dd>Lệch nặng: số ảnh mỗi nhóm chênh rất lớn giữa các nơi (Dirichlet α = 0,1)</dd></div>
        <div><dt>Ảnh để học</dt><dd>${fmtInt(trainTotal)} ảnh (PlantVillage + PlantDoc)</dd></div>
        <div><dt>Ảnh để kiểm tra</dt><dd>${fmtInt(val.plantvillage)} + ${fmtInt(val.plantdoc)} ảnh</dd></div>
        <div><dt>Số lần chạy</dt><dd>1 lần (seed ${esc(evidence.seed)})</dd></div>
      </dl>
    </section>

    <section class="mt-4">
      <h3 class="panel-title">Các con số này đáng tin tới đâu?</h3>
      <div class="trust-grid mt-3">${trustChips(summary)}</div>
      ${summary?.checklist?.available ? `<p class="field-hint mt-3">Tiến độ nghiệm thu: <b>${summary.checklist.done}/${summary.checklist.total}</b> hạng mục đã xong; ${summary.checklist.blocked} hạng mục còn lại cần file mô hình gốc hoặc máy huấn luyện GPU — <a href="#bang-chung">xem ở bước 6</a>.</p>` : ""}
    </section>

    <section class="cta-row mt-4">
      <a class="btn btn-primary" href="#non-iid">Vì sao học riêng thất bại →</a>
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
