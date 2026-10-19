// Bước 1 · Tổng quan: câu hỏi, ba trụ cột, kết quả chính, độ tin cậy.
import { getEvidence, getGallery, getSummary, modelsById } from "../core/store.js";
import { barRise, makeChartWhenVisible, pillarColors } from "../core/charts.js";
import { beat, diagramHtml, geometry, pulse } from "../core/flnet.js";
import { cancelWithin, followPointer, onMotionPrefChange, play } from "../core/motion.js";
import { onRouteChange } from "../core/router.js";
import {
  applyStyleData, MODEL_LABELS, MODEL_ORDER, byId, esc, fmtInt, fmtNum, fmtPct, formatDiseaseName, prefersReducedMotion, revealItems, setMetric, withAlpha,
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
    chips.push(`<a class="trust-chip ${re.all_within_tolerance ? "ok" : "warn"}" href="#bang-chung"><b>Đã chấm lại ${fmtInt(re.validation_count)} ảnh trên máy này</b><span>${fmtInt(re.comparisons_exact)}/${fmtInt(re.comparisons_total)} số liệu khớp tuyệt đối · ${main}</span></a>`);
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
    chips.push(`<a class="trust-chip warn" href="#ket-qua"><b>Kiểm tra lại trên ${fmtInt(ho.count)} ảnh giữ riêng</b><span>bộ chẩn đoán có sẵn của bộ dữ liệu (legacy), không dùng để chọn mô hình nhưng chưa nằm trong kiểm tra trùng ảnh · FedAvg đạt ${fmtPct(fed?.plantvillage?.top1)} (ảnh trong phòng) và ${fmtPct(fed?.plantdoc?.top1)} (ảnh ngoài đồng)</span></a>`);
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

// ───────── Hero: sơ đồ FedAvg động ─────────
// Trạng thái mặc định của CSS là hình hoàn chỉnh; hoạt ảnh chỉ "dựng dần" nó một lượt khi vào trang (không lặp).
let heroFresh = false; // mount() vừa chạy hoạt ảnh: lần onShow() ngay sau đó không chạy lại

function heroHtml() {
  return `
    <section class="hero overview-hero" id="overview-hero">
      <span class="hero-glow" id="hero-glow" aria-hidden="true"></span>
      <div class="overview-hero-copy">
        <p class="eyebrow">Đồ án học liên kết · chẩn đoán bệnh lá cây</p>
        <h2 class="hero-question">Học chung.<br>Ảnh ở lại từng nơi.</h2>
        <p class="hero-note">Mỗi client huấn luyện mô hình trên dữ liệu riêng. Máy chủ tổng hợp trọng số thành một mô hình chung; ảnh không cần tập trung về máy chủ.</p>
        <p class="viewer-scope" id="overview-scope">Thực nghiệm trên các phân vùng dữ liệu giả lập.</p>
        <div class="cta-row overview-hero-actions">
          <a class="btn btn-primary" href="#chan-doan">Thử chẩn đoán ảnh →</a>
          <a class="btn btn-secondary" href="#ket-qua">Xem kết quả</a>
        </div>
      </div>
      <figure class="overview-leaf" id="overview-leaf" aria-label="Ảnh mẫu có nhãn" hidden>
        <figcaption><span class="eyebrow">Thử trên một ảnh lá</span><small>Chọn ảnh mẫu hoặc tải ảnh của bạn.</small></figcaption>
      </figure>
      <figure class="fl-diagram" id="fl-diagram">
        <p class="overview-process-label">Quy trình FedAvg · 5 client giữ ảnh riêng</p>
        <div class="fl-figure" role="img"
          aria-label="Năm client giữ ảnh riêng, chỉ gửi trọng số mô hình tới máy chủ FedAvg; máy chủ gộp lại thành mô hình chung rồi gửi lại cho tất cả">
          ${diagramHtml({ variant: "hero" })}
        </div>
        <figcaption>
          <span>FedAvg: học tại client → gửi trọng số → gộp mô hình.</span>
          <button type="button" class="btn btn-ghost btn-sm" id="hero-replay" aria-label="Xem lại minh họa FedAvg">↻ Xem lại</button>
        </figcaption>
      </figure>
    </section>`;
}

function playHero() {
  const diagram = byId("fl-diagram");
  if (!diagram) return;
  cancelWithin(diagram);
  if (prefersReducedMotion()) return; // hiện luôn hình hoàn chỉnh
  const stage = diagram.querySelector(".fl-stage");
  const cards = stage.querySelectorAll(".fl-client .fl-node-card");
  const lit = stage.querySelectorAll(".fl-wire-lit[data-i]");
  const count = cards.length;

  // 1) client hiện lần lượt (140 ms/client)
  cards.forEach((card, i) => play(card,
    [{ opacity: 0, transform: "translateY(8px) scale(0.94)" }, { opacity: 1, transform: "translateY(0) scale(1)" }],
    { delay: i * 140, duration: 320 }));
  // 2) dây sáng dần
  lit.forEach((wire, i) => play(wire, [{ opacity: 0 }, { opacity: 1 }], { delay: 600 + i * 140, duration: 280 }));
  // 3) chấm trọng số chạy về máy chủ (hai đợt: mô tả "mỗi vòng gửi lại")
  const all = Array.from({ length: count }, (_unused, i) => i);
  const end = pulse(stage, all, { delay: 1000, perClientGap: 110, duration: 900 });
  // 4) máy chủ gộp: một nhịp; 5) mô hình chung hiện ra
  beat(stage.querySelector(".fl-server .fl-node-card"), { delay: Math.max(2350, end - 150) });
  const modelDelay = Math.max(2500, end);
  play(stage.querySelector(".fl-model .fl-node-card"),
    [{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "translateY(0)" }], { delay: modelDelay, duration: 350 });
  play(stage.querySelector(".fl-wire-model"), [{ opacity: 0 }, { opacity: 1 }], { delay: modelDelay, duration: 350 });
  // 6) mô hình chung quay lại từng client: nửa còn lại của vòng "gửi lên → gộp → gửi xuống" (minh họa, không phải số đo)
  pulse(stage, all, { delay: modelDelay + 350, perClientGap: 90, duration: 800, reverse: true });
}

function initHero() {
  const diagram = byId("fl-diagram");
  if (!diagram || diagram.dataset.ready) return;
  diagram.dataset.ready = "1";
  byId("hero-replay")?.addEventListener("click", playHero);
  followPointer(byId("overview-hero"), byId("hero-glow"));
  onMotionPrefChange(reduce => { if (reduce) cancelWithin(byId("fl-diagram")); });
  // rời bước 1: hủy hoạt ảnh (về hình hoàn chỉnh); đổi cỡ khung khi đang chạy: dừng để chấm không lệch đường dây
  onRouteChange(route => { if (route.id !== "tong-quan") cancelWithin(byId("fl-diagram")); });
  if (typeof ResizeObserver === "function") {
    let lastWidth = diagram.clientWidth;
    new ResizeObserver(() => {
      const width = diagram.clientWidth;
      if (width <= 0) return; // pane đang ẩn (rời bước 1): bỏ qua, không coi là đổi cỡ
      // chỉ hủy khi cỡ khung đổi thật (> 24 px); thanh cuộn hiện ra (~15 px) hay 0 → W lúc vào lại không được hủy hoạt ảnh vừa chạy
      if (lastWidth > 0 && Math.abs(width - lastWidth) > 24) cancelWithin(diagram);
      lastWidth = width;
    }).observe(diagram);
  }
}

/** Router gọi mỗi lần vào bước 1: chạy lại minh họa (lần đầu đã chạy ở mount()). */
export function onShow() {
  if (heroFresh) {
    heroFresh = false;
    return;
  }
  playHero();
}

export async function mount() {
  const root = byId("overview-root");
  root.innerHTML = `${heroHtml()}<div id="overview-body"><div class="skeleton skeleton-block mt-4"></div></div>`;
  applyStyleData(root); // vị trí các nút sơ đồ do --x/--y; đo hình học ngay bên dưới nên không đợi MutationObserver
  initHero();
  playHero();
  heroFresh = true;
  const body = byId("overview-body");
  const [evidence, summary, gallery] = await Promise.all([getEvidence(), getSummary().catch(() => null), getGallery()]);
  if (!evidence.available) {
    body.innerHTML = unavailable(evidence.reason);
    return;
  }

  const sample = gallery.items?.find(item => item.id === 12) || gallery.items?.[0];
  if (sample) {
    byId("overview-leaf").innerHTML = `<img src="/static/samples/demo/${esc(encodeURIComponent(sample.file))}" width="${sample.width || 256}" height="${sample.height || 256}" alt="Ảnh mẫu: ${esc(formatDiseaseName(sample.label))}">
      <figcaption><span class="eyebrow">Ảnh mẫu có nhãn</span><strong>${esc(formatDiseaseName(sample.label))}</strong><small>${esc(sample.source_name || sample.source)}${gallery.validation_membership?.status === "verified" ? " · tập kiểm tra" : ""}</small><a href="#chan-doan">Mở chẩn đoán →</a></figcaption>`;
    byId("overview-leaf").hidden = false;
    byId("overview-hero").classList.add("has-sample");
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
  const clientCount = Object.keys(evidence.shard_counts || {}).length;
  const classCount = evidence.classes?.length || summary?.shards?.classes_total;
  byId("overview-scope").textContent = `${clientCount} phân vùng giả lập${classCount ? ` · ${classCount} nhóm lá khỏe/bệnh` : ""} · chưa kiểm thử thực địa độc lập`;

  const gapFedCentral = (central.plantvillage.top1_accuracy - fed.plantvillage.top1_accuracy) * 100;
  const fedOverLocal = (fed.plantvillage.top1_accuracy - bestLocalPv) * 100;
  const domainFed = (fed.plantvillage.top1_accuracy - fed.plantdoc.top1_accuracy) * 100;

  body.innerHTML = `
    <section class="overview-headline-results" aria-label="Kết quả chính trên ảnh PlantVillage">
      <div class="overview-headline-intro"><span class="eyebrow">Trên tập kiểm tra PlantVillage</span><h3>FedAvg đạt <strong>${fmtPct(fed.plantvillage.top1_accuracy)}</strong></h3><p>${fmtInt(val.plantvillage)} ảnh · độ chính xác Top-1 (lựa chọn đầu tiên của mô hình đúng nhóm bệnh)</p></div>
      <div class="overview-headline-compare"><span>Centralized</span><strong>${fmtPct(central.plantvillage.top1_accuracy)}</strong><small>Gom ảnh một nơi</small></div>
      <div class="overview-headline-compare"><span>Local-only</span><strong>${range(pvLocals)}</strong><small>Mỗi client học riêng</small></div>
    </section>

    <p class="viewer-scope mt-3"><b class="scope-tag">Phạm vi</b> Kết quả từ một lần chạy; tập kiểm tra đã tham gia quá trình chọn mô hình, các cách học chưa huấn luyện cùng điều kiện — chưa phải độ chính xác ngoài thực địa. Xem thêm các giới hạn ở bước 6.</p>

    <section class="conclusion-grid mt-4" aria-label="Ba điều rút ra từ kết quả">
      <article class="conclusion fedavg">
        <span class="conclusion-label">Chênh lệch so với Centralized</span>
        <b>${gapFedCentral > 0.5 ? `−${fmtNum(gapFedCentral, 1)} điểm phần trăm` : "Gần ngang nhau"}</b>
        <p>So với Centralized trên PlantVillage; chênh lệch tham khảo vì điều kiện huấn luyện khác nhau.</p>
      </article>
      <article class="conclusion local">
        <span class="conclusion-label">Chênh lệch so với Local-only</span>
        <b>${fedOverLocal >= 0 ? "+" : "−"}${fmtNum(Math.abs(fedOverLocal), 1)} điểm phần trăm</b>
        <p>So với mô hình học riêng tốt nhất trên PlantVillage. <a href="#non-iid">Phân bố dữ liệu giữa các client →</a></p>
      </article>
      <article class="conclusion warn">
        <span class="conclusion-label">Chênh lệch PlantVillage → PlantDoc</span>
        <b>−${fmtNum(domainFed, 1)} điểm phần trăm</b>
        <p>FedAvg giảm khi đổi từ PlantVillage sang PlantDoc. <a href="#ket-qua">Xem phạm vi kết quả →</a></p>
      </article>
    </section>

    <details class="overview-detail mt-4 overview-chart-panel">
      <summary><span><strong>Xem biểu đồ đầy đủ của 7 mô hình</strong><small>PlantVillage và PlantDoc</small></span></summary>
      <div class="overview-detail-body">
      <div class="overview-section-heading">
        <div><p class="eyebrow">So sánh 7 mô hình</p><h3 class="panel-title" id="overview-chart-title">${fedOverLocal > 0 ? "Trong lần chạy này, học chung cao hơn mô hình học riêng" : "Trong lần chạy này, học chung không vượt mô hình học riêng"}</h3></div>
        <span class="overview-chart-unit">Tỉ lệ đoán đúng · %</span>
      </div>
      <div class="chart-container chart-tall mt-3"><canvas id="chart-overview" aria-label="So sánh tỉ lệ đoán đúng của bảy mô hình"></canvas></div>
      <p class="field-hint mt-2">Cột đậm: PlantVillage (${fmtInt(val.plantvillage)} ảnh). Cột nhạt: PlantDoc (${fmtInt(val.plantdoc)} ảnh ngoài đồng). Phạm vi đánh giá và tập giữ riêng ở bước 3.</p>
      </div>
    </details>

    <details class="overview-detail mt-4">
      <summary><span><strong>Ba cách huấn luyện được so sánh</strong><small>Centralized · FedAvg · Local-only</small></span></summary>
      <section class="pillar-grid" aria-label="Ba cách huấn luyện">
      <article class="pillar pillar-centralized">
        <header><span class="pillar-tag">Gom ảnh về một máy · mức tham chiếu</span><h3>Centralized</h3></header>
        <p class="pillar-desc">Gom toàn bộ ảnh về một máy rồi huấn luyện; dùng làm mức tham chiếu cho học liên kết.</p>
        <dl class="pillar-metrics">
          <div><dt>Ảnh chụp trong phòng <small>(PlantVillage)</small></dt><dd class="metric-big" id="m-central-pv">—</dd></div>
          <div><dt>Ảnh ngoài đồng <small>(PlantDoc)</small></dt><dd class="metric-big" id="m-central-pd">—</dd></div>
        </dl>
        ${testLine(summary?.holdout, ["centralized"])}
        <p class="pillar-foot">Ảnh gốc được gửi về máy chủ</p>
      </article>
      <article class="pillar pillar-fedavg">
        <header><span class="pillar-tag">Học liên kết · phương pháp đề xuất</span><h3>FedAvg</h3></header>
        <p class="pillar-desc">Năm client huấn luyện trên ảnh của mình và chỉ gửi trọng số; máy chủ lấy trung bình có trọng số theo số ảnh.</p>
        <dl class="pillar-metrics">
          <div><dt>Ảnh chụp trong phòng <small>(PlantVillage)</small></dt><dd class="metric-big" id="m-fed-pv">—</dd></div>
          <div><dt>Ảnh ngoài đồng <small>(PlantDoc)</small></dt><dd class="metric-big" id="m-fed-pd">—</dd></div>
        </dl>
        ${testLine(summary?.holdout, ["fedavg"])}
        <p class="pillar-foot">Ảnh gốc ở lại trên client</p>
      </article>
      <article class="pillar pillar-local">
        <header><span class="pillar-tag">Huấn luyện riêng · không chia sẻ trọng số</span><h3>Local-only</h3></header>
        <p class="pillar-desc">Mỗi client huấn luyện một mô hình riêng trên ảnh của mình. Có năm mô hình, không trao đổi trọng số.</p>
        <dl class="pillar-metrics">
          <div><dt>Ảnh chụp trong phòng <small>(PlantVillage)</small></dt><dd class="metric-big">${range(pvLocals)}</dd></div>
          <div><dt>Ảnh ngoài đồng <small>(PlantDoc)</small></dt><dd class="metric-big">${range(pdLocals)}</dd></div>
        </dl>
        ${testLine(summary?.holdout, LOCALS)}
        <p class="pillar-foot">Khoảng kết quả của 5 mô hình Local 00–04</p>
      </article>
      </section>

      <p class="field-hint mt-3">Số lớn tính trên tập kiểm tra dùng để chọn mô hình. Dòng “Tập giữ riêng” là bộ chẩn đoán có sẵn của bộ dữ liệu, không dùng để chọn mô hình, chưa nằm trong kiểm tra trùng ảnh (xem bước 3). Mới có một kịch bản và một lần chạy; đây chưa phải độ chính xác ngoài thực địa.</p>
    </details>



    <details class="overview-detail mt-4">
      <summary><span><strong>Cách triển khai, thiết lập và kiểm chứng</strong><small>Quy mô dữ liệu, cách chia và mức độ tin cậy</small></span></summary>
      <div class="overview-detail-body">
      <p class="panel-desc">Bản demo gồm phân chia dữ liệu giữa các client, điều phối FedAvg qua Flower và đối chiếu kết quả các mô hình. Mạng nền (backbone) là MobileNetV3-Small.</p><h3 class="panel-title mt-3">Thiết lập thí nghiệm</h3>
      <dl class="fact-grid mt-3">
        <div><dt>Mô hình</dt><dd>MobileNetV3-Small${classCount ? `, ${classCount} nhóm` : ""}</dd></div>
        <div><dt>Số client tham gia</dt><dd>${clientCount} client giả lập</dd></div>
        <div><dt>Cách chia ảnh</dt><dd>Lệch nặng: số ảnh mỗi nhóm chênh rất lớn giữa các client (Dirichlet α = 0,1)</dd></div>
        <div><dt>Ảnh để học</dt><dd>${fmtInt(trainTotal)} ảnh (PlantVillage + PlantDoc)</dd></div>
        <div><dt>Ảnh để kiểm tra</dt><dd>${fmtInt(val.plantvillage)} + ${fmtInt(val.plantdoc)} ảnh</dd></div>
        <div><dt>Số lần chạy</dt><dd>1 lần (seed ${esc(evidence.seed)})</dd></div>
      </dl>

      <h3 class="panel-title mt-4">Mức độ kiểm chứng của các con số</h3>
      <div class="trust-grid mt-3">${trustChips(summary)}</div>
      ${summary?.checklist?.available ? `<p class="field-hint mt-3">Danh mục kiểm tra dự án: <b>${fmtInt(summary.checklist.done)}/${fmtInt(summary.checklist.total)}</b> hạng mục đã xong. ${fmtInt(summary.checklist.blocked)} việc nghiên cứu còn lại cần file mô hình gốc hoặc máy huấn luyện GPU${summary.checklist.todo ? `; ${fmtInt(summary.checklist.todo)} hạng mục kỹ thuật cần chạy lại kiểm thử` : ""} — <a href="#bang-chung">xem ở bước 6</a>.</p>` : ""}
      </div>
    </details>

    <section class="cta-row mt-4">
      <a class="btn btn-secondary" href="#ket-qua">Xem kết quả chi tiết →</a>
      <a class="btn btn-secondary" href="#bang-chung">Kiểm chứng và giới hạn →</a>
    </section>`;

  setMetric(byId("m-central-pv"), fmtPct(central.plantvillage.top1_accuracy));
  setMetric(byId("m-central-pd"), fmtPct(central.plantdoc.top1_accuracy));
  setMetric(byId("m-fed-pv"), fmtPct(fed.plantvillage.top1_accuracy));
  setMetric(byId("m-fed-pd"), fmtPct(fed.plantdoc.top1_accuracy));
  revealItems(root.querySelectorAll(".pillar, .conclusion"), { stagger: 60 });

  // Biểu đồ nằm trong mục gập: dựng khi mục được mở. Dựng lúc đang ẩn thì hiệu ứng "mọc cột" chạy khi canvas rộng 0,
  // người xem chỉ thấy hình cuối bật ra tức thì (mất độ mượt).
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
              const role = id === "centralized" ? "gom ảnh về một chỗ" : id === "fedavg" ? "học liên kết" : "mỗi nơi tự học";
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
}
