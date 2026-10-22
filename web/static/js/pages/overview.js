// Chương 1 · Giới thiệu: bài toán, câu hỏi nghiên cứu, kết quả chính và mục lục báo cáo.
import { getEvidence, getGallery, getSummary, modelsById } from "../core/store.js";
import { factsOf, scopeText } from "../core/facts-text.js";
import { beat, diagramHtml, pulse } from "../core/flnet.js";
import { cancelWithin, followPointer, onMotionPrefChange, play } from "../core/motion.js";
import { onRouteChange } from "../core/router.js";
import {
  applyStyleData, byId, esc, fmtInt, fmtNum, fmtPct, formatDiseaseName, prefersReducedMotion, revealItems, sectionHead,
} from "../core/dom.js";

const LOCALS = ["local_00", "local_01", "local_02", "local_03", "local_04"];

function unavailable(reason) {
  return `<div class="callout callout-error" role="alert"><b>Chưa có số liệu để hiển thị.</b> ${esc(reason || "Chưa nạp bộ số liệu.")}
    Web không hiện số khi chưa đối chiếu được với bản công bố (không có số dự phòng).</div>`;
}

function range(values, digits = 1) {
  const lo = Math.min(...values) * 100;
  const hi = Math.max(...values) * 100;
  return `${fmtNum(lo, digits)}–${fmtNum(hi, digits)}%`;
}

// ───────── Hero: sơ đồ FedAvg động ─────────
// Trạng thái mặc định của CSS là hình hoàn chỉnh; hoạt ảnh chỉ "dựng dần" nó một lượt khi vào trang (không lặp).
let heroFresh = false; // mount() vừa chạy hoạt ảnh: lần onShow() ngay sau đó không chạy lại

function heroHtml() {
  return `
    <section class="hero overview-hero" id="overview-hero">
      <span class="hero-glow" id="hero-glow" aria-hidden="true"></span>
      <div class="overview-hero-copy">
        <p class="eyebrow">Đồ án tốt nghiệp · học phân tán chẩn đoán bệnh lá cây</p>
        <h2 class="hero-question">Học chung.<br>Ảnh ở lại từng nơi.</h2>
        <p class="hero-note">Mỗi client huấn luyện mô hình trên dữ liệu riêng. Máy chủ tổng hợp trọng số thành một mô hình chung; ảnh không cần tập trung về máy chủ.</p>
        <p class="viewer-scope" id="overview-scope">Thực nghiệm trên các phân vùng dữ liệu giả lập.</p>
        <div class="cta-row overview-hero-actions">
          <a class="btn btn-primary" href="#ket-qua">Xem kết quả</a>
          <a class="btn btn-secondary" href="#chan-doan">Thử chẩn đoán ảnh →</a>
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
  // rời chương 1: hủy hoạt ảnh (về hình hoàn chỉnh); đổi cỡ khung khi đang chạy: dừng để chấm không lệch đường dây
  onRouteChange(route => { if (route.id !== "gioi-thieu") cancelWithin(byId("fl-diagram")); });
  if (typeof ResizeObserver === "function") {
    let lastWidth = diagram.clientWidth;
    new ResizeObserver(() => {
      const width = diagram.clientWidth;
      if (width <= 0) return; // pane đang ẩn (rời chương 1): bỏ qua, không coi là đổi cỡ
      // chỉ hủy khi cỡ khung đổi thật (> 24 px); thanh cuộn hiện ra (~15 px) hay 0 → W lúc vào lại không được hủy hoạt ảnh vừa chạy
      if (lastWidth > 0 && Math.abs(width - lastWidth) > 24) cancelWithin(diagram);
      lastWidth = width;
    }).observe(diagram);
  }
}

/** Router gọi mỗi lần vào chương 1: chạy lại minh họa (lần đầu đã chạy ở mount()). */
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
  const [evidence, gallery, summary] = await Promise.all([getEvidence(), getGallery(), getSummary().catch(() => null)]);
  if (!evidence.available) {
    body.innerHTML = unavailable(evidence.reason);
    return;
  }

  const sample = gallery.items?.find(item => item.id === 12) || gallery.items?.[0];
  if (sample) {
    byId("overview-leaf").innerHTML = `<img src="/static/samples/demo/${esc(encodeURIComponent(sample.file))}" width="${sample.width || 256}" height="${sample.height || 256}" alt="Ảnh mẫu: ${esc(formatDiseaseName(sample.label))}">
      <figcaption><span class="eyebrow">Ảnh mẫu có nhãn</span><strong>${esc(formatDiseaseName(sample.label))}</strong><small>${esc(sample.source_name || sample.source)}${gallery.validation_membership?.status === "verified" ? " · tập kiểm tra" : ""}</small><a href="#chan-doan">Thử trên ảnh này →</a></figcaption>`;
    byId("overview-leaf").hidden = false;
    byId("overview-hero").classList.add("has-sample");
  }

  const m = modelsById(evidence);
  const central = m.centralized.metrics;
  const fed = m.fedavg.metrics;
  const pvLocals = LOCALS.map(id => m[id].metrics.plantvillage.top1_accuracy);
  const val = evidence.validation_counts || {};
  const clientCount = Object.keys(evidence.shard_counts || {}).length;
  const classCount = evidence.classes?.length;
  byId("overview-scope").textContent = `${clientCount} phân vùng giả lập${classCount ? ` · ${classCount} nhóm lá khỏe/bệnh` : ""} · chưa kiểm thử thực địa độc lập`;

  body.innerHTML = `
    <div class="report-pair mt-4">
    <section class="panel report-section" aria-labelledby="problem-title">
      ${sectionHead("1.1", "Bài toán", { id: "problem-title", lead: "Huấn luyện chung khi ảnh nằm ở nhiều nơi và không muốn gom về một máy" })}
      <p class="panel-desc mt-2">Ảnh bệnh lá cây do nhiều nơi giữ riêng. Gom hết về một máy để huấn luyện thì tốn đường truyền và mỗi nơi phải giao ảnh gốc. Học phân tán cho mỗi nơi (client) tự huấn luyện trên ảnh của mình, chỉ gửi trọng số mô hình lên máy chủ để gộp lại.</p>
      <p class="panel-desc mt-2">Khó khăn của đề tài: ảnh ở các client không giống nhau, ví dụ nơi này nhiều ảnh cà chua, nơi kia gần như không có. Với dữ liệu lệch như vậy, mô hình gộp có thể kém hơn mô hình huấn luyện trên toàn bộ ảnh.</p>
    </section>

    <section class="panel report-section" aria-labelledby="questions-title">
      ${sectionHead("1.2", "Câu hỏi nghiên cứu", { id: "questions-title", lead: "Ba câu hỏi đồ án trả lời" })}
      <ol class="question-list mt-3">
        <li><b>So với gom ảnh.</b> Khi ảnh chia lệch cho ${fmtInt(clientCount)} client, FedAvg đạt gần mô hình Centralized (gom ảnh một nơi) đến đâu?</li>
        <li><b>So với học riêng.</b> Học chung có hơn việc mỗi client tự học riêng (Local-only) không?</li>
        <li><b>Ảnh khác kiểu.</b> Mô hình học chủ yếu từ ảnh chụp trong phòng (PlantVillage) có giữ được độ chính xác trên ảnh ngoài đồng không? Ảnh ngoài đồng gồm PlantDoc${(evidence.validation_counts || {}).field_other ? " và các bộ công khai khác" : ""}.</li>
      </ol>
      <p class="field-hint mt-3">Câu trả lời nằm ở <a href="#ket-luan">chương 5 · Kết luận</a>; số liệu đầy đủ ở <a href="#ket-qua">chương 4 · Kết quả</a>.</p>
    </section>
    </div>

    <div class="mt-5">${sectionHead("1.3", "Kết quả chính", { id: "headline-title" })}</div>
    <section class="overview-headline-results mt-3" aria-labelledby="headline-title">
      <div class="overview-headline-intro"><span class="eyebrow">Trên tập kiểm tra PlantVillage</span><h4>FedAvg đạt <strong>${fmtPct(fed.plantvillage.top1_accuracy)}</strong></h4><p>${fmtInt(val.plantvillage)} ảnh · độ chính xác Top-1 (lựa chọn đầu tiên của mô hình đúng nhóm bệnh)</p></div>
      <div class="overview-headline-compare"><span>Centralized</span><strong>${fmtPct(central.plantvillage.top1_accuracy)}</strong><small>Gom ảnh một nơi</small></div>
      <div class="overview-headline-compare"><span>Local-only</span><strong>${range(pvLocals)}</strong><small>Mỗi client học riêng</small></div>
    </section>

    <p class="viewer-scope mt-3"><b class="scope-tag">Phạm vi</b> ${esc(scopeText(factsOf(summary)))} Xem hạn chế ở <a href="#ket-luan">chương 5</a>.</p>

    <section class="mt-5" aria-labelledby="toc-title">
      ${sectionHead("1.4", "Cấu trúc báo cáo", { id: "toc-title", lead: "Đọc theo thứ tự hoặc nhảy tới phần cần xem" })}
      <div class="toc-grid mt-3">
        <a class="toc-card" href="#du-lieu"><span class="toc-no">2</span><b>Dữ liệu</b><span>Bộ ảnh, cách chia tập học, kiểm tra, giữ riêng và dữ liệu lệch giữa các client.</span></a>
        <a class="toc-card" href="#phuong-phap"><span class="toc-no">3</span><b>Phương pháp</b><span>Ba cách huấn luyện, thuật toán FedAvg, thiết lập thí nghiệm và cách đánh giá.</span></a>
        <a class="toc-card" href="#ket-qua"><span class="toc-no">4</span><b>Kết quả</b><span>Bảy mô hình, tập giữ riêng và thống kê, phân tích lỗi, độ tin cậy của xác suất.</span></a>
        <a class="toc-card" href="#ket-luan"><span class="toc-no">5</span><b>Kết luận</b><span>Trả lời câu hỏi nghiên cứu, hạn chế và hướng phát triển.</span></a>
        <a class="toc-card is-demo" href="#chan-doan"><span class="toc-no">Demo</span><b>Chẩn đoán ảnh</b><span>Tải ảnh lá, ba cách nhận diện (Local only, PlantVillage, Cộng đồng) trả lời cùng lúc.</span></a>
        <a class="toc-card is-demo" href="#hoc-lien-ket"><span class="toc-no">Demo</span><b>Học phân tán</b><span>Tham gia như một client thật từ trình duyệt của bạn.</span></a>
      </div>
      <p class="field-hint mt-3">Số liệu kiểm chứng và phần chạy thử hệ thống trên máy chủ nằm ở <a href="#phu-luc">Phụ lục kỹ thuật</a>.</p>
    </section>`;

  revealItems(root.querySelectorAll(".toc-card"), { stagger: 50 });
}
