// Chương 3 · Phương pháp: ba cách huấn luyện, thuật toán FedAvg, thiết lập thí nghiệm, cách đánh giá, hệ thống demo.
// Chỉ dùng số có trong /api/demo/evidence; siêu tham số không có trong bộ số liệu đã công bố thì không ghi.
import { configsHtml, glossaryHtml } from "../core/report-training.js";
import { getEvidence, getSummary } from "../core/store.js";
import { byId, esc, fmtInt, revealItems, sectionHead, SOURCE_GROUPS } from "../core/dom.js";
import { alphaText, factsOf, recipeNote, runsText } from "../core/facts-text.js";

function pillarsHtml() {
  return `
    <div class="pillar-grid mt-3">
      <article class="pillar pillar-centralized">
        <header><span class="pillar-tag">Gom ảnh về một máy · mức tham chiếu</span><h4>Centralized</h4></header>
        <p class="pillar-desc">Gom toàn bộ ảnh về một máy rồi huấn luyện một mô hình. Dùng làm mức tham chiếu: học phân tán cố gắng tiến gần mức này mà không phải gom ảnh.</p>
        <p class="pillar-foot">Ảnh gốc được gửi về máy chủ</p>
      </article>
      <article class="pillar pillar-fedavg">
        <header><span class="pillar-tag">Học phân tán · phương pháp đề xuất</span><h4>FedAvg</h4></header>
        <p class="pillar-desc">Các client huấn luyện trên ảnh của mình và chỉ gửi trọng số; máy chủ lấy trung bình có trọng số theo số ảnh để được một mô hình chung.</p>
        <p class="pillar-foot">Ảnh gốc ở lại trên client</p>
      </article>
      <article class="pillar pillar-local">
        <header><span class="pillar-tag">Huấn luyện riêng · không chia sẻ trọng số</span><h4>Local-only</h4></header>
        <p class="pillar-desc">Mỗi client huấn luyện một mô hình riêng trên ảnh của mình, không trao đổi trọng số. Có năm mô hình; cho thấy một nơi tự học thì thiếu những nhóm bệnh nào.</p>
        <p class="pillar-foot">Năm mô hình Local 00–04</p>
      </article>
    </div>`;
}

function fedavgHtml(counts) {
  const sizes = Object.values(counts || {});
  const example = sizes.length ? `Trong thí nghiệm này, số ảnh học của ${sizes.length} client lần lượt là ${sizes.map(fmtInt).join(", ")}; client nhiều ảnh hơn đóng góp nhiều hơn vào mô hình chung.` : "";
  return `
    <ol class="method-steps mt-3">
      <li><b>Gửi xuống.</b> Máy chủ gửi mô hình chung hiện tại cho từng client.</li>
      <li><b>Học tại chỗ.</b> Mỗi client huấn luyện mô hình trên ảnh của mình trong một số epoch.</li>
      <li><b>Gửi lên.</b> Client gửi trọng số mới về máy chủ. Ảnh gốc không rời client.</li>
      <li><b>Gộp.</b> Máy chủ lấy trung bình có trọng số theo số ảnh để được mô hình chung mới. Lặp lại qua nhiều vòng.</li>
    </ol>
    <p class="formula mt-3" role="math" aria-label="w mới bằng tổng theo k của n k chia n nhân w k">
      <i>w</i><sub>mới</sub> = Σ<sub>k</sub> (<i>n</i><sub>k</sub> / <i>n</i>) · <i>w</i><sub>k</sub>
    </p>
    <p class="field-hint mt-2"><i>w</i><sub>k</sub>: trọng số client k gửi lên. <i>n</i><sub>k</sub>: số ảnh học của client k. <i>n</i> = Σ <i>n</i><sub>k</sub>. ${esc(example)}</p>`;
}

/** "label_alpha_0_1" → "0,1": lấy mức lệch từ kịch bản trong bộ số liệu thay vì gõ cứng. */
function alphaOf(evidence) {
  const match = (evidence.non_iid_scenarios || []).map(name => /alpha_(\d+)_(\d+)/.exec(String(name))).find(Boolean);
  return match ? `${match[1]},${match[2]}` : null;
}

function setupHtml(evidence, summary) {
  const alpha = alphaOf(evidence);
  const facts = factsOf(summary);
  const counts = evidence.shard_counts || {};
  const trainTotal = Object.values(counts).reduce((a, b) => a + b, 0);
  const val = evidence.validation_counts || {};
  const groups = SOURCE_GROUPS.filter(g => g.id in val);
  const trainSplit = (summary?.training?.release?.splits || []).find(s => s.name === "train");
  const trainNames = groups.filter(g => !trainSplit?.by_source || g.id in trainSplit.by_source).map(g => g.label).join(" + ");
  const classCount = evidence.classes?.length;
  return `
    <dl class="fact-grid mt-3">
      <div><dt>Mô hình</dt><dd>MobileNetV3-Small${classCount ? `, ${fmtInt(classCount)} nhóm` : ""}${/pretrained/.test(String(evidence.protocol_id)) ? "; theo mã giao thức, khởi tạo từ trọng số ImageNet (pretrained)" : ""}</dd></div>
      <div><dt>Số client</dt><dd>${fmtInt(Object.keys(counts).length)} client giả lập (năm phần của cùng một bộ ảnh)</dd></div>
      <div><dt>Cách chia ảnh</dt><dd>Lệch nặng: số ảnh mỗi nhóm chênh rất lớn giữa các client${alpha ? ` (Dirichlet α = ${alpha})` : ""}</dd></div>
      <div><dt>Ảnh để học</dt><dd>${fmtInt(trainTotal)} ảnh (${esc(trainNames)})</dd></div>
      <div><dt>Ảnh để kiểm tra</dt><dd>${groups.map(g => fmtInt(val[g.id])).join(" + ")} ảnh (${esc(groups.map(g => g.label).join(" + "))})</dd></div>
      <div><dt>Số lần chạy</dt><dd>${esc(runsText(facts))}${facts?.seeds_run?.length ? ` (${esc(facts.seeds_run.join(", "))}); web phục vụ hạt giống ${esc(evidence.seed)}` : ` (seed ${esc(evidence.seed)})`}${alphaText(facts) && !alpha ? `; ${esc(alphaText(facts))}` : ""}</dd></div>
    </dl>
    <p class="field-hint mt-2">${esc(recipeNote(facts))} Xem thêm hạn chế ở <a href="#ket-luan">chương 5</a>.</p>`;
}

function metricsHtml(summary) {
  return `
    <dl class="term-list mt-3">
      <div><dt>Top-1</dt><dd>Tỉ lệ ảnh mà lựa chọn đầu tiên của mô hình đúng nhóm bệnh. Là số chính ở chương 4.</dd></div>
      <div><dt>Macro-F1</dt><dd>Điểm trung bình giữa các nhóm, mỗi nhóm có trọng số như nhau, nên nhóm nhiều ảnh không lấn át nhóm ít ảnh.</dd></div>
      <div><dt>Đúng loại cây</dt><dd>Mô hình đoán đúng loại cây, kể cả khi đoán sai bệnh.</dd></div>
      <div><dt>Khoảng tin cậy 95%</dt><dd>Khoảng ước lượng phản ánh độ bất định của phép đo. Tính bằng bootstrap theo cụm: ảnh cùng một cảnh chụp hoặc cùng một lá không được coi là độc lập.</dd></div>
      <div><dt>So sánh theo cặp (McNemar)</dt><dd>So sánh cặp dự đoán trên cùng ảnh qua các trường hợp một mô hình đúng, mô hình kia sai. Kiểm định theo ảnh giả định các cặp độc lập; ảnh cùng cảnh có thể tương quan, nên đọc kèm khoảng tin cậy bootstrap theo cảnh.</dd></div>
      <div><dt>Độ tin cậy của xác suất (ECE)</dt><dd>Mô hình báo chắc 90% thì có đúng khoảng 90% số lần không. ECE càng thấp càng tốt; temperature scaling chỉnh xác suất mà không đổi dự đoán.</dd></div>
    </dl>
    <p class="field-hint mt-2">Tập học, tập kiểm tra và tập giữ riêng được định nghĩa ở <a href="#du-lieu">chương 2</a>. Kết quả theo từng chỉ số ở <a href="#ket-qua">chương 4</a>.</p>
    ${glossaryHtml(alphaText(factsOf(summary)))}`;
}

function systemHtml() {
  return `
    <ul class="system-list mt-3">
      <li><b>Chẩn đoán ảnh.</b> Ba cách nhận diện ngang hàng chạy trên cùng một ảnh: Local only (năm client), mô hình PlantVillage ngoài (đã đánh giá lại) và mô hình cộng đồng (phiên bản mới nhất); bảy mô hình của đồ án (kèm Centralized, FedAvg) nằm sau nút tổng hợp, có vùng ảnh mô hình dựa vào (Grad-CAM).</li>
      <li><b>Học phân tán từ trình duyệt.</b> Mỗi người tham gia là một client thật: trình duyệt tính đặc trưng ảnh bằng mạng nền đóng băng của FedAvg, huấn luyện lớp phân loại cuối (khoảng 39 nghìn tham số) và chỉ gửi trọng số lớp đó (khoảng 156 KB) lên máy chủ. Máy chủ kiểm tra rồi gộp theo FedAvg qua từng vòng.</li>
      <li><b>Kiểm thử hệ thống.</b> Phiên Flower với năm tiến trình client trên máy chủ chỉ để kiểm chứng điều phối và phép gộp, không đo độ chính xác. Xem <a href="#phu-luc">Phụ lục kỹ thuật</a>.</li>
    </ul>`;
}

export async function mount() {
  const root = byId("methods-root");
  const [evidence, summary] = await Promise.all([getEvidence().catch(error => ({ available: false, reason: error.message })), getSummary().catch(() => null)]);
  const section = (no, title, body) => `
    <section class="panel report-section mt-4" aria-labelledby="m-${no}">
      ${sectionHead(`3.${no}`, title, { id: `m-${no}` })}
      ${body}
    </section>`;
  const unavailable = !evidence.available;
  root.innerHTML = `
    ${section(1, "Ba cách huấn luyện được so sánh", pillarsHtml())}
    ${section(2, "Thuật toán FedAvg", fedavgHtml(evidence.shard_counts))}
    ${unavailable
      ? `<div class="callout callout-warn mt-4" role="note"><b>Chưa có số liệu thiết lập thí nghiệm.</b> ${esc(evidence.reason || "")}</div>`
      : section(3, "Mô hình và thiết lập thí nghiệm", setupHtml(evidence, summary) + configsHtml(summary?.training))}
    ${section(4, "Cách đánh giá", metricsHtml(summary))}
    ${section(5, "Hệ thống demo", systemHtml())}`;
  revealItems(root.querySelectorAll(".pillar"), { stagger: 60 });
}
