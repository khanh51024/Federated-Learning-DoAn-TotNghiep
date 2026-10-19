// Chương 3 · Phương pháp: ba cách huấn luyện, thuật toán FedAvg, thiết lập thí nghiệm, cách đánh giá, hệ thống demo.
// Chỉ dùng số có trong /api/demo/evidence; siêu tham số không có trong bộ số liệu đã công bố thì không ghi.
import { getEvidence } from "../core/store.js";
import { byId, esc, fmtInt, revealItems, sectionHead } from "../core/dom.js";

function pillarsHtml() {
  return `
    <div class="pillar-grid mt-3">
      <article class="pillar pillar-centralized">
        <header><span class="pillar-tag">Gom ảnh về một máy · mức tham chiếu</span><h4>Centralized</h4></header>
        <p class="pillar-desc">Gom toàn bộ ảnh về một máy rồi huấn luyện một mô hình. Dùng làm mức tham chiếu: học liên kết cố gắng tiến gần mức này mà không phải gom ảnh.</p>
        <p class="pillar-foot">Ảnh gốc được gửi về máy chủ</p>
      </article>
      <article class="pillar pillar-fedavg">
        <header><span class="pillar-tag">Học liên kết · phương pháp đề xuất</span><h4>FedAvg</h4></header>
        <p class="pillar-desc">Các client huấn luyện trên ảnh của mình và chỉ gửi trọng số; máy chủ lấy trung bình có trọng số theo số ảnh để được một mô hình chung.</p>
        <p class="pillar-foot">Ảnh gốc ở lại trên client</p>
      </article>
      <article class="pillar pillar-local">
        <header><span class="pillar-tag">Huấn luyện riêng · không chia sẻ trọng số</span><h4>Local-only</h4></header>
        <p class="pillar-desc">Mỗi client huấn luyện một mô hình riêng trên ảnh của mình, không trao đổi gì. Có năm mô hình; cho thấy một nơi tự học thì thiếu những nhóm bệnh nào.</p>
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

function setupHtml(evidence) {
  const alpha = alphaOf(evidence);
  const counts = evidence.shard_counts || {};
  const trainTotal = Object.values(counts).reduce((a, b) => a + b, 0);
  const val = evidence.validation_counts || {};
  const classCount = evidence.classes?.length;
  return `
    <dl class="fact-grid mt-3">
      <div><dt>Mô hình</dt><dd>MobileNetV3-Small${classCount ? `, ${fmtInt(classCount)} nhóm` : ""}${/pretrained/.test(String(evidence.protocol_id)) ? "; theo mã giao thức, khởi tạo từ trọng số ImageNet (pretrained)" : ""}</dd></div>
      <div><dt>Số client</dt><dd>${fmtInt(Object.keys(counts).length)} client giả lập (năm phần của cùng một bộ ảnh)</dd></div>
      <div><dt>Cách chia ảnh</dt><dd>Lệch nặng: số ảnh mỗi nhóm chênh rất lớn giữa các client${alpha ? ` (Dirichlet α = ${alpha})` : ""}</dd></div>
      <div><dt>Ảnh để học</dt><dd>${fmtInt(trainTotal)} ảnh (PlantVillage + PlantDoc)</dd></div>
      <div><dt>Ảnh để kiểm tra</dt><dd>${fmtInt(val.plantvillage)} + ${fmtInt(val.plantdoc)} ảnh</dd></div>
      <div><dt>Số lần chạy</dt><dd>1 lần (seed ${esc(evidence.seed)})</dd></div>
    </dl>
    <p class="field-hint mt-2">Ba cách huấn luyện dùng cùng một bộ ảnh và cùng kiến trúc mạng; các thiết lập huấn luyện khác (bộ tối ưu, số bước) chưa được cố định chung, nên so sánh Centralized với FedAvg chỉ mang tính tham khảo (xem hạn chế ở <a href="#ket-luan">chương 5</a>).</p>`;
}

function metricsHtml() {
  return `
    <dl class="term-list mt-3">
      <div><dt>Top-1</dt><dd>Tỉ lệ ảnh mà lựa chọn đầu tiên của mô hình đúng nhóm bệnh. Là số chính ở chương 4.</dd></div>
      <div><dt>Macro-F1</dt><dd>Điểm trung bình giữa các nhóm, mỗi nhóm có trọng số như nhau, nên nhóm nhiều ảnh không lấn át nhóm ít ảnh.</dd></div>
      <div><dt>Đúng loại cây</dt><dd>Mô hình đoán đúng loại cây, kể cả khi đoán sai bệnh.</dd></div>
      <div><dt>Khoảng tin cậy 95%</dt><dd>Khoảng mà con số thật có khả năng nằm trong. Tính bằng bootstrap theo cụm: ảnh cùng một cảnh chụp hoặc cùng một lá không được coi là độc lập.</dd></div>
      <div><dt>So sánh theo cặp (McNemar)</dt><dd>Kiểm tra hai mô hình khác nhau thật hay chỉ do ngẫu nhiên, trên đúng cùng một tập ảnh.</dd></div>
      <div><dt>Độ tin cậy của xác suất (ECE)</dt><dd>Mô hình báo chắc 90% thì có đúng khoảng 90% số lần không. ECE càng thấp càng tốt; temperature scaling chỉnh xác suất mà không đổi dự đoán.</dd></div>
    </dl>
    <p class="field-hint mt-2">Tập học, tập kiểm tra và tập giữ riêng được định nghĩa ở <a href="#du-lieu">chương 2</a>. Kết quả theo từng chỉ số ở <a href="#ket-qua">chương 4</a>.</p>`;
}

function systemHtml() {
  return `
    <ul class="system-list mt-3">
      <li><b>Chẩn đoán ảnh.</b> Máy chủ chạy bảy mô hình đã huấn luyện trên cùng một ảnh và hiện kết quả cạnh nhau, kèm vùng ảnh mô hình dựa vào (Grad-CAM).</li>
      <li><b>Học liên kết từ trình duyệt.</b> Mỗi người tham gia là một client thật: trình duyệt tính đặc trưng ảnh bằng mạng nền đóng băng của FedAvg, huấn luyện lớp phân loại cuối (khoảng 39 nghìn tham số) và chỉ gửi trọng số lớp đó (khoảng 156 KB) lên máy chủ. Máy chủ kiểm tra rồi gộp theo FedAvg qua từng vòng.</li>
      <li><b>Kiểm thử hệ thống.</b> Phiên Flower với năm tiến trình client trên máy chủ chỉ để kiểm chứng điều phối và phép gộp, không đo độ chính xác. Xem <a href="#phu-luc">Phụ lục kỹ thuật</a>.</li>
    </ul>`;
}

export async function mount() {
  const root = byId("methods-root");
  const evidence = await getEvidence().catch(error => ({ available: false, reason: error.message }));
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
      : section(3, "Mô hình và thiết lập thí nghiệm", setupHtml(evidence))}
    ${section(4, "Cách đánh giá", metricsHtml())}
    ${section(5, "Hệ thống demo", systemHtml())}`;
  revealItems(root.querySelectorAll(".pillar"), { stagger: 60 });
}
