// Chương 5 · Kết luận: trả lời câu hỏi nghiên cứu, hạn chế (nói thẳng, hiện sẵn), hướng phát triển.
import { getEvidence, getSummary, modelsById } from "../core/store.js";
import { byId, esc, fmtInt, fmtNum, fmtPct, revealItems, sectionHead } from "../core/dom.js";

const LOCALS = ["local_00", "local_01", "local_02", "local_03", "local_04"];

function baseLimits(seed) {
  return [
    ["Mới có một kịch bản và một lần chạy", `Mới thử mức lệch dữ liệu α = 0,1 với seed ${seed ?? "—"}, chưa đủ để nói kết quả ổn định khi điều kiện thay đổi (chương 2 liệt kê 12 cách chia, nhưng chỉ α = 0,1 có mô hình).`],
    ["Chưa có kiểm thử thực địa độc lập", "PlantVillage là ảnh chụp trong phòng; PlantDoc là ảnh ngoài đồng nhưng chỉ phủ một phần các nhóm và không có lá khỏe. Cả tập kiểm tra lẫn tập giữ riêng đều lấy từ cùng bộ dữ liệu."],
    ["Chỉ nhận ảnh một chiếc lá đã cắt", "Mô hình không tự tìm lá trong ảnh toàn cây; ảnh khác kiểu sẽ cho kết quả kém dù xác suất dự đoán hiển thị vẫn cao."],
    ["Xác suất dự đoán chỉ được hiệu chuẩn một phần", "Web hiện cả xác suất gốc và xác suất đã hiệu chuẩn. Việc hiệu chuẩn học trên ảnh chủ yếu là PlantVillage nên với ảnh ngoài đồng mô hình vẫn gán xác suất cao hơn độ chính xác thực (xem chương 4, mục 4.4)."],
    ["Chưa thử nghiệm client lỗi hay bị tấn công", "Máy chủ có cơ chế kiểm tra bản cập nhật (chạy thử ở Phụ lục kỹ thuật), nhưng chưa có thí nghiệm đo độ chính xác nên chưa kết luận được."],
    ["Năm client là giả lập", "Năm client là năm phần của cùng một bộ dữ liệu, không phải năm nông hộ thật; không có độ trễ mạng hay thiết bị khác nhau."],
    ["Học liên kết cộng đồng chỉ là thử nghiệm", "Người tham gia chỉ học lớp cuối trên đặc trưng của backbone đóng băng, với ảnh và nhãn tự gán (chưa có chuyên gia kiểm). Máy chủ chấm bằng tập đã dùng để chọn mô hình gốc nên con số chỉ là tín hiệu theo dõi; khi mỗi người chỉ có vài nhóm thì mô hình dễ kém đi và máy chủ sẽ từ chối vòng đó. Mô hình cộng đồng không nằm trong bảy mô hình ở chương 4."],
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
      `Phiên bản Centralized được chọn ở epoch ${fmtInt(sel.chosen_epoch)}/${fmtInt(sel.epochs)} vì có điểm PlantDoc ${sel.chosen_rank_among_epochs === 1 ? "cao nhất" : `hạng ${fmtInt(sel.chosen_rank_among_epochs)}`} trên tập kiểm tra (${fmtPct(pdVal)}; epoch cuối ${fmtPct(sel.last_plantdoc_accuracy)}), nên số ở tập này có thể hơi đẹp hơn thực tế. Tập giữ riêng (chương 4) không dùng để chọn nên không bị thiên lệch đó.${observed}`]);
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
      `${fmtInt(leaf.ungrouped_images)} ảnh PlantVillage trong tập giữ riêng không được gắn nhóm theo lá; ${fmtInt(leaf.suspected_reshoots)} ảnh trong đó có ảnh học cùng loại với số thứ tự liền kề (nghi là chụp lại cùng lá). Bỏ các ảnh đó: Centralized ${fmtPct(c.all)} → ${fmtPct(c.without_suspected_reshoots)}, FedAvg ${fmtPct(f.all)} → ${fmtPct(f.without_suspected_reshoots)} — ${holds ? "chưa thấy dấu hiệu con số bị phóng đại" : "độ chính xác giảm, con số có thể bị thổi phồng một phần"}; muốn loại hẳn rủi ro cần gom lại nhóm lá và dựng lại bộ dữ liệu.`]);
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
    "Các cách học hiện khác bộ tối ưu và số bước huấn luyện; FedAvg dừng sớm. Vì vậy chưa tách được ảnh hưởng của cách học khỏi ảnh hưởng của thiết lập. Cần huấn luyện lại cùng điều kiện, nhiều seed trước khi kết luận hai cách tương đương."]);
  return out;
}


/** Ba câu trả lời cho ba câu hỏi ở chương 1; mọi số tính từ cùng bộ số liệu của chương 4. */
function answersHtml(evidence) {
  const m = modelsById(evidence);
  const central = m.centralized.metrics;
  const fed = m.fedavg.metrics;
  const bestLocalPv = Math.max(...LOCALS.map(id => m[id].metrics.plantvillage.top1_accuracy));
  // hiệu số FedAvg trừ đối chứng, điểm phần trăm: dương là FedAvg cao hơn
  const vsCentral = (fed.plantvillage.top1_accuracy - central.plantvillage.top1_accuracy) * 100;
  const vsLocal = (fed.plantvillage.top1_accuracy - bestLocalPv) * 100;
  const vsDomain = (fed.plantdoc.top1_accuracy - fed.plantvillage.top1_accuracy) * 100;
  const signed = value => `${value >= 0 ? "+" : "−"}${fmtNum(Math.abs(value), 1)} điểm`;
  return `
    <div class="conclusion-grid mt-3">
      <article class="conclusion fedavg">
        <span class="conclusion-label">So với gom ảnh (Centralized)</span>
        <b>${signed(vsCentral)}</b>
        <span class="conclusion-sub">FedAvg so với Centralized · điểm phần trăm · PlantVillage${Math.abs(vsCentral) <= 0.5 ? " · gần ngang nhau" : ""}</span>
        <p>Trên PlantVillage: FedAvg ${fmtPct(fed.plantvillage.top1_accuracy)}, Centralized ${fmtPct(central.plantvillage.top1_accuracy)}. Chênh lệch chỉ mang tính tham khảo vì điều kiện huấn luyện hai bên khác nhau.</p>
      </article>
      <article class="conclusion local">
        <span class="conclusion-label">So với học riêng (Local-only)</span>
        <b>${signed(vsLocal)}</b>
        <span class="conclusion-sub">FedAvg so với Local-only tốt nhất · điểm phần trăm · PlantVillage</span>
        <p>So với mô hình học riêng tốt nhất trên PlantVillage. Mỗi client chỉ thấy một phần nhóm bệnh nên mô hình riêng không nhận ra nhóm chưa từng thấy (<a href="#du-lieu">chương 2</a>).</p>
      </article>
      <article class="conclusion warn">
        <span class="conclusion-label">Từ ảnh trong phòng sang ngoài đồng</span>
        <b>${signed(vsDomain)}</b>
        <span class="conclusion-sub">FedAvg trên PlantDoc so với PlantVillage · điểm phần trăm</span>
        <p>Từ PlantVillage (${fmtPct(fed.plantvillage.top1_accuracy)}) sang PlantDoc (${fmtPct(fed.plantdoc.top1_accuracy)}). Nguyên nhân chưa được kiểm chứng: có thể do ảnh khác kiểu, PlantDoc ít ảnh và chỉ phủ một phần nhóm, hoặc nhãn (<a href="#ket-qua">chương 4</a>).</p>
      </article>
    </div>`;
}

const FUTURE = [
  ["Chạy lặp lại nhiều lần", "Chạy nhiều seed và nhiều mức lệch dữ liệu (không chỉ α = 0,1) để biết kết quả có ổn định không. Cần GPU."],
  ["Huấn luyện cùng điều kiện", "Cho Centralized, FedAvg và Local-only cùng bộ tối ưu và số bước, rồi so sánh lại."],
  ["Kiểm thử ngoài đồng", "Thu thêm ảnh ngoài đồng phủ đủ các nhóm bệnh, giữ riêng một tập chưa từng dùng để chọn mô hình."],
  ["Tự tìm và cắt lá", "Thêm bước phát hiện lá trên ảnh toàn cây để người dùng không phải tự cắt ảnh."],
  ["Bảo vệ riêng tư mạnh hơn", "Thêm differential privacy hoặc secure aggregation vào phép gộp."],
  ["Thử client lỗi hoặc tấn công", "Đo độ chính xác khi có client gửi bản cập nhật sai để biết cơ chế kiểm tra có đủ không."],
];

export async function mount() {
  const root = byId("conclusion-root");
  const [evidence, summary] = await Promise.all([getEvidence().catch(() => null), getSummary().catch(() => null)]);
  const seed = evidence?.seed ?? summary?.identity?.seed;
  const base = baseLimits(seed);
  // thứ tự: phạm vi của kết quả (2 mục đầu) -> các hạn chế của phép đánh giá (theo số liệu) -> còn lại
  const limits = [...base.slice(0, 2), ...(summary?.available ? evaluationLimits(summary) : []), ...base.slice(2)];
  root.innerHTML = `
    <section class="panel report-section" aria-labelledby="c-answers">
      ${sectionHead("5.1", "Kết luận", { id: "c-answers", lead: "Câu trả lời cho ba câu hỏi nghiên cứu" })}
      ${evidence?.available ? answersHtml(evidence) : `<div class="callout callout-warn mt-3" role="note"><b>Chưa có số liệu để kết luận.</b> ${esc(evidence?.reason || "")}</div>`}
      <p class="viewer-scope mt-3"><b class="scope-tag">Phạm vi</b> Kết quả từ một lần chạy với một cách chia lệch dữ liệu; tập kiểm tra cũng dùng để chọn mô hình, nên chưa phải độ chính xác ngoài thực địa.</p>
    </section>

    <section class="panel report-section mt-4" aria-labelledby="c-limits">
      ${sectionHead("5.2", "Hạn chế", { id: "c-limits", lead: "Những gì nghiên cứu chưa làm được" })}
      <div class="viewer-limits mt-3">
        <article><h4>Chưa đánh giá qua nhiều lần chạy</h4><p>Một kịch bản, một seed; các client là phân vùng giả lập.</p></article>
        <article><h4>Chưa có kiểm thử độc lập trong điều kiện thực tế</h4><p>Ảnh ngoài đồng hiện có chỉ phủ một phần nhóm lá; kết quả chưa đại diện cho triển khai thực tế.</p></article>
        <article><h4>Cần ảnh một lá đã cắt</h4><p>Web chưa tự tìm lá trên ảnh toàn cây. Xác suất dự đoán cao vẫn có thể sai.</p></article>
      </div>
      <p class="viewer-scope mt-3">Tập kiểm tra cũng dùng để chọn mô hình; tập giữ riêng chưa được đối soát trùng ảnh. Điều kiện huấn luyện giữa các cách học còn khác nhau.</p>
      <h4 class="subhead mt-4">Danh sách đầy đủ (${limits.length} mục)</h4>
      <ol class="limit-list mt-3">${limits.map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}</ol>
    </section>

    <section class="panel report-section mt-4" aria-labelledby="c-future">
      ${sectionHead("5.3", "Hướng phát triển", { id: "c-future", lead: "Việc nên làm tiếp" })}
      <ul class="future-list mt-3">${FUTURE.map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}</ul>
      <p class="field-hint mt-3">Tình trạng các hạng mục kiểm thử và việc còn dở của dự án nằm ở <a href="#phu-luc">Phụ lục kỹ thuật</a>.</p>
    </section>`;
  revealItems(root.querySelectorAll(".conclusion"), { stagger: 60 });
}
