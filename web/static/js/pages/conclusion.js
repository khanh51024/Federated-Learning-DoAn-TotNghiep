// Chương 5 · Kết luận: trả lời câu hỏi nghiên cứu, hạn chế (nói thẳng, hiện sẵn), hướng phát triển.
import { getEvidence, getSummary, modelsById } from "../core/store.js";
import { SOURCE_GROUPS, byId, esc, fmtInt, fmtNum, fmtPct, revealItems, sectionHead } from "../core/dom.js";
import { alphaText, calibrationShift, factsOf, runsText, scopeText } from "../core/facts-text.js";

const LOCALS = ["local_00", "local_01", "local_02", "local_03", "local_04"];

function baseLimits(seed, facts, summary) {
  const alpha = alphaText(facts) || "một mức lệch";
  const runs = facts?.seeds_run?.length || 0;
  const shift = calibrationShift(summary?.reeval, SOURCE_GROUPS);
  const names = list => list.map(r => r.group.label).join(", ");
  const fairnessLimit = summary?.fairness?.available
    ? ["Nhãn PlantDoc chưa được chuyên gia rà soát; độ công bằng giữa client chỉ đo theo phân bố nhóm", "Nhãn PlantDoc được ánh xạ về 38 nhóm của PlantVillage mà chưa có chuyên gia rà lại. Độ chính xác của FedAvg và Local-only đã được đo theo việc client có hay không có ảnh của từng nhóm (chương 2, mục 2.3), nhưng năm client là phân vùng giả lập nên chưa phản ánh sự khác biệt giữa các nông hộ thật (thiết bị, ánh sáng, cách chụp)."]
    : ["Nhãn PlantDoc chưa được chuyên gia rà soát, chưa đo độ công bằng giữa client", "Nhãn PlantDoc được ánh xạ về 38 nhóm của PlantVillage mà chưa có chuyên gia rà lại. Chưa có số liệu độ chính xác của FedAvg trên dữ liệu từng client, nên chưa đánh giá được tính công bằng giữa các client."];
  const calibrationText = shift?.worsened?.length
    ? `Hiệu chuẩn làm ECE của FedAvg giảm ở ${shift.improved.length ? names(shift.improved) : "không nhóm nào"} nhưng tăng ở ${names(shift.worsened)} (xem chương 4, mục 4.4), nên với ảnh khác nguồn ảnh đã học xác suất hiển thị vẫn không đáng tin.`
    : "Với ảnh khác nguồn ảnh đã học, xác suất hiển thị vẫn có thể cao hơn độ chính xác thực (xem chương 4, mục 4.4).";
  return [
    ["Mới có một kịch bản", `Mới thử ${alpha}; ${runs > 1 ? `đã chạy ${fmtInt(runs)} hạt giống, các trang chi tiết hiển thị hạt giống ${seed ?? "—"} (bảng các hạt giống ở trang Kết quả cho thấy độ dao động giữa các lần chạy)` : `các trang chi tiết hiển thị hạt giống ${seed ?? "—"}`}, chưa đủ để nói kết quả ổn định khi mức lệch thay đổi (chương 2 liệt kê các cách chia của release, nhưng chỉ ${alpha} có mô hình).`],
    ["Chưa có kiểm thử thực địa độc lập", "PlantVillage là ảnh chụp trong phòng; PlantDoc là ảnh ngoài đồng nhưng chỉ phủ một phần các nhóm và không có lá khỏe. Cả tập kiểm tra lẫn tập giữ riêng đều lấy từ cùng bộ dữ liệu."],
    ["Chỉ nhận ảnh một chiếc lá đã cắt", "Mô hình không tự tìm lá trong ảnh toàn cây; ảnh khác kiểu sẽ cho kết quả kém dù xác suất dự đoán hiển thị vẫn cao."],
    ["Xác suất dự đoán chỉ được hiệu chuẩn một phần", `Web hiện cả xác suất gốc và xác suất đã hiệu chuẩn. ${calibrationText}`],
    ["Chưa thử nghiệm client lỗi hay bị tấn công", "Máy chủ có cơ chế kiểm tra bản cập nhật (chạy thử ở Phụ lục kỹ thuật), nhưng chưa có thí nghiệm đo độ chính xác nên chưa kết luận được."],
    ["FedAvg chưa chạy lại qua Flower", `Kết quả FedAvg dùng runner in_process_simulation (5 client mô phỏng trong một tiến trình). Lịch sử từng vòng được ghi đủ${facts?.delta_l2_recorded === false ? ", nhưng độ lệch ‖Δ‖₂ của từng client chưa được ghi" : ""}${facts?.delta_l2_recorded === false ? " và " : ", nhưng "}số byte là kích thước mô hình, không phải lưu lượng đo trên mạng. Phiên Flower trên web dùng ảnh nhiễu, chỉ kiểm tra hệ thống; Local-only chưa có lịch sử từng vòng trong báo cáo.`],
    ["Chênh lệch miền dữ liệu là vấn đề của dữ liệu", "Mô hình Centralized đạt rất cao trên PlantVillage nhưng thấp hơn nhiều trên PlantDoc, nên khoảng cách này không riêng của học liên kết. Mohanty và cộng sự (2016) cũng ghi nhận độ chính xác giảm mạnh khi ảnh được chụp trong điều kiện khác; Noyan (2022) chỉ ra PlantVillage có thiên lệch khi chụp."],
    ["BatchNorm trong học liên kết", "MobileNetV3 có lớp BatchNorm; khi đặc trưng giữa các client khác nhau, việc lấy trung bình cả thống kê BatchNorm có thể gây hại (FedBN, Li và cộng sự 2021). Đồ án chưa thử nghiệm FedBN nên đây là giả thuyết cần kiểm chứng."],
    fairnessLimit,
    ["Năm client là giả lập", "Năm client là năm phần của cùng một bộ dữ liệu, không phải năm nông hộ thật; không có độ trễ mạng hay thiết bị khác nhau."],
    ["Học phân tán cộng đồng có giới hạn đã đo", "Người tham gia chỉ học lớp cuối trên đặc trưng của backbone đóng băng, với ảnh và nhãn tự gán (chưa có chuyên gia kiểm). Thí nghiệm mô phỏng ở mục 4.5 cho thấy mức tăng khiêm tốn (khoảng 1 đến 2 điểm phần trăm khi dữ liệu người tham gia đủ đa dạng), gần như không tăng với ảnh ngoài đồng, và vòng có ít người mà mỗi người chỉ có một nhóm bệnh thì dễ bị cổng chất lượng từ chối. Người thật chưa được thử ở quy mô lớn."],
    ["Riêng tư chưa tuyệt đối", "Học phân tán chỉ giữ ảnh gốc không rời client; trọng số vẫn có thể làm lộ thông tin. Muốn bảo đảm mạnh hơn cần thêm kỹ thuật như differential privacy hoặc secure aggregation."],
  ];
}

/** Hạn chế về phép đánh giá: số liệu lấy từ báo cáo thống kê đã kiểm danh tính, không viết cứng. */
function evaluationLimits(summary) {
  const st = summary?.statistics;
  const out = [];
  const sel = st?.selection_effect;
  if (sel?.available) {
    const central = (summary?.holdout?.models || []).find(m => m.model_id === "centralized");
    // gói train_v4 chỉ ghi Macro-F1 PlantDoc theo epoch; gói cũ ghi độ chính xác: so đúng loại số với tập giữ riêng
    const macro = sel.value_kind === "macro_f1";
    const what = macro ? "Macro-F1" : "độ chính xác";
    const pdTest = macro ? central?.plantdoc?.macro_f1 : central?.plantdoc?.top1;
    const pdVal = sel.chosen_plantdoc_value;
    const observed = pdTest === undefined || pdTest === null ? ""
      : pdTest >= pdVal
        ? ` Khi chấm lại trên tập giữ riêng, Centralized đạt ${what} ${fmtPct(pdTest)} trên ảnh ngoài đồng (tập kiểm tra: ${fmtPct(pdVal)}) — chưa thấy con số bị đẩy lên, nhưng tập giữ riêng chỉ có ${fmtInt(summary.holdout.counts_by_source?.plantdoc)} ảnh ngoài đồng.`
        : ` Khi chấm lại trên tập giữ riêng, Centralized chỉ đạt ${what} ${fmtPct(pdTest)} trên ảnh ngoài đồng (tập kiểm tra: ${fmtPct(pdVal)}) — phù hợp với thiên lệch do cách chọn.`;
    const why = macro ? "theo trung bình Macro-F1 của các nguồn ảnh"
      : `vì có điểm PlantDoc ${sel.chosen_rank_among_epochs === 1 ? "cao nhất" : `hạng ${fmtInt(sel.chosen_rank_among_epochs)}`}`;
    const fed = sel.fedavg?.available
      ? ` FedAvg được chọn ở vòng ${fmtInt(sel.fedavg.chosen_epoch)}/${fmtInt(sel.fedavg.epochs)} bằng cùng cách (${what} PlantDoc ${fmtPct(sel.fedavg.chosen_plantdoc_value)}; vòng cuối ${fmtPct(sel.fedavg.last_plantdoc_value)}).`
      : "";
    out.push(["Mô hình được chọn dựa trên chính tập kiểm tra",
      `Phiên bản Centralized được chọn ở epoch ${fmtInt(sel.chosen_epoch)}/${fmtInt(sel.epochs)} ${why} trên tập kiểm tra (${what} PlantDoc ${fmtPct(pdVal)}; epoch cuối ${fmtPct(sel.last_plantdoc_value)}), nên số ở tập này có thể hơi đẹp hơn thực tế. Tập giữ riêng (chương 4) không dùng để chọn nên không bị thiên lệch đó.${observed}${fed}`]);
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
  out.push(["Công thức huấn luyện được chọn trên Centralized",
    "Ba cách dùng chung một công thức và cùng số bước gradient, nhưng các thiết lập chung (kích thước batch, tốc độ học, tăng cường) được chọn bằng thử nghiệm ngắn chỉ trên Centralized. FedAvg trên dữ liệu lệch nhãn có thể cần tốc độ học khác; chưa dò riêng nên khoảng cách Centralized − FedAvg có thể bị phóng đại một phần."]);
  return out;
}


/** Ba câu trả lời cho ba câu hỏi ở chương 1; mọi số tính từ cùng bộ số liệu của chương 4. */
function answersHtml(evidence, facts) {
  const m = modelsById(evidence);
  const central = m.centralized.metrics;
  const fed = m.fedavg.metrics;
  const bestLocalPv = Math.max(...LOCALS.map(id => m[id].metrics.plantvillage.top1_accuracy));
  // hiệu số FedAvg trừ đối chứng, điểm phần trăm: dương là FedAvg cao hơn
  const vsCentral = (fed.plantvillage.top1_accuracy - central.plantvillage.top1_accuracy) * 100;
  const vsLocal = (fed.plantvillage.top1_accuracy - bestLocalPv) * 100;
  const vsDomain = (fed.plantdoc.top1_accuracy - fed.plantvillage.top1_accuracy) * 100;
  const fieldOther = fed.field_other?.top1_accuracy;
  const signed = value => `${value >= 0 ? "+" : "−"}${fmtNum(Math.abs(value), 1)} điểm`;
  return `
    <div class="conclusion-grid mt-3">
      <article class="conclusion fedavg">
        <span class="conclusion-label">So với gom ảnh (Centralized)</span>
        <b>${signed(vsCentral)}</b>
        <span class="conclusion-sub">FedAvg so với Centralized · điểm phần trăm · PlantVillage${Math.abs(vsCentral) <= 0.5 ? " · gần ngang nhau" : ""}</span>
        <p>Trên PlantVillage: FedAvg ${fmtPct(fed.plantvillage.top1_accuracy)}, Centralized ${fmtPct(central.plantvillage.top1_accuracy)}. ${facts?.same_recipe === false ? "Chênh lệch chỉ mang tính tham khảo vì điều kiện huấn luyện hai bên khác nhau." : facts?.same_recipe === true ? `Hai bên dùng cùng thiết lập huấn luyện và cùng số bước; số liệu là của hạt giống ${esc(evidence.seed)}.` : ""}</p>
      </article>
      <article class="conclusion local">
        <span class="conclusion-label">So với học riêng (Local-only)</span>
        <b>${signed(vsLocal)}</b>
        <span class="conclusion-sub">FedAvg so với Local-only tốt nhất · điểm phần trăm · PlantVillage</span>
        <p>So với mô hình học riêng tốt nhất trên PlantVillage. Mỗi client chỉ thấy một phần nhóm bệnh nên mô hình riêng có thể nhận diện kém các nhóm chưa được học (<a href="#du-lieu">chương 2</a>).</p>
      </article>
      <article class="conclusion warn">
        <span class="conclusion-label">Từ ảnh trong phòng sang ngoài đồng</span>
        <b>${signed(vsDomain)}</b>
        <span class="conclusion-sub">FedAvg trên PlantDoc so với PlantVillage · điểm phần trăm</span>
        <p>Từ PlantVillage (${fmtPct(fed.plantvillage.top1_accuracy)}) ${fieldOther !== undefined ? `sang ảnh ngoài đồng của các bộ khác (${fmtPct(fieldOther)}) và PlantDoc` : "sang PlantDoc"} (${fmtPct(fed.plantdoc.top1_accuracy)}). Nguyên nhân chưa được kiểm chứng: có thể do ảnh khác kiểu, PlantDoc ít ảnh và chỉ phủ một phần nhóm, hoặc nhãn (<a href="#ket-qua">chương 4</a>).</p>
      </article>
    </div>`;
}

/** Hướng phát triển: bỏ mục đã làm xong (ví dụ "cùng điều kiện" khi các cách học đã cùng thiết lập). */
function futureList(facts) {
  return FUTURE.filter(([title]) => !(title === "Huấn luyện cùng điều kiện" && facts?.same_recipe === true))
    .map(([title, text]) => (title === "Chạy lặp lại nhiều lần"
      ? [title, `Chạy nhiều mức lệch dữ liệu (không chỉ ${alphaText(facts) || "một mức"}) để biết kết quả có ổn định khi điều kiện thay đổi không, và dò riêng tốc độ học cho FedAvg trên dữ liệu lệch. Cần GPU.`]
      : [title, text]));
}

const FUTURE = [
  ["Chạy lặp lại nhiều lần", ""],
  ["Đối chiếu FedAvg trên Flower", "Dùng cùng release, W0, phân hoạch và cấu hình; lưu đủ log, đối chiếu checkpoint với runner hiện tại, rồi đo mạng thật riêng."],
  ["Thuật toán chịu non-IID", "Thử FedProx, SCAFFOLD và FedBN trên cùng phân hoạch; đây là kế hoạch, chưa có kết quả để khẳng định cải thiện."],
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
  const facts = factsOf(summary);
  const base = baseLimits(seed, facts, summary);
  const future = futureList(facts);
  // thứ tự: phạm vi của kết quả (2 mục đầu) -> các hạn chế của phép đánh giá (theo số liệu) -> còn lại
  const limits = [...base.slice(0, 2), ...(summary?.available ? evaluationLimits(summary) : []), ...base.slice(2)];
  root.innerHTML = `
    <section class="panel report-section" aria-labelledby="c-answers">
      ${sectionHead("5.1", "Kết luận", { id: "c-answers", lead: "Câu trả lời cho ba câu hỏi nghiên cứu" })}
      ${evidence?.available ? answersHtml(evidence, facts) : `<div class="callout callout-warn mt-3" role="note"><b>Chưa có số liệu để kết luận.</b> ${esc(evidence?.reason || "")}</div>`}
      <p class="viewer-scope mt-3"><b class="scope-tag">Phạm vi</b> ${esc(scopeText(facts))}</p>
    </section>

    <section class="panel report-section mt-4" aria-labelledby="c-limits">
      ${sectionHead("5.2", "Hạn chế", { id: "c-limits", lead: "Những gì nghiên cứu chưa làm được" })}
      <div class="viewer-limits mt-3">
        <article><h4>Chưa đánh giá qua nhiều kịch bản</h4><p>Một mức lệch dữ liệu${facts?.alpha != null ? ` (${esc(alphaText(facts))})` : ""}, ${esc(runsText(facts))}; các client là phân vùng giả lập.</p></article>
        <article><h4>Chưa có kiểm thử độc lập trong điều kiện thực tế</h4><p>Ảnh ngoài đồng hiện có chỉ phủ một phần nhóm lá; kết quả chưa đại diện cho triển khai thực tế.</p></article>
        <article><h4>Cần ảnh một lá đã cắt</h4><p>Web chưa tự tìm lá trên ảnh toàn cây. Xác suất dự đoán cao vẫn có thể sai.</p></article>
      </div>
      <p class="viewer-scope mt-3">Tập kiểm tra cũng dùng để chọn mô hình${facts?.holdout_audited === false ? "; tập giữ riêng chưa được đối soát trùng ảnh" : ""}.${facts?.same_recipe === false ? " Điều kiện huấn luyện giữa các cách học còn khác nhau." : ""}</p>
      <h4 class="subhead mt-4">Danh sách đầy đủ (${limits.length} mục)</h4>
      <ol class="limit-list mt-3">${limits.map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}</ol>
    </section>

    <section class="panel report-section mt-4" aria-labelledby="c-future">
      ${sectionHead("5.3", "Hướng phát triển", { id: "c-future", lead: "Việc nên làm tiếp" })}
      <ul class="future-list mt-3">${future.map(([title, text]) => `<li><b>${esc(title)}.</b> ${esc(text)}</li>`).join("")}</ul>
      <p class="field-hint mt-3">Tình trạng các hạng mục kiểm thử và việc còn dở của dự án nằm ở <a href="#phu-luc">Phụ lục kỹ thuật</a>.</p>
    </section>
    <section class="panel report-section mt-4" aria-labelledby="c-progress">
      ${sectionHead("5.4", "Tiến độ và nguồn tham khảo", { id: "c-progress" })}
      <div tabindex="0" role="region" aria-label="Tiến độ theo giai đoạn" class="table-responsive mt-3"><table class="data-table wrap-head"><thead><tr><th scope="col">Giai đoạn</th><th scope="col">Đã có</th><th scope="col">Còn thiếu</th></tr></thead><tbody>
        <tr><th scope="row">Mốc so sánh</th><td>Centralized, FedAvg và 5 Local-only; ${facts?.same_recipe === true ? "cùng công thức, cùng số bước, " : ""}cùng validation; đánh giá bổ sung trên tập test giữ riêng</td><td>${facts?.same_recipe === true ? "Dò riêng tốc độ học cho FedAvg trên dữ liệu lệch" : "Đối chứng cùng bộ tối ưu và ngân sách"}</td></tr>
        <tr><th scope="row">Non-IID</th><td>Các phân hoạch của release; kết quả huấn luyện ${esc(alphaText(facts) || "một mức lệch")}, hạt giống ${seed ?? "—"} hiển thị${facts?.seeds_run?.length ? ` (${fmtInt(facts.seeds_run.length)} hạt giống đã chạy)` : ""}</td><td>Huấn luyện nhiều mức lệch</td></tr>
        <tr><th scope="row">Flower</th><td>Mã legacy và phiên localhost dùng ảnh nhiễu trên web</td><td>Chạy lại FedAvg trên Flower và đối chiếu runner</td></tr>
        <tr><th scope="row">Thuật toán và client</th><td>MobileNetV3-Small; học lớp cuối trong demo cộng đồng</td><td>FedProx / SCAFFOLD / FedBN, lượng tử hóa, thử nhiều máy và client thật</td></tr>
        <tr><th scope="row">Đánh giá triển khai</th><td>Phân tích lỗi, hiệu chuẩn, phép đo ảnh toàn cảnh trên dữ liệu hiện có</td><td>Tập thực địa mới, nhãn được chuyên gia duyệt, test cuối cùng được niêm phong</td></tr>
      </tbody></table></div>
      <p class="field-hint mt-2">Báo cáo tiến độ, không tuyên bố đã hoàn thành mọi giai đoạn. Số liệu huấn luyện truy từ checkpoint và manifest đã đối chiếu; kết quả thống kê lấy từ báo cáo đánh giá của dự án.</p>
      <h4 class="subhead mt-4">Nguồn nền tảng</h4>
      <ol class="limit-list mt-3">
        <li>Hughes và Salathé (2015), <a href="https://arxiv.org/abs/1511.08060" target="_blank" rel="noopener noreferrer">An open access repository of images on plant health</a> — nguồn PlantVillage.</li>
        <li>Mohanty, Hughes và Salathé (2016), <a href="https://doi.org/10.3389/fpls.2016.01419" target="_blank" rel="noopener noreferrer">Using Deep Learning for Image-Based Plant Disease Detection</a> — độ chính xác giảm khi ảnh khác điều kiện chụp.</li>
        <li>Noyan (2022), <a href="https://arxiv.org/abs/2206.04374" target="_blank" rel="noopener noreferrer">Uncovering Bias in the PlantVillage Dataset</a> — thiên lệch nền của PlantVillage.</li>
        <li>Zhao và cộng sự (2018), <a href="https://arxiv.org/abs/1806.00582" target="_blank" rel="noopener noreferrer">Federated Learning with Non-IID Data</a> — độ phân kỳ trọng số khi dữ liệu non-IID.</li>
        <li>Li và cộng sự, <a href="https://arxiv.org/abs/1812.06127" target="_blank" rel="noopener noreferrer">FedProx</a> (2020), <a href="https://arxiv.org/abs/1910.06378" target="_blank" rel="noopener noreferrer">SCAFFOLD</a> (Karimireddy và cộng sự, 2020), <a href="https://arxiv.org/abs/2102.07623" target="_blank" rel="noopener noreferrer">FedBN</a> (2021) — các thuật toán chịu non-IID dự kiến thử ở giai đoạn sau.</li>
        <li>Singh và cộng sự, <a href="https://arxiv.org/abs/1911.10317" target="_blank" rel="noopener noreferrer">PlantDoc: A Dataset for Visual Plant Disease Detection</a> — nguồn ảnh thực địa; các bản classification và object detection có quy mô khác nhau.</li>
        <li>McMahan và cộng sự (2017), <a href="https://proceedings.mlr.press/v54/mcmahan17a.html" target="_blank" rel="noopener noreferrer">Communication-Efficient Learning of Deep Networks from Decentralized Data</a> — FedAvg.</li>
        <li>Hsu, Qi và Brown (2019), <a href="https://arxiv.org/abs/1909.06335" target="_blank" rel="noopener noreferrer">Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification</a> — khảo sát non-IID; không chuyển số liệu CIFAR-10 thành kết quả của đồ án.</li>
        <li>Howard và cộng sự (2019), <a href="https://arxiv.org/abs/1905.02244" target="_blank" rel="noopener noreferrer">Searching for MobileNetV3</a> — kiến trúc mạng.</li>
        <li>Beutel và cộng sự (2020), <a href="https://arxiv.org/abs/2007.14390" target="_blank" rel="noopener noreferrer">Flower: A Friendly Federated Learning Research Framework</a> — nền tảng hệ thống.</li>
      </ol>
    </section>`;
  revealItems(root.querySelectorAll(".conclusion"), { stagger: 60 });
}
