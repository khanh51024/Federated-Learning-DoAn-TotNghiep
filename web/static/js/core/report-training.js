// Report additions sourced from /api/evidence/summary.training (checkpoint and manifest identity checked server-side).
import { esc, fmtInt, fmtNum, fmtPct, formatDiseaseName, modelPill } from "./dom.js";

export const modelTraining = (training, id) => training?.models?.find(m => m.model_id === id);
const missing = training => `<p class="field-hint">Chưa có dữ liệu cấu hình đã đối chiếu: ${esc(training?.reason || "báo cáo chưa khả dụng")}</p>`;
const table = (head, rows, label) => `<div tabindex="0" role="region" aria-label="${esc(label)}" class="table-responsive mt-3"><table class="data-table wrap-head"><thead><tr>${head.map(h => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;

export function releaseHtml(training) {
  if (!training?.available) return missing(training);
  const roles = {
    train: ["Huấn luyện (train)", "Cập nhật trọng số"], val: ["Kiểm định (validation)", "Chọn checkpoint"],
    calibration: ["Hiệu chỉnh (calibration)", "Fit temperature scaling sau huấn luyện"],
    legacy_diagnostic_test: ["Giữ riêng (legacy diagnostic test)", "Đánh giá bổ sung, không chọn checkpoint"],
    quarantine: ["Cách ly (quarantine)", "Loại khỏi huấn luyện và đánh giá chính"],
  };
  const rows = training.release.splits.map(s => `<tr><th scope="row">${roles[s.name][0]}</th><td>${fmtInt(s.by_source.plantdoc)}</td><td>${fmtInt(s.by_source.plantvillage)}</td><td><b>${fmtInt(s.count)}</b></td><td>${roles[s.name][1]}</td></tr>`).join("");
  return `<h4 class="subhead mt-4">Đủ năm phần của release ${esc(training.release.name)}</h4>
    ${table(["Phần dữ liệu", "PlantDoc", "PlantVillage", "Tổng ảnh", "Vai trò"], rows, "Số ảnh của năm phần release")}
    <p class="field-hint mt-2">Trên web, “tập kiểm tra” là validation (dùng chọn mô hình); “tập giữ riêng” là legacy diagnostic test. Tập giữ riêng chưa phải test thực địa mới được niêm phong. Audit hiện có chỉ bao phủ train, validation và calibration.</p>
    <h4 class="subhead mt-4">Tiền xử lý dùng cho các checkpoint công bố</h4>
    <ol class="method-steps mt-3"><li><b>Xây release.</b> Cắt lá PlantDoc theo hộp bao, nới mỗi phía 8%; ánh xạ về 38 lớp, cách ly mẫu rủi ro và cố định manifest theo nhóm ảnh gốc.</li>
    <li><b>Nạp ảnh.</b> canonical_v1: RGB → resize trực tiếp 224 × 224 (bilinear, antialias) → tensor [0, 1] → chuẩn hóa ImageNet, mean (0,485; 0,456; 0,406), std (0,229; 0,224; 0,225). Khi train có lật ngang ngẫu nhiên p = 0,5; validation và suy luận không tăng cường ngẫu nhiên.</li>
    <li><b>Lấy mẫu khi train.</b> Centralized và FedAvg ghi trong cấu hình: khoảng 25% lượt lấy từ PlantDoc, có hoàn lại. Đây là tỷ lệ lượt học; số ảnh thật và trọng số gộp n_k / Σn_k vẫn theo manifest. Checkpoint Local-only không lưu cấu hình sampler để xác minh riêng.</li></ol>
    <p class="field-hint mt-2">Word mô tả bước xoay EXIF khi nạp ảnh; bundle transform chưa xác minh được khâu đọc ảnh gốc. Web hiện đọc ảnh sang RGB nhưng chưa xoay theo EXIF: nên xoay ảnh đúng hướng trước khi tải. Resize vuông có thể làm méo tỷ lệ; mô hình không tự tìm lá trên ảnh toàn cây.</p>`;
}

export function configsHtml(training) {
  if (!training?.available) return missing(training);
  const central = modelTraining(training, "centralized");
  const fed = modelTraining(training, "fedavg");
  const c = central.config, f = fed.config;
  const rows = [
    ["Bộ tối ưu / tốc độ học ban đầu", `${esc(c.optimizer)} / ${esc(c.lr)}`, `${esc(f.client_optimizer)} / ${esc(f.client_lr)} (momentum ${esc(f.client_momentum)})`, "Không lưu trong checkpoint Local-only"],
    ["Batch / weight decay / AMP", `${esc(c.batch_size)} / ${esc(c.weight_decay)} / ${c.amp ? "Có" : "Không"}`, `${esc(f.batch_size)} / ${esc(f.client_weight_decay)} / ${f.amp ? "Có" : "Không"}`, "Chưa xác minh từ metadata"],
    ["Ngân sách → đã chạy → được chọn", `${fmtInt(c.epochs)} → ${fmtInt(central.completed_steps)} → epoch ${fmtInt(central.selected_step)}`, `${fmtInt(f.rounds)} → ${fmtInt(fed.completed_steps)} → vòng ${fmtInt(fed.selected_step)}`, "60 vòng; mỗi client chọn riêng (bảng dưới)"],
    ["Local epoch mỗi vòng", "Không áp dụng", fmtInt(f.local_epochs), "Báo cáo nhóm: 1; checkpoint không lưu cấu hình"],
    ["Runtime của kết quả công bố", "Huấn luyện tập trung", "in_process_simulation; 5 client tuần tự", "5 mô hình riêng, không gộp"],
  ].map(row => `<tr><th scope="row">${row[0]}</th>${row.slice(1).map(v => `<td>${v}</td>`).join("")}</tr>`).join("");
  const models = training.models.map(m => `<tr><th scope="row">${modelPill(m.model_id)}</th><td>${m.model_id === "centralized" ? "Epoch" : "Vòng"} ${fmtInt(m.selected_step)} / ${fmtInt(m.completed_steps)}</td><td>${fmtNum(m.checkpoint_bytes / 1e6, 2)} MB</td><td title="${esc(training.checkpoint_sha256[m.model_id])}"><code>${esc(training.checkpoint_sha256[m.model_id].slice(0, 16))}…</code></td></tr>`).join("");
  return `<h4 class="subhead mt-4">Cấu hình thực tế đã lưu</h4>${table(["Thiết lập", "Centralized v6", "FedAvg v7", "Local-only v7"], rows, "Cấu hình huấn luyện ba phương pháp")}
    <p class="field-hint mt-2">Cùng W0, seed 42, MobileNetV3-Small và canonical_v1. Các cách học khác bộ tối ưu và ngân sách: đây là mốc tham chiếu, chưa phải đối chứng chỉ thay cách phân tán. Chọn checkpoint theo macro-F1 PlantDoc khi đạt ngưỡng chất lượng PlantVillage.</p>
    <h4 class="subhead mt-4">Bảy bộ trọng số dùng khi đánh giá và nhận diện</h4>${table(["Mô hình", "Mốc chọn / đã chạy", "Dung lượng tệp", "SHA-256 (rút gọn)"], models, "Bảy checkpoint đã kiểm SHA-256")}
    <p class="field-hint mt-2">Trọng số mô hình là tensor đã học; p_k trong FedAvg là hệ số gộp theo số ảnh, hai khái niệm khác nhau. Tệp Centralized lớn hơn vì lưu kèm trạng thái AdamW; dung lượng checkpoint không phải dung lượng truyền mỗi vòng.</p>
    <h4 class="subhead mt-4">Flower nằm ở đâu?</h4>
    <p class="section-lead">Nhánh legacy có ServerApp, ClientApp và launcher gọi Flower + Ray. FedAvg v7 trong bảng kết quả dùng runner một tiến trình; chạy lại v7 qua Flower và đối chiếu vòng 27 vẫn là việc tiếp theo. Phiên Flower trên web chạy một server và năm tiến trình client localhost với ảnh nhiễu: chứng minh điều phối hoạt động, không tạo ra kết quả v7 hay đo mạng giữa năm máy.</p>
    <p class="field-hint mt-2">Xem <a href="#flower">bằng chứng Flower</a>; muốn chạy phiên mới, dùng <a href="#phu-luc">Phụ lục kỹ thuật, mục D</a>. “Đã chạy thành công” và “đang có tiến trình chạy” là hai trạng thái cần đọc riêng.</p>`;
}

export function trainingHistoryHtml(training) {
  if (!training?.available) return missing(training);
  const fed = modelTraining(training, "fedavg");
  const bytes = fed.history.reduce((sum, r) => sum + r.clients.reduce((s, c) => s + c.payload_down_bytes + c.payload_up_bytes, 0), 0);
  const seconds = fed.history.reduce((sum, r) => sum + r.duration_sec, 0);
  const first = fed.history[0].clients[0];
  const perRound = fed.history[0].clients.reduce((sum, c) => sum + c.payload_down_bytes + c.payload_up_bytes, 0);
  const durations = fed.history.map(r => r.duration_sec);
  return `<details class="overview-detail mt-4" id="fedavg-history"><summary><span><strong>Lịch sử FedAvg v7 và telemetry ${fmtInt(fed.history.length)} vòng</strong><small>Checkpoint vòng ${fmtInt(fed.selected_step)}; log không bao phủ vòng 28–32</small></span></summary>
    <div class="overview-detail-body"><p class="panel-desc">Trích từ checkpoint đã kiểm hash. Top-1 trên validation 4.613 ảnh; macro-F1 PlantDoc trên 269 ảnh. Thời gian ${fmtNum(seconds, 2)} giây chỉ tính ${fmtInt(fed.history.length)} vòng có log.</p>
    <div class="chart-container mt-3"><canvas id="chart-fedavg-history" aria-label="Top-1 validation và macro-F1 PlantDoc qua ${fed.history.length} vòng FedAvg"></canvas></div>
    ${driftHtml(fed)}
    <div class="form-group mt-4"><label for="fedavg-round">Xem cập nhật từng client ở vòng</label><select id="fedavg-round" class="input-field">${fed.history.map(r => `<option value="${r.round}"${r.round === fed.selected_step ? " selected" : ""}>Vòng ${r.round}</option>`).join("")}</select></div>
    <div id="fedavg-client-telemetry"></div>
    <p class="field-hint mt-3">Mỗi chiều của một client ≈ ${fmtNum(first.payload_down_bytes / 1e6, 2)} MB (float32); một vòng (${fed.history[0].clients.length} client × tải xuống + tải lên) ≈ ${fmtNum(perRound / 1e6, 1)} MB; thời gian một vòng ${fmtNum(Math.min(...durations), 0)}–${fmtNum(Math.max(...durations), 0)} giây. Tổng payload tải xuống + lên: ${fmtNum(bytes / 1e6, 1)} MB cho ${fed.history.length} vòng, ${fed.history[0].clients.length} client. Đây là ước tính từ tensor trong mô phỏng; không gồm overhead giao thức, không phải byte đo qua mạng. Không có RTT mạng thật cho v7. Loss và ΔL2 không phải độ trễ.</p></div></details>`;
}

export function telemetryHtml(round) {
  const rows = round.clients.map(c => `<tr><th scope="row">${esc(c.client_id)}</th><td>${fmtInt(c.sample_count_n_k)}</td><td>${fmtNum(c.weight_p_k, 4)}</td><td>${fmtNum(c.train_loss, 4)}</td><td>${fmtNum(c.delta_l2_learnable, 3)}</td><td>${fmtNum(bnNorm(c), 1)}</td><td>${fmtInt(c.optimizer_steps_applied)}</td></tr>`).join("");
  return `<p class="field-hint">Vòng ${round.round}: ${fmtNum(round.duration_sec, 2)} giây; top-1 validation ${fmtPct(round.val_accuracy)}.</p>${table(["Client", "Số ảnh n_k", "Hệ số gộp p_k", "Train loss", "‖Δ‖₂ trọng số học", "‖Δ‖₂ thống kê BatchNorm", "Bước cập nhật"], rows, "Telemetry năm client ở vòng đã chọn")}`;
}

const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
/** Phần thống kê BatchNorm của ‖w_k − w‖₂ = căn(tổng² − phần trọng số học được²). */
const bnNorm = c => Math.sqrt(Math.max(0, c.delta_l2 ** 2 - c.delta_l2_learnable ** 2));
export const localModels = training => (training?.models || []).filter(m => m.model_id.startsWith("local_"));

/** Độ trôi cập nhật của client: phần trọng số học được (‖Δ‖₂ trên weight/bias) tách khỏi thống kê BatchNorm; biểu đồ vẽ ở results.js. */
function driftHtml(fed) {
  const rows = fed.history;
  const range = (row, key, digits) => { const v = row.clients.map(c => c[key]); return `${fmtNum(Math.min(...v), digits)}–${fmtNum(Math.max(...v), digits)}`; };
  const last = rows[rows.length - 1];
  const lossMax = Math.max(...last.clients.map(c => c.train_loss));
  const bnShare = Math.floor(100 * rows.flatMap(r => r.clients).reduce((low, c) => Math.min(low, 1 - (c.delta_l2_learnable ** 2) / (c.delta_l2 ** 2)), 1));
  return `<h4 class="subhead mt-4">Độ trôi của client (client drift)</h4>
    <div class="chart-container mt-3"><canvas id="chart-fedavg-drift" aria-label="Độ lớn cập nhật trọng số học được của từng client qua các vòng FedAvg"></canvas></div>
    <p class="field-hint mt-2">Biểu đồ vẽ phần trọng số học được của cập nhật, ‖Δ‖₂ = ‖w_k − w_toàn cục‖₂ chỉ trên weight và bias, đo so với mô hình toàn cục client nhận ở đầu vòng. Phần này giảm từ khoảng ${range(rows[0], "delta_l2_learnable", 2)} ở vòng 1 xuống khoảng ${range(last, "delta_l2_learnable", 2)} ở vòng ${last.round}, chưa về 0. Tổng ‖w_k − w‖₂ gồm cả running mean/var của BatchNorm lớn hơn nhiều (${range(rows[0], "delta_l2", 0)} ở vòng 1, ${range(last, "delta_l2", 0)} ở vòng ${last.round}) và luôn có ít nhất ${bnShare}% bình phương độ lớn đó nằm ở thống kê BatchNorm, nên số tổng không đo mức học của trọng số. Loss huấn luyện cục bộ ở vòng ${last.round} của mọi client dưới ${fmtNum(Math.ceil(lossMax * 100) / 100, 2)}, trong khi macro-F1 PlantDoc của mô hình toàn cục chỉ ${fmtPct(last.plantdoc_macro_f1_supported, 1)}: mỗi client khớp tốt dữ liệu của mình nhưng các cập nhật vẫn lệch nhau. Phù hợp với mô tả về độ phân kỳ trọng số khi dữ liệu non-IID của <a href="https://arxiv.org/abs/1806.00582" target="_blank" rel="noopener noreferrer">Zhao và cộng sự (2018)</a>; đây là quan sát, chưa phải kết luận nhân quả.</p>`;
}

/** Diễn biến huấn luyện Centralized: mọi mốc đọc từ history và cấu hình đã kiểm hash. */
export function centralizedNoteHtml(training, pdVal) {
  const central = modelTraining(training, "centralized");
  if (!training?.available || !central?.history?.length) return missing(training);
  const h = central.history;
  const chosen = h.find(r => r.epoch === central.selected_step);
  const last = h[h.length - 1];
  const minLoss = h.reduce((best, r) => (r.val_loss < best.val_loss ? r : best), h[0]);
  const drops = h.filter((r, i) => i > 0 && r.lr < h[i - 1].lr);
  const early = /EARLY/.test(central.status || "");
  const pdTrain = central.train_accuracy_by_source?.plantdoc;
  return `<p class="panel-desc mt-3">Chạy ${fmtInt(central.completed_steps)}/${fmtInt(central.config.epochs)} epoch${early ? " (dừng sớm vì đã hội tụ)" : ""}, ${fmtNum(central.total_runtime_seconds, 0)} giây (≈ ${fmtNum(central.total_runtime_seconds / 3600, 1)} giờ). Loss huấn luyện giảm từ ${fmtNum(h[0].train_loss, 4)} (epoch 1) xuống ${fmtNum(chosen.train_loss, 4)} (epoch ${chosen.epoch}) và ${fmtNum(last.train_loss, 4)} (epoch ${last.epoch}). Loss kiểm định thấp nhất ${fmtNum(minLoss.val_loss, 4)} ở epoch ${minLoss.epoch}; ở epoch được chọn (${chosen.epoch}) là ${fmtNum(chosen.val_loss, 4)}, về sau quanh ${fmtNum(last.val_loss, 2)}. Macro-F1 PlantDoc không còn cải thiện sau epoch ${chosen.epoch} (${fmtPct(last.plantdoc_macro_f1_supported, 1)} ở epoch ${last.epoch}) dù top-1 chung còn nhích lên, nên quy tắc chọn giữ epoch ${chosen.epoch} thay vì epoch cuối.</p>
    <p class="field-hint mt-2">Tốc độ học giảm một nửa mỗi khi macro-F1 PlantDoc không cải thiện: lần đầu ở epoch ${drops[0]?.epoch ?? "—"} (${fmtNum(h[0].lr, 4)} → ${fmtNum(drops[0]?.lr, 4)}), tổng ${fmtInt(drops.length)} lần, còn ${last.lr.toExponential(2)} ở epoch ${last.epoch}. Dấu hiệu quá khớp trên ảnh ngoài đồng: độ chính xác trên ảnh PlantDoc của tập huấn luyện ${fmtPct(pdTrain, 2)}, trên tập kiểm định ${fmtPct(pdVal, 2)}.</p>`;
}

/** Toàn tập kiểm tra: top-1 và macro-F1 chung, mức FedAvg thu hẹp khoảng cách giữa Local-only và Centralized. */
export function overallHtml(training) {
  if (!training?.available) return missing(training);
  const o = id => modelTraining(training, id).validation_overall;
  const locals = localModels(training).map(m => m.validation_overall);
  const avg = key => mean(locals.map(v => v[key]));
  const cols = [["Centralized", o("centralized")], ["FedAvg", o("fedavg")],
    ["Local-only (trung bình 5)", { top1_accuracy: avg("top1_accuracy"), macro_f1_supported: avg("macro_f1_supported"), crop_accuracy: avg("crop_accuracy") }]];
  const row = (label, key) => `<tr><th scope="row">${label}</th>${cols.map(([, v]) => `<td>${fmtPct(v[key])}</td>`).join("")}</tr>`;
  const c = o("centralized").top1_accuracy, f = o("fedavg").top1_accuracy, l = avg("top1_accuracy");
  return `<section class="panel mt-4"><h4 class="panel-title">Toàn tập kiểm tra (${fmtInt(o("fedavg").count)} ảnh)</h4>
    ${table(["Chỉ số", ...cols.map(([name]) => name)], row("Top-1 chung", "top1_accuracy") + row("Macro-F1 chung", "macro_f1_supported") + row("Đúng loại cây", "crop_accuracy"), "Kết quả chung trên tập kiểm tra")}
    <p class="field-hint mt-2">Khoảng giữa sàn Local-only và trần Centralized là ${fmtNum((c - l) * 100, 1)} điểm % top-1 chung; FedAvg thu hẹp ${fmtNum((f - l) * 100, 1)} điểm trong đó và còn cách Centralized ${fmtNum((c - f) * 100, 2)} điểm. Top-1 chung bị PlantVillage chi phối nên cần đọc kèm cột PlantDoc ở bảng trên. Tập này dùng để chọn checkpoint nên có xu hướng lạc quan.</p></section>`;
}

/** Ảnh PlantDoc của tập giữ riêng: Centralized, FedAvg, trung bình Local-only, kèm lớp tạo ra khoảng cách lớn nhất. */
export function holdoutPlantdocHtml(holdout) {
  if (!holdout?.available) return "";
  const by = Object.fromEntries(holdout.models.map(m => [m.model_id, m]));
  const locals = holdout.models.filter(m => m.model_id.startsWith("local_"));
  const n = by.centralized.plantdoc.count;
  const avg = pick => mean(locals.map(m => pick(m.plantdoc)));
  const cell = (v, digits = 0) => fmtNum(v, Number.isInteger(v) ? 0 : digits);
  const rows = [
    ["Centralized", by.centralized.plantdoc],
    ["FedAvg", by.fedavg.plantdoc],
    ["Local-only (trung bình 5)", { correct: avg(p => p.correct), top1: avg(p => p.top1), macro_f1: avg(p => p.macro_f1), crop_accuracy: avg(p => p.crop_accuracy), high_confidence_wrong: avg(p => p.high_confidence_wrong) }],
  ].map(([label, p]) => `<tr><th scope="row">${label}</th><td>${cell(p.correct, 1)}</td><td>${fmtPct(p.top1)}</td><td>${fmtPct(p.macro_f1)}</td><td>${fmtPct(p.crop_accuracy)}</td><td>${cell(p.high_confidence_wrong, 1)}</td></tr>`).join("");
  const fedByClass = Object.fromEntries(by.fedavg.plantdoc_per_class.map(r => [r.class, r]));
  const top = by.centralized.plantdoc_per_class.map(r => ({ ...r, fed: fedByClass[r.class]?.correct ?? 0 })).sort((a, b) => (b.correct - b.fed) - (a.correct - a.fed))[0];
  const gap = by.centralized.plantdoc.correct - by.fedavg.plantdoc.correct;
  return `<section class="panel mt-4"><h4 class="panel-title">${fmtInt(n)} ảnh PlantDoc trong tập giữ riêng</h4>
    <p class="panel-desc">Ảnh ngoài đồng không dùng để cập nhật trọng số và không dùng để chọn checkpoint. Đây là phép đo tốt nhất hiện có cho ảnh ngoài đồng chưa thấy, nhưng chưa phải test thực địa mới được niêm phong và khoảng tin cậy rộng (xem bảng ở trên).</p>
    ${table(["Mô hình", `Đúng / ${fmtInt(n)}`, "Top-1 (cây + bệnh)", `Macro-F1 (${fmtInt(by.centralized.plantdoc.supported_classes)} lớp)`, "Đúng loại cây", "Sai dù tự tin ≥ 0,9"], rows, "Kết quả trên ảnh PlantDoc của tập giữ riêng")}
    <p class="field-hint mt-2">Khoảng cách top-1 Centralized − FedAvg là ${fmtNum((by.centralized.plantdoc.top1 - by.fedavg.plantdoc.top1) * 100, 2)} điểm % (${fmtInt(gap)} ảnh), nhưng macro-F1 chỉ chênh ${fmtNum((by.centralized.plantdoc.macro_f1 - by.fedavg.plantdoc.macro_f1) * 100, 2)} điểm. Lớp đóng góp nhiều nhất vào khoảng cách top-1: ${esc(formatDiseaseName(top.class))} (${fmtInt(top.support)} ảnh), Centralized đúng ${fmtInt(top.correct)}, FedAvg đúng ${fmtInt(top.fed)}. Mô hình có thể sai mà vẫn rất tự tin: ${fmtInt(by.centralized.plantdoc.high_confidence_wrong)} lỗi của Centralized và ${fmtInt(by.fedavg.plantdoc.high_confidence_wrong)} lỗi của FedAvg có xác suất từ 0,9 trở lên.</p></section>`;
}

/** Thuật ngữ dùng xuyên suốt báo cáo; định nghĩa tập kiểm tra/giữ riêng chỉ nằm ở chương 2. */
export function glossaryHtml() {
  const terms = [
    ["FL (Federated Learning)", "Học liên kết: nhiều client cùng huấn luyện, chỉ trao đổi trọng số, không chia sẻ ảnh gốc."],
    ["FedAvg", "Thuật toán gộp bằng trung bình có trọng số (theo số ảnh) các mô hình client."],
    ["non-IID", "Dữ liệu giữa các client không đồng nhất (ở đây: lệch nhóm bệnh, α = 0,1)."],
    ["Client / server", "Máy khách giữ dữ liệu / máy chủ gộp mô hình."],
    ["Shard", "Phần dữ liệu huấn luyện của một client."],
    ["Checkpoint", "Điểm lưu mô hình: bộ trọng số tại một epoch hoặc một vòng."],
    ["Epoch / vòng", "Epoch là một lượt duyệt qua dữ liệu huấn luyện. Vòng (round) là một lần phát mô hình, huấn luyện cục bộ và gộp trong FedAvg."],
    ["lr, SGD, AdamW", "Tốc độ học và hai bộ tối ưu dùng trong đồ án."],
    ["AMP", "Huấn luyện với độ chính xác số thực hỗn hợp."],
  ];
  return `<details class="fold mt-3"><summary>Thuật ngữ và viết tắt</summary><dl class="term-list mt-2">${terms.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl></details>`;
}
