// Report additions sourced from /api/evidence/summary.training (checkpoint and manifest identity checked server-side).
import { MODEL_LABELS, esc, fmtInt, fmtNum, fmtPct, formatDiseaseName, modelPill } from "./dom.js";

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
  const hasOther = training.release.splits.some(s => s.by_source.field_other);
  const rows = training.release.splits.map(s => `<tr><th scope="row">${roles[s.name][0]}</th><td>${fmtInt(s.by_source.plantdoc)}</td><td>${fmtInt(s.by_source.plantvillage)}</td>${hasOther ? `<td>${fmtInt(s.by_source.field_other || 0)}</td>` : ""}<td><b>${fmtInt(s.count)}</b></td><td>${roles[s.name][1]}</td></tr>`).join("");
  const head = ["Phần dữ liệu", "PlantDoc", "PlantVillage", ...(hasOther ? ["Ngoài đồng, bộ khác"] : []), "Tổng ảnh", "Vai trò"];
  const trainCfg = modelTraining(training, "centralized")?.config || {};
  return `<h4 class="subhead mt-4">Đủ năm phần của release ${esc(training.release.name)}</h4>
    ${table(head, rows, "Số ảnh của năm phần release")}
    <p class="field-hint mt-2">Trên web, “tập kiểm tra” là validation (dùng chọn mô hình); “tập giữ riêng” là legacy diagnostic test. Tập giữ riêng chưa phải test thực địa mới được niêm phong. Audit hiện có chỉ bao phủ train, validation và calibration.${hasOther ? " Cột “Ngoài đồng, bộ khác” gồm ảnh ngoài đồng từ các bộ công khai ngoài PlantDoc; nhãn của chúng được sàng lọc bằng mắt (không phải chuyên gia) hoặc do tác giả bộ dữ liệu gán." : ""}</p>
    <h4 class="subhead mt-4">Tiền xử lý và cách học của các checkpoint trên web</h4>
    <ol class="method-steps mt-3"><li><b>Xây release.</b> Ảnh được chia theo nhóm (cùng một lá, cùng một phiên chụp) để một lá không nằm ở cả huấn luyện lẫn kiểm tra; ảnh gần trùng với các tập đã khóa bị loại; PlantVillage giữ đủ, ảnh ngoài đồng được bổ sung từ các bộ công khai có ghi giấy phép.</li>
    <li><b>Nạp ảnh.</b> canonical_v1: RGB → resize trực tiếp 224 × 224 (bilinear, antialias) → tensor [0, 1] → chuẩn hóa ImageNet, mean (0,485; 0,456; 0,406), std (0,229; 0,224; 0,225). Validation và suy luận không tăng cường ngẫu nhiên.</li>
    <li><b>Tăng cường khi học.</b> ${esc(trainCfg.train_transform || "—")}.</li>
    <li><b>Cân bằng miền.</b> ${esc(trainCfg.loss_weighting || "—")}: tổng trọng số mất mát của ảnh phòng thí nghiệm bằng tổng của ảnh ngoài đồng, tính trên chính tập mà từng mô hình hoặc client học. Số ảnh thật và hệ số gộp n_k / Σn_k vẫn theo manifest.</li></ol>
    <p class="field-hint mt-2">Ảnh được đọc sang RGB sau khi xoay theo EXIF và dán nền đen cho ảnh trong suốt, giống lúc huấn luyện. Resize vuông có thể làm méo tỷ lệ; mô hình không tự tìm lá trong ảnh nhiều lá.</p>`;
}

export function configsHtml(training) {
  if (!training?.available) return missing(training);
  const central = modelTraining(training, "centralized");
  const fed = modelTraining(training, "fedavg");
  const local = modelTraining(training, "local_00");
  const c = central.config, f = fed.config, l = local?.config || {};
  const dash = v => (v === undefined || v === null ? "—" : esc(v));
  const rows = [
    ["Bộ tối ưu / tốc độ học ban đầu", `${dash(c.optimizer)} / ${dash(c.lr)}`, `${dash(f.client_optimizer || f.optimizer)} / ${dash(f.client_lr ?? f.lr)}${f.optimizer_state ? ` (${dash(f.optimizer_state)})` : ""}`, local ? `${dash(l.optimizer)} / ${dash(l.lr)}` : "Không lưu trong checkpoint Local-only"],
    ["Batch / độ chính xác số", `${dash(c.batch_size)} / ${dash(c.amp)}`, `${dash(f.batch_size)} / ${dash(f.amp)}`, local ? `${dash(l.batch_size)} / ${dash(l.amp)}` : "Chưa xác minh từ metadata"],
    ["Lịch tốc độ học", dash(c.scheduler_policy), dash(f.scheduler_policy), local ? dash(l.scheduler_policy) : "—"],
    ["Ngân sách → đã chạy → được chọn", `${fmtInt(c.epochs)} → ${fmtInt(central.completed_steps)} → epoch ${fmtInt(central.selected_step)}`, `${fmtInt(f.rounds)} → ${fmtInt(fed.completed_steps)} → vòng ${fmtInt(fed.selected_step)}`, `${fmtInt(l.epochs)} epoch; mỗi client chọn riêng (bảng dưới)`],
    ["Local epoch mỗi vòng", "Không áp dụng", fmtInt(f.local_epochs), "Không áp dụng"],
    ["Tăng cường / cân bằng miền", `${dash(c.train_transform)} · ${dash(c.loss_weighting)}`, `${dash(f.train_transform)} · ${dash(f.loss_weighting)}`, local ? `${dash(l.train_transform)} · ${dash(l.loss_weighting)}` : "—"],
    ["Cách chọn checkpoint", dash(c.selection), dash(f.selection), local ? dash(l.selection) : "—"],
    ["Runtime của kết quả", "Huấn luyện tập trung", "in_process_simulation; 5 client tuần tự", "5 mô hình riêng, không gộp"],
  ].map(row => `<tr><th scope="row">${row[0]}</th>${row.slice(1).map(v => `<td>${v}</td>`).join("")}</tr>`).join("");
  const models = training.models.map(m => `<tr><th scope="row">${modelPill(m.model_id)}</th><td>${m.model_id === "centralized" ? "Epoch" : "Vòng"} ${fmtInt(m.selected_step)} / ${fmtInt(m.completed_steps)}</td><td>${fmtNum(m.checkpoint_bytes / 1e6, 2)} MB</td><td title="${esc(training.checkpoint_sha256[m.model_id])}"><code>${esc(training.checkpoint_sha256[m.model_id].slice(0, 12))}…</code></td></tr>`).join("");
  return `<h4 class="subhead mt-4">Cấu hình thực tế đã lưu</h4>${table(["Thiết lập", MODEL_LABELS.centralized, MODEL_LABELS.fedavg, "Local-only"], rows, "Cấu hình huấn luyện ba phương pháp")}
    <p class="field-hint mt-2">Cùng W0, hạt giống, MobileNetV3-Small và canonical_v1. Ba cách dùng chung một công thức (bộ tối ưu, lịch tốc độ học, tăng cường, cân bằng miền) và cùng số bước gradient: Centralized học ${fmtInt(c.epochs)} epoch trên toàn tập, FedAvg ${fmtInt(f.rounds)} vòng × ${fmtInt(f.local_epochs)} epoch cục bộ trên 5 shard, mỗi Local-only ${fmtInt(l.epochs)} epoch trên shard của mình; chỉ cách phân tán dữ liệu khác nhau. Chọn checkpoint theo macro-F1 trung bình giữa các nguồn trên validation.</p>
    <h4 class="subhead mt-4">Bảy bộ trọng số dùng khi đánh giá và nhận diện</h4>${table(["Mô hình", "Mốc chọn / đã chạy", "Dung lượng tệp", "SHA-256 (rút gọn)"], models, "Bảy checkpoint đã kiểm SHA-256")}
    <p class="field-hint mt-2">Trọng số mô hình là tensor đã học; p_k trong FedAvg là hệ số gộp theo số ảnh, hai khái niệm khác nhau. Dung lượng checkpoint không phải dung lượng truyền mỗi vòng.</p>
    <h4 class="subhead mt-4">Flower nằm ở đâu?</h4>
    <p class="section-lead">Nhánh legacy có ServerApp, ClientApp và launcher gọi Flower + Ray. ${esc(MODEL_LABELS.fedavg)} trong bảng kết quả dùng runner một tiến trình; chạy lại qua Flower và đối chiếu với runner hiện tại vẫn là việc tiếp theo. Phiên Flower trên web chạy một server và năm tiến trình client localhost bằng ảnh nhiễu để kiểm tra hệ thống, không tạo ra kết quả khoa học.</p>
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
  const fullLog = fed.history.length === fed.completed_steps;
  const drift = training.delta_l2_recorded === false ? driftMissingHtml() : driftHtml(fed);
  return `<details class="overview-detail mt-4" id="fedavg-history"><summary><span><strong>Lịch sử ${esc(MODEL_LABELS.fedavg)} và telemetry ${fmtInt(fed.history.length)} vòng</strong><small>Chọn vòng ${fmtInt(fed.selected_step)}${fullLog ? `; log đủ ${fmtInt(fed.completed_steps)} vòng đã chạy` : `; log không bao phủ vòng ${fmtInt(fed.history.length + 1)}–${fmtInt(fed.completed_steps)}`}</small></span></summary>
    <div class="overview-detail-body"><p class="panel-desc">Trích từ nhật ký huấn luyện đã lưu cùng gói mô hình. Top-1 trên validation ${fmtInt(fed.validation_overall.count)} ảnh; macro-F1 PlantDoc trên phần ảnh PlantDoc của validation. Thời gian ${fmtNum(seconds, 2)} giây chỉ tính ${fmtInt(fed.history.length)} vòng có log.</p>
    <div class="chart-container mt-3"><canvas id="chart-fedavg-history" aria-label="Top-1 validation và macro-F1 PlantDoc qua ${fed.history.length} vòng FedAvg"></canvas></div>
    ${drift}
    <div class="form-group mt-4"><label for="fedavg-round">Xem cập nhật từng client ở vòng</label><select id="fedavg-round" class="input-field">${fed.history.map(r => `<option value="${r.round}"${r.round === fed.selected_step ? " selected" : ""}>Vòng ${r.round}</option>`).join("")}</select></div>
    <div id="fedavg-client-telemetry"></div>
    <p class="field-hint mt-3">Mỗi chiều của một client ≈ ${fmtNum(first.payload_down_bytes / 1e6, 2)} MB (float32); một vòng (${fed.history[0].clients.length} client × tải xuống + tải lên) ≈ ${fmtNum(perRound / 1e6, 1)} MB; thời gian một vòng ${fmtNum(Math.min(...durations), 0)}–${fmtNum(Math.max(...durations), 0)} giây. Tổng payload tải xuống + lên: ${fmtNum(bytes / 1e6, 1)} MB cho ${fed.history.length} vòng, ${fed.history[0].clients.length} client. Đây là ước tính từ tensor trong mô phỏng; không gồm overhead giao thức, không phải băng thông đo trên mạng.</p></div></details>`;
}

export function telemetryHtml(round) {
  const hasDrift = round.clients.every(c => Number.isFinite(c.delta_l2) && Number.isFinite(c.delta_l2_learnable));
  const rows = round.clients.map(c => `<tr><th scope="row">${esc(c.client_id)}</th><td>${fmtInt(c.sample_count_n_k)}</td><td>${fmtNum(c.weight_p_k, 4)}</td><td>${fmtNum(c.train_loss, 4)}</td>${hasDrift ? `<td>${fmtNum(c.delta_l2_learnable, 3)}</td><td>${fmtNum(bnNorm(c), 1)}</td>` : ""}<td>${fmtInt(c.optimizer_steps_applied)}</td></tr>`).join("");
  const head = ["Client", "Số ảnh n_k", "Hệ số gộp p_k", "Train loss", ...(hasDrift ? ["‖Δ‖₂ trọng số học", "‖Δ‖₂ thống kê BatchNorm"] : []), "Bước cập nhật"];
  return `<p class="field-hint">Vòng ${round.round}: ${fmtNum(round.duration_sec, 2)} giây; top-1 validation ${fmtPct(round.val_accuracy)}.</p>${table(head, rows, "Telemetry năm client ở vòng đã chọn")}`;
}

/** Gói train bằng scripts/train_v4.py không ghi độ lệch cập nhật của từng client: nói rõ thay vì vẽ biểu đồ không có số. */
const driftMissingHtml = () => `<p class="field-hint mt-4">Độ lệch cập nhật ‖w_k − w_toàn cục‖₂ của từng client chưa được ghi lại ở đợt huấn luyện này, nên không vẽ biểu đồ độ trôi (client drift). Mất mát huấn luyện và số bước của từng client vẫn có trong bảng bên dưới.</p>`;

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

/** Diễn biến huấn luyện Centralized: mọi mốc đọc từ history và cấu hình đã lưu cùng gói mô hình. */
export function centralizedNoteHtml(training, pdVal) {
  const central = modelTraining(training, "centralized");
  if (!training?.available || !central?.history?.length) return missing(training);
  const h = central.history;
  const chosen = h.find(r => r.epoch === central.selected_step);
  const last = h[h.length - 1];
  const withValLoss = h.filter(r => Number.isFinite(r.val_loss));
  const minLoss = withValLoss.length ? withValLoss.reduce((best, r) => (r.val_loss < best.val_loss ? r : best), withValLoss[0]) : null;
  const pdTrain = central.train_accuracy_by_source?.plantdoc;
  const policy = central.config.scheduler_policy ? ` Tốc độ học: ${esc(central.config.scheduler_policy)}; còn ${last.lr.toExponential(2)} ở epoch ${last.epoch}.` : "";
  const overfit = Number.isFinite(pdTrain) ? ` Dấu hiệu quá khớp trên ảnh ngoài đồng: độ chính xác trên ảnh PlantDoc của tập huấn luyện ${fmtPct(pdTrain, 2)}, trên tập kiểm định ${fmtPct(pdVal, 2)}.` : "";
  return `<p class="panel-desc mt-3">Chạy ${fmtInt(central.completed_steps)}/${fmtInt(central.config.epochs)} epoch, ${fmtNum(central.total_runtime_seconds, 0)} giây (≈ ${fmtNum(central.total_runtime_seconds / 3600, 1)} giờ). Loss huấn luyện giảm từ ${fmtNum(h[0].train_loss, 4)} (epoch 1) xuống ${fmtNum(chosen.train_loss, 4)} (epoch ${chosen.epoch}) và ${fmtNum(last.train_loss, 4)} (epoch ${last.epoch}).${minLoss ? ` Loss kiểm định thấp nhất ${fmtNum(minLoss.val_loss, 4)} ở epoch ${minLoss.epoch}.` : ""} Top-1 kiểm định ở epoch được chọn (${chosen.epoch}): ${fmtPct(chosen.val_accuracy, 2)}.</p>
    <p class="field-hint mt-2">Checkpoint được chọn theo macro-F1 trung bình giữa các nguồn trên validation.${policy}${overfit}</p>`;
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

/** Bảng các hạt giống: trung bình ± độ lệch chuẩn của mỗi cách, từng hạt giống, và hạt giống mà web đang phục vụ. Chỉ hiện khi máy chủ xác nhận cùng release. */
export function seedsHtml(seeds) {
  if (!seeds?.available) return "";
  const names = { centralized: MODEL_LABELS.centralized, fedavg: MODEL_LABELS.fedavg, local: "Local-only (TB 5 client)" };
  const methods = Object.keys(names).filter(m => seeds.methods?.[m]);
  const pct = v => (v === null || v === undefined ? "—" : fmtPct(v, 1));
  const cell = (m, key) => {
    const x = seeds.methods[m]?.[key];
    if (!x) return "<td>—</td>";
    const spread = Number.isFinite(x.std) ? ` <small>± ${fmtNum(x.std * 100, 1)}</small>` : "";
    const each = seeds.seeds.map(sd => `${sd}: ${pct(x.by_seed[String(sd)])}`).join(" · ");
    return `<td title="${esc(each)}"><b>${pct(x.mean)}</b>${spread}</td>`;
  };
  const rows = seeds.metric_order.filter(key => methods.some(m => seeds.methods[m]?.[key]))
    .map(key => `<tr><th scope="row">${esc(methods.map(m => seeds.methods[m]?.[key]?.label).find(Boolean))}</th>${methods.map(m => cell(m, key)).join("")}</tr>`).join("");
  return `<section class="panel mt-4"><h4 class="panel-title">Ba hạt giống: kết quả ổn định đến đâu</h4>
    <p class="panel-desc">Mỗi cách được huấn luyện lại với ${fmtInt(seeds.seeds.length)} hạt giống (${esc(seeds.seeds.join(", "))}); web phục vụ mô hình của hạt giống ${esc(seeds.served_seed)}. Số lớn là trung bình, số nhỏ là độ lệch chuẩn (điểm %); rê chuột lên ô để xem từng hạt giống.</p>
    ${table(["Chỉ số", ...methods.map(m => esc(names[m]))], rows, "Kết quả theo hạt giống")}
    <p class="field-hint mt-2">${esc(seeds.note || "")}</p></section>`;
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
export function glossaryHtml(alpha = null) {
  const terms = [
    ["FL (Federated Learning)", "Học liên kết: nhiều client cùng huấn luyện, chỉ trao đổi trọng số, không chia sẻ ảnh gốc."],
    ["FedAvg", "Thuật toán gộp bằng trung bình có trọng số (theo số ảnh) các mô hình client."],
    ["non-IID", `Dữ liệu giữa các client không đồng nhất (ở đây: lệch nhóm bệnh${alpha ? `, ${alpha}` : ""}).`],
    ["Client / server", "Máy khách giữ dữ liệu / máy chủ gộp mô hình."],
    ["Shard", "Phần dữ liệu huấn luyện của một client."],
    ["Checkpoint", "Điểm lưu mô hình: bộ trọng số tại một epoch hoặc một vòng."],
    ["Epoch / vòng", "Epoch là một lượt duyệt qua dữ liệu huấn luyện. Vòng (round) là một lần phát mô hình, huấn luyện cục bộ và gộp trong FedAvg."],
    ["lr, SGD", "Tốc độ học và bộ tối ưu (SGD có động lượng) dùng cho cả ba cách."],
    ["AMP", "Huấn luyện với độ chính xác số thực hỗn hợp."],
  ];
  return `<details class="fold mt-3"><summary>Thuật ngữ và viết tắt</summary><dl class="term-list mt-2">${terms.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl></details>`;
}
