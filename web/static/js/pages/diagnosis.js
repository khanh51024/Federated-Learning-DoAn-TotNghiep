// Bước 4 · Chẩn đoán: một ảnh, bảy mô hình, đúng/sai và lý do (ảnh train của lớp đó trong shard).
import { getEvidence, getGallery, getModels, getSummary } from "../core/store.js";
import { ApiError, isAbort, postForm } from "../core/api.js";
import {
  $$, MODEL_LABELS, MODEL_ORDER, byId, cropOf, esc, fmtInt, fmtNum, fmtPct, fmtPctValue, formatApiError,
  formatDiseaseName, modelPill, notify, pillarOf, prefersReducedMotion, revealItems, setBar, setDisabled, setHtml,
  setText, toggleHidden,
} from "../core/dom.js";

const LOCAL_SOURCES = ["local_00", "local_01", "local_02", "local_03", "local_04"];
const ALL_SOURCES = ["fedavg", ...LOCAL_SOURCES];
const LEVEL_LABEL = { very_easy: "Rất dễ", easy: "Dễ", medium: "Vừa", hard: "Khó", very_hard: "Rất khó" };

const DIAGNOSE_HTML = '<svg class="svg-icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Chẩn đoán';

const state = {
  file: null,
  item: null, // mục catalog nếu chọn ảnh mẫu
  previewUrl: null,
  mode: "all",
  quickModel: "fedavg",
  filter: "all",
  runId: 0,
  selectToken: 0, // chỉ lần chọn ảnh MỚI NHẤT được áp dụng (fetch ảnh mẫu có thể về không đúng thứ tự)
  showStats: false, // chỉ hiện độ khó/recall từ catalog khi bundle khả dụng và ảnh đã đối chiếu manifest validation
  controller: null,
  rows: new Map(), // model_id -> kết quả hoặc {error}
  explain: false, // người dùng bật Grad-CAM
  explanations: new Map(), // model_id -> {class, image}
  allEndpoint: null, // null: chưa biết; false: máy chủ cũ không có /api/inference/all → gọi từng cặp
  timing: null,
  busy: false, // đang có yêu cầu chẩn đoán
  runExplain: false, // yêu cầu đang chạy có kèm Grad-CAM không
  classes: [],
  shards: null,
  gallery: [],
};

// ───────── Thư viện ảnh mẫu ─────────
const FILTERS = {
  all: () => true,
  easy: i => ["very_easy", "easy"].includes(i.difficulty_key),
  medium: i => i.difficulty_key === "medium",
  hard: i => ["hard", "very_hard"].includes(i.difficulty_key),
  plantvillage: i => i.source === "plantvillage",
  plantdoc: i => i.source === "plantdoc",
};

function renderGallery() {
  const grid = byId("gallery-grid");
  const items = state.gallery.filter(FILTERS[state.filter] || FILTERS.all);
  grid.removeAttribute("aria-busy");
  if (!state.gallery.length) {
    grid.innerHTML = '<p class="field-hint">Chưa có thư viện ảnh mẫu trên máy chủ này — hãy tải ảnh lên.</p>';
    return;
  }
  if (!items.length) {
    grid.innerHTML = '<p class="field-hint">Không có ảnh nào khớp bộ lọc.</p>';
    return;
  }
  grid.innerHTML = items.map(item => {
    const id = Number(item.id);
    const level = state.showStats ? `<span class="level level-${esc(item.difficulty_key)}">${esc(item.difficulty)}</span>` : "";
    const levelLabel = state.showStats ? `, ${esc(item.difficulty)}` : "";
    return `
    <button type="button" class="gallery-item${state.item?.id === item.id ? " is-selected" : ""}" data-id="${id}"
      aria-pressed="${state.item?.id === item.id}"
      aria-label="Ảnh mẫu ${id}: ${esc(formatDiseaseName(item.label))}${levelLabel}${item.field_photo ? ", ảnh ngoài đồng" : ""}">
      <img src="/static/samples/demo/${esc(item.thumb)}" alt="" loading="lazy" width="88" height="88">
      ${level}
      <span class="gallery-cap">${esc(formatDiseaseName(item.label))}</span>
      <span class="gallery-src">${item.source === "plantdoc" ? "PlantDoc · ngoài đồng" : "PlantVillage"}</span>
    </button>`;
  }).join("");
}

/** Đổi ô đang chọn tại chỗ: không mất focus bàn phím và không reset vị trí cuộn của khung thư viện. */
function markSelected() {
  $$("#gallery-grid .gallery-item").forEach(button => {
    const on = state.item !== null && String(state.item.id) === button.dataset.id;
    button.classList.toggle("is-selected", on);
    button.setAttribute("aria-pressed", String(on));
  });
}

// ───────── Chọn ảnh ─────────
function setPreview(file, metaHtml) {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  const img = byId("preview-image");
  img.src = state.previewUrl;
  byId("selected-stage").classList.remove("hidden");
  resetStage();
  byId("selected-empty").classList.add("hidden");
  const meta = byId("selected-meta");
  meta.innerHTML = metaHtml;
  meta.classList.remove("hidden");
  setDisabled(byId("btn-diagnose"), false);
  setDisabled(byId("btn-plantvillage"), false);
  byId("plantvillage-result").textContent = "";
}

function resetResults() {
  state.controller?.abort();
  state.runId += 1;
  state.rows.clear();
  state.explanations.clear();
  state.timing = null;
  state.busy = false;
  resetStage();
  byId("diag-result").classList.add("hidden");
  byId("btn-print").classList.add("hidden");
  byId("diag-empty").classList.remove("hidden");
  byId("diag-progress-badge").classList.add("hidden");
  setHtml(byId("btn-diagnose"), DIAGNOSE_HTML); // phiên chạy cũ bị hủy giữa chừng không còn dòng "Đang chẩn đoán…"
}

function itemMeta(item) {
  const verified = state.showStats;
  const level = verified ? `<span class="level level-${esc(item.difficulty_key)}">${esc(item.difficulty)}</span>` : "";
  const stats = verified
    ? `<p class="field-hint">Trên ${fmtInt(item.class_val_count)} ảnh kiểm tra ${item.source === "plantdoc" ? "PlantDoc" : "PlantVillage"} của nhóm này,
      FedAvg nhận đúng ${fmtPctValue(item.recall_fedavg_pct, 0)}, Centralized ${fmtPctValue(item.recall_centralized_pct, 0)}.</p>`
    : '<p class="field-hint">Chưa có số liệu đã kiểm hash để xếp độ khó cho ảnh này.</p>';
  return `
    <div class="meta-chips">
      ${level}
      <span class="chip chip-static">${item.source === "plantdoc" ? "PlantDoc · ảnh ngoài đồng" : "PlantVillage"}</span>
      <span class="chip chip-static">ảnh mẫu #${Number(item.id)}${verified ? " · thuộc tập kiểm tra" : " · chưa đối chiếu được"}</span>
    </div>
    <p class="meta-label">${esc(formatDiseaseName(item.label))}</p>
    ${stats}`;
}

async function selectGallery(item) {
  const ticket = ++state.selectToken;
  try {
    const res = await fetch(`/static/samples/demo/${encodeURIComponent(item.file)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (ticket !== state.selectToken) return; // người dùng đã chọn ảnh khác trong lúc tải
    resetResults();
    state.file = new File([blob], item.file, { type: blob.type || "image/jpeg" });
    state.item = item;
    byId("inference-true-class").value = item.label;
    setPreview(state.file, itemMeta(item));
    markSelected();
    await run();
  } catch (error) {
    notify("error", "Không thể tải ảnh mẫu", item.file, { detail: error.message });
  }
}

async function selectUpload(file) {
  if (file.type && !file.type.startsWith("image/")) {
    notify("warning", "Tệp không phải ảnh", `"${file.name}" có định dạng ${file.type}. Hãy chọn ảnh JPG, PNG hoặc WEBP.`);
    return;
  }
  state.selectToken += 1;
  resetResults();
  state.file = file;
  state.item = null;
  byId("inference-true-class").value = "";
  setPreview(file, `
    <div class="meta-chips"><span class="chip chip-static">Ảnh của bạn</span></div>
    <p class="meta-label">${esc(file.name)}</p>
    <p class="field-hint">${fmtNum(file.size / 1024, 0)} KB · chưa có nhãn thật — chọn nhãn bên dưới nếu bạn biết; kết quả được chấm lại ngay.</p>`);
  markSelected();
  await run(); // giống ảnh mẫu: chọn xong là chạy, không bắt người dùng cuộn tìm nút
}

// ───────── Bảng kết quả ─────────
function shardInfo(modelId, classKey) {
  const row = state.shards?.classes?.find(r => r.class === classKey);
  if (!row) return null;
  if (modelId.startsWith("local_")) {
    const n = row.per_client[modelId.replace("local_", "client_")];
    return { count: n, total: row.total, scope: "ảnh của nơi này" };
  }
  return { count: row.total, total: row.total, scope: modelId === "fedavg" ? "gộp từ 5 nơi" : "toàn bộ ảnh" };
}

function targetClass() {
  return byId("inference-true-class").value || "";
}

function verdict(result, trueClass) {
  if (!trueClass || !result || result.error) return null;
  if (result.predicted_class === trueClass) return "correct";
  return cropOf(result.predicted_class) === cropOf(trueClass) ? "wrong-disease" : "wrong";
}

function rowHtml(modelId) {
  const result = state.rows.get(modelId);
  const label = MODEL_LABELS[modelId];
  const phase = modelId === "centralized" ? "Gom ảnh" : modelId === "fedavg" ? "Học liên kết" : "Học riêng";
  if (!result) {
    return `<th scope="row">${modelPill(modelId)}<span class="row-sub">${phase}</span></th>
      <td colspan="3"><span class="skeleton skeleton-line"></span></td>`;
  }
  if (result.error) {
    return `<th scope="row">${modelPill(modelId)}<span class="row-sub">${phase}</span></th>
      <td colspan="3" class="text-rose">Không có kết quả: ${esc(result.error)}</td>`;
  }
  const trueClass = targetClass();
  const v = verdict(result, trueClass);
  const verdictHtml = v === "correct"
    ? '<span class="verdict ok">✓ Đúng</span>'
    : v === "wrong-disease" ? '<span class="verdict mid" title="Đúng loại cây nhưng sai bệnh">✕ Sai bệnh</span>'
      : v === "wrong" ? '<span class="verdict bad" title="Sai cả loại cây">✕ Sai loại cây</span>'
        : '<span class="verdict none" title="Chọn nhãn thật để chấm">chưa biết nhãn</span>';
  const classKey = trueClass || result.predicted_class;
  const info = shardInfo(modelId, classKey);
  const shardHtml = info
    ? `<span class="shard-line" title="Số ảnh của nhóm ${trueClass ? "thật" : "được dự đoán"} «${esc(formatDiseaseName(classKey))}» mà mô hình này đã được học">
        <span class="shard-count ${info.count === 0 ? "is-zero" : ""} mono">${fmtInt(info.count)}</span>
        <span class="row-sub">${esc(info.scope)}</span></span>`
    : '<span class="text-muted">—</span>';
  const uncertain = result.is_uncertain ? ' <span class="pill-flag" title="Xác suất cao nhất dưới 50% hoặc hai nhóm đầu quá sát nhau">không chắc</span>' : "";
  return `<th scope="row">${modelPill(modelId)}<span class="row-sub">${phase}</span></th>
    <td class="cell-pred">
      <b>${esc(formatDiseaseName(result.predicted_class))}</b>${uncertain}
      <div class="conf"><div class="conf-track"><span class="conf-fill ${pillarOf(modelId)}" style="transform:scaleX(0)" data-pct="${result.confidence_pct}"></span></div><span class="mono">${fmtNum(result.confidence_pct, 1)}%</span></div>
      ${result.calibrated ? `<span class="conf-cal" title="Đã hiệu chuẩn bằng temperature scaling, T = ${fmtNum(result.calibrated.temperature, 2)} (học trên tập calibration riêng)">hiệu chuẩn <b class="mono">${fmtNum(result.calibrated.confidence_pct, 1)}%</b></span>` : ""}
    </td>
    <td>${verdictHtml}</td>
    <td>${shardHtml}</td>`;
}

function detailHtml(modelId) {
  const result = state.rows.get(modelId);
  if (!result || result.error) return "";
  const val = result.model_validation;
  return `<td colspan="4"><div class="detail-box">
      <b>Top 3 xác suất</b>
      <ol class="top3">${(result.top3 || []).map((t, i) => {
        const cal = result.calibrated?.top3?.[i];
        return `<li><span>${esc(formatDiseaseName(t.class))}</span><b class="mono">${fmtNum(t.confidence, 2)}%</b>${cal ? `<span class="mono text-muted">hiệu chuẩn ${fmtNum(cal.confidence, 1)}%</span>` : ""}</li>`;
      }).join("")}</ol>
      <p class="field-hint">Độ chính xác công bố của mô hình này trên tập kiểm tra: PlantVillage ${fmtPct(val?.plantvillage?.top1_accuracy)} · PlantDoc ${fmtPct(val?.plantdoc?.top1_accuracy)}.
        ${result.calibrated ? `Số hiệu chuẩn chia logit cho T = ${fmtNum(result.calibrated.temperature, 2)}; thứ tự các nhóm không đổi.` : "Xác suất softmax chưa hiệu chuẩn."}</p>
    </div></td>`;
}

function renderTable(modelIds, { animateIds = [] } = {}) {
  const body = byId("diag-body");
  const expanded = new Set($$("tr.detail-row:not(.hidden)", body).map(r => r.dataset.model));
  body.innerHTML = modelIds.map(id => `
    <tr class="diag-row row-${pillarOf(id)}${state.rows.get(id) && !state.rows.get(id).error ? " is-clickable" : ""}" data-model="${id}" tabindex="${state.rows.get(id) && !state.rows.get(id).error ? 0 : -1}"${state.rows.get(id) && !state.rows.get(id).error ? ` aria-expanded="${expanded.has(id)}"` : ""}>${rowHtml(id)}</tr>
    <tr class="detail-row${expanded.has(id) ? "" : " hidden"}" data-model="${id}">${detailHtml(id)}</tr>`).join("");
  // thanh độ tin cậy chạy từ 0 → giá trị (một lần reflow để chốt style ban đầu)
  void body.offsetWidth;
  $$(".conf-fill", body).forEach(el => setBar(el, Number(el.dataset.pct)));
  if (animateIds.length) {
    const rows = animateIds.map(id => body.querySelector(`tr.diag-row[data-model="${id}"]`));
    revealItems(rows, { stagger: 60, offset: 6, duration: 260 });
  }
}

function summarize(modelIds) {
  const trueClass = targetClass();
  const results = modelIds.map(id => [id, state.rows.get(id)]).filter(([, r]) => r && !r.error);
  const box = byId("diag-summary");
  const explain = byId("diag-explain");
  if (!results.length) {
    box.innerHTML = '<span class="verdict bad">Không có mô hình nào trả kết quả.</span>';
    toggleHidden(explain, true);
    return;
  }
  const pre = state.item
    ? `<p class="summary-line">Ảnh mẫu #${Number(state.item.id)} · <b>${esc(formatDiseaseName(state.item.label))}</b>${state.showStats ? ` · ${esc(state.item.difficulty)}` : ""}</p>` : "";

  if (trueClass) {
    const ok = results.filter(([id, r]) => verdict(r, trueClass) === "correct");
    const locals = results.filter(([id]) => id.startsWith("local_"));
    const localsOk = locals.filter(([, r]) => verdict(r, trueClass) === "correct");
    const mark = id => {
      const r = state.rows.get(id);
      if (!r || r.error) return "—";
      return verdict(r, trueClass) === "correct" ? "✓" : "✕";
    };
    box.innerHTML = `${pre}
      <p class="summary-big"><b>${ok.length}/${results.length}</b> mô hình đúng với nhãn <b>${esc(formatDiseaseName(trueClass))}</b></p>
      <div class="summary-pillars">
        ${state.rows.has("centralized") ? `<span class="summary-pill centralized">Centralized ${mark("centralized")}</span>` : ""}
        ${state.rows.has("fedavg") || state.mode === "quick" ? `<span class="summary-pill fedavg">${state.mode === "quick" && state.quickModel !== "fedavg" ? MODEL_LABELS[state.quickModel] : "FedAvg"} ${mark(state.mode === "quick" ? state.quickModel : "fedavg")}</span>` : ""}
        ${locals.length ? `<span class="summary-pill local">Local-only ${localsOk.length}/${locals.length}</span>` : ""}
      </div>`;
    explain.innerHTML = explanation(results, trueClass);
    toggleHidden(explain, !explain.innerHTML);
    box.insertAdjacentHTML("beforeend", timingHtml());
  } else {
    const tally = new Map();
    for (const [, r] of results) tally.set(r.predicted_class, (tally.get(r.predicted_class) || 0) + 1);
    const ranked = [...tally.entries()].sort((a, b) => b[1] - a[1]);
    box.innerHTML = `${pre}
      <p class="summary-big">Các mô hình dự đoán: ${ranked.map(([c, n]) => `<b>${esc(formatDiseaseName(c))}</b> (${n}/${results.length})`).join(" · ")}</p>
      <p class="field-hint">Chưa biết nhãn thật nên chưa chấm đúng/sai. Chọn nhãn ở ô "Nhãn thật" phía trên để thấy mô hình nào sai và vì sao.</p>${timingHtml()}`;
    toggleHidden(explain, true);
    explain.innerHTML = "";
  }
}

function explanation(results, trueClass) {
  if (!state.shards) return ""; // không có số ảnh theo shard thì không được suy ra "vì sao"
  const name = esc(formatDiseaseName(trueClass));
  const lines = [];
  const locals = results.filter(([id]) => id.startsWith("local_"));
  const wrongZero = [];
  const wrongSome = [];
  const rightLocals = [];
  for (const [id, r] of locals) {
    const info = shardInfo(id, trueClass);
    if (verdict(r, trueClass) === "correct") rightLocals.push({ id, info });
    else if (info && info.count === 0) wrongZero.push(id);
    else wrongSome.push({ id, info });
  }
  const names = ids => ids.map(id => MODEL_LABELS[id]).join(", ");
  if (wrongZero.length) {
    const many = wrongZero.length > 1;
    const they = many ? "chúng" : "nó";
    lines.push(`<b>${names(wrongZero)}</b> sai: dữ liệu của ${they} có <b>0 ảnh</b> thuộc nhóm «${name}», ${they} chưa từng thấy nhóm này.`);
  }
  for (const { id, info } of rightLocals) {
    const share = info && info.total ? ` (${fmtNum((info.count / info.total) * 100, 0)}% tổng ảnh của nhóm này)` : "";
    lines.push(`<b>${MODEL_LABELS[id]}</b> đúng: dữ liệu của nó có <b>${fmtInt(info?.count)} ảnh</b> của nhóm này${share}.`);
  }
  for (const { id, info } of wrongSome) {
    const share = info && info.total ? ` (${fmtNum((info.count / info.total) * 100, 1)}% tổng ảnh của nhóm này)` : "";
    lines.push(`<b>${MODEL_LABELS[id]}</b> chỉ có ${fmtInt(info?.count)} ảnh của nhóm này${share} và vẫn sai.`);
  }
  const fed = state.rows.get("fedavg");
  if (fed && !fed.error) {
    const info = shardInfo("fedavg", trueClass);
    lines.push(verdict(fed, trueClass) === "correct"
      ? `<b>FedAvg</b> đúng: học từ cả năm client, tổng cộng ${fmtInt(info?.count)} ảnh của nhóm này mà không ảnh nào rời khỏi client.`
      : `<b>FedAvg</b> sai dù đã tổng hợp ${fmtInt(info?.count)} ảnh của nhóm này từ các client — ảnh này khó với cả mô hình học liên kết.`);
  }
  return lines.length ? `<b>Vì sao?</b><ul>${lines.map(l => `<li>${l}</li>`).join("")}</ul>` : "";
}

// ───────── Chạy chẩn đoán ─────────
function plannedModels() {
  return state.mode === "all" ? MODEL_ORDER : ["centralized", state.quickModel];
}

function timingHtml() {
  const t = state.timing;
  if (!t) return "";
  const parts = [`${fmtNum(t.predict / 1000, 2)} s cho 7 mô hình`];
  if (t.explain !== undefined) parts.push(`${fmtNum(t.explain / 1000, 2)} s cho Grad-CAM`);
  return `<p class="field-hint">Máy chủ xử lý: ${parts.join(" · ")}.</p>`;
}

function renderGradcam(models) {
  const box = byId("diag-gradcam");
  const shown = models.filter(id => state.explanations.has(id));
  if (!state.explain || !shown.length) {
    toggleHidden(box, true);
    return;
  }
  byId("diag-gradcam-grid").innerHTML = shown.map(id => {
    const item = state.explanations.get(id);
    return `<figure class="gradcam-card row-${pillarOf(id)}">
        <img src="${esc(item.image)}" alt="Grad-CAM của ${esc(MODEL_LABELS[id])} cho nhóm ${esc(formatDiseaseName(item.class))}" width="224" height="224" loading="lazy">
        <figcaption>${modelPill(id)}<span>${esc(formatDiseaseName(item.class))}</span></figcaption>
      </figure>`;
  }).join("");
  toggleHidden(box, false);
}

/** Vẽ lại toàn bộ khối kết quả từ state (đổi chế độ xem / nhãn thật không cần gọi lại mô hình). */
// ───────── Khung ảnh: ảnh gốc → vùng FedAvg dựa vào ─────────
// Vạch quét = đang xử lý; crossfade 320 ms sang Grad-CAM chỉ khi có kết quả thật (cần bật Grad-CAM).
const STAGE_MODEL = "fedavg";

function setStageView(view) {
  const stage = byId("selected-stage");
  if (!stage) return;
  const cam = view === "cam";
  stage.classList.toggle("is-cam", cam);
  byId("stage-tag").textContent = cam ? "Grad-CAM của FedAvg v7: vùng ảnh ảnh hưởng tới nhóm nó đoán" : "";
  $$("[data-stage-view]").forEach(button => {
    const on = button.dataset.stageView === view;
    button.classList.toggle("is-active", on);
    button.setAttribute("aria-checked", String(on));
  });
}

let stageStamp = 0; // tăng mỗi lần đổi ảnh/xóa kết quả: giải mã ảnh Grad-CAM đến muộn của ảnh cũ bị bỏ qua

function resetStage() {
  stageStamp += 1;
  const stage = byId("selected-stage");
  if (!stage) return;
  stage.classList.remove("is-cam", "is-scanning");
  const cam = byId("preview-cam");
  cam.removeAttribute("src");
  cam.alt = "";
  cam.setAttribute("aria-hidden", "true"); // chưa có ảnh vùng: không để trình đọc màn hình đọc một ảnh không tồn tại
  byId("stage-views").classList.add("hidden");
  byId("stage-tag").textContent = "";
  setStageView("original");
}

function setScanning(on) {
  const stage = byId("selected-stage");
  if (!stage) return;
  stage.classList.toggle("is-scanning", on);
  if (on) byId("stage-tag").textContent = prefersReducedMotion() ? "Đang chẩn đoán…" : "";
  else if (!stage.classList.contains("is-cam")) byId("stage-tag").textContent = "";
}

async function renderStage() {
  const item = state.explanations.get(STAGE_MODEL);
  const views = byId("stage-views");
  if (!state.explain || !item?.image) {
    // Grad-CAM đang tắt hoặc chưa có ảnh: quay về ảnh gốc, không để lại ảnh vùng cũ
    views.classList.add("hidden");
    setStageView("original");
    return;
  }
  const cam = byId("preview-cam");
  const stamp = stageStamp;
  const wasHidden = views.classList.contains("hidden");
  let fresh = wasHidden;
  if (cam.getAttribute("src") !== item.image) {
    fresh = true;
    cam.src = item.image;
    try { await cam.decode(); } catch (_error) { /* ảnh vẫn hiển thị được */ }
    // ảnh khác đã được chọn / kết quả bị xóa / Grad-CAM bị tắt trong lúc giải mã: bỏ qua
    if (stamp !== stageStamp || !state.explain || state.explanations.get(STAGE_MODEL)?.image !== item.image) return;
  }
  cam.alt = "Vùng ảnh FedAvg dựa vào để chẩn đoán (Grad-CAM)";
  cam.removeAttribute("aria-hidden");
  views.classList.remove("hidden");
  // Chỉ tự chuyển sang "vùng mô hình nhìn" khi có ảnh mới/vừa bật lại; vẽ lại do đổi nhãn thật/chế độ xem thì giữ lựa chọn của người dùng.
  if (fresh) setStageView("cam");
}

function renderAll({ animate = false } = {}) {
  const models = plannedModels();
  renderTable(models, { animateIds: animate ? models : [] });
  summarize(models);
  renderGradcam(models);
  renderStage();
}

function startRun(models) {
  state.busy = true;
  setScanning(true);
  byId("diag-empty").classList.add("hidden");
  byId("diag-result").classList.remove("hidden");
  // trên màn hình hẹp, kết quả nằm dưới khung chọn ảnh: cuộn tới để người dùng thấy ngay
  if (window.matchMedia("(max-width: 1100px)").matches) {
    byId("diag-result").scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
  byId("diag-summary").innerHTML = '<p class="summary-big"><span class="skeleton skeleton-line"></span></p>';
  toggleHidden(byId("diag-explain"), true);
  toggleHidden(byId("diag-gradcam"), true);
  byId("btn-print").classList.add("hidden");
  renderTable(models);
  const button = byId("btn-diagnose");
  button.disabled = true;
  setHtml(button, '<svg class="svg-icon icon-sm spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Đang chẩn đoán…');
  toggleHidden(byId("diag-progress-badge"), false);
}

function finishRun() {
  state.busy = false;
  setScanning(false);
  toggleHidden(byId("diag-progress-badge"), true);
  const button = byId("btn-diagnose");
  button.disabled = false;
  setHtml(button, '<svg class="svg-icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Chẩn đoán lại');
  byId("btn-print").classList.toggle("hidden", ![...state.rows.values()].some(r => r && !r.error));
}

/** Một yêu cầu cho cả 7 mô hình (mỗi mô hình chạy đúng một lần trên máy chủ). Trả false nếu máy chủ cũ không có endpoint. */
async function runAll(runId, controller, form) {
  setText(byId("diag-progress-badge"), state.explain ? "Đang chạy 7 mô hình + Grad-CAM" : "Đang chạy 7 mô hình");
  try {
    const data = await postForm(`/api/inference/all?explain=${state.explain ? "true" : "false"}`, form, { signal: controller.signal, timeoutMs: 120000 });
    if (runId !== state.runId) return true;
    for (const id of MODEL_ORDER) {
      const result = data.models?.[id];
      state.rows.set(id, result?.ready ? result : { error: result?.error || "Mô hình chưa sẵn sàng" });
    }
    for (const [id, item] of Object.entries(data.explain || {})) {
      if (item?.image && item.image.startsWith("data:image/")) state.explanations.set(id, item);
    }
    state.timing = data.timing_ms || null;
    renderAll({ animate: true });
    return true;
  } catch (error) {
    if (isAbort(error) || runId !== state.runId) return true;
    if (error instanceof ApiError && (error.status === 404 || error.status === 405)) {
      state.allEndpoint = false;
      return false;
    }
    for (const id of MODEL_ORDER) state.rows.set(id, { error: error.message });
    renderAll();
    notify("error", "Không thể chẩn đoán ảnh", "Kiểm tra ảnh đầu vào hoặc kết nối rồi thử lại.", { detail: error.message });
    return true;
  }
}

/** Cách cũ (máy chủ chưa cập nhật): Centralized + từng mô hình, mỗi lần một yêu cầu; không có Grad-CAM. */
async function runPairs(runId, controller, form) {
  const models = plannedModels();
  const sources = state.mode === "all" ? ALL_SOURCES : [state.quickModel];
  const badge = byId("diag-progress-badge");
  let done = 0;
  let failed = 0;
  for (const source of sources) {
    if (runId !== state.runId) return;
    setText(badge, `Đang chạy ${done + 1}/${sources.length}`);
    try {
      const data = await postForm(`/api/inference?federated_source=${encodeURIComponent(source)}`, form, { signal: controller.signal, timeoutMs: 90000 });
      if (runId !== state.runId) return;
      const fresh = [source];
      if (!state.rows.has("centralized") && data.centralized?.ready) {
        state.rows.set("centralized", data.centralized);
        fresh.unshift("centralized");
      }
      if (data.federated?.ready) state.rows.set(source, data.federated);
      else state.rows.set(source, { error: data.federated?.error || "Mô hình chưa sẵn sàng" });
      renderTable(models, { animateIds: fresh });
    } catch (error) {
      if (isAbort(error) || runId !== state.runId) return;
      failed += 1;
      state.rows.set(source, { error: error.message });
      renderTable(models);
    }
    done += 1;
  }
  if (runId !== state.runId) return;
  if (!state.rows.has("centralized")) state.rows.set("centralized", { error: "Không có kết quả từ máy chủ" });
  renderAll();
  if (state.explain) notify("info", "Máy chủ chưa hỗ trợ Grad-CAM", "Bản máy chủ này chưa có /api/inference/all; bảng kết quả vẫn đầy đủ.");
  if (failed === sources.length) {
    notify("error", "Không thể chẩn đoán ảnh", "Kiểm tra ảnh đầu vào hoặc kết nối rồi thử lại.");
  } else if (failed > 0) {
    notify("warning", "Một số mô hình không trả kết quả", `${failed}/${sources.length} yêu cầu bị lỗi; bảng ghi rõ từng dòng.`);
  }
}

async function run() {
  if (!state.file) return;
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const runId = ++state.runId;
  state.rows.clear();
  state.explanations.clear();
  state.timing = null;
  state.runExplain = state.explain;

  const models = plannedModels();
  const form = new FormData();
  form.append("file", state.file);
  startRun(models);
  try {
    const handled = state.allEndpoint !== false && await runAll(runId, controller, form);
    if (!handled && runId === state.runId) await runPairs(runId, controller, form);
  } finally {
    if (runId === state.runId) finishRun();
  }
}

/** Đã có đủ kết quả cho chế độ xem hiện tại (và Grad-CAM nếu đang bật) thì chỉ cần vẽ lại. */
function haveResultsFor(models) {
  const complete = models.every(id => state.rows.has(id));
  const gradcamOk = !state.explain || state.allEndpoint === false || models.every(id => state.explanations.has(id) || state.rows.get(id)?.error);
  return complete && gradcamOk;
}

async function runPlantVillageReference() {
  if (!state.file) return;
  const button = byId("btn-plantvillage");
  const output = byId("plantvillage-result");
  const file = state.file;
  button.disabled = true;
  output.textContent = "Đang nhận diện bằng PlantVillage...";
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await postForm("/api/inference/plantvillage", form);
    if (file !== state.file) return; // ảnh đã đổi: không ghi kết quả của ảnh cũ cạnh ảnh mới
    const top = data.top3.map((r, i) => `${i + 1}. ${formatDiseaseName(r.class)}: ${fmtNum(r.confidence, 2)}%`).join("\n");
    output.textContent = `Dự đoán: ${formatDiseaseName(data.predicted_class)} (${fmtNum(data.confidence_pct, 2)}%)\nTop 3:\n${top}\n${data.is_uncertain ? "⚠ Mô hình không chắc chắn.\n" : ""}Lưu ý: độ tin cậy chưa hiệu chuẩn; kết quả có thể sai trên ảnh thực tế.`;
  } catch (error) {
    if (file !== state.file) return;
    output.textContent = `Không thể nhận diện: ${error.message}`;
    notify("error", "Không thể nhận diện bằng PlantVillage", "", { detail: error.message });
  } finally {
    button.disabled = state.file === null;
  }
}

// ───────── Khởi tạo ─────────
async function loadModelOptions() {
  const select = byId("inference-fed-source");
  try {
    const payload = await getModels();
    const choices = (payload.models || []).filter(m => m.role === "current" && m.inference_key && m.inference_key !== "centralized");
    if (!choices.length) throw new Error("Registry chưa cho phép model nào");
    select.replaceChildren();
    for (const model of choices) {
      const option = document.createElement("option");
      option.value = model.inference_key;
      option.textContent = `${model.display_name} — ${model.display_label}`;
      option.selected = model.inference_key === "fedavg";
      select.appendChild(option);
    }
    state.quickModel = select.value;
  } catch (error) {
    select.disabled = true;
    select.innerHTML = `<option value="">Không tải được danh sách model: ${esc(error.message)}</option>`;
  }
}

function setMode(mode) {
  state.mode = mode;
  $$("[data-mode]").forEach(button => {
    const on = button.dataset.mode === mode;
    button.classList.toggle("is-active", on);
    button.setAttribute("aria-checked", String(on));
  });
  toggleHidden(byId("quick-model-group"), mode !== "quick");
  rerender();
}

/** Đổi chế độ xem / mô hình so sánh / Grad-CAM: vẽ lại nếu đã có đủ kết quả, chỉ gọi máy chủ khi còn thiếu. */
function rerender() {
  if (!state.file || (!state.busy && !state.rows.size)) return; // chưa chẩn đoán lần nào
  if (state.busy) {
    // yêu cầu "cả 7 mô hình" đang chạy sẽ vẽ theo chế độ xem mới; chỉ chạy lại khi cần Grad-CAM mà yêu cầu đó không có
    if (state.allEndpoint === false || (state.explain && !state.runExplain)) run();
    return;
  }
  if (haveResultsFor(plannedModels())) renderAll();
  else run();
}

export async function mount() {
  const [evidence, summary, gallery] = await Promise.all([
    getEvidence().catch(() => null),
    getSummary().catch(() => null),
    getGallery(),
  ]);
  state.classes = evidence?.classes || [];
  state.shards = summary?.shards?.available ? summary.shards : null;
  state.gallery = gallery.items || [];
  state.showStats = evidence?.available === true && gallery.validation_membership?.status === "verified";
  if (!state.showStats) {
    // không có số đã kiểm hash để xếp độ khó: ẩn các bộ lọc theo độ khó
    $$("#gallery-filters [data-filter='easy'], #gallery-filters [data-filter='medium'], #gallery-filters [data-filter='hard']")
      .forEach(chip => chip.classList.add("hidden"));
    const note = byId("gallery-note");
    if (note) note.textContent = "Ảnh mẫu có nhãn thật; chưa đối chiếu được với tập kiểm tra nên không xếp độ khó.";
  }

  const trueSelect = byId("inference-true-class");
  for (const cls of state.classes) {
    const option = document.createElement("option");
    option.value = cls;
    option.textContent = formatDiseaseName(cls);
    trueSelect.appendChild(option);
  }
  trueSelect.addEventListener("change", () => {
    // đổi nhãn sau khi đã có kết quả: chấm lại ngay, không gọi lại mô hình
    if (state.rows.size && !state.busy) renderAll();
  });

  await loadModelOptions();
  renderGallery();

  byId("gallery-grid").addEventListener("click", event => {
    const button = event.target.closest(".gallery-item");
    if (!button) return;
    const item = state.gallery.find(i => String(i.id) === button.dataset.id);
    if (item) selectGallery(item);
  });
  byId("gallery-filters").addEventListener("click", event => {
    const chip = event.target.closest("[data-filter]");
    if (!chip) return;
    state.filter = chip.dataset.filter;
    $$("#gallery-filters .chip").forEach(c => c.classList.toggle("is-active", c === chip));
    renderGallery();
  });

  const input = byId("leaf-image-input");
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      selectUpload(input.files[0]);
      input.value = ""; // cho phép chọn lại đúng tệp đó lần sau
    }
  });
  const zone = byId("upload-dropzone");
  ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, () => zone.classList.add("is-dragover")));
  ["dragleave", "drop", "dragend"].forEach(name => zone.addEventListener(name, () => zone.classList.remove("is-dragover")));

  $$("[data-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
  byId("inference-fed-source").addEventListener("change", event => {
    state.quickModel = event.target.value || "fedavg";
    if (state.mode === "quick") rerender();
  });
  byId("explain-toggle").addEventListener("change", event => {
    state.explain = event.target.checked;
    rerender();
  });
  byId("btn-print").addEventListener("click", () => window.print());
  $$("[data-stage-view]").forEach(button => button.addEventListener("click", () => setStageView(button.dataset.stageView)));
  byId("btn-diagnose").addEventListener("click", run);
  byId("btn-plantvillage").addEventListener("click", runPlantVillageReference);

  // bấm hoặc Enter/Space trên một dòng kết quả để mở top-3
  const body = byId("diag-body");
  const toggleRow = row => {
    const detail = body.querySelector(`tr.detail-row[data-model="${row.dataset.model}"]`);
    if (detail && detail.innerHTML.trim()) {
      const open = detail.classList.toggle("hidden") === false;
      row.setAttribute("aria-expanded", String(open));
    }
  };
  body.addEventListener("click", event => {
    const row = event.target.closest("tr.diag-row");
    if (row) toggleRow(row);
  });
  body.addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest("tr.diag-row");
    if (row) { event.preventDefault(); toggleRow(row); }
  });
}
