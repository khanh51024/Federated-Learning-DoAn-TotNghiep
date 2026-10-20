// Demo · Chẩn đoán ảnh: một ảnh, bảy mô hình, đúng/sai và lý do (ảnh train của lớp đó trong shard).
import { getEvidence, getGallery, getModels, getSummary } from "../core/store.js";
import { ApiError, isAbort, postForm } from "../core/api.js";
import {
  $$, MODEL_LABELS, MODEL_ORDER, NARROW_QUERY, byId, cropOf, diseaseOf, esc, fmtInt, fmtNum, fmtPct, fmtPctValue, formatApiError,
  formatDiseaseName, modelPill, notify, pillarOf, prefersReducedMotion, revealItems, setBar, setDisabled, setHtml,
  openOnWide, setText, toggleHidden,
} from "../core/dom.js";

const LOCAL_SOURCES = ["local_00", "local_01", "local_02", "local_03", "local_04"];
const ALL_SOURCES = ["fedavg", ...LOCAL_SOURCES];
const LEVEL_LABEL = { very_easy: "Nhận đúng rất cao", easy: "Nhận đúng cao", medium: "Nhận đúng trung bình", hard: "Nhận đúng thấp", very_hard: "Nhận đúng rất thấp" };
// Nhãn theo tỉ lệ nhận đúng của nhóm bệnh (recall lớp ở FedAvg), không phải độ khó của riêng ảnh
const levelText = item => LEVEL_LABEL[item.difficulty_key] || item.difficulty;
// Thẻ ảnh mẫu rộng ~88 px trên điện thoại: nhãn đầy đủ ("Nhận đúng trung bình") xuống 3 dòng. Thẻ dùng bản ngắn (mức nhận đúng đã được giải thích ở chú thích dưới thư viện ảnh); bản đầy đủ nằm ở aria-label và title.
const LEVEL_SHORT = { very_easy: "Rất cao", easy: "Cao", medium: "Vừa", hard: "Thấp", very_hard: "Rất thấp" };
const levelShort = item => LEVEL_SHORT[item.difficulty_key] || levelText(item);

// Giới hạn dung lượng ảnh tải lên: phải khớp MAX_UPLOAD_BYTES trong web/routers/inference.py (có test kiểm hai số này).
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

const DIAGNOSE_HTML = '<svg class="svg-icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Chẩn đoán bằng cả 3 cách';

const state = {
  file: null,
  item: null, // mục catalog nếu chọn ảnh mẫu
  previewUrl: null,
  mode: "all",
  showLocals: false, // bảng gom nhóm: mở/thu năm hàng Local-only (mặc định thu về một dòng tóm tắt)
  quickModel: "fedavg",
  filter: "all",
  galleryAll: false, // đang xem toàn bộ thư viện ảnh mẫu (mặc định chỉ 8 ảnh đầu)
  runId: 0,
  selectToken: 0, // chỉ lần chọn ảnh MỚI NHẤT được áp dụng (fetch ảnh mẫu có thể về không đúng thứ tự)
  showStats: false, // chỉ hiện độ khó/recall từ catalog khi bundle khả dụng và ảnh đã đối chiếu manifest validation
  controller: null,
  rows: new Map(), // model_id -> kết quả hoặc {error}
  runCrop: "", // loại cây của lần chẩn đoán đang chạy (đọc từ ô chọn lúc bấm), vd "Corn_(maize)"; rỗng = để mô hình tự đoán
  explain: false, // người dùng bật Grad-CAM
  explanations: new Map(), // model_id -> {class, image}
  allEndpoint: null, // null: chưa biết; false: máy chủ cũ không có /api/inference/all → gọi từng cặp
  timing: null,
  busy: false, // đang có yêu cầu chẩn đoán
  runExplain: false, // yêu cầu đang chạy có kèm Grad-CAM không
  classes: [],
  shards: null,
  gallery: [],
  pv: null, // kết quả cách nhận PlantVillage: null | {loading} | {data} | {error}
  community: null, // kết quả cách nhận cộng đồng, cùng dạng
  pvEvidence: null, // báo cáo đánh giá lại PlantVillage (từ /api/evidence/summary)
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

const GALLERY_PAGE = 8; // UI-05: mặc định 8 ảnh đại diện + "Xem tất cả"; bộ lọc vẫn áp lên toàn bộ thư viện

function renderGallery() {
  const grid = byId("gallery-grid");
  const matching = state.gallery.filter(FILTERS[state.filter] || FILTERS.all);
  let items = state.galleryAll ? matching : matching.slice(0, GALLERY_PAGE);
  if (state.item && !items.includes(state.item) && matching.includes(state.item)) items = [...items, state.item]; // ảnh đang chọn luôn hiện
  const more = byId("gallery-more");
  if (more) {
    toggleHidden(more, matching.length <= GALLERY_PAGE);
    more.textContent = state.galleryAll ? `Chỉ hiện ${GALLERY_PAGE} ảnh` : `Xem tất cả ${matching.length} ảnh`;
    more.setAttribute("aria-expanded", String(state.galleryAll));
  }
  grid.removeAttribute("aria-busy");
  grid.classList.toggle("is-expanded", state.galleryAll);
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
    const level = state.showStats ? `<span class="level level-${esc(item.difficulty_key)}" title="${esc(levelText(item))}">${esc(levelShort(item))}</span>` : "";
    const levelLabel = state.showStats ? `, ${esc(levelText(item))}` : "";
    return `
    <button type="button" class="gallery-item${state.item?.id === item.id ? " is-selected" : ""}" data-id="${id}"
      aria-pressed="${state.item?.id === item.id}"
      aria-label="Ảnh mẫu ${id}: ${esc(formatDiseaseName(item.label))}${levelLabel}${item.field_photo ? ", ảnh ngoài đồng" : ""}">
      <img src="/static/samples/demo/${esc(item.thumb)}" alt="" loading="lazy" width="88" height="88">
      <span class="gallery-cap" title="${esc(formatDiseaseName(item.label))}">${esc(formatDiseaseName(item.label))}</span>
      <span class="gallery-src">${item.source === "plantdoc" ? "PlantDoc · ngoài đồng" : "PlantVillage"}</span>
      ${level}
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
const BIG_PREVIEW_BYTES = 1.5 * 1024 * 1024;

/** Ảnh lớn: thu về 960 px bằng createImageBitmap (giải mã ngoài luồng chính) trước khi gắn vào <img>;
 *  giải mã thẳng ảnh ~10 MB làm đứng luồng chính hơn 1 s. Lỗi thì rơi về ảnh gốc. */
async function downscaledPreviewUrl(file) {
  const bitmap = await createImageBitmap(file, { resizeWidth: 960, resizeQuality: "medium" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  bitmap.close?.();
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
  if (!blob) throw new Error("toBlob");
  return URL.createObjectURL(blob);
}

function setPreview(file, metaHtml) {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  const img = byId("preview-image");
  img.decoding = "async";
  if (file.size > BIG_PREVIEW_BYTES && typeof createImageBitmap === "function") {
    const original = URL.createObjectURL(file);
    state.previewUrl = original;
    img.removeAttribute("src");
    downscaledPreviewUrl(file).then(url => {
      if (state.file !== file) { URL.revokeObjectURL(url); return; } // đã chọn ảnh khác
      URL.revokeObjectURL(original);
      state.previewUrl = url;
      img.src = url;
    }).catch(() => { if (state.file === file) img.src = original; });
  } else {
    state.previewUrl = URL.createObjectURL(file);
    img.src = state.previewUrl;
  }
  byId("selected-stage").classList.remove("hidden");
  resetStage();
  byId("selected-empty").classList.add("hidden");
  const meta = byId("selected-meta");
  meta.innerHTML = metaHtml;
  meta.classList.remove("hidden");
  setDisabled(byId("btn-diagnose"), false);
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
  state.pv = null;
  state.community = null;
  byId("local-view").innerHTML = "";
  renderPv();
  renderCommunity();
  renderTabBadges();
  setDisabled(byId("btn-all-models"), true);
  setHtml(byId("btn-diagnose"), DIAGNOSE_HTML); // phiên chạy cũ bị hủy giữa chừng không còn dòng "Đang chẩn đoán…"
}

function itemMeta(item) {
  const verified = state.showStats;
  const level = verified ? `<span class="level level-${esc(item.difficulty_key)}">${esc(levelText(item))}</span>` : "";
  const stats = verified
    ? `<p class="field-hint">Trên ${fmtInt(item.class_val_count)} ảnh kiểm tra ${item.source === "plantdoc" ? "PlantDoc" : "PlantVillage"} của nhóm này,
      FedAvg nhận đúng ${fmtPctValue(item.recall_fedavg_pct, 0)}, Centralized ${fmtPctValue(item.recall_centralized_pct, 0)}.</p>`
    : '<p class="field-hint">Chưa có số liệu để xếp mức nhận đúng cho ảnh này.</p>';
  return `
    <div class="meta-chips">
      ${level}
      <span class="chip chip-static">${item.source === "plantdoc" ? "PlantDoc · ảnh ngoài đồng" : "PlantVillage"}</span>
      <span class="chip chip-static">ảnh mẫu #${Number(item.id)}${verified ? " · thuộc tập kiểm tra" : " · chưa đối chiếu được"}</span>
    </div>
    <p class="meta-label">${esc(formatDiseaseName(item.label))}</p>
    <details class="fold"><summary>Thống kê của nhóm ảnh</summary>${stats}<p class="field-hint">Dễ/Khó theo recall của lớp ở FedAvg, không phải độ khó đo riêng ảnh này.</p></details>`;
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
  // chặn trước resetResults(): ảnh đang chọn và kết quả của nó không bị mất vì một tệp bị từ chối
  if (file.size === 0) {
    notify("warning", "Tệp rỗng", `"${file.name}" không có dữ liệu. Hãy chọn một ảnh khác.`);
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    notify("warning", "Ảnh quá lớn", `"${file.name}" nặng ${fmtNum(file.size / 1048576, 1)} MB, vượt giới hạn ${MAX_UPLOAD_BYTES / 1048576} MB. Hãy chọn ảnh nhỏ hơn.`);
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

function rowHtml(modelId, expanded = false) {
  const result = state.rows.get(modelId);
  const label = MODEL_LABELS[modelId];
  const phase = modelId === "centralized" ? "Gom ảnh" : modelId === "fedavg" ? "Học phân tán" : "Học riêng";
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
  // aria-expanded chỉ hợp lệ trên hàng của treegrid, không phải của table: dùng một nút thật trong ô tên mô hình
  return `<th scope="row"><button type="button" class="row-toggle" aria-expanded="${expanded}" aria-controls="detail-${modelId}">${modelPill(modelId)}<span class="row-sub">${phase}</span><span class="sr-only">Xem top 3 xác suất</span></button></th>
    <td class="cell-pred">
      <b>${esc(formatDiseaseName(result.predicted_class))}</b>${uncertain}
      <div class="conf"><div class="conf-track"><span class="conf-fill ${pillarOf(modelId)}" data-pct="${result.confidence_pct}"></span></div><span class="mono">${fmtNum(result.confidence_pct, 1)}%</span></div>
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
        ${result.calibrated ? `Số hiệu chuẩn chia logit cho T = ${fmtNum(result.calibrated.temperature, 2)}; thứ tự các nhóm không đổi.` : result.crop_restricted ? cropRestrictedHint(result.crop_restricted) : "Xác suất softmax chưa hiệu chuẩn."}</p>
    </div></td>`;
}

// Ba cách huấn luyện của đồ án, theo thứ tự đọc: mức tham chiếu, phương pháp đề xuất, các nơi tự học riêng.
const GROUPS = [
  { pillar: "centralized", title: "Centralized", sub: "Gom ảnh về một chỗ · mức tham chiếu" },
  { pillar: "fedavg", title: "FedAvg · học phân tán", sub: "Phương pháp đề xuất · ảnh ở lại client" },
  { pillar: "local", title: "Local-only", sub: "Mỗi nơi tự học riêng, không chia sẻ trọng số" },
];

/** Dòng tóm tắt của nhóm Local-only khi đang thu gọn: số nơi nhận đúng, hoặc nhóm được dự đoán nhiều nhất khi chưa biết nhãn thật. */
function localSummaryText(localIds) {
  const results = localIds.map(id => state.rows.get(id)).filter(r => r && !r.error);
  if (!results.length) return state.busy ? "đang chạy…" : "chưa có kết quả";
  const trueClass = targetClass();
  if (trueClass) {
    const ok = results.filter(r => verdict(r, trueClass) === "correct").length;
    return `${ok}/${results.length} nơi nhận đúng`;
  }
  const tally = new Map();
  for (const r of results) tally.set(r.predicted_class, (tally.get(r.predicted_class) || 0) + 1);
  const [name, n] = [...tally.entries()].sort((a, b) => b[1] - a[1])[0];
  return `${n}/${results.length} nơi cùng dự đoán ${formatDiseaseName(name)}`;
}

function groupRowHtml(group, ids) {
  const collapsible = group.pillar === "local" && ids.length > 1;
  const summary = collapsible
    ? `<span class="group-summary-text">${esc(localSummaryText(ids))}</span>
       <button type="button" class="btn btn-ghost btn-sm" data-toggle-locals aria-expanded="${state.showLocals}" aria-controls="diag-body">${state.showLocals ? "Thu gọn" : `Xem ${ids.length} mô hình Local`}</button>`
    : "";
  return `<tr class="group-row group-${group.pillar}"><th scope="colgroup" colspan="4">
      <span class="group-head"><span class="group-title">${esc(group.title)}${ids.length > 1 ? ` · ${ids.length} mô hình` : ""}</span><span class="group-sub">${esc(group.sub)}</span></span>
      ${summary}</th></tr>`;
}

function renderTable(modelIds, { animateIds = [] } = {}) {
  const body = byId("diag-body");
  const expanded = new Set($$("tr.detail-row:not(.hidden)", body).map(r => r.dataset.model));
  const rowsHtml = GROUPS.map(group => {
    const ids = modelIds.filter(id => pillarOf(id) === group.pillar);
    if (!ids.length) return "";
    const collapsed = group.pillar === "local" && ids.length > 1 && !state.showLocals;
    const member = collapsed ? " is-collapsed" : "";
    return groupRowHtml(group, ids) + ids.map(id => `
    <tr class="diag-row row-${pillarOf(id)}${state.rows.get(id) && !state.rows.get(id).error ? " is-clickable" : ""}${member}" data-model="${id}">${rowHtml(id, expanded.has(id))}</tr>
    <tr class="detail-row${expanded.has(id) ? "" : " hidden"}${member}" id="detail-${id}" data-model="${id}">${detailHtml(id)}</tr>`).join("");
  }).join("");
  body.innerHTML = rowsHtml;
  // thanh độ tin cậy chạy từ 0 → giá trị (một lần reflow để chốt style ban đầu)
  void body.offsetWidth;
  $$(".conf-fill", body).forEach(el => setBar(el, Number(el.dataset.pct)));
  if (animateIds.length) {
    const rows = animateIds.map(id => body.querySelector(`tr.diag-row[data-model="${id}"]`)).filter(row => row && !row.classList.contains("is-collapsed"));
    revealItems(rows, { stagger: 60, offset: 6, duration: 260 });
  }
}

function toggleLocals() {
  state.showLocals = !state.showLocals;
  if (!byId("diag-result").classList.contains("hidden")) renderTable(plannedModels());
}

/** Kết quả chính (UI-03): mô hình học phân tán đứng đầu với nhãn dự đoán, đúng/sai và độ tin cậy; các mô hình còn lại chỉ là đối chiếu. */
/** Số mô hình (trong các mô hình đang hiển thị) trùng với mô hình chính về loại cây và về nhóm bệnh. */
function agreement(refId) {
  const ref = state.rows.get(refId);
  const rows = plannedModels().map(id => state.rows.get(id)).filter(r => r && !r.error);
  const refCrop = cropOf(ref.predicted_class);
  return {
    total: rows.length,
    sameCrop: rows.filter(r => cropOf(r.predicted_class) === refCrop).length,
    sameClass: rows.filter(r => r.predicted_class === ref.predicted_class).length,
  };
}

function headlineHtml(trueClass) {
  const ref = state.mode === "quick" && state.quickModel ? state.quickModel : "fedavg";
  const result = state.rows.get(ref);
  if (!result || result.error) return "";
  const v = verdict(result, trueClass);
  const tone = v === "correct" ? "is-correct" : v ? "is-wrong" : "is-unknown";
  const verdictText = v === "correct" ? '<span class="verdict ok">✓ Đúng</span>'
    : v === "wrong-disease" ? '<span class="verdict mid">✕ Sai bệnh (đúng loại cây)</span>'
      : v === "wrong" ? '<span class="verdict bad">✕ Sai cả loại cây</span>'
        : '<span class="verdict none">chưa biết nhãn thật nên chưa chấm</span>';
  const calibrated = result.calibrated ? `sau hiệu chuẩn ${fmtNum(result.calibrated.confidence_pct, 1)}%` : result.crop_restricted ? "chuẩn hóa trong nhóm cây" : "chưa hiệu chuẩn";
  const uncertain = result.is_uncertain ? ' <span class="pill-flag" title="Xác suất cao nhất dưới 50% hoặc hai nhóm đầu quá sát nhau">không chắc</span>' : "";
  const agree = agreement(ref);
  const agreeText = count => (agree.total > 1 ? `${count}/${agree.total} mô hình đồng ý` : "chỉ một mô hình");
  const truth = trueClass && v !== "correct"
    ? `<p class="headline-truth">Nhãn thật: <b>${esc(formatDiseaseName(trueClass))}</b></p>` : "";
  return `<div class="diag-headline ${tone}">
      <p class="summary-line">Kết quả chính, mô hình ${esc(MODEL_LABELS[ref])}${ref === "fedavg" ? " (học phân tán)" : ""}</p>
      <div class="verdict-grid">
        <div class="vg-cell"><span class="vg-label">Loại cây</span><b class="vg-value">${esc(cropOf(result.predicted_class))}</b><span class="vg-sub">${agreeText(agree.sameCrop)}</span></div>
        <div class="vg-cell"><span class="vg-label">Nhóm</span><b class="vg-value headline-name">${esc(diseaseOf(result.predicted_class))}${uncertain}</b><span class="vg-sub">${agreeText(agree.sameClass)}</span></div>
        <div class="vg-cell"><span class="vg-label">Xác suất dự đoán</span><b class="vg-value mono">${fmtNum(result.confidence_pct, 1)}%</b><span class="vg-sub">${calibrated}</span></div>
      </div>
      <p class="headline-meta">${verdictText}</p>
      ${truth}
      <p class="field-hint">Độ tin cậy của mô hình cho nhóm đã chọn; ảnh thực tế có thể sai khác dù xác suất cao.</p>
    </div>`;
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
    ? `<p class="summary-line">Ảnh mẫu #${Number(state.item.id)}, <b>${esc(formatDiseaseName(state.item.label))}</b>${state.showStats ? `, ${esc(levelText(state.item))}` : ""}</p>`
    : '<p class="summary-line">Ảnh của bạn (ngoài bộ dữ liệu): kết quả chỉ để tham khảo.</p>';

  if (trueClass) {
    const ok = results.filter(([id, r]) => verdict(r, trueClass) === "correct");
    const locals = results.filter(([id]) => id.startsWith("local_"));
    const localsOk = locals.filter(([, r]) => verdict(r, trueClass) === "correct");
    const mark = id => {
      const r = state.rows.get(id);
      if (!r || r.error) return "—";
      return verdict(r, trueClass) === "correct" ? "✓" : "✕";
    };
    box.innerHTML = `${pre}${headlineHtml(trueClass)}
      <p class="summary-big">Đối chiếu: <b>${ok.length}/${results.length}</b> mô hình nhận đúng ảnh này</p>
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
    box.innerHTML = `${pre}${headlineHtml(null)}
      <p class="summary-big">Đối chiếu, các mô hình dự đoán: ${ranked.map(([c, n]) => `<b>${esc(formatDiseaseName(c))}</b> (${n}/${results.length})`).join(" · ")}</p>
      <p class="field-hint">Chưa biết nhãn thật nên chưa chấm đúng/sai. Nếu biết nhãn thật, chọn ở ô “Nhãn thật” phía trên để đối chiếu.</p>${timingHtml()}`;
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
      : `<b>FedAvg</b> sai dù đã tổng hợp ${fmtInt(info?.count)} ảnh của nhóm này từ các client — ảnh này khó với cả mô hình học phân tán.`);
  }
  const overview = [];
  if (wrongZero.length) overview.push(`${names(wrongZero)} nhận sai và không có ảnh của nhóm này trong phần dữ liệu học.`);
  if (fed && !fed.error) overview.push(`FedAvg ${verdict(fed, trueClass) === "correct" ? "nhận đúng" : "nhận sai"}; dữ liệu của các client có tổng ${fmtInt(shardInfo("fedavg", trueClass)?.count)} ảnh của nhóm này.`);
  return lines.length ? `<b>Liên hệ với dữ liệu huấn luyện</b><p class="panel-desc mt-2">${overview.map(esc).join(" ")}</p><p class="field-hint mt-2">Thiếu dữ liệu có thể góp phần vào lỗi; đây chưa phải kiểm chứng nguyên nhân của dự đoán.</p><details class="fold mt-2"><summary>Đối chiếu từng client</summary><ul>${lines.map(l => `<li>${l}</li>`).join("")}</ul></details>` : "";
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

/** Dòng giải thích dưới danh sách top-3 khi kết quả bị giới hạn theo cây: kèm phần xác suất gốc mô hình đặt vào cây đó. */
function cropRestrictedHint(info) {
  const crop = String(info.crop).replace(/_/g, " ");
  const low = info.low_crop_mass ? " Mô hình gần như không thấy cây này trong ảnh: kết quả không đáng tin." : "";
  return `Đã giới hạn theo cây ${esc(crop)} do bạn chọn; xác suất chuẩn hóa lại trong nhóm cây (${fmtNum(info.classes_in_crop, 0)} nhóm). Mô hình đặt ${fmtNum(info.crop_mass_pct, 1)}% xác suất gốc vào cây này.${low}`;
}

/** Kết quả đã bị giới hạn theo cây người dùng chọn: nói rõ để không ai đọc xác suất như softmax thô của 38 lớp. */
function renderCropNote(models) {
  const results = models.map(id => state.rows.get(id)).filter(r => r && !r.error);
  const restricted = results.find(r => r.crop_restricted);
  const summary = byId("diag-summary");
  const note = document.createElement("p");
  note.className = "field-hint mt-2";
  note.id = "diag-crop-note";
  if (restricted) {
    const low = results.some(r => r.crop_restricted?.low_crop_mass);
    note.textContent = `Đã giới hạn theo cây ${restricted.crop_restricted.crop.replace(/_/g, " ")} do bạn chọn: xác suất được chuẩn hóa lại trong nhóm cây này, không còn là softmax của 38 lớp.${low ? " Ít nhất một mô hình đặt dưới 50% xác suất gốc vào cây này, nên kết quả đó không đáng tin." : ""} Nếu ảnh không phải cây này, kết quả vô nghĩa.`;
  } else if (state.runCrop && results.length) {
    // máy chủ cũ hoặc đường gọi từng cặp bỏ qua tham số crop: nói rõ thay vì im lặng
    note.textContent = "Lựa chọn loại cây không được áp dụng: máy chủ này chưa hỗ trợ giới hạn theo cây, kết quả ở trên là kết quả đầy đủ 38 nhóm.";
  } else {
    return;
  }
  summary.appendChild(note);
}

function renderAll({ animate = false } = {}) {
  const models = plannedModels();
  renderTable(models, { animateIds: animate ? models : [] });
  summarize(models);
  renderCropNote(models);
  renderGradcam(models);
  renderStage();
  renderLocal();
  renderPv();
  renderCommunity();
  renderTabBadges();
}

function startRun(models) {
  state.busy = true;
  setScanning(true);
  byId("diag-empty").classList.add("hidden");
  byId("diag-result").classList.remove("hidden");
  // trên màn hình hẹp, kết quả nằm dưới khung chọn ảnh: cuộn tới để người dùng thấy ngay
  if (window.matchMedia(NARROW_QUERY).matches) {
    byId("diag-result").scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
  byId("diag-summary").innerHTML = '<p class="summary-big"><span class="skeleton skeleton-line"></span></p>';
  toggleHidden(byId("diag-explain"), true);
  toggleHidden(byId("diag-gradcam"), true);
  byId("btn-print").classList.add("hidden");
  setDisabled(byId("btn-all-models"), true);
  renderTable(models);
  renderLocal();
  renderTabBadges();
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
  const any = [...state.rows.values()].some(r => r && !r.error);
  byId("btn-print").classList.toggle("hidden", !any);
  setDisabled(byId("btn-all-models"), !any);
  renderTabBadges();
}

/** Một yêu cầu cho cả 7 mô hình (mỗi mô hình chạy đúng một lần trên máy chủ). Trả false nếu máy chủ cũ không có endpoint. */
async function runAll(runId, controller, form) {
  setText(byId("diag-progress-badge"), state.explain ? "Đang chạy 7 mô hình + Grad-CAM" : "Đang chạy 7 mô hình");
  try {
    const cropQuery = state.runCrop ? `&crop=${encodeURIComponent(state.runCrop)}` : "";
    const data = await postForm(`/api/inference/all?explain=${state.explain ? "true" : "false"}${cropQuery}`, form, { signal: controller.signal, timeoutMs: 120000 });
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
  state.runCrop = byId("inference-crop")?.value || "";

  const models = plannedModels();
  const form = new FormData();
  form.append("file", state.file);
  startRun(models);
  const file = state.file;
  runPv(runId, file); // PlantVillage và mô hình cộng đồng chạy song song với bảy mô hình; mỗi cách có trạng thái riêng
  runCommunity(runId, file);
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

// ───────── Ba cách nhận diện ngang hàng: Local only · PlantVillage · Cộng đồng ─────────
const DIAG_TABS = ["local", "pv", "community"];
const TAB_NAME = { local: "Local only", pv: "PlantVillage", community: "Cộng đồng" };

function applyBars(root) {
  void root.offsetWidth; // một lần reflow để thanh độ tin cậy chạy từ 0 → giá trị
  $$(".conf-fill", root).forEach(el => setBar(el, Number(el.dataset.pct)));
}

function confBarHtml(pct, pillar) {
  return `<div class="conf"><div class="conf-track"><span class="conf-fill ${pillar}" data-pct="${Number(pct) || 0}"></span></div><span class="mono">${fmtNum(pct, 1)}%</span></div>`;
}

function verdictHtml(v) {
  return v === "correct" ? '<span class="verdict ok">✓ Đúng</span>'
    : v === "wrong-disease" ? '<span class="verdict mid" title="Đúng loại cây nhưng sai bệnh">✕ Sai bệnh</span>'
      : v === "wrong" ? '<span class="verdict bad" title="Sai cả loại cây">✕ Sai loại cây</span>'
        : '<span class="verdict none" title="Chọn nhãn thật để chấm">chưa biết nhãn</span>';
}

function toneOf(verdicts) {
  const known = verdicts.filter(Boolean);
  if (!known.length) return "is-unknown";
  if (known.every(v => v === "correct")) return "is-correct";
  if (known.every(v => v !== "correct")) return "is-wrong";
  return "is-unknown";
}

function setBadge(tab, text, tone = "") {
  const badge = byId(`badge-${tab}`);
  if (!badge) return;
  badge.textContent = text;
  badge.className = `tab-badge${tone ? ` is-${tone}` : ""}`;
  toggleHidden(badge, !text);
}

// ── Local only ──
function localCardHtml(id, trueClass) {
  const result = state.rows.get(id);
  const client = `Client ${id.replace("local_", "").replace(/^0/, "")}`;
  if (!result) return `<article class="local-card"><header>${modelPill(id)}<span class="row-sub">${client}</span></header><span class="skeleton skeleton-line"></span></article>`;
  if (result.error) return `<article class="local-card"><header>${modelPill(id)}<span class="row-sub">${client}</span></header><p class="text-rose">Không có kết quả: ${esc(result.error)}</p></article>`;
  const v = verdict(result, trueClass);
  const classKey = trueClass || result.predicted_class;
  const info = shardInfo(id, classKey);
  const shard = info
    ? `<p class="local-shard" title="Số ảnh của nhóm ${trueClass ? "thật" : "được dự đoán"} «${esc(formatDiseaseName(classKey))}» mà client này đã học"><span class="shard-count ${info.count === 0 ? "is-zero" : ""} mono">${fmtInt(info.count)}</span> ảnh của nhóm này nằm trong dữ liệu client</p>`
    : "";
  const cam = state.explain && state.explanations.get(id)?.image
    ? `<img class="local-cam" src="${esc(state.explanations.get(id).image)}" alt="Vùng ${esc(MODEL_LABELS[id])} dựa vào" width="112" height="112" loading="lazy">` : "";
  const uncertain = result.is_uncertain ? ' <span class="pill-flag" title="Xác suất cao nhất dưới 50% hoặc hai nhóm đầu quá sát nhau">không chắc</span>' : "";
  return `<article class="local-card ${v === "correct" ? "is-correct" : v ? "is-wrong" : ""}">
      <header>${modelPill(id)}<span class="row-sub">${client}</span>${verdictHtml(v)}</header>
      <p class="local-pred"><b>${esc(formatDiseaseName(result.predicted_class))}</b>${uncertain}</p>
      ${confBarHtml(result.confidence_pct, "local")}
      ${result.calibrated ? `<span class="conf-cal">hiệu chuẩn <b class="mono">${fmtNum(result.calibrated.confidence_pct, 1)}%</b></span>` : ""}
      ${shard}${cam}
    </article>`;
}

/** Lý do các client sai/đúng, chỉ dựa trên số ảnh của nhóm đó trong dữ liệu của từng client (không nói về Centralized/FedAvg). */
function localExplanation(trueClass) {
  if (!state.shards || !trueClass) return "";
  const name = esc(formatDiseaseName(trueClass));
  const wrongZero = [];
  const wrongSome = [];
  const right = [];
  for (const id of LOCAL_SOURCES) {
    const r = state.rows.get(id);
    if (!r || r.error) continue;
    const info = shardInfo(id, trueClass);
    if (verdict(r, trueClass) === "correct") right.push({ id, info });
    else if (info && info.count === 0) wrongZero.push(id);
    else wrongSome.push({ id, info });
  }
  const lines = [];
  if (wrongZero.length) {
    const many = wrongZero.length > 1;
    lines.push(`<b>${wrongZero.map(id => MODEL_LABELS[id]).join(", ")}</b> sai: dữ liệu của ${many ? "chúng" : "nó"} có <b>0 ảnh</b> thuộc nhóm «${name}», ${many ? "chúng" : "nó"} chưa từng thấy nhóm này.`);
  }
  for (const { id, info } of right) {
    const share = info && info.total ? ` (${fmtNum((info.count / info.total) * 100, 0)}% tổng ảnh của nhóm này)` : "";
    lines.push(`<b>${MODEL_LABELS[id]}</b> đúng: dữ liệu của nó có <b>${fmtInt(info?.count)} ảnh</b> của nhóm này${share}.`);
  }
  for (const { id, info } of wrongSome) {
    const share = info && info.total ? ` (${fmtNum((info.count / info.total) * 100, 1)}% tổng ảnh của nhóm này)` : "";
    lines.push(`<b>${MODEL_LABELS[id]}</b> chỉ có ${fmtInt(info?.count)} ảnh của nhóm này${share} và vẫn sai.`);
  }
  if (!lines.length) return "";
  return `<div class="callout mt-3" role="note"><b>Vì sao các client khác nhau</b><ul class="local-why">${lines.map(l => `<li>${l}</li>`).join("")}</ul>
    <p class="field-hint mt-2">Thiếu dữ liệu có thể góp phần vào lỗi; đây chưa phải kiểm chứng nguyên nhân.</p></div>`;
}

function renderLocal() {
  const box = byId("local-view");
  const present = LOCAL_SOURCES.filter(id => state.rows.has(id));
  if (!present.length) {
    box.innerHTML = `<div class="local-grid">${LOCAL_SOURCES.map(id => localCardHtml(id, "")).join("")}</div>`;
    return;
  }
  const trueClass = targetClass();
  const ok = LOCAL_SOURCES.filter(id => { const r = state.rows.get(id); return r && !r.error && verdict(r, trueClass) === "correct"; });
  const results = LOCAL_SOURCES.map(id => state.rows.get(id)).filter(r => r && !r.error);
  const verdicts = results.map(r => verdict(r, trueClass));
  const tally = new Map();
  for (const r of results) tally.set(r.predicted_class, (tally.get(r.predicted_class) || 0) + 1);
  const [topClass, topCount] = [...tally.entries()].sort((a, b) => b[1] - a[1])[0] || ["", 0];
  const crops = new Map();
  for (const r of results) crops.set(cropOf(r.predicted_class), (crops.get(cropOf(r.predicted_class)) || 0) + 1);
  const [topCrop, cropCount] = [...crops.entries()].sort((a, b) => b[1] - a[1])[0] || ["", 0];
  const first = trueClass
    ? `<div class="vg-cell"><span class="vg-label">Nhận đúng nhóm thật</span><b class="vg-value">${ok.length}/${results.length} client</b><span class="vg-sub">${esc(formatDiseaseName(trueClass))}</span></div>`
    : `<div class="vg-cell"><span class="vg-label">Chấm đúng/sai</span><b class="vg-value">chưa chấm</b><span class="vg-sub">chọn nhãn thật ở thanh trên</span></div>`;
  byId("local-view").innerHTML = `
    <div class="diag-headline ${toneOf(verdicts)}">
      <div class="verdict-grid">
        ${first}
        <div class="vg-cell"><span class="vg-label">Loại cây nhiều client chọn</span><b class="vg-value">${esc(topCrop)}</b><span class="vg-sub">${cropCount}/${results.length} client</span></div>
        <div class="vg-cell"><span class="vg-label">Nhóm nhiều client chọn</span><b class="vg-value headline-name">${esc(diseaseOf(topClass))}</b><span class="vg-sub">${topCount}/${results.length} client</span></div>
      </div>
    </div>
    <div class="local-grid mt-3">${LOCAL_SOURCES.map(id => localCardHtml(id, trueClass)).join("")}</div>
    ${localExplanation(trueClass)}
    ${timingHtml()}`;
  applyBars(box);
}

// ── PlantVillage ──
function pvEvidenceHtml() {
  const ev = state.pvEvidence;
  if (!ev?.available) {
    return '<p class="field-hint">Chưa có báo cáo đánh giá lại của mô hình này trên máy chủ.</p>';
  }
  const by = ev.holdout?.by_source || {};
  const pd = by.plantdoc;
  const pv = by.plantvillage;
  const ci = pd?.top1_ci95;
  return `<dl class="kv-list">
      <div><dt>Nguồn</dt><dd>${esc(ev.model?.origin || "")} · ${esc(ev.model?.architecture || "")}</dd></div>
      <div><dt>Dữ liệu mô hình đã học</dt><dd>${esc(ev.model?.training_data || "")}</dd></div>
      <div><dt>PlantDoc, test giữ riêng (${fmtInt(pd?.count)} ảnh, mô hình chưa từng thấy)</dt><dd class="mono"><b>${fmtPct(pd?.top1_accuracy, 1)}</b>${ci && ci.low !== null ? ` · khoảng tin cậy 95%: ${fmtPct(ci.low, 1)} đến ${fmtPct(ci.high, 1)}` : ""}</dd></div>
      <div><dt>PlantVillage, test giữ riêng (${fmtInt(pv?.count)} ảnh)</dt><dd class="mono">${fmtPct(pv?.top1_accuracy, 1)} <span class="pill-flag" title="${esc(ev.leakage?.note || "")}">có thể trùng dữ liệu đã học</span></dd></div>
      <div><dt>Macro-F1 (PlantVillage / PlantDoc)</dt><dd class="mono">${fmtNum(pv?.macro_f1_supported, 3)} / ${fmtNum(pd?.macro_f1_supported, 3)}</dd></div>
    </dl>
    <p class="field-hint mt-2">${esc(ev.leakage?.note || "")}</p>`;
}

function topListHtml(top3, pillar, calibrated) {
  return `<ol class="top3-bars mt-2">${(top3 || []).map((t, i) => {
    const cal = calibrated?.top3?.[i];
    return `<li><span>${esc(formatDiseaseName(t.class))}</span>${confBarHtml(t.confidence, pillar)}${cal ? `<span class="conf-cal">hiệu chuẩn <b class="mono">${fmtNum(cal.confidence, 1)}%</b></span>` : ""}</li>`;
  }).join("")}</ol>`;
}

function headlineBlock(title, data, trueClass, extra = "") {
  const v = verdict(data, trueClass);
  const calibrated = data.calibrated ? `sau hiệu chuẩn ${fmtNum(data.calibrated.confidence_pct, 1)}%` : data.crop_restricted ? "chuẩn hóa trong nhóm cây" : "chưa hiệu chuẩn";
  const uncertain = data.is_uncertain ? ' <span class="pill-flag" title="Xác suất cao nhất dưới 50% hoặc hai nhóm đầu quá sát nhau">không chắc</span>' : "";
  const truth = trueClass && v !== "correct" ? `<p class="headline-truth">Nhãn thật: <b>${esc(formatDiseaseName(trueClass))}</b></p>` : "";
  return `<div class="diag-headline ${v === "correct" ? "is-correct" : v ? "is-wrong" : "is-unknown"}">
      <p class="summary-line">${title}</p>
      <div class="verdict-grid">
        <div class="vg-cell"><span class="vg-label">Loại cây</span><b class="vg-value">${esc(cropOf(data.predicted_class))}</b></div>
        <div class="vg-cell"><span class="vg-label">Nhóm</span><b class="vg-value headline-name">${esc(diseaseOf(data.predicted_class))}${uncertain}</b></div>
        <div class="vg-cell"><span class="vg-label">Xác suất dự đoán</span><b class="vg-value mono">${fmtNum(data.confidence_pct, 1)}%</b><span class="vg-sub">${calibrated}</span></div>
      </div>
      <p class="headline-meta">${verdictHtml(v)}</p>${truth}${extra}
    </div>`;
}

function renderPv() {
  const box = byId("pv-view");
  const st = state.pv;
  if (!st) { box.innerHTML = '<p class="field-hint">Chọn ảnh lá để nhận diện.</p>'; setBadge("pv", ""); return; }
  if (st.loading) { box.innerHTML = '<span class="skeleton skeleton-block"></span>'; setBadge("pv", "…"); return; }
  if (st.error) {
    box.innerHTML = `<div class="callout callout-error" role="alert"><b>Chưa nhận diện được bằng PlantVillage.</b> ${esc(st.error)}</div>${pvEvidenceDetails()}`;
    setBadge("pv", "lỗi", "bad");
    return;
  }
  const d = st.data;
  const trueClass = targetClass();
  const v = verdict(d, trueClass);
  box.innerHTML = `${headlineBlock("Kết quả chính, PlantVillage (MobileNetV2 · Hugging Face)", d, trueClass)}
    <h4 class="subhead mt-3">Ba nhóm có xác suất cao nhất</h4>${topListHtml(d.top3, "centralized", d.calibrated)}
    ${pvEvidenceDetails()}`;
  applyBars(box);
  setBadge("pv", v === "correct" ? "✓" : v ? "✕" : "xong", v === "correct" ? "ok" : v ? "bad" : "");
}

function pvEvidenceDetails() {
  return `<details class="fold mt-3"><summary>Mô hình này được đánh giá thế nào</summary><div class="fold-body">${pvEvidenceHtml()}</div></details>`;
}

// ── Cộng đồng ──
function renderCommunity() {
  const box = byId("comm-view");
  const st = state.community;
  if (!st) { box.innerHTML = '<p class="field-hint">Chọn ảnh lá để nhận diện.</p>'; setBadge("community", ""); return; }
  if (st.loading) { box.innerHTML = '<span class="skeleton skeleton-block"></span>'; setBadge("community", "…"); return; }
  if (st.error) {
    box.innerHTML = `<div class="callout callout-error" role="alert"><b>Chưa nhận diện được bằng mô hình cộng đồng.</b> ${esc(st.error)}</div>`;
    setBadge("community", "lỗi", "bad");
    return;
  }
  const d = st.data;
  const trueClass = targetClass();
  const v = verdict(d, trueClass);
  const baseVerdict = verdict({ predicted_class: d.baseline.predicted_class }, trueClass);
  const note = d.version === 0
    ? "Chưa có vòng cộng đồng nào được nhận nên phiên bản hiện hành trùng FedAvg v7."
    : `Phiên bản ${esc(d.version)} · độ chính xác trên tập giám sát ${fmtPct(d.metrics?.accuracy, 1)} (mô hình gốc ${fmtPct(d.base_metrics?.accuracy, 1)}).`;
  const compare = `<p class="field-hint mt-2">FedAvg v7 gốc trên cùng ảnh: <b>${esc(formatDiseaseName(d.baseline.predicted_class))}</b> (${fmtNum(d.baseline.confidence_pct, 1)}%) ${baseVerdict ? verdictHtml(baseVerdict) : ""} · ${d.same_as_baseline ? "cùng kết luận" : "khác kết luận"}.</p>`;
  box.innerHTML = `${headlineBlock(`Kết quả chính, mô hình cộng đồng · phiên bản ${esc(d.version)}`, d, trueClass, compare)}
    <h4 class="subhead mt-3">Ba nhóm có xác suất cao nhất</h4>${topListHtml(d.top3, "fedavg", null)}
    <p class="field-hint mt-3">${note}</p>
    <p class="field-hint">${esc(d.caveat || "")} Trọng số, độ trễ và độ chính xác của từng vòng: <a href="#hoc-lien-ket">trang Học phân tán</a>.</p>`;
  applyBars(box);
  setBadge("community", v === "correct" ? "✓" : v ? "✕" : "xong", v === "correct" ? "ok" : v ? "bad" : "");
}

/** Hai cách còn lại chạy song song với bảy mô hình; mỗi cách có trạng thái riêng nên một cách lỗi không kéo theo cách khác. */
async function runPv(runId, file) {
  state.pv = { loading: true };
  renderPv();
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await postForm("/api/inference/plantvillage", form, { timeoutMs: 90000 });
    if (runId !== state.runId) return;
    state.pv = { data };
  } catch (error) {
    if (isAbort(error) || runId !== state.runId) return;
    state.pv = { error: error.message };
  }
  renderPv();
}

async function runCommunity(runId, file) {
  state.community = { loading: true };
  renderCommunity();
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await postForm("/api/community/predict", form, { timeoutMs: 90000 });
    if (runId !== state.runId) return;
    state.community = { data };
  } catch (error) {
    if (isAbort(error) || runId !== state.runId) return;
    state.community = { error: error.status === 503 ? "Mô hình cộng đồng chưa được bật trên máy chủ này." : error.message };
  }
  renderCommunity();
}

function renderTabBadges() {
  const trueClass = targetClass();
  if (state.busy && !LOCAL_SOURCES.some(id => state.rows.has(id))) { setBadge("local", "…"); return; }
  const results = LOCAL_SOURCES.map(id => state.rows.get(id)).filter(r => r && !r.error);
  if (!results.length) { setBadge("local", ""); return; }
  if (!trueClass) { setBadge("local", "xong"); return; }
  const ok = results.filter(r => verdict(r, trueClass) === "correct").length;
  setBadge("local", `${ok}/${results.length}`, ok === results.length ? "ok" : ok === 0 ? "bad" : "mid");
}

/** Ảnh đang chọn chỉ nằm ở thanh trên cùng, dùng chung cho cả ba tab. */
function activateDiagTab(name, { focus = false } = {}) {
  byId("diag-output").dataset.activeTab = name;
  for (const id of DIAG_TABS) {
    const tab = byId(`dtab-${id}`);
    const on = id === name;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    toggleHidden(byId(`dpanel-${id}`), !on);
    if (on && focus) tab.focus();
  }
}

function bindDiagTabs() {
  DIAG_TABS.forEach((id, index) => {
    const tab = byId(`dtab-${id}`);
    tab.addEventListener("click", () => activateDiagTab(id));
    tab.addEventListener("keydown", event => {
      const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        activateDiagTab(DIAG_TABS[event.key === "Home" ? 0 : DIAG_TABS.length - 1], { focus: true });
      } else if (step) {
        event.preventDefault();
        activateDiagTab(DIAG_TABS[(index + step + DIAG_TABS.length) % DIAG_TABS.length], { focus: true });
      }
    });
  });
  const dialog = byId("diag-all-dialog");
  byId("btn-all-models").addEventListener("click", () => { if (typeof dialog.showModal === "function") dialog.showModal(); else dialog.setAttribute("open", ""); });
  byId("btn-all-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); }); // bấm ra ngoài khung thì đóng
}

// ───────── Khởi tạo ─────────
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

/** Dải "Mô hình nhận được gì": số nhóm/loại cây từ danh sách nhóm; độ chính xác từ số liệu đã kiểm hash. */
function renderBrief(evidence, summary) {
  const classes = state.classes;
  if (classes.length) {
    const crops = new Map();
    for (const cls of classes) {
      const crop = cropOf(cls);
      if (!crops.has(crop)) crops.set(crop, []);
      crops.get(crop).push(cls);
    }
    const healthy = classes.filter(cls => /___healthy$/.test(cls)).length;
    setText(byId("brief-classes-title"), `${classes.length} nhóm, ${crops.size} loại cây`);
    setText(byId("brief-classes-text"), `${classes.length - healthy} nhóm bệnh, ${healthy} nhóm lá khỏe. Cây hoặc bệnh ngoài danh mục vẫn bị xếp vào nhóm gần nhất, nên kết quả không còn ý nghĩa.`);
    setHtml(byId("brief-list-body"), [...crops.entries()].map(([crop, list]) =>
      `<p><b>${esc(crop)}</b>: ${list.map(cls => esc(diseaseOf(cls))).join("; ")}</p>`).join(""));
  }
  const fed = evidence?.available ? (evidence.models || []).find(m => m.model_id === "fedavg")?.metrics : null;
  if (fed) {
    setText(byId("brief-crop-title"), `Nhận đúng cây: ${fmtPct(fed.plantvillage.crop_accuracy, 1)} (trong phòng), ${fmtPct(fed.plantdoc.crop_accuracy, 1)} (ngoài đồng)`);
    setText(byId("brief-crop-text"), `Đúng nhóm bệnh: ${fmtPct(fed.plantvillage.top1_accuracy, 1)} (trong phòng), ${fmtPct(fed.plantdoc.top1_accuracy, 1)} (ngoài đồng).`);
    const fullFrame = summary?.fullframe?.available ? summary.fullframe.splits?.val?.full_frame?.fedavg : null;
    setText(byId("brief-own-title"), "Ảnh ngoài bộ dữ liệu");
    setText(byId("brief-own-text"), `Nên dùng ảnh một lá chụp rõ${fullFrame !== null && fullFrame !== undefined ? ` (ảnh nhiều lá giảm còn khoảng ${fmtPct(fullFrame, 1)})` : ""}. Kết quả mang tính tham khảo.`);
  }
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
  state.pvEvidence = summary?.plantvillage_reference || null;
  state.showStats = evidence?.available === true && gallery.validation_membership?.status === "verified";
  renderBrief(evidence, summary);
  openOnWide(byId("diag-model-details")); // màn rộng: bảng dự đoán từng mô hình mở sẵn (in ấn vẫn khôi phục đúng nhờ beforeprint)
  if (!state.showStats) {
    // không có số liệu để xếp mức nhận đúng: ẩn các bộ lọc theo độ khó
    $$("#gallery-filters [data-filter='easy'], #gallery-filters [data-filter='medium'], #gallery-filters [data-filter='hard']")
      .forEach(chip => chip.classList.add("hidden"));
    const note = byId("gallery-note");
    if (note) note.textContent = "Ảnh mẫu có nhãn thật; chưa đối chiếu được với tập kiểm tra nên không xếp độ khó.";
  }

  const cropSelect = byId("inference-crop");
  for (const crop of [...new Set(state.classes.map(cls => cls.split("___")[0]))].sort()) {
    const option = document.createElement("option");
    option.value = crop;
    option.textContent = crop.replace(/_/g, " ");
    cropSelect.appendChild(option);
  }
  // Đổi cây chỉ có hiệu lực ở lần chẩn đoán SAU (đọc giá trị lúc bấm); không gọi lại mô hình ngầm.

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

  renderGallery();

  byId("gallery-grid").addEventListener("click", event => {
    const button = event.target.closest(".gallery-item");
    if (!button) return;
    const item = state.gallery.find(i => String(i.id) === button.dataset.id);
    if (item) selectGallery(item);
  });
  byId("gallery-more")?.addEventListener("click", () => {
    state.galleryAll = !state.galleryAll;
    renderGallery();
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
  // Phải cancel dragover thì trình duyệt mới giao sự kiện drop cho zone; nếu không, kéo ảnh vào
  // ngoài vùng sẽ làm trình duyệt điều hướng tới file:/// và dỡ cả ứng dụng đang chạy.
  const cancelDrag = event => event.preventDefault();
  ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, event => {
    cancelDrag(event);
    zone.classList.add("is-dragover");
  }));
  ["dragleave", "dragend"].forEach(name => zone.addEventListener(name, () => zone.classList.remove("is-dragover")));
  zone.addEventListener("drop", event => {
    cancelDrag(event);
    zone.classList.remove("is-dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file) selectUpload(file);
  });

  byId("explain-toggle").addEventListener("change", event => {
    state.explain = event.target.checked;
    rerender();
  });
  byId("btn-print").addEventListener("click", () => window.print());
  let restoreModelDetails = false;
  window.addEventListener("beforeprint", () => {
    const details = byId("diag-model-details");
    restoreModelDetails = !details.open;
    details.open = true;
  });
  window.addEventListener("afterprint", () => {
    if (restoreModelDetails) byId("diag-model-details").open = false;
    restoreModelDetails = false;
  });
  $$("[data-stage-view]").forEach(button => button.addEventListener("click", () => setStageView(button.dataset.stageView)));
  byId("btn-diagnose").addEventListener("click", run);
  bindDiagTabs();

  // bấm dòng kết quả (hoặc Enter/Space trên nút trong ô tên mô hình) để mở top-3
  const body = byId("diag-body");
  const toggleRow = row => {
    const detail = body.querySelector(`tr.detail-row[data-model="${row.dataset.model}"]`);
    if (detail && detail.innerHTML.trim()) {
      const open = detail.classList.toggle("hidden") === false;
      row.querySelector(".row-toggle")?.setAttribute("aria-expanded", String(open));
    }
  };
  body.addEventListener("click", event => {
    if (event.target.closest("[data-toggle-locals]")) { toggleLocals(); return; }
    const row = event.target.closest("tr.diag-row");
    if (row) toggleRow(row);
  });
}
