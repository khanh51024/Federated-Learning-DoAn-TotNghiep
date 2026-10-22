// Chương 2 · Dữ liệu: bộ ảnh, cách chia tập, rồi phân bố non-IID (heatmap 38 nhóm × 5 client) và vì sao Local-only không nhận ra bệnh chưa từng thấy.
import { releaseHtml } from "../core/report-training.js";
import { getEvidence, getPartitions, getSummary, modelsById } from "../core/store.js";
import { attachCrosshair } from "../core/crosshair.js";
import {
  SOURCE_GROUPS, byId, cropOf, diseaseOf, esc, fmtInt, fmtNum, fmtPct, formatDiseaseName, revealItems, sectionHead,
} from "../core/dom.js";
import { alphaText, factsOf } from "../core/facts-text.js";

const LOCALS = ["local_00", "local_01", "local_02", "local_03", "local_04"];

function clientNumber(clientId) {
  // chỉ giữ ký tự an toàn: giá trị này được ghép vào HTML, thuộc tính title và khóa tra cứu
  return String(clientId).replace("client_", "").replace(/[^0-9A-Za-z_-]/g, "");
}

/** Tương quan giữa "client có 0 ảnh lớp này" và recall của Local-only tương ứng (tính từ hai báo cáo đã kiểm). */
function recallByPresence(shards, pvRecall) {
  const absent = [];
  const present = [];
  for (const row of shards.classes) {
    for (const [client, count] of Object.entries(row.per_client)) {
      const recall = pvRecall?.[`local_${clientNumber(client)}`]?.[row.class];
      if (recall === undefined || recall === null) continue;
      (count === 0 ? absent : present).push(recall);
    }
  }
  const mean = list => (list.length ? list.reduce((a, b) => a + b, 0) / list.length : null);
  return { absentMean: mean(absent), absentN: absent.length, presentMean: mean(present), presentN: present.length };
}

function pickExample(shards) {
  const concentrated = shards.classes.filter(r => (r.top_share_pct || 0) >= 90);
  if (!concentrated.length) return null;
  const preferred = concentrated.find(r => r.class === "Tomato___Bacterial_spot");
  return preferred || concentrated.reduce((a, b) => (b.total > a.total ? b : a));
}

/** Tập đánh giá cho phần công bằng: ưu tiên tập giữ riêng (không dùng chọn mô hình), không có thì tập kiểm tra. */
function fairnessSplit(fairness) {
  if (!fairness?.available) return null;
  for (const name of ["test", "val"]) {
    if (fairness.splits?.[name]?.available) return { name, label: name === "test" ? "tập giữ riêng" : "tập kiểm tra", block: fairness.splits[name] };
  }
  return null;
}

/** Nhận xét tổng hợp từ số đo (khoảng min–max giữa các client), không viết sẵn con số. */
function fairnessText(split) {
  const rows = Object.values(split.block.by_client);
  const range = pick => {
    const values = rows.map(pick).filter(v => v !== null && v !== undefined);
    return values.length ? `${fmtPct(Math.min(...values), 1)}–${fmtPct(Math.max(...values), 1)}` : "—";
  };
  const fedAll = rows.flatMap(r => [r.classes_with_images.fedavg, r.classes_without_images.fedavg]).filter(v => v !== null && v !== undefined);
  const gap = fedAll.length ? (Math.max(...fedAll) - Math.min(...fedAll)) * 100 : null;
  return `Trên ${split.label}, ở các nhóm mà client <b>không có ảnh học</b>, mô hình Local-only của client đó đúng ${range(r => r.classes_without_images.local)}`
    + ` còn FedAvg đúng ${range(r => r.classes_without_images.fedavg)}; ở các nhóm client <b>có ảnh</b>, Local-only đúng ${range(r => r.classes_with_images.local)}`
    + ` và FedAvg ${range(r => r.classes_with_images.fedavg)}.${gap !== null ? ` Giữa các client, độ chính xác của FedAvg chênh nhau tối đa ${fmtNum(gap, 1)} điểm phần trăm.` : ""}`
    + " Đây là số đo theo việc client có ảnh của nhóm hay không, chưa phải kiểm chứng nguyên nhân.";
}

function clientCards(evidence, shards, byModel, fairness) {
  const counts = evidence.shard_counts || {};
  const split = fairnessSplit(fairness);
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return Object.keys(counts).sort().map((client, i) => {
    const model = byModel[LOCALS[i]];
    const shard = shards?.clients?.find(c => c.client === client);
    return `
      <article class="client-tile">
        <header><b>Client ${clientNumber(client)}</b><span class="model-pill local">Local ${clientNumber(client)}</span></header>
        <p class="client-tile-n mono">${fmtInt(counts[client])} <span>ảnh để học · ${fmtNum((counts[client] / total) * 100, 1)}%</span></p>
        <p class="client-tile-sub">${shard ? `${shard.classes_with_images}/${shards.classes_total} nhóm có ảnh` : "Số nhóm có ảnh: chưa có báo cáo"}</p>
        <dl class="client-tile-metrics">
          <div><dt>Học riêng · ảnh trong phòng</dt><dd class="mono">${fmtPct(model.metrics.plantvillage.top1_accuracy, 1)}</dd></div>
          <div><dt>Học riêng · ngoài đồng</dt><dd class="mono">${fmtPct(model.metrics.plantdoc.top1_accuracy, 1)}</dd></div>
          ${split ? `<div><dt>Nhóm client có ảnh · Local / FedAvg</dt><dd class="mono">${fmtPct(split.block.by_client[client]?.classes_with_images.local, 1)} / ${fmtPct(split.block.by_client[client]?.classes_with_images.fedavg, 1)}</dd></div>
          <div><dt>Nhóm client thiếu ảnh · Local / FedAvg</dt><dd class="mono">${fmtPct(split.block.by_client[client]?.classes_without_images.local, 1)} / ${fmtPct(split.block.by_client[client]?.classes_without_images.fedavg, 1)}</dd></div>` : ""}
        </dl>
      </article>`;
  }).join("");
}

function heatmapTable(shards, pvRecall, view) {
  const clients = shards.clients.map(c => c.client);
  const showRecall = view === "recall";
  const head = showRecall
    ? `${clients.map(c => `<th scope="col">Local ${clientNumber(c)}</th>`).join("")}<th scope="col" class="col-global">FedAvg</th><th scope="col" class="col-global">Centralized</th>`
    : `${clients.map(c => `<th scope="col">Client ${clientNumber(c)}</th>`).join("")}<th scope="col">Tổng</th><th scope="col">Ghi chú</th>`;
  const columns = 1 + clients.length + 2;

  let lastCrop = null;
  const rows = [];
  for (const row of shards.classes) {
    const crop = cropOf(row.class);
    if (crop !== lastCrop) {
      rows.push(`<tr class="crop-row"><th colspan="${columns}" scope="colgroup">${esc(crop)}</th></tr>`);
      lastCrop = crop;
    }
    let cells;
    if (showRecall) {
      const recallCells = clients.map(c => {
        const r = pvRecall?.[`local_${clientNumber(c)}`]?.[row.class];
        const count = row.per_client[c];
        return r === undefined || r === null
          ? `<td class="heat-cell is-empty" title="Không có ảnh kiểm tra của nhóm này">—</td>`
          : `<td class="heat-cell heat-recall${r < 0.5 ? " is-low" : ""}" data-style="--p:${r}" title="Local ${clientNumber(c)}: nhận đúng ${fmtPct(r, 1)} · client này có ${fmtInt(count)} ảnh của nhóm này">${fmtNum(r * 100, 0)}</td>`;
      }).join("");
      const globals = ["fedavg", "centralized"].map(id => {
        const r = pvRecall?.[id]?.[row.class];
        return r === undefined || r === null
          ? `<td class="heat-cell is-empty">—</td>`
          : `<td class="heat-cell heat-recall${r < 0.5 ? " is-low" : ""} col-global" data-style="--p:${r}" title="${id === "fedavg" ? "FedAvg" : "Centralized"}: nhận đúng ${fmtPct(r, 1)}">${fmtNum(r * 100, 0)}</td>`;
      }).join("");
      cells = recallCells + globals;
    } else {
      const countCells = clients.map(c => {
        const n = row.per_client[c];
        const share = row.total ? n / row.total : 0;
        const cls = n === 0 ? "is-zero" : share >= 0.9 ? "is-dominant" : "";
        return `<td class="heat-cell heat-count ${cls}" data-style="--p:${share.toFixed(3)}" title="Client ${clientNumber(c)} có ${fmtInt(n)}/${fmtInt(row.total)} ảnh nhóm ${esc(formatDiseaseName(row.class))} (${fmtNum(share * 100, 1)}%)">${fmtInt(n)}</td>`;
      }).join("");
      const concentration = row.top_share_pct !== null && row.top_share_pct >= 90
        ? `<span class="pill-flag" title="Hơn 90% ảnh của nhóm này nằm ở một client">${fmtNum(row.top_share_pct, 0)}% ở client ${clientNumber(row.top_client)}</span>` : "";
      cells = `${countCells}<td class="heat-total mono">${fmtInt(row.total)}</td><td class="heat-note">${concentration}</td>`;
    }
    rows.push(`<tr><th scope="row" class="heat-class">${esc(diseaseOf(row.class))}</th>${cells}</tr>`);
  }
  return `<table class="heat-table"><thead><tr><th scope="col">Nhóm</th>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

// ───────── Trình xem các cách chia của release (chỉ dữ liệu) ─────────
const FAMILY_ORDER = ["iid", "label", "quantity", "feature", "mixed"];

function schemeLabel(name, scheme) {
  if (scheme.family === "iid") return "IID — chia đều";
  if (scheme.alpha === null || scheme.alpha === undefined) return scheme.family_label;
  return `${scheme.family_label.replace(/\s*\(.*\)$/, "")} · α = ${fmtNum(scheme.alpha, scheme.alpha < 1 ? 1 : 0)}`;
}

/** Đổi một cách chia sang đúng dạng `shards` mà heatmapTable dùng. */
function schemeAsShards(partitions, scheme) {
  const clients = Object.keys(scheme.clients).sort();
  const classes = partitions.classes.map((cls, i) => {
    const perClient = Object.fromEntries(clients.map(c => [c, scheme.clients[c].per_class[i]]));
    const total = Object.values(perClient).reduce((a, b) => a + b, 0);
    const top = clients.reduce((a, b) => (perClient[b] > perClient[a] ? b : a), clients[0]);
    return { class: cls, per_client: perClient, total, top_client: top, top_share_pct: total ? (perClient[top] / total) * 100 : null };
  });
  return { clients: clients.map(client => ({ client })), classes };
}

function schemeSummary(scheme) {
  const clients = Object.keys(scheme.clients).sort();
  const total = clients.reduce((a, c) => a + scheme.clients[c].n_k, 0) || 1;
  return clients.map(c => {
    const d = scheme.clients[c];
    return `<li class="scheme-client">
        <b>Client ${clientNumber(c)}</b>
        <span class="scheme-bar"><span data-style="transform:scaleX(${(d.n_k / total).toFixed(3)})"></span></span>
        <span class="mono">${fmtInt(d.n_k)} ảnh</span>
        <span>${fmtInt(d.classes_with_images)}/38 nhóm</span>
        <span>PlantDoc ${fmtPct(d.plantdoc_share, 1)}</span>
      </li>`;
  }).join("");
}

async function mountExplorer() {
  const host = byId("scheme-explorer-body");
  let partitions;
  try {
    partitions = await getPartitions();
  } catch (error) {
    host.innerHTML = `<p class="field-hint">Không tải được dữ liệu các cách chia: ${esc(error.message)}</p>`;
    return;
  }
  if (!partitions.available) {
    host.innerHTML = `<p class="field-hint">${esc(partitions.reason || "Chưa có báo cáo các cách chia.")}</p>`;
    return;
  }
  const names = Object.keys(partitions.schemes).sort((a, b) => {
    const sa = partitions.schemes[a];
    const sb = partitions.schemes[b];
    const fa = FAMILY_ORDER.indexOf(sa.family);
    const fb = FAMILY_ORDER.indexOf(sb.family);
    return fa !== fb ? fa - fb : (sb.alpha ?? 0) - (sa.alpha ?? 0); // trong cùng họ: α lớn (gần IID) trước
  });
  const groups = FAMILY_ORDER.map(family => {
    const items = names.filter(n => partitions.schemes[n].family === family);
    if (!items.length) return "";
    const label = partitions.schemes[items[0]].family_label;
    return `<optgroup label="${esc(label)}">${items.map(n => `<option value="${esc(n)}"${n === partitions.trained_scheme ? " selected" : ""}>${esc(schemeLabel(n, partitions.schemes[n]))}${partitions.schemes[n].has_trained_models ? " (đã huấn luyện)" : ""}</option>`).join("")}</optgroup>`;
  }).join("");
  host.innerHTML = `
    <div class="scheme-controls">
      <label for="scheme-select" class="form-label">Cách chia</label>
      <select id="scheme-select" class="input-field">${groups}</select>
      <span id="scheme-badge"></span>
    </div>
    <p class="field-hint mt-2" id="scheme-desc"></p>
    <ul class="scheme-clients mt-3" id="scheme-clients"></ul>
    <div tabindex="0" role="region" aria-label="Phân bố nhóm ảnh theo client của cách chia đang chọn, cuộn được bằng phím mũi tên" class="table-responsive heat-wrap mt-3" id="scheme-wrap"></div>`;
  const select = byId("scheme-select");
  attachCrosshair(byId("scheme-wrap"));
  const render = () => {
    const scheme = partitions.schemes[select.value];
    if (!scheme) return;
    byId("scheme-badge").innerHTML = scheme.has_trained_models
      ? '<span class="status-badge success">Có 7 mô hình đã huấn luyện</span>'
      : '<span class="status-badge is-neutral">Chỉ có dữ liệu, chưa huấn luyện</span>';
    byId("scheme-desc").textContent = `Độ lệch phân bố nhóm trung bình giữa client và toàn bộ (Jensen–Shannon): ${scheme.mean_js_divergence === null ? "0 (chia đều)" : fmtNum(scheme.mean_js_divergence, 3)} — càng lớn càng non-IID. Tổng ${fmtInt(partitions.total_train)} ảnh huấn luyện.`;
    byId("scheme-clients").innerHTML = schemeSummary(scheme);
    byId("scheme-wrap").innerHTML = heatmapTable(schemeAsShards(partitions, scheme), null, "count");
  };
  select.addEventListener("change", render);
  render();
}

/** Bộ dữ liệu và ba tập ảnh: nơi DUY NHẤT trên web định nghĩa tập kiểm tra và tập giữ riêng (các trang khác chỉ nhắc tên). */
function datasetHtml(evidence, summary) {
  const facts = factsOf(summary);
  const counts = evidence.shard_counts || {};
  const trainTotal = Object.values(counts).reduce((a, b) => a + b, 0);
  const val = evidence.validation_counts || {};
  const classCount = evidence.classes?.length || summary?.shards?.classes_total;
  const ho = summary?.holdout?.available ? summary.holdout : null;
  const groups = SOURCE_GROUPS.filter(g => g.id in val);
  const trainSplit = (summary?.training?.release?.splits || []).find(s => s.name === "train");
  const trainNames = groups.filter(g => !trainSplit?.by_source || g.id in trainSplit.by_source).map(g => g.label).join(" + ");
  const numberWord = ["", "Một", "Hai", "Ba", "Bốn"][groups.length] || String(groups.length);
  const otherCard = groups.some(g => g.id === "field_other") ? `
        <article class="source-card">
          <h4>Ngoài đồng, bộ khác</h4>
          <p>Ảnh lá ngoài đồng từ các bộ dữ liệu công khai khác PlantDoc (nguồn và giấy phép ghi trong manifest của release). Nhãn do tác giả bộ dữ liệu gán hoặc được sàng lọc bằng mắt, không phải chuyên gia. Ảnh được chia theo nhóm ảnh nên một nhóm không nằm ở cả tập học và tập kiểm tra.</p>
        </article>` : "";
  return `
    <section class="panel report-section" aria-labelledby="dataset-title">
      ${sectionHead("2.1", "Bộ dữ liệu", { id: "dataset-title", lead: `${numberWord} nguồn ảnh, ${classCount ? `${fmtInt(classCount)} nhóm lá khỏe/bệnh` : "nhiều nhóm lá khỏe/bệnh"}` })}
      <div class="source-grid mt-3">
        <article class="source-card">
          <h4>PlantVillage</h4>
          <p>Ảnh một chiếc lá chụp trong phòng, nền đơn giản (bản màu, 54.305 ảnh gốc, 38 nhóm). Là phần lớn dữ liệu học và kiểm tra.</p>
        </article>
        <article class="source-card">
          <h4>PlantDoc</h4>
          <p>Ảnh lá ngoài đồng, nền phức tạp, ít ảnh hơn và chỉ phủ một phần các nhóm. Bản phát hiện đối tượng có hộp bao lá; mỗi hộp được cắt thành một ảnh lá (nới 8% mỗi phía), nên một ảnh gốc có thể cho nhiều ảnh. Dùng để xem mô hình có giữ được độ chính xác khi ảnh khác kiểu.</p>
        </article>${otherCard}
      </div>
      <h4 class="subhead mt-4">Ba tập ảnh và mục đích của từng tập</h4>
      <dl class="term-list mt-3">
        <div><dt>Tập học</dt><dd><b>${fmtInt(trainTotal)} ảnh</b> (${esc(trainNames)}), chia cho ${Object.keys(counts).length} client. Mô hình chỉ cập nhật trọng số trên tập này.</dd></div>
        <div><dt>Tập kiểm tra</dt><dd>${groups.map(g => `<b>${fmtInt(val[g.id])}</b> ảnh ${esc(g.label)}`).join(" + ")}. Dùng trong lúc huấn luyện để chọn phiên bản mô hình tốt nhất, nên số đo trên tập này có thể hơi đẹp hơn thực tế.</dd></div>
        <div><dt>Tập giữ riêng</dt><dd>${ho ? `<b>${fmtInt(ho.count)} ảnh</b> (${groups.map(g => `${fmtInt(ho.counts_by_source?.[g.id])} ${esc(g.label)}`).join("; ")}). ` : ""}Là bộ chẩn đoán có sẵn của bộ dữ liệu (legacy), không dùng để chọn mô hình${facts?.holdout_audited === false ? ", nhưng chưa nằm trong kiểm tra trùng ảnh" : ""}. Dùng làm phép kiểm tra bổ sung ở chương 4.</dd></div>
      </dl>
      ${releaseHtml(summary?.training)}
    </section>`;
}

export async function mount() {
  const root = byId("noniid-root");
  const [evidence, summary] = await Promise.all([getEvidence(), getSummary().catch(() => null)]);
  if (!evidence.available) {
    root.innerHTML = `<div class="callout callout-error" role="alert"><b>Chưa có số liệu để hiển thị.</b> ${esc(evidence.reason || "")}</div>`;
    return;
  }
  const shards = summary?.shards?.available ? summary.shards : null;
  const re = summary?.reeval?.available ? summary.reeval : null;
  const byModel = modelsById(evidence);
  const example = shards ? pickExample(shards) : null;
  const presence = shards && re ? recallByPresence(shards, re.pv_recall) : null;

  let exampleHtml = "";
  if (example) {
    const holder = clientNumber(example.top_client);
    const holderCount = example.per_client[example.top_client];
    const others = Object.entries(example.per_client).filter(([c]) => c !== example.top_client);
    const othersTotal = others.reduce((sum, [, n]) => sum + n, 0);
    const zeros = others.filter(([, n]) => n === 0).map(([c]) => clientNumber(c));
    const zeroText = zeros.length ? `, trong đó client ${zeros.join(", ")} có <b>0 ảnh</b>` : "";
    const recallOf = id => {
      const r = re?.pv_recall?.[id]?.[example.class];
      return r === undefined || r === null ? "—" : fmtPct(r, 1);
    };
    // Mô hình | ảnh nhóm này đã học | tỉ lệ nhận đúng: Local k học đúng phần của client k; FedAvg và Centralized học từ cả năm phần
    const recallRows = re
      ? [
          ...Object.entries(example.per_client).map(([client, count]) => [`Local ${clientNumber(client)}`, count, recallOf(`local_${clientNumber(client)}`), client === example.top_client]),
          ["FedAvg", example.total, recallOf("fedavg"), false],
          ["Centralized", example.total, recallOf("centralized"), false],
        ]
      : [];
    const recallTable = recallRows.length ? `
        <div tabindex="0" role="group" aria-label="Tỉ lệ nhận đúng nhóm này theo mô hình, cuộn được bằng phím mũi tên" class="table-responsive example-table mt-3">
          <table class="data-table">
            <thead><tr><th scope="col">Mô hình</th><th scope="col">Ảnh nhóm này đã học</th><th scope="col">Tỉ lệ nhận đúng</th></tr></thead>
            <tbody>${recallRows.map(([name, count, recall, holds]) => `<tr${holds ? ' class="is-holder"' : ""}><th scope="row">${esc(name)}</th><td class="mono">${fmtInt(count)}</td><td class="mono">${recall}</td></tr>`).join("")}</tbody>
          </table>
        </div>` : "";
    exampleHtml = `
      <section class="viewer-example mt-4" aria-label="Ví dụ dữ liệu lệch giữa các client">
        <div><p class="eyebrow">Ví dụ</p><h4 class="panel-title">${esc(formatDiseaseName(example.class))}</h4>
        <p class="panel-desc">Client ${holder} giữ <b>${fmtNum(example.top_share_pct, 0)}%</b> ảnh của nhóm này. Các client còn lại có ít hoặc không có ảnh của nhóm này.</p></div>
        <ul class="viewer-bars" aria-label="Số ảnh của nhóm này tại từng client">
          ${Object.entries(example.per_client).map(([client, count]) => `<li><span>Client ${clientNumber(client)}</span><span class="viewer-bar-track"><span data-style="--share:${example.total ? count / example.total : 0}"></span></span><b>${fmtInt(count)} ảnh</b></li>`).join("")}
        </ul>
        <p class="panel-desc">Client ${holder}: ${fmtInt(holderCount)} ảnh; các client khác: ${fmtInt(othersTotal)} ảnh${zeroText}.${recallTable ? "" : " Chưa có kết quả theo nhóm."}</p>
        ${recallTable}
        ${recallTable ? '<p class="field-hint mt-2">Tỉ lệ nhận đúng trên ảnh kiểm tra PlantVillage của nhóm này. Đây là sự liên hệ với phân bố dữ liệu, chưa phải kiểm chứng nguyên nhân.</p>' : ""}
      </section>`;
  }

  root.innerHTML = `
    ${datasetHtml(evidence, summary)}
    <section class="panel noniid-intro mt-4">
      <div>
        ${sectionHead("2.2", "Dữ liệu lệch giữa các client (non-IID)", { lead: "Số ảnh mỗi nhóm bệnh khác nhau giữa các client" })}
        <p class="panel-desc">Dữ liệu non-IID là dữ liệu có phân bố khác nhau giữa các client: mỗi client có thể thiếu hẳn một số nhóm bệnh.</p>
      </div>
      ${shards ? `<div class="noniid-intro-stat"><strong>${fmtInt(shards.single_client_classes)}/${fmtInt(shards.classes_total)}</strong><span>nhóm có từ 90% ảnh trở lên ở một client</span></div>` : ""}
    </section>
    ${exampleHtml}

    <section class="mt-5" aria-labelledby="clients-title">
      ${sectionHead("2.3", "Dữ liệu của từng client", { id: "clients-title" })}
      <p class="panel-desc">${Object.keys(evidence.shard_counts || {}).length} phân vùng giả lập từ cùng một bộ ảnh, kèm kết quả của mô hình học riêng trên phân vùng đó.</p>
      <div class="client-grid mt-3" aria-label="Các client">${clientCards(evidence, shards, byModel, summary?.fairness)}</div>
      ${fairnessSplit(summary?.fairness) ? `<p class="field-hint mt-3">${fairnessText(fairnessSplit(summary.fairness))}</p>` : ""}
    </section>

    ${shards ? `
    <details class="overview-detail mt-4">
      <summary><span><strong>Xem bảng đầy đủ ${fmtInt(shards.classes_total)} nhóm × ${shards.clients.length} client</strong><small>Số ảnh và tỉ lệ nhận đúng theo nhóm</small></span></summary>
      <div class="overview-detail-body">
      <div class="panel-header-flex">
        <div>
          <h4 class="panel-title">Phân bố ảnh theo client</h4>
          <p class="panel-desc" id="heat-desc">Số ảnh của mỗi nhóm ở từng client. Ô càng đậm thì client đó giữ phần càng lớn của nhóm ấy.</p>
        </div>
        <div class="segmented" role="radiogroup" aria-label="Chọn bảng hiển thị">
          <button type="button" class="segmented-option is-active" role="radio" aria-checked="true" data-heat="count">Số ảnh để học</button>
          <button type="button" class="segmented-option" role="radio" aria-checked="false" data-heat="recall">Tỉ lệ nhận đúng</button>
        </div>
      </div>
      <div tabindex="0" role="region" aria-label="Số ảnh theo client và nhóm, cuộn được bằng phím mũi tên" class="table-responsive heat-wrap mt-3" id="heat-wrap">${heatmapTable(shards, re?.pv_recall, "count")}</div>
      <p class="field-hint mt-2">Tỉ lệ nhận đúng (recall) lấy từ lần chấm lại trên ảnh kiểm tra PlantVillage; chỉ có ở nhóm có ảnh kiểm tra.</p>
      ${presence && presence.absentMean !== null ? `
      <div class="callout mt-3">
        <b>Khớp với bảng số ảnh:</b> ở ${fmtInt(presence.absentN)} ô (client × nhóm) <b>không có ảnh</b>, tỉ lệ nhận đúng của mô hình học riêng
        ${presence.absentMean === 0 ? "luôn bằng <b>0%</b>" : `trung bình chỉ <b>${fmtPct(presence.absentMean, 1)}</b>`};
        ở ${fmtInt(presence.presentN)} ô có ảnh, tỉ lệ nhận đúng trung bình là <b>${fmtPct(presence.presentMean, 1)}</b>.
      </div>` : ""}
      </div>
    </details>` : `
    <div class="callout callout-warn mt-4" role="note">
      <b>Chưa có báo cáo phân bố nhóm theo client.</b> ${esc(summary?.shards?.reason || "Chạy scripts/report_client_shards.py trên máy có dataset.")}
    </div>`}

    <details class="panel panel-fold mt-4" id="scheme-explorer">
      <summary>Xem các cách chia dữ liệu có sẵn</summary>
      <p class="panel-desc mt-2">Release dựng sẵn các cách chia cho 5 client (số lượng và loại — IID, lệch nhãn... — tùy release).
        Đây <b>chỉ là dữ liệu</b> — mọi kết quả trên web là của cách chia lệch nhãn${alphaText(factsOf(summary)) ? ` ${alphaText(factsOf(summary))}` : ""}; các cách chia khác chưa được huấn luyện.</p>
      <div class="mt-3" id="scheme-explorer-body"><div class="skeleton skeleton-block"></div></div>
    </details>

    `;

  revealItems(root.querySelectorAll(".client-tile"), { stagger: 50 });

  const explorer = byId("scheme-explorer");
  let explorerMounted = false;
  explorer.addEventListener("toggle", () => {
    if (explorer.open && !explorerMounted) {
      explorerMounted = true;
      mountExplorer();
    }
  });

  if (shards) {
    const wrap = byId("heat-wrap");
    attachCrosshair(wrap);
    const desc = byId("heat-desc");
    root.querySelectorAll("[data-heat]").forEach(button => {
      button.addEventListener("click", () => {
        const view = button.dataset.heat;
        root.querySelectorAll("[data-heat]").forEach(b => {
          const on = b === button;
          b.classList.toggle("is-active", on);
          b.setAttribute("aria-checked", String(on));
        });
        wrap.innerHTML = heatmapTable(shards, re?.pv_recall, view);
        desc.textContent = view === "recall"
          ? "Tỉ lệ ảnh của nhóm đó được mô hình nhận đúng (%, trên ảnh kiểm tra chụp trong phòng). Hai cột cuối là mô hình chung."
          : "Số ảnh của mỗi nhóm ở từng client. Ô càng đậm thì client đó giữ phần càng lớn của nhóm ấy.";
      });
    });
  }
}
