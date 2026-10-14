// Bước 2 · Dữ liệu phân tán (non-IID): heatmap 38 lớp × 5 client và vì sao Local-only không nhận ra bệnh chưa từng thấy.
import { getEvidence, getPartitions, getSummary, modelsById } from "../core/store.js";
import { attachCrosshair } from "../core/crosshair.js";
import {
  byId, cropOf, diseaseOf, esc, fmtInt, fmtNum, fmtPct, fmtPctValue, formatDiseaseName, revealItems,
} from "../core/dom.js";

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

function clientCards(evidence, shards, byModel) {
  const counts = evidence.shard_counts || {};
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
          : `<td class="heat-cell heat-recall${r < 0.5 ? " is-low" : ""}" style="--p:${r}" title="Local ${clientNumber(c)}: nhận đúng ${fmtPct(r, 1)} · client này có ${fmtInt(count)} ảnh của nhóm này">${fmtNum(r * 100, 0)}</td>`;
      }).join("");
      const globals = ["fedavg", "centralized"].map(id => {
        const r = pvRecall?.[id]?.[row.class];
        return r === undefined || r === null
          ? `<td class="heat-cell is-empty">—</td>`
          : `<td class="heat-cell heat-recall${r < 0.5 ? " is-low" : ""} col-global" style="--p:${r}" title="${id === "fedavg" ? "FedAvg" : "Centralized"}: nhận đúng ${fmtPct(r, 1)}">${fmtNum(r * 100, 0)}</td>`;
      }).join("");
      cells = recallCells + globals;
    } else {
      const countCells = clients.map(c => {
        const n = row.per_client[c];
        const share = row.total ? n / row.total : 0;
        const cls = n === 0 ? "is-zero" : share >= 0.9 ? "is-dominant" : "";
        return `<td class="heat-cell heat-count ${cls}" style="--p:${share.toFixed(3)}" title="Client ${clientNumber(c)} có ${fmtInt(n)}/${fmtInt(row.total)} ảnh nhóm ${esc(formatDiseaseName(row.class))} (${fmtNum(share * 100, 1)}%)">${fmtInt(n)}</td>`;
      }).join("");
      const concentration = row.top_share_pct !== null && row.top_share_pct >= 90
        ? `<span class="pill-flag" title="Hơn 90% ảnh của nhóm này nằm ở một client">${fmtNum(row.top_share_pct, 0)}% ở client ${clientNumber(row.top_client)}</span>` : "";
      cells = `${countCells}<td class="heat-total mono">${fmtInt(row.total)}</td><td class="heat-note">${concentration}</td>`;
    }
    rows.push(`<tr><th scope="row" class="heat-class">${esc(diseaseOf(row.class))}</th>${cells}</tr>`);
  }
  return `<table class="heat-table"><thead><tr><th scope="col">Nhóm</th>${head}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

// ───────── Trình xem 12 cách chia của release (chỉ dữ liệu) ─────────
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
        <span class="scheme-bar"><span style="transform:scaleX(${(d.n_k / total).toFixed(3)})"></span></span>
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
    byId("scheme-desc").textContent = `Độ lệch phân bố nhóm trung bình giữa client và toàn bộ (Jensen–Shannon): ${scheme.mean_js_divergence === null ? "0 (chia đều)" : fmtNum(scheme.mean_js_divergence, 3)} — càng lớn càng non-IID. Tổng ${fmtInt(partitions.total_train)} ảnh train.`;
    byId("scheme-clients").innerHTML = schemeSummary(scheme);
    byId("scheme-wrap").innerHTML = heatmapTable(schemeAsShards(partitions, scheme), null, "count");
  };
  select.addEventListener("change", render);
  render();
}

export async function mount() {
  const root = byId("noniid-root");
  const [evidence, summary] = await Promise.all([getEvidence(), getSummary().catch(() => null)]);
  if (!evidence.available) {
    root.innerHTML = `<div class="callout callout-error" role="alert"><b>Chưa có bộ số liệu đã kiểm hash.</b> ${esc(evidence.reason || "")}</div>`;
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
    const recalls = re
      ? others.map(([c]) => {
          const r = re.pv_recall?.[`local_${clientNumber(c)}`]?.[example.class];
          return `Local ${clientNumber(c)}: ${r === undefined || r === null ? "—" : fmtPct(r, 0)}`;
        }).join(" · ")
      : "";
    exampleHtml = `
      <section class="viewer-example mt-4" aria-label="Ví dụ dữ liệu lệch giữa các client">
        <div><p class="eyebrow">Một ví dụ cụ thể</p><h3 class="panel-title">${esc(formatDiseaseName(example.class))}</h3>
        <p class="panel-desc">Client ${holder} giữ <b>${fmtNum(example.top_share_pct, 0)}%</b> ảnh của nhóm này. Các nơi còn lại có rất ít hoặc không có ảnh để học.</p></div>
        <ul class="viewer-bars" aria-label="Số ảnh của nhóm này tại từng client">
          ${Object.entries(example.per_client).map(([client, count]) => `<li><span>Client ${clientNumber(client)}</span><span class="viewer-bar-track"><span style="--share:${example.total ? count / example.total : 0}"></span></span><b>${fmtInt(count)} ảnh</b></li>`).join("")}
        </ul>
        <details class="fold"><summary>Đối chiếu với kết quả học riêng</summary><p class="panel-desc mt-2">Client ${holder}: ${fmtInt(holderCount)} ảnh; các client khác: ${fmtInt(othersTotal)} ảnh${zeroText}. ${recalls ? `Tỉ lệ nhận đúng nhóm này: ${recalls}.` : "Chưa có kết quả theo nhóm."} Đây là sự liên hệ với phân bố dữ liệu, chưa phải kiểm chứng nguyên nhân.</p></details>
      </section>`;
  }

  root.innerHTML = `
    <section class="panel noniid-intro">
      <div>
        <p class="eyebrow">Dữ liệu non-IID</p>
        <h3 class="panel-title">Mỗi client chỉ thấy một phần dữ liệu</h3>
        <p class="panel-desc">Mỗi nơi có thể thiếu ảnh của một số nhóm. Học chung giúp kết hợp dữ liệu từ các client.</p>
      </div>
      ${shards ? `<div class="noniid-intro-stat"><strong>${fmtInt(shards.single_client_classes)}/${fmtInt(shards.classes_total)}</strong><span>nhóm có ≥ 90% ảnh ở một client</span></div>` : ""}
    </section>
    ${exampleHtml}

    <details class="overview-detail mt-4"><summary><span><strong>Xem dữ liệu và mô hình của từng client</strong><small>${Object.keys(evidence.shard_counts || {}).length} phân vùng giả lập</small></span></summary>
      <div class="overview-detail-body"><section class="client-grid" aria-label="Các client">${clientCards(evidence, shards, byModel)}</section></div>
    </details>

    ${shards ? `
    <details class="overview-detail mt-4">
      <summary><span><strong>Xem bảng đầy đủ ${fmtInt(shards.classes_total)} nhóm × ${shards.clients.length} client</strong><small>Số ảnh và tỉ lệ nhận đúng theo nhóm</small></span></summary>
      <div class="overview-detail-body">
      <div class="panel-header-flex">
        <div>
          <h3 class="panel-title">Phân bố ảnh theo client</h3>
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
      <summary>Khám phá 12 cách chia dữ liệu có sẵn trong bộ dữ liệu</summary>
      <p class="panel-desc mt-2">Bộ dữ liệu dựng sẵn 12 cách chia cho 5 client: IID, lệch nhãn, lệch số lượng, lệch nguồn ảnh và kết hợp.
        Đây <b>chỉ là dữ liệu</b> — mọi kết quả trên web là của cách chia lệch nhãn α = 0,1; các cách chia khác chưa được huấn luyện.</p>
      <div class="mt-3" id="scheme-explorer-body"><div class="skeleton skeleton-block"></div></div>
    </details>

    <section class="cta-row mt-4">
      <a class="btn btn-primary" href="#ket-qua">Xem kết quả chi tiết →</a>
      <a class="btn btn-secondary" href="#chan-doan">Tự thử trên một ảnh →</a>
    </section>`;

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
