// Published 2026-10-03 evidence is rendered only after the API verifies all bytes.
// The Flower localhost smoke session is a separate technical demonstration.
// Helper dùng chung (setText, setMetric, notify, esc…) nằm trong main.js, được nạp trước file này.

// Danh sách model chọn được lấy từ registry (chỉ model đã qua cổng web_visible), kèm nhãn ngữ cảnh.
async function loadModelSelector() {
  const select = document.getElementById("inference-fed-source");
  if (!select) return;
  const previous = select.value; // giữ lựa chọn của người dùng khi danh sách được nạp lại sau mỗi tác vụ
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const choices = payload.models.filter(m => m.role === "current" && m.inference_key && m.inference_key !== "centralized");
    select.replaceChildren();
    if (!choices.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Registry chưa cho phép model nào — chạy scripts/update_status.py";
      select.appendChild(option);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    const keepPrevious = choices.some(m => m.inference_key === previous);
    for (const model of choices) {
      const option = document.createElement("option");
      option.value = model.inference_key;
      option.textContent = `${model.display_name} — ${model.display_label}`;
      if (keepPrevious ? model.inference_key === previous : model.inference_key === "fedavg") option.selected = true;
      select.appendChild(option);
    }
  } catch (error) {
    select.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = `Không tải được danh sách model: ${error.message}`;
    select.appendChild(option);
    select.disabled = true;
  }
}

let publishedDemoFailureNotified = false;

async function loadPublishedDemo() {
  await loadModelSelector();
  const badge = document.getElementById("published-evidence-badge");
  const note = document.getElementById("published-evidence-note");
  const body = document.getElementById("compare-body");
  const clientsContainer = document.getElementById("clients-container");
  const metricIds = ["dash-acc", "dash-loss", "cent-acc", "cent-f1", "cent-loss"];
  const setMetricById = (id, value) => setMetric(document.getElementById(id), value);
  const clearCharts = () => {
    for (const chart of [distributionChart, centralizedChart, federatedChart, alphaChart, faultyChart, convergenceChart]) {
      if (chart) { chart.data.labels = []; chart.data.datasets.forEach(dataset => dataset.data = []); chart.update("none"); }
    }
  };
  const unavailable = reason => {
    if (badge) { badge.textContent = "ARTIFACT CHƯA HỢP LỆ"; badge.className = "status-badge error"; }
    if (note) note.textContent = `Không hiển thị metric/checkpoint: ${reason}`;
    for (const id of metricIds) setMetricById(id, "—");
    if (body) {
      body.removeAttribute("aria-busy");
      body.innerHTML = '<tr><td colspan="4" class="text-muted">Chưa có bộ artifact hợp lệ.</td></tr>';
    }
    if (clientsContainer) {
      clientsContainer.removeAttribute("aria-busy");
      clientsContainer.textContent = "Không có shard/model Local-only đã kiểm hash.";
    }
    clearCharts();
    document.querySelectorAll("#inference-fed-source option").forEach(option => option.disabled = true);
    if (!publishedDemoFailureNotified) {
      publishedDemoFailureNotified = true;
      notify("error", "Artifact chưa hợp lệ", "Web không hiển thị số liệu khi chưa kiểm được SHA-256.", { detail: reason });
    }
  };

  try {
    const response = await fetch("/api/demo/evidence", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const evidence = await response.json();
    if (!evidence.available) { unavailable(evidence.reason || "Chưa nhập artifact"); return; }
    publishedDemoFailureNotified = false;

    if (badge) { badge.textContent = "SHA-256 ĐÃ ĐỐI SOÁT"; badge.className = "status-badge success"; }
    if (note) note.textContent = "Validation nội bộ từ ba nhánh GitHub, seed 42; SHA-256 file và checkpoint đã khớp. Chưa có bộ ảnh gốc để tái đánh giá. GĐ2 chỉ có một kịch bản lệch nhãn α=0,1; chưa hoàn tất nhiều seed.";
    const trueSelect = document.getElementById("inference-true-class");
    if (trueSelect && Array.isArray(evidence.classes) && trueSelect.options.length <= 1) {
      for (const cls of evidence.classes) {
        const option = document.createElement("option");
        option.value = cls;
        option.textContent = cls.replace("___", " — ").replaceAll("_", " ");
        trueSelect.appendChild(option);
      }
    }
    const models = Object.fromEntries(evidence.models.map(model => [model.model_id, model]));
    setMetricById("dash-acc", (models.fedavg.metrics.plantvillage.top1_accuracy * 100).toFixed(2) + "%");
    setMetricById("dash-loss", (models.fedavg.metrics.plantdoc.top1_accuracy * 100).toFixed(2) + "%");
    setMetricById("cent-acc", (models.centralized.metrics.plantvillage.top1_accuracy * 100).toFixed(2) + "%");
    setMetricById("cent-f1", models.centralized.metrics.plantvillage.macro_f1_supported.toFixed(4));
    setMetricById("cent-loss", (models.centralized.metrics.plantdoc.top1_accuracy * 100).toFixed(2) + "%");

    if (clientsContainer) {
      const fragment = document.createDocumentFragment();
      for (let clientId = 0; clientId < 5; clientId++) {
        const model = models[`local_${String(clientId).padStart(2, "0")}`];
        const shardName = `client_${String(clientId).padStart(2, "0")}`;
        const card = document.createElement("div");
        card.className = "client-card";
        const title = document.createElement("div");
        title.className = "client-card-title";
        title.textContent = `Client ${String(clientId).padStart(2, "0")} · Local-only v7`;
        card.appendChild(title);
        for (const value of [
          `${evidence.shard_counts[shardName].toLocaleString()} ảnh train trong shard α=0,1 đã công bố`,
          `PlantVillage validation: ${(model.metrics.plantvillage.top1_accuracy * 100).toFixed(2)}%`,
          `PlantDoc validation: ${(model.metrics.plantdoc.top1_accuracy * 100).toFixed(2)}%`,
          `Checkpoint SHA-256: ${model.checkpoint_sha256.slice(0, 16)}…`,
        ]) {
          const row = document.createElement("div");
          row.className = "client-stat-row";
          row.textContent = value;
          card.appendChild(row);
        }
        fragment.appendChild(card);
      }
      clientsContainer.replaceChildren(fragment);
      clientsContainer.removeAttribute("aria-busy");
      revealItems(clientsContainer.children, { stagger: 60 });
    }

    if (body) {
      const fragment = document.createDocumentFragment();
      for (const model of evidence.models) {
        for (const source of ["plantdoc", "plantvillage"]) {
          const metric = model.metrics[source];
          const tr = document.createElement("tr");
          const values = [
            `${model.name} (${model.phase})`,
            `${source === "plantdoc" ? "PlantDoc" : "PlantVillage"} (${metric.count})`,
            `${(metric.top1_accuracy * 100).toFixed(2)}% / ${metric.macro_f1_supported.toFixed(4)}`,
            `${model.branch} @ ${model.commit.slice(0, 8)} · SHA ${model.checkpoint_sha256.slice(0, 12)}…`,
          ];
          values.forEach(value => { const td = document.createElement("td"); td.textContent = value; tr.appendChild(td); });
          fragment.appendChild(tr);
        }
      }
      body.replaceChildren(fragment);
      body.removeAttribute("aria-busy");
    }
    if (distributionChart) {
      distributionChart.data.labels = Object.keys(evidence.shard_counts).map(name => name.replace("client_", "Client "));
      distributionChart.data.datasets[0].data = Object.values(evidence.shard_counts);
      distributionChart.update();
    }
    if (alphaChart) {
      alphaChart.data.labels = ["Lệch nhãn α=0,1"];
      alphaChart.data.datasets[0].data = [(models.fedavg.metrics.plantvillage.top1_accuracy * 100).toFixed(2)];
      alphaChart.update();
    }
    if (centralizedChart) {
      centralizedChart.data.labels = evidence.centralized_history.map(row => `E${row.epoch}`);
      centralizedChart.data.datasets[0].label = "Validation accuracy (%) · toàn bộ nguồn";
      centralizedChart.data.datasets[0].data = evidence.centralized_history.map(row => +(row.val_accuracy * 100).toFixed(2));
      centralizedChart.data.datasets[1].data = [];
      centralizedChart.update();
    }
    // FedAvg v7 publishes only selected-checkpoint validation, no per-round
    // history. Do not synthesize a convergence curve from a single endpoint.
    for (const chart of [federatedChart, convergenceChart, faultyChart]) {
      if (chart) { chart.data.labels = []; chart.data.datasets.forEach(dataset => dataset.data = []); chart.update("none"); }
    }
    document.querySelectorAll("#inference-fed-source option").forEach(option => option.disabled = false);
    setText(document.getElementById("cent-live-badge"), "Artifact Centralized v6 · validation");
  } catch (error) {
    unavailable(error.message);
  }
}
