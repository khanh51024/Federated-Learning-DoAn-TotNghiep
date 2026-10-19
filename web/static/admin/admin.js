// Trang quản trị học liên kết cộng đồng. Mọi quyền nằm ở token quản trị mà máy chủ kiểm; trang này chỉ là giao diện cho
// các API /api/community/admin/*. Token chỉ giữ trong sessionStorage của tab; mã vừa tạo chỉ giữ trong bộ nhớ.
import { getJson, postJson } from "/static/js/core/api.js";
import { esc, fmtNum, formatApiError } from "/static/js/core/dom.js";
import {
  DEFAULT_USER, authHeader, clearCredential, isRemembered, readCredential, storeCredential,
} from "/static/js/core/admin-auth.js";

const REFRESH_MS = 10000;          // /admin/state bị nginx giới hạn 20 yêu cầu/phút: 10 giây là đủ thoải mái
const HISTORY_EVERY = 3;            // lịch sử công khai: mỗi 3 lần làm mới (30 giây) hoặc sau một thao tác

const $ = id => document.getElementById(id);

const S = {
  token: null,         // mật khẩu quản trị
  user: "admin",
  state: null,
  history: null,
  dirty: false,        // người dùng đang sửa cấu hình: không ghi đè ô nhập khi làm mới
  deadlineAt: 0,
  ticks: 0,
  timer: 0,
  busy: false,
  lastCodes: [],
};

// ───────── Khai báo trường cấu hình ─────────
// to/from: đổi giữa đơn vị của máy chủ và đơn vị hiển thị (phút, điểm phần trăm).
const MIN = { to: v => v / 60, from: v => v * 60 };
const PCT = { to: v => v * 100, from: v => v / 100 };
const FIELDS = [
  { key: "k_target", label: "Số người mỗi vòng (K)", unit: "người", step: 1, main: true,
    hint: "Đủ ngần này cập nhật hợp lệ thì gộp ngay." },
  { key: "k_min", label: "Số người tối thiểu khi hết giờ", unit: "người", step: 1, main: true,
    hint: "Hết giờ mà có ít nhất ngần này thì vẫn gộp; chưa đủ thì gia hạn." },
  { key: "round_ttl_seconds", label: "Thời gian một vòng", unit: "phút", step: 0.5, conv: MIN, main: true,
    hint: "Áp dụng cho cả vòng đang mở." },
  { key: "clip_norm", label: "Ngưỡng cắt độ lớn cập nhật", unit: "", step: 0.1, main: true,
    hint: "Cập nhật lớn hơn bị thu nhỏ về ngưỡng này (chống một người kéo lệch cả mô hình)." },
  { key: "n_min", label: "Số ảnh tối thiểu mỗi người", unit: "ảnh", step: 1 },
  { key: "n_cap", label: "Trần trọng số mỗi người", unit: "ảnh", step: 1, hint: "Một người có nhiều ảnh cũng chỉ được tính tối đa ngần này." },
  { key: "epsilon", label: "Cho phép giảm độ chính xác mỗi vòng", unit: "điểm %", step: 0.1, conv: PCT },
  { key: "epsilon_macro", label: "Cho phép giảm macro-recall mỗi vòng", unit: "điểm %", step: 0.1, conv: PCT },
  { key: "epsilon_plantdoc", label: "Cho phép giảm ảnh ngoài đồng mỗi vòng", unit: "điểm %", step: 0.1, conv: PCT },
  { key: "budget", label: "Ngân sách tụt tích lũy (chính xác)", unit: "điểm %", step: 0.1, conv: PCT,
    hint: "So với mức tốt nhất từng đạt, chặn trôi dần qua nhiều vòng nhỏ." },
  { key: "budget_macro", label: "Ngân sách tụt tích lũy (macro-recall)", unit: "điểm %", step: 0.1, conv: PCT },
  { key: "budget_plantdoc", label: "Ngân sách tụt tích lũy (ảnh ngoài đồng)", unit: "điểm %", step: 0.1, conv: PCT },
  { key: "local_epochs", label: "Số epoch huấn luyện trên máy người dùng", unit: "epoch", step: 1 },
  { key: "local_lr", label: "Tốc độ học trên máy người dùng", unit: "", step: 0.001 },
  { key: "local_batch", label: "Kích thước batch", unit: "ảnh", step: 1 },
  { key: "holdout_fraction", label: "Phần ảnh giữ lại để kiểm tra", unit: "%", step: 1, conv: PCT },
  { key: "dp_enabled", label: "Bật bảo vệ riêng tư tùy chọn (cắt chuẩn + nhiễu)", bool: true,
    hint: "Khuyến nghị cho client; máy chủ không ép được. Chưa phải differential privacy hình thức." },
  { key: "dp_clip", label: "Ngưỡng cắt khi bật riêng tư", unit: "", step: 0.1 },
  { key: "dp_noise_multiplier", label: "Hệ số nhiễu khi bật riêng tư", unit: "", step: 0.05 },
];

const EVENT_LABELS = {
  join_ok: "Có người vào", join_failed: "Nhập sai mã", update_ok: "Nhận cập nhật", update_rejected: "Từ chối cập nhật",
  round_opened: "Mở vòng mới", round_aggregated: "Vòng đã gộp", round_rejected: "Vòng bị từ chối", round_cancelled: "Vòng bị hủy",
  admin_open: "Quản trị: đổi trạng thái phiên", admin_policy: "Quản trị: đổi cấu hình", admin_codes: "Quản trị: tạo mã",
  admin_revoke_code: "Quản trị: thu hồi mã", admin_revoke_participant: "Quản trị: thu hồi người",
  admin_close_round: "Quản trị: gộp vòng ngay", admin_rollback: "Quản trị: khôi phục phiên bản",
};
const KIND_LABELS = { base: "Gốc (FedAvg v7)", aggregate: "Gộp từ vòng", rollback: "Khôi phục" };

// ───────── Tiện ích ─────────
const pct = v => (v === null || v === undefined ? "—" : `${fmtNum(Number(v) * 100, 1)}%`);
const when = ts => (ts ? new Date(Number(ts) * 1000).toLocaleString("vi-VN") : "—");
const clock = seconds => {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const text = (id, value) => { const el = $(id); if (el && el.textContent !== String(value)) el.textContent = String(value); };
const show = (id, on) => $(id)?.classList.toggle("hidden", !on);

function flash(kind, message) {
  const box = document.createElement("div");
  box.className = `callout${kind === "error" ? " callout-error" : kind === "warn" ? " callout-warn" : ""}`;
  box.textContent = message;
  const host = $("flash");
  host.prepend(box);
  while (host.children.length > 3) host.lastChild.remove();
  setTimeout(() => box.remove(), kind === "error" ? 12000 : 6000);
}

const headers = () => authHeader(S.user, S.token);

async function api(method, path, payload) {
  const url = `/api/community/admin/${path}`;
  if (method === "GET") return getJson(url, { headers: headers(), timeoutMs: 20000 });
  return postJson(url, payload ?? {}, { headers: headers(), timeoutMs: 30000 });
}

function describe(error) {
  return formatApiError(error?.body, error?.message || "Đã xảy ra lỗi không xác định.");
}

/** Nút nguy hiểm: bấm lần một đổi chữ, bấm lần hai trong 5 giây mới chạy. */
function confirmable(button, label, action) {
  let timer = 0;
  const original = button.textContent;
  button.addEventListener("click", () => {
    if (!timer) {
      button.textContent = label;
      timer = setTimeout(() => { timer = 0; button.textContent = original; }, 5000);
      return;
    }
    clearTimeout(timer);
    timer = 0;
    button.textContent = original;
    action();
  });
}

// ───────── Đăng nhập ─────────
// Lưu/đọc/xóa thông tin đăng nhập: core/admin-auth.js (dùng chung với các nút huấn luyện ở bước 5).
const saveToken = remember => storeCredential(S.user, S.token, remember);
const loadToken = readCredential;
const clearToken = clearCredential;

function showLoginError(message) {
  const el = $("login-error");
  el.textContent = message;
  show("login-error", !!message);
}

function enterApp() {
  show("login-panel", false);
  show("admin-app", true);
  show("btn-logout", true);
}

function leaveApp(message) {
  clearInterval(S.timer);
  S.token = null;
  S.state = null;
  clearToken();
  show("admin-app", false);
  show("btn-logout", false);
  show("login-panel", true);
  text("session-badge-top", "Chưa đăng nhập");
  $("session-badge-top").className = "status-badge is-neutral";
  if (message) showLoginError(message);
  $("token-input").value = "";
  $("token-input").focus();
}

async function login(event) {
  event.preventDefault();
  const value = $("token-input").value.trim();
  if (!value) return;
  S.user = $("user-input").value.trim() || DEFAULT_USER;
  S.token = value;
  $("btn-login").disabled = true;
  showLoginError("");
  try {
    S.state = await api("GET", "state");
    saveToken($("remember-token").checked);
    $("token-input").value = "";
    enterApp();
    await afterState({ firstTime: true });
    await loadHistory();
    renderAll();
    startPolling();
  } catch (error) {
    S.token = null;
    const status = error?.status;
    showLoginError(status === 401 ? "Tài khoản hoặc mật khẩu không đúng."
      : status === 429 ? describe(error)
      : status === 503 ? describe(error)
      : `Không đăng nhập được: ${describe(error)}`);
  } finally {
    $("btn-login").disabled = false;
  }
}

// ───────── Dữ liệu ─────────
async function loadState() {
  try {
    S.state = await api("GET", "state");
    return true;
  } catch (error) {
    if (error?.status === 401) { leaveApp("Phiên đăng nhập không còn hiệu lực. Hãy đăng nhập lại."); return false; }
    if (error?.status === 429) { flash("warn", describe(error)); return false; }
    flash("error", `Không tải được trạng thái: ${describe(error)}`);
    return false;
  }
}

async function loadHistory() {
  try {
    S.history = await getJson("/api/community/history?limit=15", { timeoutMs: 15000 });
  } catch {
    /* lịch sử là phần phụ: giữ bản cũ nếu lỗi */
  }
}

async function afterState({ firstTime = false } = {}) {
  if (!S.dirty || firstTime) fillPolicy();
}

async function refresh({ withHistory = true } = {}) {
  if (!S.token || S.busy) return;
  if (!(await loadState())) return;
  S.ticks += 1;
  if (withHistory) await loadHistory();
  await afterState();
  renderAll();
}

function startPolling() {
  clearInterval(S.timer);
  S.timer = setInterval(() => {
    if (document.hidden) return;
    refresh({ withHistory: S.ticks % HISTORY_EVERY === 0 });
  }, REFRESH_MS);
}

// ───────── Hiển thị ─────────
function renderAll() {
  if (!S.state) return;
  renderSession();
  renderCodesTable();
  renderVersions();
  renderEvents();
}

function renderSession() {
  const st = S.state;
  const round = st.round;
  const open = !!st.open;
  const badge = $("session-badge");
  badge.textContent = open ? "Đang mở" : "Đang đóng";
  badge.className = `status-badge ${open ? "success" : "is-neutral"}`;
  const top = $("session-badge-top");
  top.textContent = open ? "Phiên đang mở" : "Phiên đang đóng";
  top.className = `status-badge ${open ? "success" : "is-neutral"}`;
  text("session-note", open
    ? "Người có mã có thể vào bước 5 của trang chính và tham gia."
    : "Phiên đang đóng: không ai vào được cho tới khi bạn mở.");
  const toggle = $("btn-toggle-session");
  toggle.textContent = open ? "Đóng phiên" : "Mở phiên";
  toggle.className = `btn ${open ? "btn-danger" : "btn-primary"}`;
  toggle.disabled = S.busy;

  text("kpi-version", `v${st.current_version}`);
  text("kpi-participants", fmtNum(st.participants.total, 0));
  if (round) {
    S.deadlineAt = performance.now() + round.seconds_left * 1000;
    text("kpi-round", `#${round.round_id}`);
    text("kpi-received", `${round.received} / ${round.k_target}`);
    tickClock();
  } else {
    text("kpi-round", "—");
    text("kpi-received", "—");
    text("kpi-left", "—");
    S.deadlineAt = 0;
  }
  $("btn-close-round").disabled = S.busy || !round || round.received === 0;

  const versions = S.history?.versions || [];
  const latest = versions[versions.length - 1];
  const base = versions[0];
  if (latest?.metrics) {
    text("kpi-accuracy", pct(latest.metrics.accuracy));
    text("metric-note", latest.version === 0
      ? "Chưa có vòng nào được nhận: mô hình hiện hành là FedAvg v7 gốc."
      : `Mô hình gốc FedAvg v7: ${pct(base?.metrics?.accuracy)}. Tính trên tập validation đã dùng để chọn mô hình gốc — chỉ là tín hiệu theo dõi.`);
  }
  const select = $("rollback-version");
  const wanted = versions.map(v => String(v.version));
  if (select.dataset.versions !== wanted.join(",")) {
    select.dataset.versions = wanted.join(",");
    select.innerHTML = versions.slice().reverse()
      .map(v => `<option value="${esc(v.version)}">v${esc(v.version)} — ${esc(KIND_LABELS[v.kind] || v.kind)}${v.metrics ? ` · ${esc(pct(v.metrics.accuracy))}` : ""}</option>`)
      .join("");
  }
  $("btn-rollback").disabled = S.busy || versions.length === 0;
}

function tickClock() {
  if (S.deadlineAt) text("kpi-left", clock((S.deadlineAt - performance.now()) / 1000));
}

function codeStatus(code) {
  const now = Date.now() / 1000;
  if (code.revoked) return ["Đã thu hồi", "is-bad"];
  if (code.expires_at <= now) return ["Hết hạn", "is-warn"];
  if (code.uses >= code.max_uses) return ["Hết lượt", "is-warn"];
  return ["Dùng được", "is-ok"];
}

function renderCodesTable() {
  const codes = S.state.codes || [];
  const body = $("codes-body");
  if (!codes.length) {
    body.innerHTML = '<tr><td colspan="6" class="text-center text-muted">Chưa có mã nào.</td></tr>';
    return;
  }
  body.innerHTML = codes.map(code => {
    const [label, cls] = codeStatus(code);
    const usable = !code.revoked;
    return `<tr>
      <td class="mono">${esc(code.id)}</td>
      <td>${esc(code.label || "—")}</td>
      <td class="mono">${esc(code.uses)} / ${esc(code.max_uses)}</td>
      <td>${esc(when(code.expires_at))}</td>
      <td><span class="row-state ${cls}">${esc(label)}</span></td>
      <td>${usable ? `<button type="button" class="btn btn-ghost btn-sm" data-revoke="${esc(code.id)}">Thu hồi</button>` : ""}</td>
    </tr>`;
  }).join("");
}

function renderVersions() {
  const versions = S.history?.versions || [];
  $("versions-body").innerHTML = versions.length
    ? versions.slice().reverse().map(v => `<tr>
        <td class="fw-600">v${esc(v.version)}</td>
        <td>${esc(KIND_LABELS[v.kind] || v.kind)}</td>
        <td class="mono">${esc(v.clients)}</td><td class="mono">${esc(v.examples)}</td>
        <td class="mono">${esc(pct(v.metrics?.accuracy))}</td>
        <td class="mono">${esc(pct(v.metrics?.macro_recall))}</td>
        <td class="mono">${esc(pct(v.metrics?.plantdoc_accuracy))}</td>
        <td>${esc(when(v.created_at))}</td></tr>`).join("")
    : '<tr><td colspan="8" class="text-center text-muted">Chưa có dữ liệu.</td></tr>';

  const rounds = S.history?.rounds || [];
  $("rounds-body").innerHTML = rounds.length
    ? rounds.map(r => {
      let result = "Đã hủy";
      let cls = "is-warn";
      if (r.status === "aggregated") { result = `Đã nhận → v${r.version}`; cls = "is-ok"; }
      else if (r.status === "rejected") { result = `Bị từ chối: ${(r.messages || []).join("; ") || "không đạt cổng"}`; cls = "is-bad"; }
      return `<tr>
        <td class="fw-600">#${esc(r.round_id)}</td><td class="mono">v${esc(r.parent_version)}</td>
        <td class="mono">${esc(r.clients)}</td><td class="mono">${esc(r.examples)}</td><td class="mono">${esc(r.clipped)}</td>
        <td><span class="row-state ${cls}">${esc(result)}</span></td>
        <td class="mono">${r.metrics ? `${esc(pct(r.parent_metrics?.accuracy))} → ${esc(pct(r.metrics.accuracy))}` : "—"}</td></tr>`;
    }).join("")
    : '<tr><td colspan="7" class="text-center text-muted">Chưa có vòng nào hoàn tất.</td></tr>';
}

// Khóa kỹ thuật của nhật ký → nhãn tiếng Việt; khóa lạ giữ nguyên để không mất thông tin.
const DETAIL_LABELS = {
  round_id: "vòng", parent_version: "từ phiên bản", version: "phiên bản", new_version: "phiên bản mới", to: "về phiên bản",
  clients: "số người", examples: "số ảnh", updates: "số cập nhật", n: "số ảnh", norm: "độ lớn", clipped: "đã cắt",
  ip: "địa chỉ IP", code_id: "mã số", code: "mã lỗi", reason: "lý do", reasons: "lý do", message: "thông điệp", count: "số lượng",
  max_uses: "lượt dùng tối đa", ttl_hours: "hạn (giờ)", label: "nhãn", open: "phiên mở", changed: "đã đổi", suffix: "đuôi", changes: "thay đổi",
};
const detailValue = value => {
  if (typeof value === "boolean") return value ? "có" : "không";
  if (typeof value === "number") return fmtNum(value, Number.isInteger(value) ? 0 : 4); // dấu thập phân kiểu Việt
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
};

function eventDetail(detail) {
  return Object.entries(detail || {})
    .map(([key, value]) => {
      if (key === "participant") return `người …${String(value).slice(-4)}`;
      return `${DETAIL_LABELS[key] || key} ${detailValue(value)}`;
    })
    .join(" · ");
}

function renderEvents() {
  const events = S.state.events || [];
  $("events-body").innerHTML = events.length
    ? events.map(e => `<tr><td>${esc(when(e.ts))}</td><td>${esc(EVENT_LABELS[e.kind] || e.kind)}</td>
        <td class="text-secondary">${esc(eventDetail(e.detail))}</td></tr>`).join("")
    : '<tr><td colspan="3" class="text-center text-muted">Chưa có sự kiện.</td></tr>';
}

// ───────── Cấu hình ─────────
function fieldHtml(field, policy, bounds) {
  const id = `pf-${field.key}`;
  const hint = field.hint ? `<p class="field-hint">${esc(field.hint)}</p>` : "";
  if (field.bool) {
    return `<div class="form-group"><label class="check-label" for="${id}">
      <input type="checkbox" id="${id}" data-policy="${esc(field.key)}"${policy[field.key] ? " checked" : ""}> ${esc(field.label)}</label>${hint}</div>`;
  }
  const conv = field.conv || { to: v => v };
  const range = bounds[field.key] || [null, null];
  const min = range[0] === null ? "" : ` min="${conv.to(range[0])}"`;
  const max = range[1] === null ? "" : ` max="${conv.to(range[1])}"`;
  const unit = field.unit ? ` <span class="admin-unit">(${esc(field.unit)})</span>` : "";
  return `<div class="form-group"><label for="${id}">${esc(field.label)}${unit}</label>
    <input type="number" id="${id}" class="input-field" data-policy="${esc(field.key)}" step="${field.step}"${min}${max} required>${hint}</div>`;
}

let policyBuilt = false;
function buildPolicyForm() {
  const bounds = S.state.policy_bounds || {};
  const policy = S.state.policy || {};
  $("policy-main").innerHTML = FIELDS.filter(f => f.main).map(f => fieldHtml(f, policy, bounds)).join("");
  $("policy-advanced").innerHTML = FIELDS.filter(f => !f.main).map(f => fieldHtml(f, policy, bounds)).join("");
  policyBuilt = true;
}

function setPolicyInputs(values) {
  for (const field of FIELDS) {
    const input = $(`pf-${field.key}`);
    if (!input) continue;
    const value = values[field.key];
    if (field.bool) input.checked = !!value;
    else if (value !== undefined && value !== null) {
      const shown = (field.conv || { to: v => v }).to(Number(value));
      input.value = String(Math.round(shown * 1e6) / 1e6);
    }
  }
}

function fillPolicy() {
  if (!S.state) return;
  if (!policyBuilt) buildPolicyForm();
  setPolicyInputs(S.state.policy || {});
  S.dirty = false;
}

function collectPolicyChanges() {
  const changes = {};
  const current = S.state.policy || {};
  const bounds = S.state.policy_bounds || {};
  for (const field of FIELDS) {
    const input = $(`pf-${field.key}`);
    if (!input) continue;
    let value;
    if (field.bool) value = input.checked;
    else {
      if (input.value === "" || !Number.isFinite(Number(input.value))) throw new Error(`${field.label}: hãy nhập một số.`);
      value = (field.conv || { from: v => v }).from(Number(input.value));
      const range = bounds[field.key];
      if (range && (value < range[0] - 1e-12 || value > range[1] + 1e-12)) {
        const conv = field.conv || { to: v => v };
        throw new Error(`${field.label}: nằm ngoài khoảng cho phép (${conv.to(range[0])} – ${conv.to(range[1])}).`);
      }
      if (Number.isInteger(current[field.key]) && !Number.isInteger(value)) {
        if (field.conv === MIN) value = Math.round(value);
        else if (Math.abs(value - Math.round(value)) > 1e-9) throw new Error(`${field.label}: phải là số nguyên.`);
        else value = Math.round(value);
      }
    }
    const before = current[field.key];
    const changed = field.bool ? value !== before : Math.abs(Number(value) - Number(before)) > 1e-9;
    if (changed) changes[field.key] = value;
  }
  return changes;
}

// ───────── Hành động ─────────
async function act(button, work) {
  if (S.busy) return;
  S.busy = true;
  const before = button?.textContent;
  if (button) button.disabled = true;
  try {
    await work();
  } catch (error) {
    if (error?.status === 401) leaveApp("Phiên đăng nhập không còn hiệu lực. Hãy đăng nhập lại.");
    else flash("error", describe(error));
  } finally {
    S.busy = false;
    if (button) {
      button.disabled = false; // các nút có điều kiện riêng (gộp ngay, khôi phục, mở/đóng) được renderAll() đặt lại ngay dưới
      if (before !== undefined) button.textContent = before;
    }
    if (S.token) { await loadState(); await loadHistory(); await afterState(); renderAll(); }
  }
}

let closeArmed = 0;
async function toggleSession() {
  const open = !S.state.open;
  const received = S.state.round?.received || 0;
  if (!open && received > 0) {
    const button = $("btn-toggle-session");
    if (!closeArmed) {
      button.textContent = `Bấm lần nữa để đóng (vòng đang có ${received} người đã gửi)`;
      closeArmed = setTimeout(() => { closeArmed = 0; renderSession(); }, 5000);
      return;
    }
    clearTimeout(closeArmed);
    closeArmed = 0;
  }
  await act($("btn-toggle-session"), async () => {
    await api("POST", "settings", { open, changes: {} });
    flash("ok", open ? "Đã mở phiên." : "Đã đóng phiên. Vòng đang mở (nếu có) đã được gộp hoặc hủy.");
  });
}

async function savePolicy(event) {
  event.preventDefault();
  const error = $("policy-error");
  show("policy-error", false);
  let changes;
  try {
    changes = collectPolicyChanges();
  } catch (problem) {
    error.textContent = problem.message;
    show("policy-error", true);
    return;
  }
  if (!Object.keys(changes).length) { flash("ok", "Không có gì thay đổi."); return; }
  await act($("btn-save-policy"), async () => {
    try {
      await api("POST", "settings", { open: null, changes });
    } catch (failure) {
      error.textContent = describe(failure);
      show("policy-error", true);
      throw failure;
    }
    // Không điền lại form ở đây: S.state còn là bản TRƯỚC khi lưu. act() tải lại trạng thái rồi mới điền (afterState);
    // trong khoảng chờ, ô nhập vẫn giữ đúng giá trị vừa lưu.
    S.dirty = false;
    flash("ok", `Đã lưu: ${Object.keys(changes).join(", ")}.`);
  });
}

function joinLink(code) {
  return `${location.origin}/?code=${encodeURIComponent(code)}#huan-luyen`;
}

async function createCodes(event) {
  event.preventDefault();
  const payload = {
    count: Number($("codes-count").value),
    max_uses: Number($("codes-uses").value),
    ttl_hours: Number($("codes-hours").value),
    label: $("codes-label").value.trim(),
  };
  if (!Number.isInteger(payload.count) || payload.count < 1 || payload.count > 500) { flash("error", "Số mã phải từ 1 đến 500."); return; }
  await act($("btn-create-codes"), async () => {
    const data = await api("POST", "codes", payload);
    S.lastCodes = data.codes || [];
    $("codes-list").innerHTML = S.lastCodes.map(code => `<li class="code-item">
        <span class="code-value">${esc(code)}</span><span class="code-link">${esc(joinLink(code))}</span></li>`).join("");
    show("codes-result", true);
    $("codes-result").scrollIntoView({ block: "nearest" });
    flash("ok", `Đã tạo ${S.lastCodes.length} mã.`);
  });
}

async function copy(textValue, what) {
  try {
    await navigator.clipboard.writeText(textValue);
    flash("ok", `Đã sao chép ${what}.`);
  } catch {
    const area = document.createElement("textarea");
    area.value = textValue;
    area.setAttribute("readonly", "");
    area.className = "sr-only";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    area.remove();
    flash(ok ? "ok" : "warn", ok ? `Đã sao chép ${what}.` : "Trình duyệt không cho sao chép tự động: hãy bôi đen và sao chép thủ công.");
  }
}

async function revokeCode(id) {
  await act(null, async () => {
    await api("POST", `codes/${encodeURIComponent(id)}/revoke`, {});
    flash("ok", `Đã thu hồi mã #${id} và những người đã vào bằng mã đó.`);
  });
}

async function closeRound() {
  await act($("btn-close-round"), async () => {
    const result = await api("POST", "round/close", {});
    if (result.status === "aggregated") flash("ok", `Vòng #${result.round_id} đã gộp từ ${result.clients} người → phiên bản ${result.version}.`);
    else if (result.status === "rejected") flash("warn", `Vòng #${result.round_id} bị từ chối: ${(result.messages || []).join("; ")}.`);
    else flash("warn", `Vòng #${result.round_id} đã hủy vì chưa có cập nhật nào.`);
  });
}

async function rollback() {
  const version = Number($("rollback-version").value);
  if (!Number.isInteger(version)) return;
  await act($("btn-rollback"), async () => {
    const result = await api("POST", "rollback", { version });
    flash("ok", `Đã khôi phục: phiên bản mới v${result.version} có nội dung của v${result.restored_from}.`);
  });
}

// ───────── Khởi động ─────────
function init() {
  $("login-form").addEventListener("submit", login);
  $("btn-logout").addEventListener("click", () => leaveApp(""));
  $("btn-refresh").addEventListener("click", () => refresh({ withHistory: true }));
  $("btn-toggle-session").addEventListener("click", toggleSession);
  $("policy-form").addEventListener("submit", savePolicy);
  $("policy-form").addEventListener("input", () => { S.dirty = true; });
  $("btn-reset-policy").addEventListener("click", () => {
    setPolicyInputs(S.state?.policy_defaults || {});
    S.dirty = true;
    flash("ok", "Đã điền giá trị mặc định vào biểu mẫu. Bấm “Lưu cấu hình” để áp dụng.");
  });
  $("codes-form").addEventListener("submit", createCodes);
  $("btn-copy-codes").addEventListener("click", () => copy(S.lastCodes.join("\n"), "danh sách mã"));
  $("btn-copy-links").addEventListener("click", () => copy(S.lastCodes.map(joinLink).join("\n"), "các link tham gia"));
  $("btn-print-codes").addEventListener("click", () => window.print());
  $("codes-body").addEventListener("click", event => {
    const button = event.target.closest("[data-revoke]");
    if (!button) return;
    if (button.dataset.armed !== "1") {
      button.dataset.armed = "1";
      button.textContent = "Bấm lần nữa để thu hồi";
      setTimeout(() => { if (button.isConnected) { button.dataset.armed = ""; button.textContent = "Thu hồi"; } }, 5000);
      return;
    }
    revokeCode(button.dataset.revoke);
  });
  confirmable($("btn-close-round"), "Bấm lần nữa để gộp ngay", closeRound);
  confirmable($("btn-rollback"), "Bấm lần nữa để khôi phục", rollback);
  setInterval(tickClock, 1000);

  // `#token=` không còn được nhận: mật khẩu nằm trong URL là thói quen xấu (lộ khi chia sẻ màn hình,
  // lọt vào bookmark). Vẫn xóa fragment để link cũ ai đó đã lưu không để mật khẩu lại trên thanh địa chỉ.
  const linkParams = new URLSearchParams(location.hash.replace(/^#/, ""));
  if (linkParams.has("token")) {
    try { history.replaceState(null, "", `${location.pathname}${location.search}`); } catch { /* bỏ qua */ }
  }
  const stored = loadToken();
  if (stored) {
    S.user = stored.user || DEFAULT_USER;
    S.token = stored.password;
    $("token-input").value = "";
    api("GET", "state").then(async state => {
      S.state = state;
      saveToken(isRemembered() || $("remember-token").checked);
      enterApp();
      await afterState({ firstTime: true });
      await loadHistory();
      renderAll();
      startPolling();
    }).catch(error => {
      S.token = null;
      clearToken();
      if (error?.status === 401) showLoginError("Mật khẩu đã lưu không còn đúng. Hãy nhập lại.");
    });
  }
}

init();
