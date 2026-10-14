// Poll /api/status: trạng thái server và tác vụ huấn luyện nền.
// - Chỉ ghi DOM khi giá trị đổi; dừng khi tab trình duyệt ẩn; không chồng yêu cầu.
// - Chỉ lỗi mạng/HTTP mới là "mất kết nối"; lỗi dựng giao diện chỉ ghi console.
import { getJson } from "./api.js";
import { byId, notify, setClass, setText } from "./dom.js";

let pollMs = 4000;
const POLL_DEFAULT_MS = 4000;

let timer = null;
let serverOnline = null;
let offlineToast = null;
let previousStatus = "idle";
let lastTask = null;
let startSignature = null; // dấu vân tay của tác vụ cũ ngay lúc gửi lệnh: poll còn trả đúng nó thì đó là dữ liệu cũ
let startedAt = 0;
const STALE_WINDOW_MS = 10000;

const taskSignature = task => JSON.stringify([task.status, task.message, task.progress, task.type, task.experiment, (task.live_progress || []).length]);
const listeners = new Set();

/** fn(data, { task, network, finished, previousStatus }) được gọi sau mỗi lần poll thành công. */
export function onStatus(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Nhịp poll (ms). Bước 5 hạ xuống ~1,5 s khi có phiên chạy để sơ đồ bám sát trạng thái thật; trả về 4 s khi xong. */
export function setPollInterval(ms) {
  pollMs = Number.isFinite(ms) && ms >= 1000 ? ms : POLL_DEFAULT_MS;
}

export function currentTask() {
  return lastTask;
}

/** Gọi ngay sau khi chính ta gửi lệnh huấn luyện: nếu tác vụ kết thúc trong một chu kỳ poll vẫn phải báo. */
export function markTaskStarted() {
  startSignature = lastTask ? taskSignature(lastTask) : taskSignature({ status: "idle" });
  startedAt = Date.now();
  previousStatus = "running";
}

function kindLabel(task) {
  if (task.type === "centralized" || task.experiment === "Centralized") return "Centralized";
  if (task.type === "federated" || task.type === "federated-network" || task.experiment === "Federated") return "FedAvg";
  return "Tác vụ";
}

function notifyFinished(task) {
  const kind = kindLabel(task);
  if (task.status === "completed") {
    notify("success", `Huấn luyện ${kind} hoàn tất`, task.message || "");
  } else if (task.status === "cancelled") {
    notify("warning", `Đã dừng huấn luyện ${kind}`, task.message || "Tiến trình đã dừng theo yêu cầu.");
  } else if (task.status === "failed") {
    notify("error", `Huấn luyện ${kind} thất bại`, task.message || "Tác vụ nền báo lỗi.");
  }
}

function markOffline(error) {
  const indicator = byId("server-status-text");
  setText(indicator, "Máy chủ: mất kết nối");
  const holderOff = byId("server-status");
  if (holderOff) holderOff.title = "Máy chủ: mất kết nối";
  indicator?.parentElement?.classList.add("offline");
  const badge = byId("task-badge");
  setText(badge, "Mất kết nối API");
  setClass(badge, "status-badge error");
  if (serverOnline !== false) {
    offlineToast = notify("error", "Mất kết nối máy chủ", "Giao diện sẽ tự thử lại sau vài giây.", { detail: error.message, duration: 0, sticky: true });
  }
  serverOnline = false;
}

function renderHeader(data) {
  const indicator = byId("server-status-text");
  setText(indicator, `Máy chủ: ${data.server_status === "online" ? "đang hoạt động" : "không phản hồi"}`);
  const holder = byId("server-status");
  if (holder) holder.title = indicator.textContent;
  indicator?.parentElement?.classList.remove("offline");
  const flower = data.flower_server_active ? "Flower đang chạy" : "Flower chờ";
  setText(byId("system-device"), `Thiết bị: ${data.device} · ${flower}`);

  const task = data.task || { status: "idle", message: "Sẵn sàng" };
  const badge = byId("task-badge");
  // Phiên bị dừng vẫn được giữ trong /api/status để bước Huấn luyện xem lại,
  // nhưng thanh trạng thái chung chỉ cần báo hệ thống đã sẵn sàng chạy phiên mới.
  setText(badge, task.status === "cancelled" ? "Sẵn sàng" : (task.message || "Sẵn sàng"));
  let cls = "status-badge";
  if (task.status === "running") cls += " running";
  else if (task.status === "completed") cls += " success";
  else if (task.status === "failed") cls += " error";
  setClass(badge, cls);
}

export async function pollStatus() {
  let data;
  try {
    data = await getJson("/api/status", { timeoutMs: 8000 });
  } catch (error) {
    markOffline(error);
    return;
  }
  if (serverOnline === false) {
    if (offlineToast) offlineToast.close();
    offlineToast = null;
    notify("success", "Đã kết nối lại máy chủ");
  }
  serverOnline = true;

  try {
    renderHeader(data);
    const task = data.task || { status: "idle", message: "Sẵn sàng" };
    const network = data.network_processes || {};
    lastTask = task;
    // Endpoint Centralized/simulation đặt status=running trong BackgroundTask SAU khi trả lời, nên một lần poll ngay
    // sau khi bấm Bắt đầu có thể còn thấy tác vụ cũ ("completed"). Đó không phải tác vụ vừa kết thúc.
    let stale = false;
    if (startSignature !== null) {
      if (task.status === "running" || taskSignature(task) !== startSignature || Date.now() - startedAt > STALE_WINDOW_MS) {
        startSignature = null;
      } else {
        stale = true;
      }
    }
    const finished = !stale && previousStatus === "running" && task.status !== "running";
    if (finished) notifyFinished(task);
    for (const fn of listeners) {
      try {
        fn(data, { task, network, finished, previousStatus, stale });
      } catch (error) {
        console.error("[Status] listener lỗi:", error);
      }
    }
    if (!stale) previousStatus = task.status || "idle";
  } catch (error) {
    console.error("[Status] Lỗi dựng trạng thái hệ thống:", error);
  }
}

function schedule() {
  clearTimeout(timer);
  if (document.hidden) return;
  timer = setTimeout(async () => {
    await pollStatus();
    schedule();
  }, pollMs);
}

export function startStatusPolling() {
  pollStatus().finally(schedule);
  document.addEventListener("visibilitychange", () => {
    clearTimeout(timer);
    if (!document.hidden) pollStatus().finally(schedule);
  });
}
