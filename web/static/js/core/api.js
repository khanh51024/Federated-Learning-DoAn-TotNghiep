// Một chỗ gọi API: timeout, hủy yêu cầu, thông điệp lỗi thống nhất. Không tự bật toast (nơi gọi quyết định).
import { formatApiError } from "./dom.js";

export class ApiError extends Error {
  constructor(message, status = 0, body = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/** Câu mặc định khi máy chủ trả lỗi mà không có thông điệp JSON (trang lỗi HTML của proxy, 502/503...). */
function fallbackMessage(status) {
  if (status >= 500) return `Máy chủ gặp lỗi (HTTP ${status}). Hãy thử lại sau ít phút.`;
  return `Yêu cầu không thành công (HTTP ${status}).`;
}

async function request(url, { method = "GET", body, headers, signal, timeoutMs = 60000, cache } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException("timeout", "TimeoutError")), timeoutMs);
  const forward = () => controller.abort(signal.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", forward, { once: true });
  }
  try {
    const response = await fetch(url, { method, body, headers, signal: controller.signal, cache });
    const data = await response.json().catch(() => null);
    if (!response.ok || data === null) {
      throw new ApiError(formatApiError(data, fallbackMessage(response.status)), response.status, data);
    }
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error && (error.name === "AbortError" || error.name === "TimeoutError")) {
      if (signal && signal.aborted) throw error; // do người gọi hủy: giữ nguyên để họ bỏ qua
      throw new ApiError("Máy chủ không phản hồi kịp (quá thời gian chờ).", 0);
    }
    // fetch báo mất mạng/máy chủ không lên bằng TypeError với chữ tiếng Anh của trình duyệt ("Failed to fetch"): đổi sang câu dễ hiểu
    if (error instanceof TypeError) throw new ApiError("Không kết nối được máy chủ. Hãy kiểm tra mạng hoặc máy chủ rồi thử lại.", 0);
    throw new ApiError(error && error.message ? error.message : "Không kết nối được máy chủ.", 0);
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", forward);
  }
}

export const getJson = (url, options) => request(url, { ...options, method: "GET" });

export const postJson = (url, payload, options) =>
  request(url, {
    ...options,
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    body: JSON.stringify(payload ?? {}),
  });

export const postForm = (url, formData, options) => request(url, { ...options, method: "POST", body: formData });

export const isAbort = error => !!error && (error.name === "AbortError" || error.name === "TimeoutError") && !(error instanceof ApiError);
