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
      throw new ApiError(formatApiError(data, `HTTP ${response.status}`), response.status, data);
    }
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error && (error.name === "AbortError" || error.name === "TimeoutError")) {
      if (signal && signal.aborted) throw error; // do người gọi hủy: giữ nguyên để họ bỏ qua
      throw new ApiError("Máy chủ không phản hồi kịp (quá thời gian chờ).", 0);
    }
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });

export const postForm = (url, formData, options) => request(url, { ...options, method: "POST", body: formData });

export const isAbort = error => !!error && (error.name === "AbortError" || error.name === "TimeoutError") && !(error instanceof ApiError);
