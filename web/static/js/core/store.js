// Dữ liệu dùng chung giữa các bước: nạp một lần, mọi trang đọc cùng một bản (số liệu không bao giờ lệch nhau).
import { getJson } from "./api.js";

const memo = new Map();

function once(key, loader, force = false) {
  if (force || !memo.has(key)) {
    const promise = loader().catch(error => {
      memo.delete(key); // lần sau thử lại thay vì giữ lỗi
      throw error;
    });
    memo.set(key, promise);
  }
  return memo.get(key);
}

/** /api/demo/evidence: danh tính bundle, 7 mô hình + metric đã kiểm hash, shard_counts, lịch sử Centralized. */
export const getEvidence = (force = false) => once("evidence", () => getJson("/api/demo/evidence", { cache: "no-store" }), force);

/** /api/evidence/summary: tái đánh giá, PlantDoc theo lớp, hiệu chuẩn, shard, audit rò rỉ, checklist. */
export const getSummary = (force = false) => once("summary", () => getJson("/api/evidence/summary", { cache: "no-store" }), force);

/** /api/evidence/details: ma trận nhầm lẫn (val + test giữ riêng) và thư viện ảnh "sai mà tự tin". */
export const getDetails = (force = false) => once("details", () => getJson("/api/evidence/details", { cache: "no-store" }), force);

/** /api/evidence/partitions: phân bố lớp theo client của 12 cách chia (chỉ dữ liệu). */
export const getPartitions = (force = false) => once("partitions", () => getJson("/api/evidence/partitions", { cache: "no-store" }), force);

/** /api/models: danh sách mô hình đã qua cổng web_visible. */
export const getModels = (force = false) => once("models", () => getJson("/api/models", { cache: "no-store" }), force);

/** Catalog ảnh mẫu (tệp tĩnh). Thiếu catalog thì trang Chẩn đoán vẫn dùng được bằng tải ảnh lên. */
export const getGallery = (force = false) =>
  once("gallery", async () => {
    try {
      return await getJson("/static/samples/demo/catalog.json", { cache: "no-store" });
    } catch (error) {
      memo.delete("gallery"); // lỗi thoáng qua: lần mở sau thử lại thay vì nhớ "không có thư viện"
      return { count: 0, items: [], unavailable: true, reason: error.message };
    }
  }, force);

export function invalidateAll() {
  memo.clear();
}

/** Số liệu đã kiểm hash của một mô hình: { plantdoc: {...}, plantvillage: {...} } hoặc null. */
export function publishedMetrics(evidence, modelId) {
  const model = (evidence?.models || []).find(m => m.model_id === modelId);
  return model ? model.metrics : null;
}

export function modelsById(evidence) {
  return Object.fromEntries((evidence?.models || []).map(m => [m.model_id, m]));
}
