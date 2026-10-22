// Câu phạm vi / hạn chế về cách huấn luyện, dựng từ `summary.facts` (web/services/evidence_service.py::_facts).
// Không viết sẵn "một lần chạy", "chưa cùng điều kiện"...: đổi gói mô hình thì câu chữ đổi theo số liệu của gói.
import { fmtInt, fmtNum } from "./dom.js";

/** facts hợp lệ của summary, hoặc null (báo cáo thiếu: giao diện dùng câu trung tính, không khẳng định gì về cách huấn luyện). */
export const factsOf = summary => (summary?.facts?.available ? summary.facts : null);

const RECIPE_LABELS = {
  optimizer: "bộ tối ưu", lr: "tốc độ học", batch_size: "kích thước batch", label_smoothing: "label smoothing", precision: "kiểu số",
  train_transform: "tăng cường ảnh", loss_weighting: "trọng số mất mát", selection: "cách chọn checkpoint", steps_total: "số bước", local_epochs: "số epoch cục bộ",
};

export const alphaText = facts => (facts?.alpha !== null && facts?.alpha !== undefined ? `α = ${fmtNum(facts.alpha, 1)}` : null);

export function runsText(facts) {
  const n = facts?.seeds_run?.length || 0;
  if (n > 1) return `${fmtInt(n)} hạt giống`;
  return n === 1 ? "một hạt giống" : "một lần chạy";
}

/** Vế "điều kiện huấn luyện": rỗng khi chưa biết, nói rõ điểm khác khi các cách học không cùng thiết lập. */
export function conditionsClause(facts) {
  if (!facts || facts.same_recipe === null || facts.same_recipe === undefined) return "";
  if (facts.same_recipe) return "";
  const parts = (facts.recipe_differences || []).map(key => RECIPE_LABELS[key] || key);
  return `các cách học chưa huấn luyện cùng điều kiện${parts.length ? ` (khác ở ${parts.join(", ")})` : ""}`;
}

/** Câu "Phạm vi" dưới các số chính ở Giới thiệu, Kết quả và Kết luận. */
export function scopeText(facts) {
  if (!facts) return "Tập kiểm tra cũng dùng để chọn mô hình, nên chưa phải độ chính xác ngoài thực địa.";
  const n = facts.seeds_run?.length || 0;
  const head = n > 1 ? `Số liệu trang này là của hạt giống ${facts.seed_served} (đã chạy ${fmtInt(n)} hạt giống)` : "Kết quả từ một lần chạy";
  const alpha = alphaText(facts);
  const conditions = conditionsClause(facts);
  return `${head}${alpha ? `, một mức lệch dữ liệu (${alpha})` : ""}; tập kiểm tra cũng dùng để chọn mô hình${conditions ? ` và ${conditions}` : ""}, nên chưa phải độ chính xác ngoài thực địa.`;
}

/** Nhóm nguồn nào ECE của FedAvg giảm / tăng sau hiệu chuẩn (summary.reeval); null khi báo cáo thiếu. Dùng cho mục 4.4 và hạn chế về xác suất. */
export function calibrationShift(reeval, groups) {
  const fed = reeval?.available ? reeval.models.find(m => m.model_id === "fedavg") : null;
  if (!fed) return null;
  const rows = groups.filter(g => fed.calibration?.[g.id])
    .map(g => ({ group: g, raw: fed.calibration[g.id].ece_raw, scaled: fed.calibration[g.id].ece_temperature_scaled }));
  return { improved: rows.filter(r => r.scaled < r.raw), worsened: rows.filter(r => r.scaled > r.raw), temperature: fed.temperature };
}

/** Đoạn dưới bảng thiết lập ở chương 3. */
export function recipeNote(facts) {
  if (!facts || facts.same_recipe === null || facts.same_recipe === undefined) {
    return "Thiết lập huấn luyện của từng cách nằm ở bảng dưới.";
  }
  if (facts.same_recipe) {
    return "Ba cách huấn luyện dùng cùng một bộ ảnh, cùng kiến trúc mạng và cùng thiết lập (bộ tối ưu, tốc độ học, kích thước batch, tăng cường ảnh, trọng số mất mát, số epoch hoặc vòng); chỉ khác ở nơi ảnh nằm: gom về một chỗ, chia cho các client rồi gộp, hoặc mỗi client học riêng.";
  }
  const parts = (facts.recipe_differences || []).map(key => RECIPE_LABELS[key] || key);
  return `Ba cách huấn luyện dùng cùng một bộ ảnh và cùng kiến trúc mạng nhưng khác nhau ở ${parts.join(", ") || "thiết lập huấn luyện"} như bảng dưới, nên so sánh chỉ mang tính tham khảo.`;
}
