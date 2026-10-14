// Phần số học của học liên kết cộng đồng, thuần JS (không DOM) để chạy trong Web Worker và kiểm bằng Node:
// tiền xử lý ảnh trùng canonical_v1 của PyTorch, huấn luyện softmax regression cho lớp cuối, mã hóa head.
//
// Vì sao tự viết resize: `createImageBitmap(..., { resizeQuality: "high" })` KHÔNG khớp torchvision Resize(antialias=True)
// (đo trên 49 ảnh: cos đặc trưng tối thiểu 0,948, 2/49 dự đoán đổi). Bộ lọc tam giác có co giãn của PIL, cố định 22 bit,
// khớp từng byte; tests/test_community_js.py kiểm đúng điều này so với Pillow.

export const NUM_CLASSES = 38;
export const FEATURE_DIM = 1024;
export const INPUT_SIZE = 224;
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];
const PRECISION_BITS = 22;

// ───────── Tiền xử lý ─────────
function pilCoefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale; // bộ lọc tam giác có support 1,0
  const invScale = 1 / filterScale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const coeffs = new Int32Array(outSize * ksize);
  for (let out = 0; out < outSize; out++) {
    const center = (out + 0.5) * scale;
    let first = Math.trunc(center - support + 0.5);
    if (first < 0) first = 0;
    let last = Math.trunc(center + support + 0.5);
    if (last > inSize) last = inSize;
    const count = last - first;
    const weights = new Float64Array(count);
    let total = 0;
    for (let x = 0; x < count; x++) {
      const d = Math.abs((x + first - center + 0.5) * invScale);
      const w = d < 1 ? 1 - d : 0;
      weights[x] = w;
      total += w;
    }
    for (let x = 0; x < count; x++) {
      const w = total === 0 ? weights[x] : weights[x] / total;
      coeffs[out * ksize + x] = Math.trunc(w * (1 << PRECISION_BITS) + (w < 0 ? -0.5 : 0.5));
    }
    bounds[out * 2] = first;
    bounds[out * 2 + 1] = count;
  }
  return { bounds, coeffs, ksize };
}

const clip8 = v => (v < 0 ? 0 : v > 255 ? 255 : v);

/** Ảnh RGBA (đã có nền) -> RGB 8 bit kích thước đích, bilinear + antialias đúng thuật toán PIL (ngang trước, dọc sau). */
export function resizeRgbaToRgb(rgba, width, height, outWidth = INPUT_SIZE, outHeight = INPUT_SIZE) {
  const horizontal = pilCoefficients(width, outWidth);
  const vertical = pilCoefficients(height, outHeight);
  const half = 1 << (PRECISION_BITS - 1);
  const middle = new Uint8Array(outWidth * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < outWidth; x++) {
      const first = horizontal.bounds[x * 2];
      const count = horizontal.bounds[x * 2 + 1];
      const base = x * horizontal.ksize;
      let r = half, g = half, b = half;
      for (let k = 0; k < count; k++) {
        const w = horizontal.coeffs[base + k];
        const p = (y * width + first + k) * 4;
        r += rgba[p] * w; g += rgba[p + 1] * w; b += rgba[p + 2] * w;
      }
      const o = (y * outWidth + x) * 3;
      middle[o] = clip8(r >> PRECISION_BITS);
      middle[o + 1] = clip8(g >> PRECISION_BITS);
      middle[o + 2] = clip8(b >> PRECISION_BITS);
    }
  }
  const out = new Uint8Array(outWidth * outHeight * 3);
  for (let y = 0; y < outHeight; y++) {
    const first = vertical.bounds[y * 2];
    const count = vertical.bounds[y * 2 + 1];
    const base = y * vertical.ksize;
    for (let x = 0; x < outWidth; x++) {
      let r = half, g = half, b = half;
      for (let k = 0; k < count; k++) {
        const w = vertical.coeffs[base + k];
        const p = ((first + k) * outWidth + x) * 3;
        r += middle[p] * w; g += middle[p + 1] * w; b += middle[p + 2] * w;
      }
      const o = (y * outWidth + x) * 3;
      out[o] = clip8(r >> PRECISION_BITS);
      out[o + 1] = clip8(g >> PRECISION_BITS);
      out[o + 2] = clip8(b >> PRECISION_BITS);
    }
  }
  return out;
}

/** Ảnh có kênh alpha được đặt lên nền ĐEN như bộ tải ảnh của nhóm train (PIL: paste với mask). Sửa tại chỗ. */
export function flattenOnBlack(rgba) {
  for (let i = 3; i < rgba.length; i += 4) {
    const a = rgba[i];
    if (a === 255) continue;
    for (let c = 1; c <= 3; c++) {
      const tmp = rgba[i - c] * a + 128;
      rgba[i - c] = ((tmp >> 8) + tmp) >> 8;
    }
    rgba[i] = 255;
  }
  return rgba;
}

/** RGB HWC uint8 -> tensor float32 NCHW đã chuẩn hóa ImageNet (ToTensor + Normalize). */
export function rgbToTensor(rgb, size = INPUT_SIZE) {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  const mean = MEAN.map(Math.fround); // PyTorch giữ mean/std ở float32
  const std = STD.map(Math.fround);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = Math.fround(Math.fround(Math.fround(rgb[i * 3 + c] / 255) - mean[c]) / std[c]);
    }
  }
  return out;
}

// ───────── Ngẫu nhiên có hạt giống (tái lập được trong test) ─────────
export function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffleInPlace(array, rng) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = array[i]; array[i] = array[j]; array[j] = tmp;
  }
  return array;
}

/** Gaussian chuẩn từ nguồn ngẫu nhiên `uniform()` (Box–Muller). */
export function gaussian(uniform) {
  let u = 0;
  while (u === 0) u = uniform();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
}

// ───────── Softmax regression cho lớp cuối ─────────
/**
 * features: Float32Array (n × dim) hàng liên tiếp; labels: Int32Array(n); weight: Float32Array(classes × dim); bias: Float32Array(classes).
 * Mini-batch SGD theo thứ tự xáo mỗi epoch (hạt giống cố định). Không sửa weight/bias đầu vào; trả bản mới.
 */
export function trainSoftmax({ features, labels, weight, bias, dim = FEATURE_DIM, classes = NUM_CLASSES, epochs, lr, batch, seed = 1, onEpoch }) {
  const n = labels.length;
  const w = Float32Array.from(weight);
  const b = Float32Array.from(bias);
  const rng = makeRng(seed);
  const order = Array.from({ length: n }, (_unused, i) => i);
  const logits = new Float64Array(classes);
  const gradW = new Float64Array(classes * dim);
  const gradB = new Float64Array(classes);
  const history = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleInPlace(order, rng);
    let lossSum = 0;
    for (let start = 0; start < n; start += batch) {
      const end = Math.min(start + batch, n);
      const size = end - start;
      gradW.fill(0); gradB.fill(0);
      for (let s = start; s < end; s++) {
        const row = order[s] * dim;
        let max = -Infinity;
        for (let c = 0; c < classes; c++) {
          let z = b[c];
          const o = c * dim;
          for (let d = 0; d < dim; d++) z += w[o + d] * features[row + d];
          logits[c] = z;
          if (z > max) max = z;
        }
        let sum = 0;
        for (let c = 0; c < classes; c++) { logits[c] = Math.exp(logits[c] - max); sum += logits[c]; }
        const target = labels[order[s]];
        lossSum += -Math.log(Math.max(logits[target] / sum, 1e-12));
        for (let c = 0; c < classes; c++) {
          const g = (logits[c] / sum - (c === target ? 1 : 0)) / size;
          gradB[c] += g;
          const o = c * dim;
          for (let d = 0; d < dim; d++) gradW[o + d] += g * features[row + d];
        }
      }
      for (let k = 0; k < w.length; k++) w[k] -= lr * gradW[k];
      for (let c = 0; c < classes; c++) b[c] -= lr * gradB[c];
    }
    history.push(lossSum / Math.max(n, 1));
    if (onEpoch) onEpoch(epoch + 1, history[history.length - 1]);
  }
  return { weight: w, bias: b, lossHistory: history };
}

/** Độ chính xác và loss trung bình trên một tập đặc trưng. */
export function evaluateHead({ features, labels, weight, bias, dim = FEATURE_DIM, classes = NUM_CLASSES }) {
  const n = labels.length;
  if (n === 0) return { accuracy: null, loss: null, count: 0 };
  let correct = 0, loss = 0;
  const logits = new Float64Array(classes);
  for (let i = 0; i < n; i++) {
    const row = i * dim;
    let max = -Infinity, arg = 0;
    for (let c = 0; c < classes; c++) {
      let z = bias[c];
      const o = c * dim;
      for (let d = 0; d < dim; d++) z += weight[o + d] * features[row + d];
      logits[c] = z;
      if (z > max) { max = z; arg = c; }
    }
    let sum = 0;
    for (let c = 0; c < classes; c++) sum += Math.exp(logits[c] - max);
    loss += -Math.log(Math.max(Math.exp(logits[labels[i]] - max) / sum, 1e-12));
    if (arg === labels[i]) correct++;
  }
  return { accuracy: correct / n, loss: loss / n, count: n };
}

/** Tách một phần ảnh giữ lại để đo trước/sau trung thực; chia đều theo lớp (mỗi lớp ≥ 2 ảnh góp 1 ảnh trở lên vào phần giữ lại). */
export function splitHoldout(labels, fraction, seed = 7, minTotal = 15) {
  const n = labels.length;
  const all = Array.from({ length: n }, (_unused, i) => i);
  if (n < minTotal || !(fraction > 0)) return { train: all, holdout: [] };
  const rng = makeRng(seed);
  const byClass = new Map();
  for (const i of all) {
    if (!byClass.has(labels[i])) byClass.set(labels[i], []);
    byClass.get(labels[i]).push(i);
  }
  const holdout = [];
  for (const members of byClass.values()) {
    if (members.length < 2) continue; // lớp chỉ có một ảnh: giữ cho phần học
    shuffleInPlace(members, rng);
    holdout.push(...members.slice(0, Math.max(1, Math.round(members.length * fraction))));
  }
  const held = new Set(holdout);
  const train = all.filter(i => !held.has(i));
  if (train.length === 0 || holdout.length === 0) return { train: all, holdout: [] };
  return { train, holdout };
}

export function deltaNorm(weight0, bias0, weight1, bias1) {
  let sum = 0;
  for (let i = 0; i < weight0.length; i++) { const d = weight1[i] - weight0[i]; sum += d * d; }
  for (let i = 0; i < bias0.length; i++) { const d = bias1[i] - bias0[i]; sum += d * d; }
  return Math.sqrt(sum);
}

/**
 * Riêng tư tùy chọn ở phía client: cắt chuẩn delta tại `clip` rồi cộng nhiễu Gaussian để chuẩn L2 kỳ vọng của nhiễu bằng
 * `noiseMultiplier × clip`. Không phải bảo đảm differential privacy hình thức (chưa có bộ kế toán ε).
 */
export function privatize({ weight0, bias0, weight, bias, clip, noiseMultiplier, uniform }) {
  const norm = deltaNorm(weight0, bias0, weight, bias);
  const scale = norm > clip ? clip / norm : 1;
  const dims = weight.length + bias.length;
  const sigma = (noiseMultiplier * clip) / Math.sqrt(dims);
  const w = new Float32Array(weight.length);
  const b = new Float32Array(bias.length);
  for (let i = 0; i < w.length; i++) w[i] = weight0[i] + (weight[i] - weight0[i]) * scale + (sigma > 0 ? sigma * gaussian(uniform) : 0);
  for (let i = 0; i < b.length; i++) b[i] = bias0[i] + (bias[i] - bias0[i]) * scale + (sigma > 0 ? sigma * gaussian(uniform) : 0);
  return { weight: w, bias: b, clipped: scale < 1, normBefore: norm };
}

// ───────── Mã hóa trên đường truyền: float32 little-endian, weight (hàng liên tiếp) rồi bias, base64 ─────────
export function encodeHead(weight, bias) {
  const bytes = new Uint8Array((weight.length + bias.length) * 4);
  const view = new DataView(bytes.buffer);
  let o = 0;
  for (let i = 0; i < weight.length; i++, o += 4) view.setFloat32(o, weight[i], true);
  for (let i = 0; i < bias.length; i++, o += 4) view.setFloat32(o, bias[i], true);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export function decodeHead(text, classes = NUM_CLASSES, dim = FEATURE_DIM) {
  const binary = atob(text);
  const expected = (classes * dim + classes) * 4;
  if (binary.length !== expected) throw new Error(`Head có ${binary.length} byte, cần ${expected}.`);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const weight = new Float32Array(classes * dim);
  const bias = new Float32Array(classes);
  let o = 0;
  for (let i = 0; i < weight.length; i++, o += 4) weight[i] = view.getFloat32(o, true);
  for (let i = 0; i < bias.length; i++, o += 4) bias[i] = view.getFloat32(o, true);
  return { weight, bias };
}

export async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
}
