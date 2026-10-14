// Web Worker của học liên kết cộng đồng: mọi việc nặng chạy ở đây để giao diện không giật.
//   init     tải/kiểm SHA-256 mô hình đặc trưng ONNX (có cache), tạo phiên onnxruntime-web (wasm, 1 luồng)
//   extract  ảnh -> tensor canonical_v1 (resize kiểu PIL) -> vector 1024 chiều; giữ trong bộ nhớ worker, ảnh không đi đâu cả
//   train    softmax regression cho lớp cuối trên đặc trưng đã giữ, đo trước/sau trên phần ảnh giữ lại
import {
  FEATURE_DIM, INPUT_SIZE, NUM_CLASSES, deltaNorm, evaluateHead, flattenOnBlack, privatize, resizeRgbaToRgb, rgbToTensor,
  sha256Hex, splitHoldout, trainSoftmax,
} from "../core/community-math.js";

const ORT_MODULE = "/static/vendor/ort/ort.wasm.min.mjs";
const ORT_BASE = "/static/vendor/ort/";
const CACHE_NAME = "cropfed-community-v1";
const MAX_CANVAS_PIXELS = 16_000_000; // iOS Safari giới hạn canvas ~16,7 triệu điểm ảnh

let ort = null;
let session = null;
const features = new Map(); // khóa ảnh -> Float32Array(1024)
let queue = Promise.resolve();

const post = (message, transfer = []) => self.postMessage(message, transfer);

class UserFacingError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

// ───────── Mô hình ─────────
async function readWithProgress(response, expectedBytes) {
  const total = Number(response.headers.get("content-length")) || expectedBytes || 0;
  if (!response.body || !response.body.getReader) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const parts = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.length;
    post({ type: "progress", phase: "download", done: received, total });
  }
  const merged = new Uint8Array(received);
  let offset = 0;
  for (const part of parts) { merged.set(part, offset); offset += part.length; }
  return merged;
}

async function verified(bytes, sha256) {
  if (!sha256) return true;
  if (!self.crypto || !self.crypto.subtle) return true; // ngữ cảnh không an toàn (http): bỏ qua, máy chủ vẫn kiểm đầu ra
  return (await sha256Hex(bytes)) === sha256;
}

async function loadModelBytes({ url, sha256, bytes: expected }) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* Cache API bị chặn: tải mỗi lần */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const cached = new Uint8Array(await hit.arrayBuffer());
      if (await verified(cached, sha256)) {
        post({ type: "progress", phase: "download", done: cached.length, total: cached.length, cached: true });
        return cached;
      }
      await cache.delete(url);
    }
  }
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new UserFacingError("download", `Không tải được mô hình đặc trưng (HTTP ${response.status}).`);
  const bytes = await readWithProgress(response, expected);
  if (!(await verified(bytes, sha256))) {
    throw new UserFacingError("sha", "Mô hình tải về không khớp mã SHA-256 mà máy chủ công bố. Đã dừng để an toàn.");
  }
  if (cache) { try { await cache.put(url, new Response(bytes)); } catch { /* hết dung lượng: bỏ qua */ } }
  return bytes;
}

async function init({ model }) {
  const started = performance.now();
  if (typeof OffscreenCanvas === "undefined" || typeof createImageBitmap === "undefined") {
    throw new UserFacingError("unsupported", "Trình duyệt này chưa hỗ trợ xử lý ảnh nền (OffscreenCanvas). Hãy dùng bản Chrome, Edge, Firefox hoặc Safari mới.");
  }
  if (session) { post({ type: "ready", ms: 0, reused: true }); return; }
  ort = await import(ORT_MODULE);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1; // một luồng: không cần cô lập cross-origin (COOP/COEP)
  ort.env.wasm.proxy = false;
  const bytes = await loadModelBytes(model);
  post({ type: "progress", phase: "session", done: 0, total: 1 });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  post({ type: "ready", ms: Math.round(performance.now() - started), input: session.inputNames[0], output: session.outputNames[0] });
}

// ───────── Ảnh -> đặc trưng ─────────
async function decodeToRgba(blob) {
  const options = { imageOrientation: "from-image", colorSpaceConversion: "none" };
  let bitmap = await createImageBitmap(blob, options);
  const pixels = bitmap.width * bitmap.height;
  if (pixels > MAX_CANVAS_PIXELS) {
    const scale = Math.sqrt(MAX_CANVAS_PIXELS / pixels);
    const resized = await createImageBitmap(blob, {
      ...options, resizeWidth: Math.max(1, Math.floor(bitmap.width * scale)), resizeHeight: Math.max(1, Math.floor(bitmap.height * scale)),
      resizeQuality: "high",
    });
    bitmap.close();
    bitmap = resized;
  }
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const image = context.getImageData(0, 0, width, height);
  return { rgba: flattenOnBlack(image.data), width, height };
}

async function extract({ items }) {
  if (!session) throw new UserFacingError("not_ready", "Mô hình đặc trưng chưa sẵn sàng.");
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  for (let index = 0; index < items.length; index++) {
    const { key, file } = items[index];
    const started = performance.now();
    try {
      const { rgba, width, height } = await decodeToRgba(file);
      const tensor = rgbToTensor(resizeRgbaToRgb(rgba, width, height, INPUT_SIZE, INPUT_SIZE));
      const result = await session.run({ [inputName]: new ort.Tensor("float32", tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]) });
      const vector = Float32Array.from(result[outputName].data);
      if (vector.length !== FEATURE_DIM) throw new Error(`Đặc trưng có ${vector.length} chiều, cần ${FEATURE_DIM}.`);
      features.set(key, vector);
      post({ type: "feature", key, index: index + 1, total: items.length, ms: Math.round(performance.now() - started) });
    } catch (error) {
      post({ type: "feature_error", key, index: index + 1, total: items.length, message: error && error.message ? error.message : String(error) });
    }
  }
  post({ type: "extract_done", cached: features.size });
}

// ───────── Huấn luyện ─────────
function secureUniform() {
  const buffer = new Uint32Array(1);
  self.crypto.getRandomValues(buffer);
  return (buffer[0] + 0.5) / 4294967296;
}

function train({ keys, labels, weight, bias, config }) {
  const started = performance.now();
  const missing = keys.filter(key => !features.has(key));
  if (missing.length) throw new UserFacingError("missing_features", `Còn ${missing.length} ảnh chưa được trích đặc trưng.`);
  const n = keys.length;
  const labelArray = Int32Array.from(labels);
  const { train: trainIndex, holdout: holdIndex } = splitHoldout(labelArray, config.holdout_fraction, config.seed ?? 7);
  const gather = indices => {
    const matrix = new Float32Array(indices.length * FEATURE_DIM);
    indices.forEach((source, row) => matrix.set(features.get(keys[source]), row * FEATURE_DIM));
    return { features: matrix, labels: Int32Array.from(indices, source => labelArray[source]) };
  };
  const learn = gather(trainIndex);
  const hold = gather(holdIndex);
  const before = evaluateHead({ ...hold, weight, bias });
  const trainBefore = evaluateHead({ ...learn, weight, bias });
  const epochs = config.epochs;
  const result = trainSoftmax({
    ...learn, weight, bias, epochs, lr: config.lr, batch: config.batch, seed: config.seed ?? 1,
    onEpoch: (done, loss) => post({ type: "progress", phase: "train", done, total: epochs, loss }),
  });
  let { weight: w, bias: b } = result;
  const after = evaluateHead({ ...hold, weight: w, bias: b });
  const trainAfter = evaluateHead({ ...learn, weight: w, bias: b });
  const rawNorm = deltaNorm(weight, bias, w, b);
  let clipped = false, dpApplied = false;
  if (config.dp && config.dp.enabled) {
    const out = privatize({ weight0: weight, bias0: bias, weight: w, bias: b, clip: config.dp.clip, noiseMultiplier: config.dp.noise_multiplier, uniform: secureUniform });
    w = out.weight; b = out.bias; clipped = out.clipped; dpApplied = true;
  }
  const counts = new Array(NUM_CLASSES).fill(0);
  for (const label of learn.labels) counts[label]++;
  post({
    type: "trained", weight: w, bias: b, nTrain: trainIndex.length, nHoldout: holdIndex.length, total: n, labelCounts: counts,
    accBefore: before.accuracy, accAfter: after.accuracy, lossBefore: before.loss, lossAfter: after.loss,
    trainAccBefore: trainBefore.accuracy, trainAccAfter: trainAfter.accuracy,
    norm: rawNorm, normSent: deltaNorm(weight, bias, w, b), clipped, dpApplied, lossHistory: result.lossHistory,
    trainMs: Math.round(performance.now() - started),
  }, [w.buffer, b.buffer]);
}

function forget({ keys }) {
  if (!keys) features.clear();
  else keys.forEach(key => features.delete(key));
  post({ type: "forgotten", cached: features.size });
}

const HANDLERS = { init, extract, train, forget };

self.onmessage = event => {
  const message = event.data || {};
  queue = queue.then(async () => {
    try {
      const handler = HANDLERS[message.type];
      if (!handler) throw new UserFacingError("unknown", `Lệnh không hợp lệ: ${message.type}`);
      await handler(message);
    } catch (error) {
      post({ type: "error", for: message.type, code: error.code || "failed", message: error && error.message ? error.message : String(error) });
    }
  });
};
