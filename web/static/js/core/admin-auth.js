// Thông tin đăng nhập quản trị, dùng chung giữa trang /quan-tri và các nút Bắt đầu/Dừng huấn luyện ở Phụ lục kỹ thuật.
//
// Lưu ở sessionStorage của tab; khi người dùng chọn "Ghi nhớ trên máy này" còn một bản trong localStorage (cùng khóa) nên tab
// khác của cùng trình duyệt dùng lại được. Quyền thật do máy chủ kiểm bằng tiêu đề Authorization; đây chỉ là nơi giữ mật khẩu.
const KEY = "cropfed.admin.token";

export const DEFAULT_USER = "admin";
export const ADMIN_PAGE = "/quan-tri";

/** `tài khoản:mật khẩu` -> base64 an toàn với mọi ký tự (kể cả tiếng Việt): đi qua UTF-8 chứ không dùng btoa trực tiếp. */
export function basicCredential(user, password) {
  const bytes = new TextEncoder().encode(`${user}:${password}`);
  let binary = "";
  bytes.forEach(byte => { binary += String.fromCharCode(byte); });
  return btoa(binary);
}

export function authHeader(user, password) {
  return { Authorization: `Basic ${basicCredential(user || DEFAULT_USER, password)}` };
}

/** {user, password} đã lưu, hoặc null. Chấp nhận cả giá trị cũ (chỉ là chuỗi mật khẩu, tài khoản mặc định). */
export function readCredential() {
  let raw = null;
  try { raw = sessionStorage.getItem(KEY) || localStorage.getItem(KEY); } catch { return null; }
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.p === "string" && parsed.p) return { user: String(parsed.u || DEFAULT_USER), password: parsed.p };
  } catch { /* giá trị cũ: chuỗi thuần */ }
  return { user: DEFAULT_USER, password: raw };
}

export function storeCredential(user, password, remember) {
  try {
    const value = JSON.stringify({ u: user || DEFAULT_USER, p: password });
    sessionStorage.setItem(KEY, value);
    if (remember) localStorage.setItem(KEY, value);
    else localStorage.removeItem(KEY);
  } catch { /* bị chặn */ }
}

/** Có bản ghi nhớ lâu dài (localStorage) không. */
export function isRemembered() {
  try { return !!localStorage.getItem(KEY); } catch { return false; }
}

export function clearCredential() {
  try { sessionStorage.removeItem(KEY); localStorage.removeItem(KEY); } catch { /* bị chặn */ }
}

/** Tiêu đề Authorization từ thông tin đã lưu, hoặc {} nếu chưa đăng nhập. */
export function storedAuthHeader() {
  const saved = readCredential();
  return saved ? authHeader(saved.user, saved.password) : {};
}
