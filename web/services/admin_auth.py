"""Xác thực quản trị dùng chung cho mọi API cần quyền admin (học phân tán cộng đồng và điều khiển huấn luyện).

Nguyên tắc:
- Kiểm ở BACKEND bằng tiêu đề `Authorization` (Basic `tài khoản:mật khẩu` hoặc Bearer `mật khẩu`), không dùng cookie nên
  không có CSRF; trang không bao giờ gửi `WWW-Authenticate` nên trình duyệt không tự đính kèm thông tin đăng nhập.
- Mật khẩu là `COMMUNITY_ADMIN_TOKEN` (>= 12 ký tự), tài khoản là `COMMUNITY_ADMIN_USER` (mặc định "admin").
- Một bộ đếm thất bại duy nhất (`LIMITER`): 5 lần sai / 10 phút / IP và 60 lần / 10 phút toàn cục, dùng chung cho mọi API
  quản trị để kẻ đoán mật khẩu không có nhiều lượt hơn khi đổi sang API khác.
- Chế độ cục bộ: nếu `COMMUNITY_ADMIN_TOKEN` chưa được cấu hình (máy dev), các API huấn luyện chỉ mở cho
  kết nối được nhìn thấy là từ chính máy này (`is_local_client`); request từ IP khác nhận 503 `admin_disabled`.
  Trên máy chủ công khai biến này luôn được đặt nên cả ba API huấn luyện đều đòi đăng nhập.
- LƯU Ý khi chạy sau reverse proxy: `client_ip` đọc `request.client.host`, giá trị này chỉ đúng khi uvicorn
  được bật `--proxy-headers` (deploy/cropfed-demo.service dùng `--proxy-headers --forwarded-allow-ips 127.0.0.1`).
  Thiếu cờ đó thì mọi request đều hiện ra là IP của proxy (thường 127.0.0.1) và bị coi là cục bộ.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import os
from typing import Optional

from fastapi import HTTPException, Request

from web.services.community_store import RateLimiter

ADMIN_MIN_LENGTH = 12
FAIL_PER_IP = 5
FAIL_GLOBAL = 60
FAIL_WINDOW = 600.0

LIMITER = RateLimiter()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# "testclient" là host giả mà Starlette TestClient đặt; mọi request thật đều có IP định tuyến được.
LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def is_local_client(request: Request) -> bool:
    return client_ip(request) in LOCAL_CLIENT_HOSTS


def admin_user() -> str:
    return os.environ.get("COMMUNITY_ADMIN_USER") or "admin"


def configured_password() -> Optional[str]:
    """Mật khẩu quản trị đã cấu hình, hoặc None nếu chưa đặt."""
    return os.environ.get("COMMUNITY_ADMIN_TOKEN") or None


def auth_required() -> bool:
    """Các API huấn luyện có đòi đăng nhập không (có cấu hình mật khẩu quản trị)."""
    return configured_password() is not None


def admin_secret(request: Request) -> Optional[str]:
    """Mật khẩu từ tiêu đề Authorization: `Bearer <mật khẩu>` hoặc `Basic base64(tài khoản:mật khẩu)`.

    Với Basic, tài khoản phải khớp; sai tài khoản coi như không có mật khẩu để thông báo lỗi và bộ đếm thất bại giống hệt
    khi sai mật khẩu (không lộ tài khoản nào tồn tại)."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    scheme = scheme.lower()
    if scheme == "bearer":
        return value.strip() or None
    if scheme == "basic":
        try:
            decoded = base64.b64decode(value.strip(), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        user, _, password = decoded.partition(":")
        if not hmac.compare_digest(user.encode("utf-8"), admin_user().encode("utf-8")):
            return None
        return password or None
    return None


def locked_out(limiter: RateLimiter, ip: str) -> bool:
    return limiter.count(f"adminfail:{ip}", FAIL_WINDOW) >= FAIL_PER_IP or limiter.count("adminfail:*", FAIL_WINDOW) >= FAIL_GLOBAL


def record_failure(limiter: RateLimiter, ip: str) -> None:
    limiter.allow(f"adminfail:{ip}", 10**6, FAIL_WINDOW)
    limiter.allow("adminfail:*", 10**6, FAIL_WINDOW)


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def require_admin(request: Request) -> None:
    """Dependency của FastAPI: chặn nếu chưa đăng nhập admin. Đặt `request.state.admin_user` để ghi nhật ký."""
    expected = configured_password()
    if expected is None:
        if is_local_client(request):
            request.state.admin_user = "local"
            return
        raise _error(
            503,
            "admin_disabled",
            "Chưa cấu hình COMMUNITY_ADMIN_TOKEN nên API huấn luyện chỉ mở cho kết nối từ chính máy này. "
            "Hãy đặt mật khẩu quản trị trước khi cho truy cập từ xa.",
        )
    ip = client_ip(request)
    if locked_out(LIMITER, ip):
        raise _error(429, "locked", "Đăng nhập quản trị sai quá nhiều lần. Hãy thử lại sau ít phút.")
    if len(expected) < ADMIN_MIN_LENGTH:
        raise _error(503, "admin_disabled", f"Quản trị chưa được bật: mật khẩu quản trị phải dài ít nhất {ADMIN_MIN_LENGTH} ký tự.")
    supplied = admin_secret(request)
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        # Chỉ đếm khi CÓ gửi thông tin đăng nhập mà sai (đoán mật khẩu). Yêu cầu không có tiêu đề không phải lần đoán: nếu đếm,
        # bot gọi vu vơ có thể khóa admin thật và đẩy bộ đếm toàn cục.
        if request.headers.get("authorization"):
            record_failure(LIMITER, ip)
        raise _error(401, "bad_admin", "Cần đăng nhập quản trị. Tài khoản hoặc mật khẩu không đúng.")
    header = request.headers.get("authorization", "")
    request.state.admin_user = admin_user() if header.lower().startswith("basic ") else "bearer"
