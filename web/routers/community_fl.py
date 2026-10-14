"""Router học liên kết cộng đồng: mỗi người vào web là một client, huấn luyện lớp cuối trong trình duyệt.

Giao thức (JSON qua HTTPS, ảnh KHÔNG bao giờ rời máy người dùng):
  GET  /api/community/status          công khai: phiên đang mở không, vòng hiện tại, phiên bản mô hình, cấu hình client
  POST /api/community/join            {code} → token phiên (HMAC, hết hạn sau vài giờ)
  GET  /api/community/feature-model   file ONNX đóng băng (tĩnh, SHA-256, cache dài)
  GET  /api/community/round           (token) head cha hiện hành + hạn + số cập nhật đã nhận
  POST /api/community/update          (token) head mới + số ảnh + phân bố nhãn → nhận/từ chối kèm lý do
  GET  /api/community/history         các vòng đã gộp/bị từ chối và các phiên bản
  POST /api/community/predict         ảnh → dự đoán của mô hình cộng đồng (thử nghiệm) so với FedAvg v7
  /api/community/admin/*              (tài khoản COMMUNITY_ADMIN_USER, mặc định "admin" + mật khẩu COMMUNITY_ADMIN_TOKEN; Bearer
                                      <mật khẩu> hoặc Basic) mở/đóng phiên, đặt K/hạn, phát mã, thu hồi, khôi phục

Biến môi trường: COMMUNITY_SECRET (ký token, băm mã; thiếu thì không ai tham gia được), COMMUNITY_ADMIN_TOKEN (thiếu thì
quản trị bị tắt). Phiên mặc định ĐÓNG cho tới khi quản trị mở.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from src import registry as reg
from src.federated.community import CommunityError
from web.services import community_inference
from web.services.community_store import AssetsUnavailable, CommunityAssets, CommunityStore

router = APIRouter(prefix="/api/community", tags=["community"])
log = logging.getLogger("community")

MAX_UPDATE_BODY_BYTES = 320_000  # 155.800 byte float32 → ~207.800 ký tự base64 + phần còn lại
MAX_SMALL_BODY_BYTES = 8_000

HTTP_STATUS = {
    "bad_code": 403, "bad_token": 401, "closed": 409, "disabled": 503, "stale_round": 409, "already_submitted": 409,
    "stale_parent": 409, "too_few": 422, "too_many": 422, "no_change": 422, "non_finite": 422, "bad_labels": 422,
    "bad_shape": 422, "bad_size": 422, "bad_encoding": 422, "bad_structure": 422, "admin_disabled": 503,
    "bad_admin": 401, "no_round": 409, "no_version": 404, "bad_setting": 422, "no_updates": 409,
}


def _scaled(value: int) -> int:
    try:
        scale = max(0.1, float(os.environ.get("COMMUNITY_RATE_SCALE", "1")))
    except ValueError:
        scale = 1.0
    return max(1, int(value * scale))


# Hạn mức theo IP phải rộng: một lớp học/sự kiện thường ra Internet qua MỘT địa chỉ NAT. Chặn thật sự nằm ở hạn mức theo người
# tham gia (token) và ở nginx; COMMUNITY_RATE_SCALE nhân tất cả hạn mức theo IP khi cần (ví dụ 3 cho hội trường 100 người).
PUBLIC_PER_MINUTE = 1500       # status/history/round/... gộp theo IP
JOIN_PER_10_MIN = 80           # lượt nhập mã (kể cả đúng) theo IP
JOIN_FAILS_PER_15_MIN = 30     # nhập sai mã theo IP: quá thì khóa tạm
UPDATES_PER_10_MIN_IP = 400    # lượt gửi cập nhật theo IP
MODEL_DOWNLOADS_PER_10_MIN = 200
PREDICTIONS_PER_MINUTE = 60
PER_PARTICIPANT_PER_MINUTE = 120
UPDATES_PER_10_MIN_PARTICIPANT = 12


# ---------------------------------------------------------------------------
# Store dùng chung (một tiến trình): nạp lười, lỗi artifact được giữ lại để báo rõ lý do
# ---------------------------------------------------------------------------
@dataclass
class StoreState:
    store: Optional[CommunityStore]
    reason: Optional[str]            # công khai (/status): KHÔNG chứa đường dẫn hay chi tiết nội bộ
    detail: Optional[str] = None     # chi tiết cho người vận hành (chỉ ghi nhật ký)


_STATE: Optional[StoreState] = None
_STATE_LOCK = threading.Lock()


PUBLIC_REASON = "Học liên kết cộng đồng chưa được bật trên máy chủ này."


def _unavailable(detail: str) -> StoreState:
    log.warning("community disabled: %s", detail)
    return StoreState(None, PUBLIC_REASON, detail)


def _build_state() -> StoreState:
    try:
        assets = CommunityAssets.from_directory()
    except AssetsUnavailable as exc:
        return _unavailable(str(exc))
    state_dir = Path(os.environ["COMMUNITY_STATE_DIR"]) if os.environ.get("COMMUNITY_STATE_DIR") else reg.abs_path(reg.COMMUNITY_STATE_DIR)
    try:
        store = CommunityStore(
            state_dir, assets,
            secret=os.environ.get("COMMUNITY_SECRET") or None,
            admin_token=os.environ.get("COMMUNITY_ADMIN_TOKEN") or None,
        )
    except (AssetsUnavailable, OSError) as exc:
        return _unavailable(f"cannot initialise community state: {exc}")
    if store.secret_too_short:
        log.warning("COMMUNITY_SECRET is shorter than 16 characters: joining is disabled")
    if store.admin_token_too_short:
        log.warning("COMMUNITY_ADMIN_TOKEN is shorter than 12 characters: admin API is disabled")
    return StoreState(store, None)


def store_state() -> StoreState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = _build_state()
        return _STATE


def reset_store_state() -> None:
    """Dành cho test/triển khai lại: bỏ store đã nạp để lần gọi sau dựng lại từ đĩa."""
    global _STATE
    with _STATE_LOCK:
        _STATE = None


def require_store(state: StoreState = Depends(store_state)) -> CommunityStore:
    if state.store is None:
        raise HTTPException(status_code=503, detail=state.reason or PUBLIC_REASON)
    return state.store


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _fail(error: CommunityError) -> HTTPException:
    return HTTPException(status_code=HTTP_STATUS.get(error.code, 400), detail={"code": error.code, "message": error.message})


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limit(store: CommunityStore, key: str, limit: int, window: float) -> None:
    allowed, wait = store.limiter.allow(key, limit, window)
    if not allowed:
        raise HTTPException(
            status_code=429, detail=f"Bạn thao tác quá nhanh. Hãy thử lại sau {int(wait)} giây.",
            headers={"Retry-After": str(int(wait))},
        )


def _bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header[:7].lower() == "bearer " else None


def _admin_secret(request: Request) -> Optional[str]:
    """Mật khẩu quản trị từ tiêu đề Authorization: `Bearer <mật khẩu>` (curl/script) hoặc `Basic base64(tài khoản:mật khẩu)`
    (trang /quan-tri). Với Basic, tài khoản phải khớp COMMUNITY_ADMIN_USER (mặc định "admin"); sai tài khoản thì coi như
    không có mật khẩu, để thông báo lỗi và bộ đếm thất bại giống hệt khi sai mật khẩu."""
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
        expected = os.environ.get("COMMUNITY_ADMIN_USER") or "admin"
        if not hmac.compare_digest(user.encode("utf-8"), expected.encode("utf-8")):
            return None
        return password or None
    return None


def _participant(request: Request, store: CommunityStore) -> str:
    _limit(store, f"ip:{_client_ip(request)}", _scaled(PUBLIC_PER_MINUTE), 60)
    try:
        participant = store.authenticate(_bearer(request))
    except CommunityError as error:
        raise _fail(error) from error
    _limit(store, f"p:{participant}", PER_PARTICIPANT_PER_MINUTE, 60)
    return participant


async def _read_json(request: Request, limit: int) -> Any:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=f"Yêu cầu vượt giới hạn {limit // 1000} KB.")
    received = bytearray()
    async for chunk in request.stream():
        received.extend(chunk)
        if len(received) > limit:
            raise HTTPException(status_code=413, detail=f"Yêu cầu vượt giới hạn {limit // 1000} KB.")
    try:
        return json.loads(bytes(received))
    except (ValueError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Nội dung yêu cầu không phải JSON hợp lệ.") from exc


def _admin(request: Request, store: CommunityStore) -> None:
    ip = _client_ip(request)
    if store.limiter.count(f"adminfail:{ip}", 600) >= 5 or store.limiter.count("adminfail:*", 600) >= 60:
        raise HTTPException(status_code=429, detail="Đăng nhập quản trị sai quá nhiều lần. Hãy thử lại sau ít phút.")
    try:
        store.check_admin(_admin_secret(request))
    except CommunityError as error:
        if error.code == "bad_admin":
            store.limiter.allow(f"adminfail:{ip}", 10**6, 600)
            store.limiter.allow("adminfail:*", 10**6, 600)
        raise _fail(error) from error


# ---------------------------------------------------------------------------
# Mô hình dữ liệu
# ---------------------------------------------------------------------------
class JoinBody(BaseModel):
    code: str = Field(min_length=4, max_length=40)


class UpdateBody(BaseModel):
    round_id: int
    parent_sha: str = Field(min_length=64, max_length=64)
    head: str = Field(min_length=100, max_length=250_000)
    num_examples: int = Field(ge=0, le=1_000_000)
    label_counts: list[int] = Field(min_length=38, max_length=38)
    local: Optional[Dict[str, Any]] = None
    dp_applied: bool = False


LOCAL_KEYS = ("epochs", "lr", "batch", "n_train", "n_holdout", "acc_before", "acc_after", "loss_before", "loss_after",
              "extract_ms", "train_ms")


def _clean_local(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in LOCAL_KEYS:
        value = (raw or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) < 1e12:
            out[key] = float(value)
    return out


class SettingsBody(BaseModel):
    open: Optional[bool] = None
    changes: Dict[str, Any] = Field(default_factory=dict)


class CodesBody(BaseModel):
    count: int = Field(default=1, ge=1, le=500)
    max_uses: int = Field(default=3, ge=1, le=100000)
    ttl_hours: float = Field(default=24, gt=0, le=24 * 365)
    label: str = Field(default="", max_length=80)


class RollbackBody(BaseModel):
    version: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Công khai
# ---------------------------------------------------------------------------
@router.get("/status")
def get_status(request: Request, state: StoreState = Depends(store_state)) -> Dict[str, Any]:
    """Không cần token. Khi chưa bật (thiếu artifact) vẫn trả 200 kèm `enabled=false` để giao diện hiện lý do, không báo lỗi."""
    if state.store is None:
        return {"enabled": False, "open": False, "joinable": False, "reason": state.reason}
    _limit(state.store, f"ip:{_client_ip(request)}", _scaled(PUBLIC_PER_MINUTE), 60)
    return state.store.public_status()


@router.post("/join")
async def join(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    ip = _client_ip(request)
    if store.limiter.count(f"joinfail:{ip}", 900) >= _scaled(JOIN_FAILS_PER_15_MIN):
        raise HTTPException(status_code=429, detail="Nhập sai mã quá nhiều lần. Hãy thử lại sau ít phút.",
                            headers={"Retry-After": "300"})
    _limit(store, f"join:{ip}", _scaled(JOIN_PER_10_MIN), 600)
    raw = await _read_json(request, MAX_SMALL_BODY_BYTES)
    try:
        body = JoinBody.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Hãy nhập mã tham gia (khoảng 10 ký tự).") from exc
    try:
        return await run_in_threadpool(store.join, body.code, ip)
    except CommunityError as error:
        if error.code == "bad_code":
            store.limiter.allow(f"joinfail:{ip}", 10**6, 900)
        raise _fail(error) from error


@router.get("/feature-model")
def get_feature_model(request: Request, state: StoreState = Depends(store_state)) -> FileResponse:
    store = require_store(state)
    _limit(store, f"fm:{_client_ip(request)}", _scaled(MODEL_DOWNLOADS_PER_10_MIN), 600)
    assets = store.assets
    if assets.feature_model_path is None or not assets.feature_model_path.is_file():
        raise HTTPException(status_code=503, detail="Máy chủ chưa có file mô hình đặc trưng.")
    return FileResponse(
        assets.feature_model_path, media_type="application/octet-stream", filename="feature_model.onnx",
        headers={"Cache-Control": "public, max-age=31536000, immutable", "ETag": f'"{assets.feature_model_sha}"',
                 "X-Content-SHA256": str(assets.feature_model_sha)},
    )


@router.get("/round")
def get_round(request: Request, have: Optional[str] = Query(None, max_length=64),
              store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    participant = _participant(request, store)
    return store.round_view(participant, have)


@router.post("/update")
async def post_update(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    participant = await run_in_threadpool(_participant, request, store)
    _limit(store, f"upd:{participant}", UPDATES_PER_10_MIN_PARTICIPANT, 600)
    _limit(store, f"updip:{_client_ip(request)}", _scaled(UPDATES_PER_10_MIN_IP), 600)
    raw = await _read_json(request, MAX_UPDATE_BODY_BYTES)
    try:
        body = UpdateBody.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()))
        raise HTTPException(status_code=422, detail=f"Dữ liệu gửi lên không hợp lệ ({field or 'body'}).") from exc
    payload = body.model_dump()
    payload["local"] = _clean_local(body.local)
    try:
        return await run_in_threadpool(store.submit_update, participant, payload)
    except CommunityError as error:
        raise _fail(error) from error


@router.get("/history")
def get_history(request: Request, limit: int = Query(20, ge=1, le=200),
                store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    _limit(store, f"ip:{_client_ip(request)}", _scaled(PUBLIC_PER_MINUTE), 60)
    return store.history(limit)


@router.post("/predict")
def post_predict(request: Request, file: UploadFile = File(...), store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    from web.routers.inference import _read_image  # cùng giới hạn dung lượng/điểm ảnh với trang Chẩn đoán

    _limit(store, f"pred:{_client_ip(request)}", _scaled(PREDICTIONS_PER_MINUTE), 60)
    image = _read_image(file)
    version, head, sha, metrics = store.current_head()
    try:
        return community_inference.predict(image, version, head, sha, store.base_head(), metrics, store.base_metrics())
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Mô hình nền chưa sẵn sàng: {exc}") from exc


# ---------------------------------------------------------------------------
# Quản trị
# ---------------------------------------------------------------------------
@router.get("/admin/state")
def admin_state(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    _admin(request, store)
    return store.admin_state()


async def _admin_body(request: Request, store: CommunityStore, model: Any) -> Any:
    """Xác thực quản trị TRƯỚC, rồi mới đọc thân (tối đa 8 KB) và kiểm kiểu: người lạ không bắt máy chủ parse thân lớn."""
    _admin(request, store)
    raw = await _read_json(request, MAX_SMALL_BODY_BYTES)
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Dữ liệu quản trị không hợp lệ.") from exc


@router.post("/admin/settings")
async def admin_settings(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    body = await _admin_body(request, store, SettingsBody)
    try:
        return await run_in_threadpool(store.admin_configure, body.open, body.changes)
    except CommunityError as error:
        raise _fail(error) from error


@router.post("/admin/codes")
async def admin_codes(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    body = await _admin_body(request, store, CodesBody)
    try:
        codes = await run_in_threadpool(store.admin_create_codes, body.count, body.max_uses, body.ttl_hours, body.label)
    except CommunityError as error:
        raise _fail(error) from error
    return {"codes": codes, "max_uses": body.max_uses, "ttl_hours": body.ttl_hours,
            "note": "Mã chỉ hiện một lần; máy chủ chỉ lưu giá trị băm. Mỗi mã chỉ gửi được một cập nhật mỗi vòng."}


@router.post("/admin/codes/{code_id}/revoke")
def admin_revoke_code(code_id: int, request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    _admin(request, store)
    store.admin_revoke_code(code_id)
    return {"revoked": code_id}


@router.post("/admin/participants/{suffix}/revoke")
def admin_revoke_participant(suffix: str, request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    _admin(request, store)
    return {"revoked": store.admin_revoke_participant(suffix)}


@router.post("/admin/round/close")
def admin_close_round(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    _admin(request, store)
    try:
        return store.admin_close_round()
    except CommunityError as error:
        raise _fail(error) from error


@router.post("/admin/rollback")
async def admin_rollback(request: Request, store: CommunityStore = Depends(require_store)) -> Dict[str, Any]:
    body = await _admin_body(request, store, RollbackBody)
    try:
        return await run_in_threadpool(store.admin_rollback, body.version)
    except CommunityError as error:
        raise _fail(error) from error
