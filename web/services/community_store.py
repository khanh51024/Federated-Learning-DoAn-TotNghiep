"""Trạng thái và điều phối vòng của học liên kết cộng đồng (SQLite + file head theo phiên bản).

Một tiến trình uvicorn duy nhất: mọi thay đổi trạng thái đi qua `self._lock` (RLock) và một giao dịch SQLite, nên một vòng
chỉ được gộp đúng một lần và khởi động lại không làm mất vòng. Logic số học nằm ở src/federated/community.py.

Vòng đời một vòng: mở khi có người cần (head cha = phiên bản hiện hành) → nhận tối đa một cập nhật hợp lệ mỗi người →
đủ K thì gộp ngay; hết hạn mà có ≥ K_min thì gộp, chưa đủ thì gia hạn → head ứng viên được chấm trên tập đặc trưng giám sát
và chỉ được nhận nếu không thoái lui (`gate_decision`); bị từ chối thì giữ head cũ và ghi lý do.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src import registry as reg
from src.federated import community as core
from src.federated.community import CommunityError, EvalSet, Head, Policy

TOKEN_TTL_SECONDS = 6 * 3600
MAX_IMAGES_HINT = 300          # gợi ý trần số ảnh mỗi lần cho giao diện (trọng số máy chủ bị chặn ở policy.n_cap)
MAX_EXAMPLES_ACCEPTED = 5000   # số ảnh client khai báo vượt mức này bị từ chối (chặn khai man số lớn)
EVENTS_KEPT = 3000             # sự kiện quan trọng (vòng, quản trị, tham gia thành công)
NOISY_EVENTS = ("join_failed", "update_rejected")
NOISY_EVENTS_KEPT = 1000       # sự kiện nhiễu (nhập sai mã, cập nhật bị từ chối): không được đẩy sự kiện quan trọng ra
HEADS_KEPT = 200               # chỉ giữ file head của ngần này phiên bản mới nhất (+ phiên bản gốc); hàng CSDL vẫn còn
HISTORY_VERSIONS = 60          # số phiên bản gần nhất trả về ở /history
SECRET_MIN_LENGTH = 16
ADMIN_TOKEN_MIN_LENGTH = 12   # mật khẩu quản trị; hạn mức thử sai (5/10 phút/IP, 60/10 phút toàn cục) lo phần còn lại
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"  # bỏ ký tự dễ nhầm (0/O, 1/I/L, U)
CODE_LENGTH = 10


class AssetsUnavailable(RuntimeError):
    """Thiếu hoặc sai artifact (ONNX / đặc trưng giám sát / head khởi điểm): cộng đồng bị tắt, không đoán."""


# ---------------------------------------------------------------------------
# Tài sản bất biến: head khởi điểm, tập giám sát, file ONNX (đều kiểm SHA-256 theo manifest)
# ---------------------------------------------------------------------------
@dataclass
class CommunityAssets:
    classes: List[str]
    base_head: Head
    evalset: EvalSet
    manifest: Dict[str, Any]
    feature_model_path: Optional[Path] = None
    feature_model_sha: Optional[str] = None
    feature_model_bytes: int = 0

    @classmethod
    def from_directory(cls, directory: Optional[Path] = None) -> "CommunityAssets":
        root = directory or reg.abs_path(reg.COMMUNITY_FL_DIR)
        manifest_path = root / Path(reg.COMMUNITY_MANIFEST_JSON).name
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AssetsUnavailable(
                f"Chưa có {reg.COMMUNITY_MANIFEST_JSON} — chạy scripts/export_community_feature_model.py và "
                "scripts/build_community_eval_features.py rồi chép thư mục lên máy chủ."
            ) from exc
        feature = manifest.get("feature_model") or {}
        head_info = manifest.get("base_head") or {}
        evals = manifest.get("eval_features") or {}
        classes = manifest.get("classes") or []
        if len(classes) != core.NUM_CLASSES or not feature or not head_info or not evals:
            raise AssetsUnavailable("community_manifest.json thiếu mục feature_model / base_head / eval_features / classes.")
        if evals.get("feature_model_sha256") != feature.get("sha256"):
            raise AssetsUnavailable("Đặc trưng giám sát được sinh bằng một file ONNX khác — chạy lại build_community_eval_features.py.")

        onnx_path = root / feature["file"]
        try:
            if core.file_sha256(onnx_path) != feature["sha256"]:
                raise AssetsUnavailable(f"{onnx_path.name}: SHA-256 khác manifest.")
            head_path = root / head_info["file"]
            if core.file_sha256(head_path) != head_info["sha256"]:
                raise AssetsUnavailable(f"{head_path.name}: SHA-256 khác manifest.")
            with np.load(head_path, allow_pickle=False) as data:
                base_head = core.as_head(data["weight"], data["bias"])
            if core.head_sha(base_head) != head_info["head_sha"]:
                raise AssetsUnavailable("Head khởi điểm có hash khác manifest.")
            evalset = core.load_eval_set(root / evals["file"], evals["sha256"])
        except AssetsUnavailable:
            raise
        except (OSError, ValueError, KeyError, CommunityError) as exc:
            raise AssetsUnavailable(f"Không nạp được artifact cộng đồng: {exc}") from exc
        if len(evalset.labels) != int(evals["count"]):
            raise AssetsUnavailable("Số mẫu của tập giám sát khác manifest.")
        if not (evalset.sources == core.SOURCE_PLANTDOC).any() or not (evalset.sources == core.SOURCE_PLANTVILLAGE).any():
            raise AssetsUnavailable("Tập giám sát phải có cả ảnh PlantVillage lẫn PlantDoc.")
        return cls(
            classes=list(classes), base_head=base_head, evalset=evalset, manifest=manifest,
            feature_model_path=onnx_path, feature_model_sha=feature["sha256"], feature_model_bytes=int(feature["bytes"]),
        )


# ---------------------------------------------------------------------------
# Giới hạn tần suất trong tiến trình (nginx lo lớp ngoài; đây là lớp ứng dụng, một worker duy nhất)
# ---------------------------------------------------------------------------
class RateLimiter:
    MAX_KEYS = 100_000          # trần cứng: bị quét bằng nhiều IPv6 cũng không phình bộ nhớ
    PRUNE_EVERY_SECONDS = 30.0

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._window: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_prune = 0.0

    def _prune(self, now: float) -> None:
        """Bỏ khóa mà MỌI lần ghi đều đã quá cửa sổ riêng của khóa đó; chỉ chạy tối đa mỗi 30 giây (không O(n) mỗi lệnh gọi)."""
        if now - self._last_prune < self.PRUNE_EVERY_SECONDS and len(self._hits) < self.MAX_KEYS:
            return
        self._last_prune = now
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= now - self._window.get(k, 0.0)]:
            self._hits.pop(key, None)
            self._window.pop(key, None)
        if len(self._hits) >= self.MAX_KEYS:  # vẫn đầy: bỏ nửa số khóa cũ nhất
            ordered = sorted(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0.0)
            for key in ordered[: len(ordered) // 2]:
                self._hits.pop(key, None)
                self._window.pop(key, None)

    def allow(self, key: str, limit: int, window: float) -> Tuple[bool, float]:
        """Ghi một lần gọi và cho biết có còn trong hạn mức không; trả thêm số giây phải đợi nếu bị chặn."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            hits = self._hits.setdefault(key, deque())
            self._window[key] = max(window, self._window.get(key, 0.0))
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1.0, hits[0] + window - now)
            hits.append(now)
            return True, 0.0

    def count(self, key: str, window: float) -> int:
        """Số lần đã ghi trong cửa sổ (không ghi thêm): dùng để chặn sau N lần thất bại."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 0
            while hits and hits[0] <= now - window:
                hits.popleft()
            return len(hits)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._window.clear()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def normalize_code(raw: str) -> str:
    return "".join(ch for ch in str(raw).upper() if ch.isalnum())


class CommunityStore:
    def __init__(
        self,
        state_dir: Path,
        assets: CommunityAssets,
        secret: Optional[str],
        admin_token: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.assets = assets
        self.state_dir = Path(state_dir)
        self.heads_dir = self.state_dir / "heads"
        self.db_path = self.state_dir / "community.sqlite3"
        self._secret = secret.encode("utf-8") if secret and len(secret) >= SECRET_MIN_LENGTH else None
        self.secret_too_short = bool(secret) and len(secret) < SECRET_MIN_LENGTH
        self._admin_token = admin_token if admin_token and len(admin_token) >= ADMIN_TOKEN_MIN_LENGTH else None
        self.admin_token_too_short = bool(admin_token) and len(admin_token) < ADMIN_TOKEN_MIN_LENGTH
        self._clock = clock
        self._lock = threading.RLock()
        self.limiter = RateLimiter(clock)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.heads_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            self._create_schema(conn)
            self._bootstrap(conn)
            self._bind_assets(conn)

    # ------------------------------------------------------------------ hạ tầng
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Một giao dịch ghi duy nhất (đã giữ khóa tiến trình): hoặc ghi hết hoặc không gì cả."""
        with self._lock, self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS heads(
              version INTEGER PRIMARY KEY, sha TEXT NOT NULL, kind TEXT NOT NULL, parent_version INTEGER,
              round_id INTEGER, created_at REAL NOT NULL, file TEXT NOT NULL,
              n_clients INTEGER NOT NULL DEFAULT 0, n_examples INTEGER NOT NULL DEFAULT 0, metrics TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rounds(
              round_id INTEGER PRIMARY KEY AUTOINCREMENT, parent_version INTEGER NOT NULL, parent_sha TEXT NOT NULL,
              status TEXT NOT NULL, k_target INTEGER NOT NULL, k_min INTEGER NOT NULL, opened_at REAL NOT NULL,
              deadline REAL NOT NULL, extensions INTEGER NOT NULL DEFAULT 0, closed_at REAL,
              n_updates INTEGER NOT NULL DEFAULT 0, n_examples INTEGER NOT NULL DEFAULT 0, n_clipped INTEGER NOT NULL DEFAULT 0,
              result_version INTEGER, reason TEXT, candidate_metrics TEXT, parent_metrics TEXT);
            CREATE TABLE IF NOT EXISTS codes(
              id INTEGER PRIMARY KEY AUTOINCREMENT, digest TEXT NOT NULL UNIQUE, label TEXT,
              max_uses INTEGER NOT NULL, uses INTEGER NOT NULL DEFAULT 0, expires_at REAL NOT NULL,
              revoked INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS participants(
              id TEXT PRIMARY KEY, code_id INTEGER NOT NULL, created_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS updates(
              id INTEGER PRIMARY KEY AUTOINCREMENT, round_id INTEGER NOT NULL, participant_id TEXT NOT NULL,
              received_at REAL NOT NULL, num_examples INTEGER NOT NULL, label_counts TEXT NOT NULL, local TEXT,
              norm_raw REAL NOT NULL, norm_used REAL NOT NULL, clipped INTEGER NOT NULL, suspicious INTEGER NOT NULL,
              dp_applied INTEGER NOT NULL DEFAULT 0, head BLOB, code_id INTEGER NOT NULL DEFAULT 0,
              UNIQUE(round_id, participant_id), UNIQUE(round_id, code_id));
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
            """
        )

    def _bootstrap(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            if conn.execute("SELECT 1 FROM heads WHERE version = 0").fetchone():
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                head = self.assets.base_head
                self._write_head_file(0, head)
                metrics = core.evaluate_head(head, self.assets.evalset)
                conn.execute(
                    "INSERT INTO heads(version, sha, kind, parent_version, round_id, created_at, file, n_clients, n_examples, metrics) "
                    "VALUES(0, ?, 'base', NULL, NULL, ?, ?, 0, 0, ?)",
                    (core.head_sha(head), self._clock(), self._head_filename(0), json.dumps(metrics)),
                )
                self._meta_set(conn, "current_version", "0")
                self._meta_set(conn, "open", "0")
                self._meta_set(conn, "policy", json.dumps(Policy().to_dict()))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _fingerprint(self) -> str:
        parts = [self.assets.feature_model_sha or "", self.assets.evalset.sha256, core.head_sha(self.assets.base_head)]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _bind_assets(self, conn: sqlite3.Connection) -> None:
        """Trạng thái (head, chỉ số, vòng) chỉ có nghĩa với đúng ONNX + tập giám sát + head gốc đã sinh ra nó."""
        with self._lock:
            stored = self._meta_get(conn, "assets_fingerprint")
            current = self._fingerprint()
            if stored is None:
                self._meta_set(conn, "assets_fingerprint", current)
            elif stored != current:
                raise AssetsUnavailable(
                    "Trạng thái cộng đồng đã lưu thuộc một bộ artifact khác (ONNX, đặc trưng giám sát hoặc head gốc đã đổi). "
                    "Hãy lưu lại rồi dùng thư mục trạng thái mới (COMMUNITY_STATE_DIR) cho bộ artifact mới.")

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    @staticmethod
    def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def _log(self, conn: sqlite3.Connection, kind: str, **detail: Any) -> None:
        conn.execute("INSERT INTO events(ts, kind, detail) VALUES(?, ?, ?)", (self._clock(), kind, json.dumps(detail, ensure_ascii=False)))
        marks = ",".join("?" * len(NOISY_EVENTS))
        noisy = kind in NOISY_EVENTS
        op = "IN" if noisy else "NOT IN"
        conn.execute(
            f"DELETE FROM events WHERE kind {op} ({marks}) AND id NOT IN "
            f"(SELECT id FROM events WHERE kind {op} ({marks}) ORDER BY id DESC LIMIT ?)",
            (*NOISY_EVENTS, *NOISY_EVENTS, NOISY_EVENTS_KEPT if noisy else EVENTS_KEPT),
        )

    # ------------------------------------------------------------------ head trên đĩa
    @staticmethod
    def _head_filename(version: int) -> str:
        return f"v{version:06d}.npz"

    def _write_head_file(self, version: int, head: Head) -> None:
        path = self.heads_dir / self._head_filename(version)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, weight=head[core.WEIGHT_KEY], bias=head[core.BIAS_KEY])
        os.replace(temporary, path)

    def _load_head(self, version: int, expected_sha: Optional[str] = None) -> Head:
        path = self.heads_dir / self._head_filename(version)
        with np.load(path, allow_pickle=False) as data:
            head = core.as_head(data["weight"], data["bias"])
        if expected_sha and core.head_sha(head) != expected_sha:
            raise AssetsUnavailable(f"Head phiên bản {version} bị đổi so với hash đã ghi.")
        return head

    # ------------------------------------------------------------------ cấu hình
    def policy(self, conn: Optional[sqlite3.Connection] = None) -> Policy:
        def read(c: sqlite3.Connection) -> Policy:
            raw = self._meta_get(c, "policy")
            try:
                return core.policy_from(json.loads(raw) if raw else None)
            except (ValueError, CommunityError):
                return Policy()

        if conn is not None:
            return read(conn)
        with self._conn() as c:
            return read(c)

    def is_open(self, conn: sqlite3.Connection) -> bool:
        return self._meta_get(conn, "open", "0") == "1"

    # ------------------------------------------------------------------ danh tính
    def _sign(self, payload: str) -> str:
        if self._secret is None:
            raise CommunityError("disabled", "Máy chủ chưa cấu hình COMMUNITY_SECRET nên chưa nhận người tham gia.")
        return _b64url(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())

    def _code_digest(self, normalized: str) -> str:
        if self._secret is None:
            raise CommunityError("disabled", "Máy chủ chưa cấu hình COMMUNITY_SECRET nên chưa nhận người tham gia.")
        return hmac.new(self._secret, b"code:" + normalized.encode("ascii"), hashlib.sha256).hexdigest()

    def join(self, code: str, ip: Optional[str] = None) -> Dict[str, Any]:
        normalized = normalize_code(code)
        failure: Optional[CommunityError] = None
        result: Dict[str, Any] = {}
        with self._tx() as conn:
            if not self.is_open(conn):
                raise CommunityError("closed", "Phiên học liên kết cộng đồng hiện đang đóng.")
            row = None
            if len(normalized) == CODE_LENGTH and all(ch in CODE_ALPHABET for ch in normalized):
                row = conn.execute("SELECT * FROM codes WHERE digest = ?", (self._code_digest(normalized),)).fetchone()
            now = self._clock()
            if row is None or row["revoked"] or row["expires_at"] <= now or row["uses"] >= row["max_uses"]:
                self._log(conn, "join_failed", ip=ip)  # giao dịch vẫn COMMIT để nhật ký thất bại được lưu
                failure = CommunityError("bad_code", "Mã tham gia không đúng, đã hết hạn hoặc đã hết lượt dùng.")
            else:
                participant = secrets.token_hex(8)
                conn.execute("UPDATE codes SET uses = uses + 1 WHERE id = ?", (row["id"],))
                conn.execute("INSERT INTO participants(id, code_id, created_at) VALUES(?, ?, ?)", (participant, row["id"], now))
                expires = now + TOKEN_TTL_SECONDS
                body = _b64url(json.dumps({"p": participant, "e": int(expires)}, separators=(",", ":")).encode("ascii"))
                token = f"v1.{body}.{self._sign('v1.' + body)}"
                self._log(conn, "join_ok", participant=participant, code_id=row["id"], ip=ip)
                result = {"token": token, "expires_at": int(expires), "participant": participant[-4:]}
        if failure is not None:
            raise failure
        return result

    def authenticate(self, token: Optional[str]) -> str:
        """Trả participant_id nếu token hợp lệ, chưa hết hạn và chưa bị thu hồi."""
        bad = CommunityError("bad_token", "Phiên tham gia không hợp lệ hoặc đã hết hạn. Hãy nhập lại mã tham gia.")
        if not token or not token.isascii() or token.count(".") != 2:
            raise bad
        version, body, signature = token.split(".")
        if version != "v1" or not hmac.compare_digest(self._sign(f"{version}.{body}"), signature):
            raise bad
        try:
            payload = json.loads(_b64url_decode(body))
            participant, expires = str(payload["p"]), float(payload["e"])
        except (ValueError, KeyError, TypeError) as exc:
            raise bad from exc
        if expires <= self._clock():
            raise bad
        with self._conn() as conn:
            row = conn.execute("SELECT revoked FROM participants WHERE id = ?", (participant,)).fetchone()
        if row is None or row["revoked"]:
            raise bad
        return participant

    def check_admin(self, supplied: Optional[str]) -> None:
        if not self._admin_token:
            raise CommunityError(
                "admin_disabled",
                "Quản trị chưa được bật: cần đặt COMMUNITY_ADMIN_TOKEN dài ít nhất "
                f"{ADMIN_TOKEN_MIN_LENGTH} ký tự.")
        if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), self._admin_token.encode("utf-8")):
            raise CommunityError("bad_admin", "Tài khoản hoặc mật khẩu không đúng.")

    # ------------------------------------------------------------------ vòng
    def _current(self, conn: sqlite3.Connection) -> sqlite3.Row:
        version = int(self._meta_get(conn, "current_version", "0"))
        return conn.execute("SELECT * FROM heads WHERE version = ?", (version,)).fetchone()

    def _open_round(self, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
        return conn.execute("SELECT * FROM rounds WHERE status = 'open' ORDER BY round_id DESC LIMIT 1").fetchone()

    def _ensure_round(self, conn: sqlite3.Connection, now: float) -> Optional[sqlite3.Row]:
        """Đóng/gộp vòng theo hạn, rồi bảo đảm có một vòng mở khi phiên đang mở."""
        rnd = self._open_round(conn)
        policy = self.policy(conn)
        if rnd is not None:
            count = conn.execute("SELECT COUNT(*) FROM updates WHERE round_id = ?", (rnd["round_id"],)).fetchone()[0]
            if not self.is_open(conn):
                if count >= rnd["k_min"]:
                    self._aggregate_safely(conn, rnd, now)  # đã đủ người tối thiểu: không bỏ công của họ
                else:
                    self._close_round(conn, rnd, "cancelled",
                                      f"Phiên bị đóng khi vòng mới có {count}/{rnd['k_min']} người tối thiểu; các cập nhật bị bỏ.", now)
                rnd = None
            elif count >= rnd["k_target"] or (now >= rnd["deadline"] and count >= rnd["k_min"]):
                self._aggregate_safely(conn, rnd, now)
                rnd = None
            elif now >= rnd["deadline"]:
                conn.execute("UPDATE rounds SET deadline = ?, extensions = extensions + 1 WHERE round_id = ?",
                             (now + policy.round_ttl_seconds, rnd["round_id"]))
                rnd = self._open_round(conn)
        if rnd is None and self.is_open(conn):
            current = self._current(conn)
            cursor = conn.execute(
                "INSERT INTO rounds(parent_version, parent_sha, status, k_target, k_min, opened_at, deadline) "
                "VALUES(?, ?, 'open', ?, ?, ?, ?)",
                (current["version"], current["sha"], policy.k_target, min(policy.k_min, policy.k_target), now,
                 now + policy.round_ttl_seconds),
            )
            self._log(conn, "round_opened", round_id=cursor.lastrowid, parent_version=current["version"])
            rnd = self._open_round(conn)
        return rnd

    def _close_round(self, conn: sqlite3.Connection, rnd: sqlite3.Row, status: str, reason: str, now: float) -> None:
        conn.execute(
            "UPDATE rounds SET status = ?, reason = ?, closed_at = ?, "
            "n_updates = (SELECT COUNT(*) FROM updates WHERE round_id = ?) WHERE round_id = ?",
            (status, reason, now, rnd["round_id"], rnd["round_id"]))
        conn.execute("UPDATE updates SET head = NULL WHERE round_id = ?", (rnd["round_id"],))
        self._log(conn, f"round_{status}", round_id=rnd["round_id"], reason=reason)

    def _prune_heads(self, conn: sqlite3.Connection, newest: int) -> None:
        """Giữ file head của HEADS_KEPT phiên bản mới nhất và phiên bản gốc; hàng CSDL (chỉ số, lịch sử) vẫn còn."""
        stale = conn.execute(
            "SELECT version, file FROM heads WHERE version <= ? AND kind != 'base' AND file != ''", (newest - HEADS_KEPT,)).fetchall()
        for row in stale:
            try:
                (self.heads_dir / row["file"]).unlink(missing_ok=True)
            except OSError:
                continue
            conn.execute("UPDATE heads SET file = '' WHERE version = ?", (row["version"],))

    def _aggregate_safely(self, conn: sqlite3.Connection, rnd: sqlite3.Row, now: float) -> None:
        """Gộp trong SAVEPOINT: lỗi nội bộ hủy riêng vòng đó (ghi lý do) thay vì làm kẹt mọi yêu cầu sau."""
        conn.execute("SAVEPOINT aggregate")
        try:
            self._aggregate(conn, rnd, now)
        except Exception as exc:  # noqa: BLE001 — bất kỳ lỗi nào cũng không được để vòng mở mãi
            conn.execute("ROLLBACK TO aggregate")
            self._close_round(conn, rnd, "cancelled", f"Lỗi nội bộ khi gộp ({type(exc).__name__}); vòng bị hủy.", now)
        finally:
            conn.execute("RELEASE aggregate")

    def _aggregate(self, conn: sqlite3.Connection, rnd: sqlite3.Row, now: float) -> Dict[str, Any]:
        rows = conn.execute(
            "SELECT id, num_examples, clipped, head FROM updates WHERE round_id = ? AND head IS NOT NULL "
            "AND participant_id NOT IN (SELECT id FROM participants WHERE revoked = 1) ORDER BY id",
            (rnd["round_id"],),
        ).fetchall()
        policy = self.policy(conn)
        parent_row = conn.execute("SELECT * FROM heads WHERE version = ?", (rnd["parent_version"],)).fetchone()
        parent = self._load_head(parent_row["version"], parent_row["sha"])
        updates = [{"head": core.head_from_bytes(bytes(r["head"])), "num_examples": r["num_examples"]} for r in rows]
        n_examples = int(sum(r["num_examples"] for r in rows))
        n_clipped = int(sum(r["clipped"] for r in rows))
        candidate = core.aggregate_heads(updates, parent, policy)
        candidate_metrics = core.evaluate_head(candidate, self.assets.evalset)
        parent_metrics = core.evaluate_head(parent, self.assets.evalset)
        best = core.best_of([json.loads(r["metrics"]) for r in conn.execute("SELECT metrics FROM heads").fetchall()])
        accepted, reasons = core.gate_decision(candidate_metrics, parent_metrics, policy, best)
        conn.execute("UPDATE updates SET head = NULL WHERE round_id = ?", (rnd["round_id"],))
        common = (len(rows), n_examples, n_clipped, json.dumps(candidate_metrics), json.dumps(parent_metrics))
        if not accepted:
            text = "; ".join(core.GATE_MESSAGES[r] for r in reasons)
            conn.execute(
                "UPDATE rounds SET status = 'rejected', closed_at = ?, reason = ?, n_updates = ?, n_examples = ?, n_clipped = ?, "
                "candidate_metrics = ?, parent_metrics = ? WHERE round_id = ?",
                (now, ",".join(reasons), *common, rnd["round_id"]),
            )
            self._log(conn, "round_rejected", round_id=rnd["round_id"], reasons=reasons, message=text)
            return {"accepted": False, "reasons": reasons, "metrics": candidate_metrics}
        version = int(conn.execute("SELECT MAX(version) FROM heads").fetchone()[0]) + 1
        self._write_head_file(version, candidate)
        conn.execute(
            "INSERT INTO heads(version, sha, kind, parent_version, round_id, created_at, file, n_clients, n_examples, metrics) "
            "VALUES(?, ?, 'aggregate', ?, ?, ?, ?, ?, ?, ?)",
            (version, core.head_sha(candidate), rnd["parent_version"], rnd["round_id"], now, self._head_filename(version),
             len(rows), n_examples, json.dumps(candidate_metrics)),
        )
        conn.execute(
            "UPDATE rounds SET status = 'aggregated', closed_at = ?, result_version = ?, n_updates = ?, n_examples = ?, "
            "n_clipped = ?, candidate_metrics = ?, parent_metrics = ? WHERE round_id = ?",
            (now, version, *common, rnd["round_id"]),
        )
        self._meta_set(conn, "current_version", str(version))
        self._prune_heads(conn, version)
        self._log(conn, "round_aggregated", round_id=rnd["round_id"], version=version, clients=len(rows), examples=n_examples)
        return {"accepted": True, "version": version, "metrics": candidate_metrics}

    # ------------------------------------------------------------------ dữ liệu trả cho giao diện
    @staticmethod
    def _metrics_view(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not metrics:
            return None
        return {key: metrics.get(key) for key in ("accuracy", "macro_recall", "plantvillage_accuracy", "plantdoc_accuracy", "count")}

    def _round_summary(self, conn: sqlite3.Connection, rnd: sqlite3.Row, now: float) -> Dict[str, Any]:
        received = conn.execute("SELECT COUNT(*) FROM updates WHERE round_id = ?", (rnd["round_id"],)).fetchone()[0]
        return {
            "round_id": rnd["round_id"], "status": rnd["status"], "parent_version": rnd["parent_version"],
            "parent_sha": rnd["parent_sha"], "k_target": rnd["k_target"], "k_min": rnd["k_min"], "received": received,
            "deadline": rnd["deadline"], "seconds_left": max(0, int(rnd["deadline"] - now)), "extensions": rnd["extensions"],
        }

    def _client_config(self, policy: Policy) -> Dict[str, Any]:
        return {
            "epochs": policy.local_epochs, "lr": policy.local_lr, "batch": policy.local_batch,
            "holdout_fraction": policy.holdout_fraction, "min_images": policy.n_min, "max_images": MAX_IMAGES_HINT,
            "dp": {"enabled": policy.dp_enabled, "clip": policy.dp_clip, "noise_multiplier": policy.dp_noise_multiplier},
        }

    def public_status(self) -> Dict[str, Any]:
        with self._tx() as conn:
            now = self._clock()
            open_flag = self.is_open(conn)
            rnd = self._ensure_round(conn, now)
            policy = self.policy(conn)
            current = self._current(conn)
            base = conn.execute("SELECT metrics FROM heads WHERE version = 0").fetchone()
            data = {
                "enabled": True,
                "open": open_flag,
                "joinable": open_flag and self._secret is not None,
                "version": current["version"], "head_sha": current["sha"],
                "metrics": self._metrics_view(json.loads(current["metrics"])),
                "base_metrics": self._metrics_view(json.loads(base["metrics"])),
                "round": self._round_summary(conn, rnd, now) if rnd else None,
                "config": self._client_config(policy),
                "classes": self.assets.classes,
                "feature_model": self.feature_model_info(),
                "caveat": (
                    "Học liên kết chỉ lớp cuối trên đặc trưng của backbone đóng băng. Độ chính xác giám sát tính trên tập "
                    "validation đã dùng để chọn mô hình gốc: chỉ là tín hiệu theo dõi, không phải số liệu khoa học."
                ),
            }
        return data

    def feature_model_info(self) -> Optional[Dict[str, Any]]:
        if not self.assets.feature_model_path:
            return None
        return {
            "sha256": self.assets.feature_model_sha, "bytes": self.assets.feature_model_bytes,
            "url": f"/api/community/feature-model?v={str(self.assets.feature_model_sha)[:16]}",
            "input_size": 224, "dim": core.FEATURE_DIM, "transform": "canonical_v1",
        }

    def round_view(self, participant: str, have_sha: Optional[str] = None) -> Dict[str, Any]:
        with self._tx() as conn:
            now = self._clock()
            rnd = self._ensure_round(conn, now)
            policy = self.policy(conn)
            current = self._current(conn)
            if rnd is None:
                return {"open": False, "round": None, "config": self._client_config(policy), "version": current["version"]}
            parent_sha = rnd["parent_sha"]
            submitted = conn.execute(
                "SELECT 1 FROM updates WHERE round_id = ? AND code_id = (SELECT code_id FROM participants WHERE id = ?)",
                (rnd["round_id"], participant),
            ).fetchone() is not None
            summary = self._round_summary(conn, rnd, now)
            payload: Dict[str, Any] = {
                "open": True, "round": summary, "already_submitted": submitted, "config": self._client_config(policy),
                "version": current["version"], "head": None,
                "head_encoding": "float32-le: weight[38][1024] row-major, then bias[38]; base64",
            }
            if have_sha != parent_sha:
                payload["head"] = core.encode_head(self._load_head(rnd["parent_version"], parent_sha))
            return payload

    def _check_submission(
        self, conn: sqlite3.Connection, rnd: Optional[sqlite3.Row], policy: Policy, participant: str, body: Dict[str, Any],
    ) -> Tuple[core.UpdateCheck, int, int]:
        """Mọi kiểm tra một cập nhật trước khi ghi; ném CommunityError (đã có mã + thông điệp tiếng Việt) nếu từ chối."""
        if rnd is None:
            raise CommunityError("closed", "Phiên học liên kết cộng đồng hiện đang đóng.")
        if int(body["round_id"]) != rnd["round_id"]:
            raise CommunityError(
                "stale_round", "Vòng này đã đóng trong lúc bạn huấn luyện. Hãy huấn luyện lại trên mô hình mới nhất.")
        who = conn.execute("SELECT code_id FROM participants WHERE id = ?", (participant,)).fetchone()
        if who is None:
            raise CommunityError("bad_token", "Phiên tham gia không hợp lệ hoặc đã hết hạn. Hãy nhập lại mã tham gia.")
        code_id = int(who["code_id"])
        # MỘT cập nhật mỗi MÃ mỗi vòng: một mã dùng nhiều lần (nhiều token) không được chiếm cả quorum.
        if conn.execute("SELECT 1 FROM updates WHERE round_id = ? AND code_id = ?", (rnd["round_id"], code_id)).fetchone():
            raise CommunityError(
                "already_submitted",
                "Mã tham gia của bạn đã được dùng để gửi cập nhật cho vòng này. Hãy đợi vòng kế tiếp.")
        count = int(body["num_examples"])
        if count < policy.n_min:
            raise CommunityError("too_few", f"Cần ít nhất {policy.n_min} ảnh đã gán nhãn để đóng góp (bạn có {count}).")
        if count > MAX_EXAMPLES_ACCEPTED:
            raise CommunityError("too_many", "Số ảnh khai báo vượt mức máy chủ chấp nhận.")
        core.label_counts_ok(body["label_counts"], count)
        parent = self._load_head(rnd["parent_version"], rnd["parent_sha"])
        child = core.decode_head(body["head"])
        return core.check_update(parent, child, str(body.get("parent_sha") or ""), policy), count, code_id

    def submit_update(self, participant: str, body: Dict[str, Any]) -> Dict[str, Any]:
        failure: Optional[CommunityError] = None
        response: Dict[str, Any] = {}
        with self._tx() as conn:
            now = self._clock()
            rnd = self._ensure_round(conn, now)
            policy = self.policy(conn)
            try:
                check, count, code_id = self._check_submission(conn, rnd, policy, participant, body)
            except CommunityError as error:
                self._log(conn, "update_rejected", participant=participant, code=error.code)
                failure = error  # giao dịch vẫn COMMIT: giữ nhật ký và hiệu ứng của _ensure_round (vòng vừa tới hạn)
            else:
                conn.execute(
                    "INSERT INTO updates(round_id, participant_id, received_at, num_examples, label_counts, local, norm_raw, "
                    "norm_used, clipped, suspicious, dp_applied, head, code_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rnd["round_id"], participant, now, count, json.dumps(body["label_counts"]),
                     json.dumps(body.get("local") or {}), check.norm_raw, check.norm_used, int(check.clipped),
                     int(check.suspicious), int(bool(body.get("dp_applied"))), core.head_to_bytes(check.head), code_id),
                )
                self._log(conn, "update_ok", participant=participant, round_id=rnd["round_id"], n=count,
                          norm=round(check.norm_raw, 4), clipped=check.clipped)
                after = self._ensure_round(conn, now)  # đủ K thì gộp ngay trong cùng giao dịch
                finished = conn.execute("SELECT * FROM rounds WHERE round_id = ?", (rnd["round_id"],)).fetchone()
                aggregated = finished["status"] != "open"
                response = {
                    "accepted": True, "clipped": check.clipped, "norm": round(check.norm_raw, 4),
                    "norm_used": round(check.norm_used, 4), "aggregated": aggregated,
                    "result": self._round_result(finished) if aggregated else None,
                    "round": self._round_summary(conn, after, now) if after else None,
                }
        if failure is not None:
            raise failure
        return response

    @staticmethod
    def _round_result(row: sqlite3.Row) -> Dict[str, Any]:
        reasons = [r for r in (row["reason"] or "").split(",") if r] if row["status"] == "rejected" else []
        return {
            "round_id": row["round_id"], "status": row["status"], "version": row["result_version"],
            "clients": row["n_updates"], "examples": row["n_examples"], "clipped": row["n_clipped"],
            "reasons": reasons, "messages": [core.GATE_MESSAGES.get(r, r) for r in reasons],
            "metrics": CommunityStore._metrics_view(json.loads(row["candidate_metrics"]) if row["candidate_metrics"] else None),
            "parent_metrics": CommunityStore._metrics_view(json.loads(row["parent_metrics"]) if row["parent_metrics"] else None),
        }

    def history(self, limit: int = 20) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        with self._tx() as conn:
            self._ensure_round(conn, self._clock())
            rounds = [
                {**self._round_result(r), "opened_at": r["opened_at"], "closed_at": r["closed_at"],
                 "parent_version": r["parent_version"]}
                for r in conn.execute(
                    "SELECT * FROM rounds WHERE status != 'open' ORDER BY round_id DESC LIMIT ?", (limit,)).fetchall()
            ]
            versions = [
                {"version": r["version"], "kind": r["kind"], "created_at": r["created_at"], "clients": r["n_clients"],
                 "examples": r["n_examples"], "sha": r["sha"], "metrics": self._metrics_view(json.loads(r["metrics"]))}
                for r in conn.execute(
                    "SELECT * FROM heads WHERE version = 0 OR version > (SELECT MAX(version) FROM heads) - ? ORDER BY version",
                    (HISTORY_VERSIONS,)).fetchall()
            ]
            totals = conn.execute(
                "SELECT COUNT(*) AS rounds, SUM(status = 'aggregated') AS accepted, SUM(status = 'rejected') AS rejected "
                "FROM rounds WHERE status != 'open'").fetchone()
            return {
                "rounds": rounds, "versions": versions,
                "totals": {"rounds": totals["rounds"] or 0, "accepted": totals["accepted"] or 0,
                           "rejected": totals["rejected"] or 0},
                "current_version": int(self._meta_get(conn, "current_version", "0")),
            }

    def current_head(self) -> Tuple[int, Head, str, Dict[str, Any]]:
        with self._conn() as conn:
            row = self._current(conn)
        return row["version"], self._load_head(row["version"], row["sha"]), row["sha"], json.loads(row["metrics"])

    def base_head(self) -> Head:
        return self.assets.base_head

    def base_metrics(self) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("SELECT metrics FROM heads WHERE version = 0").fetchone()
        return json.loads(row["metrics"])

    # ------------------------------------------------------------------ quản trị
    def admin_state(self) -> Dict[str, Any]:
        with self._tx() as conn:
            now = self._clock()
            rnd = self._ensure_round(conn, now)
            events = [
                {"ts": r["ts"], "kind": r["kind"], "detail": json.loads(r["detail"])}
                for r in conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 100").fetchall()
            ]
            codes = [
                {"id": r["id"], "label": r["label"], "uses": r["uses"], "max_uses": r["max_uses"],
                 "created_at": r["created_at"], "expires_at": r["expires_at"], "revoked": bool(r["revoked"])}
                for r in conn.execute("SELECT * FROM codes ORDER BY id DESC LIMIT 100").fetchall()
            ]
            participants = conn.execute(
                "SELECT COUNT(*) AS n, SUM(revoked) AS revoked FROM participants").fetchone()
            return {
                "open": self.is_open(conn), "policy": self.policy(conn).to_dict(),
                "policy_defaults": Policy().to_dict(),
                "policy_bounds": {k: list(v) for k, v in core.POLICY_BOUNDS.items()},
                "current_version": int(self._meta_get(conn, "current_version", "0")),
                "round": self._round_summary(conn, rnd, now) if rnd is not None else None,
                "participants": {"total": participants["n"] or 0, "revoked": participants["revoked"] or 0},
                "codes": codes, "events": events, "server_time": now,
            }

    def admin_configure(self, open_flag: Optional[bool], changes: Dict[str, Any]) -> Dict[str, Any]:
        with self._tx() as conn:
            if changes:
                merged = {**self.policy(conn).to_dict(), **changes}
                policy = core.policy_from(merged)  # kiểm kiểu + giới hạn; ném CommunityError nếu sai
                self._meta_set(conn, "policy", json.dumps(policy.to_dict()))
                conn.execute(
                    "UPDATE rounds SET k_target = ?, k_min = ? WHERE status = 'open'", (policy.k_target, policy.k_min))
                if "round_ttl_seconds" in changes:
                    conn.execute("UPDATE rounds SET deadline = opened_at + ? WHERE status = 'open'", (policy.round_ttl_seconds,))
                self._log(conn, "admin_policy", changes=changes)
            if open_flag is not None:
                self._meta_set(conn, "open", "1" if open_flag else "0")
                self._log(conn, "admin_open", open=open_flag)
            self._ensure_round(conn, self._clock())
        return self.admin_state()

    def admin_create_codes(self, count: int, max_uses: int, ttl_hours: float, label: str = "") -> List[str]:
        if not 1 <= count <= 500 or not 1 <= max_uses <= 100000 or not 0 < ttl_hours <= 24 * 365:
            raise CommunityError("bad_setting", "count 1–500, max_uses 1–100000, ttl_hours trong (0, 8760].")
        out: List[str] = []
        with self._tx() as conn:
            now = self._clock()
            for _ in range(count):
                raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
                conn.execute(
                    "INSERT INTO codes(digest, label, max_uses, expires_at, created_at) VALUES(?, ?, ?, ?, ?)",
                    (self._code_digest(raw), label[:80], max_uses, now + ttl_hours * 3600, now),
                )
                out.append(f"{raw[:5]}-{raw[5:]}")
            self._log(conn, "admin_codes", count=count, max_uses=max_uses, ttl_hours=ttl_hours, label=label[:80])
        return out

    def admin_revoke_code(self, code_id: int) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE codes SET revoked = 1 WHERE id = ?", (code_id,))
            conn.execute("UPDATE participants SET revoked = 1 WHERE code_id = ?", (code_id,))
            conn.execute("DELETE FROM updates WHERE head IS NOT NULL AND participant_id IN (SELECT id FROM participants WHERE revoked = 1)")
            self._log(conn, "admin_revoke_code", code_id=code_id)

    def admin_revoke_participant(self, suffix: str) -> int:
        with self._tx() as conn:
            changed = conn.execute(
                "UPDATE participants SET revoked = 1 WHERE substr(id, -4) = ? OR id = ?", (suffix[-4:], suffix)).rowcount
            conn.execute("DELETE FROM updates WHERE head IS NOT NULL AND participant_id IN (SELECT id FROM participants WHERE revoked = 1)")
            self._log(conn, "admin_revoke_participant", suffix=suffix[-4:], changed=changed)
            return changed

    def admin_close_round(self) -> Dict[str, Any]:
        """Gộp ngay với những gì đã nhận (cần ≥ 1 cập nhật); không có gì thì hủy vòng."""
        with self._tx() as conn:
            now = self._clock()
            rnd = self._open_round(conn)
            if rnd is None:
                raise CommunityError("no_round", "Hiện không có vòng nào đang mở.")
            count = conn.execute("SELECT COUNT(*) FROM updates WHERE round_id = ?", (rnd["round_id"],)).fetchone()[0]
            if count == 0:
                self._close_round(conn, rnd, "cancelled", "Quản trị đóng vòng khi chưa có cập nhật.", now)
            else:
                self._aggregate_safely(conn, rnd, now)
            self._log(conn, "admin_close_round", round_id=rnd["round_id"], updates=count)
            done = conn.execute("SELECT * FROM rounds WHERE round_id = ?", (rnd["round_id"],)).fetchone()
            return self._round_result(done)

    def admin_rollback(self, version: int) -> Dict[str, Any]:
        """Tạo phiên bản mới có nội dung của `version` (lịch sử tuyến tính, không xóa gì) và hủy vòng đang mở."""
        with self._tx() as conn:
            now = self._clock()
            row = conn.execute("SELECT * FROM heads WHERE version = ?", (version,)).fetchone()
            if row is None:
                raise CommunityError("no_version", f"Không có phiên bản {version}.")
            if not row["file"]:
                raise CommunityError(
                    "no_version", f"Phiên bản {version} đã bị dọn khỏi đĩa (chỉ giữ {HEADS_KEPT} phiên bản mới nhất).")
            head = self._load_head(row["version"], row["sha"])
            new_version = int(conn.execute("SELECT MAX(version) FROM heads").fetchone()[0]) + 1
            self._write_head_file(new_version, head)
            conn.execute(
                "INSERT INTO heads(version, sha, kind, parent_version, round_id, created_at, file, n_clients, n_examples, metrics) "
                "VALUES(?, ?, 'rollback', ?, NULL, ?, ?, 0, 0, ?)",
                (new_version, row["sha"], version, now, self._head_filename(new_version), row["metrics"]),
            )
            self._meta_set(conn, "current_version", str(new_version))
            rnd = self._open_round(conn)
            if rnd is not None:
                self._close_round(conn, rnd, "cancelled", f"Quản trị khôi phục về phiên bản {version}.", now)
            self._log(conn, "admin_rollback", to=version, new_version=new_version)
            return {"version": new_version, "restored_from": version, "sha": row["sha"]}
