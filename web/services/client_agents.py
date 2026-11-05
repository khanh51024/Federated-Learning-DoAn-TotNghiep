"""Máy đã cài bộ client tự báo danh với web và nhận phần việc khi quản trị bắt đầu lần đo (deploy/lan_client/client_agent.py là phía máy).

Web chỉ giao số client đầu và số lượng. Máy tự biết máy chủ đo và thư mục ảnh từ lúc cài, nên web không làm máy nối sang chỗ khác hay chạy lệnh khác được.
Ai cũng báo danh được vì bộ cài công khai, nên báo danh không đem lại quyền gì: quản trị viên phải tự đánh dấu máy nào tham gia, không máy nào được chọn sẵn.
Máy được nhận biết bằng băm đầy đủ của khóa: mã 6 ký tự trên trang chỉ để người đọc, có thể tìm ra khóa giả trùng mã trong vài chục giây nên không được dùng làm định danh.
Trang công khai không thấy khóa, IP hay tên máy. Dữ liệu nằm trong bộ nhớ: web khởi động lại thì máy báo danh lại sau vài giây.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

POLL_S = 5
ONLINE_S = 20       # lỡ tới 3 nhịp báo danh mới coi là mất kết nối
FORGET_S = 600
MAX_AGENTS = 64
MAX_ONLINE_PER_SOURCE = 16     # một IP không thể làm đầy danh sách; cả phòng máy chung IP vẫn đủ


class AgentsFull(Exception):
    """Quá nhiều máy lạ cùng lúc; máy đã có trong danh sách vẫn báo danh bình thường."""


class CannotAssign(Exception):
    pass


@dataclass
class _Agent:
    agent_id: str
    first_seen: float
    last_seen: float
    source: str = ""
    cores: Optional[int] = None
    ram_gb: Optional[float] = None
    ram_free_gb: Optional[float] = None
    job: Optional[str] = None
    first: int = 0
    count: int = 0
    accepted: bool = False          # máy đã nhận phần việc: không gửi lại, kẻo máy khởi động lại rồi mở trùng client
    busy: bool = False              # nhịp gần nhất máy còn chạy phần việc này
    result: Optional[str] = None    # kết quả phần việc gần nhất: ok hoặc failed


_lock = threading.Lock()
_agents: dict[str, _Agent] = {}


def agent_id_of(key: str) -> str:
    return hashlib.sha256(key.encode("ascii")).hexdigest()


def code_of(agent_id: str) -> str:
    return agent_id[:6]


def split(clients: int, machines: int) -> list[tuple[int, int]]:
    """(client đầu, số lượng) của từng máy: chia đều, máy trước nhận phần dư, số client liền nhau."""
    size, extra = divmod(clients, machines)
    parts, first = [], 0
    for index in range(machines):
        count = size + (index < extra)
        parts.append((first, count))
        first += count
    return parts


def _sync(run_active: bool, now: float) -> None:
    """Hết lần đo thì thu lại phần việc, và quên máy đã tắt quá lâu. Phải giữ khóa khi gọi."""
    for agent_id in [agent_id for agent_id, agent in _agents.items() if now - agent.last_seen > FORGET_S and not agent.job]:
        del _agents[agent_id]
    if not run_active:
        for agent in _agents.values():
            agent.job = None
            agent.accepted = agent.busy = False


def _make_room(source: str, now: float) -> None:
    """Máy mới chỉ bị từ chối khi nguồn của nó đã có quá nhiều máy đang nối, hoặc danh sách đầy máy đang nối. Máy đã tắt thì nhường chỗ. Phải giữ khóa khi gọi."""
    if sum(agent.source == source and now - agent.last_seen <= ONLINE_S for agent in _agents.values()) >= MAX_ONLINE_PER_SOURCE:
        raise AgentsFull
    if len(_agents) < MAX_AGENTS:
        return
    idle = [agent for agent in _agents.values() if now - agent.last_seen > ONLINE_S and not agent.job]
    if not idle:
        raise AgentsFull
    del _agents[min(idle, key=lambda agent: agent.last_seen).agent_id]


def heartbeat(key: str, *, source: str, cores: Optional[int], ram_gb: Optional[float], ram_free_gb: Optional[float], running: Optional[str],
              result: Optional[str], result_job: Optional[str], run_active: bool, now: Optional[float] = None) -> dict[str, Any]:
    now = time.monotonic() if now is None else now
    agent_id = agent_id_of(key)
    with _lock:
        _sync(run_active, now)
        agent = _agents.get(agent_id)
        if agent is None:
            _make_room(source, now)
            agent = _agents[agent_id] = _Agent(agent_id=agent_id, first_seen=now, last_seen=now)
        agent.source, agent.last_seen, agent.cores, agent.ram_gb, agent.ram_free_gb = source, now, cores, ram_gb, ram_free_gb
        agent.busy = bool(agent.job) and running == agent.job
        finished_job = bool(agent.job) and result_job == agent.job and running != agent.job
        if finished_job:        # máy có thể hỏng ngay từ đầu, chưa kịp báo đang chạy lần nào; kết quả của phần việc cũ thì bỏ qua
            agent.result = result
        if agent.busy or finished_job:
            agent.accepted = True
        job = {"id": agent.job, "first": agent.first, "count": agent.count} if agent.job and not agent.accepted else None
        return {"code": code_of(agent_id), "poll_s": POLL_S, "job": job, "stop": running is not None and running != agent.job}


def _state(agent: _Agent, now: float) -> str:
    if now - agent.last_seen > ONLINE_S:
        return "offline"
    if agent.job and not agent.accepted:
        return "starting"
    if agent.job and agent.busy:
        return "running"
    if agent.result == "failed":
        return "failed"
    return "done" if agent.job else "ready"


def view(run_active: bool, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Máy đang có phần việc đứng trước, theo số client: tên Máy A, Máy B trùng với cách run_view đặt tên trong lần đo. Máy đã tắt xếp cuối."""
    now = time.monotonic() if now is None else now
    with _lock:
        _sync(run_active, now)
        ordered = sorted(_agents.values(), key=lambda agent: (agent.job is None, now - agent.last_seen > ONLINE_S, agent.first if agent.job else 0, agent.first_seen))
        return [{
            "id": agent.agent_id,
            "code": code_of(agent.agent_id),
            "label": f"Máy {chr(ord('A') + index)}",
            "state": _state(agent, now),
            "cores": agent.cores,
            "ram_gb": agent.ram_gb,
            "ram_free_gb": agent.ram_free_gb,
            "clients": list(range(agent.first, agent.first + agent.count)) if agent.job else [],
        } for index, agent in enumerate(ordered)]


def check_assignable(agent_ids: list[str], clients: int, now: Optional[float] = None) -> None:
    """Gọi trước khi mở máy chủ đo, để không mở cổng cho một lần đo không thể đủ client."""
    now = time.monotonic() if now is None else now
    if len(set(agent_ids)) != len(agent_ids):
        raise CannotAssign("Danh sách máy bị trùng.")
    if len(agent_ids) > clients:
        raise CannotAssign(f"Chọn {len(agent_ids)} máy nhưng chỉ có {clients} client.")
    with _lock:
        gone = [code_of(agent_id) for agent_id in agent_ids if agent_id not in _agents or now - _agents[agent_id].last_seen > ONLINE_S]
    if gone:
        raise CannotAssign(f"Máy {', '.join(gone)} không còn nối với trang. Tải lại trang rồi chọn lại.")


def assign(agent_ids: list[str], clients: int) -> list[dict[str, Any]]:
    job = secrets.token_hex(8)
    parts = split(clients, len(agent_ids))
    with _lock:
        for agent in _agents.values():
            agent.job, agent.accepted, agent.busy = None, False, False
        for agent_id, (first, count) in zip(agent_ids, parts):
            agent = _agents.get(agent_id)
            if agent is not None:       # vừa bị quên giữa lúc kiểm và lúc giao: lần đo sẽ thiếu client, trang vẫn báo đang chờ
                agent.job, agent.first, agent.count, agent.result = job, first, count, None
    return [{"id": agent_id, "first": first, "count": count} for agent_id, (first, count) in zip(agent_ids, parts)]
