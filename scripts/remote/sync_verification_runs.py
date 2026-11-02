"""Pull completed runs via OpenSSH, verify hashes, and publish each directory atomically.

Keep this running locally while the remote tmux batch is active. A local DONE
marker becomes visible only after the checkpoint, logits, and metadata match
their remote SHA-256 hashes. No dataset files are sent to the remote machine.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath

FILES = ("checkpoint_best.pt", "logits_best.npz", "config.json", "history.jsonl", "final.json", "DONE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--remote-root", default="/root/cropfed/runs/v4_sweep")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=18)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = PurePosixPath(args.remote_root)
    if not root.is_absolute() or ".." in root.parts or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(root)):
        parser.error("remote-root must be a simple absolute path")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = ["-i", str(args.key.resolve()), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
              "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
    ssh = ["ssh", *common, "-p", str(args.port), args.host]
    scp = ["scp", *common, "-P", str(args.port)]

    def remote(argv: list[str]) -> str:
        return subprocess.check_output([*ssh, shlex.join(argv)], text=True, timeout=120)

    while True:
        try:
            paths = remote(["find", str(root), "-mindepth", "2", "-maxdepth", "2", "-type", "f", "-name", "DONE", "-printf", "%h\n"])
            for line in sorted(paths.splitlines()):
                path = PurePosixPath(line)
                if path.parent != root or not re.fullmatch(r"(?:fedavg|centralized|local_c[0-4])_s(?:42|123|2026)_[a-z0-9_]+", path.name):
                    raise ValueError(f"unexpected run path: {line}")
                target = out / path.name
                if (target / "DONE").is_file():
                    continue
                if target.exists():
                    raise ValueError(f"refusing to overwrite incomplete local directory: {target}")
                hashes = remote(["sha256sum", *[str(path / name) for name in FILES]])
                stage = Path(tempfile.mkdtemp(prefix=".incoming_", dir=out))
                subprocess.run([*scp, "-r", f"{args.host}:{path}", str(stage)], check=True, timeout=1200)
                source = stage / path.name
                for row in hashes.splitlines():
                    expected, name = row.split(maxsplit=1)
                    relative = PurePosixPath(name.strip()).relative_to(path)
                    if relative.as_posix() not in FILES:
                        raise ValueError(f"unexpected hash path: {name}")
                    with (source / relative.name).open("rb") as stream:
                        actual = hashlib.file_digest(stream, "sha256").hexdigest()
                    if actual != expected:
                        raise ValueError(f"download hash mismatch: {path.name}/{relative.name}")
                # Verify both resolved paths remain inside the intended workspace output.
                if not source.resolve().is_relative_to(out) or not target.resolve().is_relative_to(out):
                    raise ValueError("refusing to move a directory outside the output root")
                source.rename(target)
                stage.rmdir()
                print(f"verified and pulled {path.name}", flush=True)
            have = len(list(out.glob("*/DONE")))
            print(f"local completed runs: {have}/{args.expected}", flush=True)
            if args.once or have >= args.expected:
                return 0
        except (OSError, subprocess.SubprocessError) as error:
            print(f"sync retry after connection error: {error}", flush=True)
            if args.once:
                return 1
        time.sleep(max(10, min(args.interval, 60)))


if __name__ == "__main__":
    raise SystemExit(main())
