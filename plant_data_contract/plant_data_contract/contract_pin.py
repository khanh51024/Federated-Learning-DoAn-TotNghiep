"""Contract pin: prove which plant_data_contract a launcher actually loaded.

verify_loaded_contract() checks that EVERY loaded module named plant_data_contract or
plant_data_contract.* comes from this package's own directory (or, when the workspace bridge is used,
from the bridge __init__.py one level up), and that the digest of this directory's *.py files equals
CONTRACT_LOCK.json next to this file. It prints the directory and digest to stderr and returns them.
It changes no state and imports nothing from the contract.
"""
import hashlib
import json
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCK = os.path.join(_PKG_DIR, "CONTRACT_LOCK.json")


def contract_digest(pkg_dir=_PKG_DIR):
    """sha256 over sorted 'name NUL sha256(file) LF' for every *.py file directly in pkg_dir."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(pkg_dir)):
        if name.endswith(".py") and os.path.isfile(os.path.join(pkg_dir, name)):
            with open(os.path.join(pkg_dir, name), "rb") as fh:
                file_sha = hashlib.sha256(fh.read()).hexdigest()
            h.update(name.encode("utf-8") + b"\0" + file_sha.encode("ascii") + b"\n")
    return h.hexdigest()


def loaded_contract_files():
    bridge = os.path.join(os.path.dirname(_PKG_DIR), "__init__.py")
    files, foreign = {}, {}
    for name, mod in list(sys.modules.items()):
        if name != "plant_data_contract" and not name.startswith("plant_data_contract."):
            continue
        f = getattr(mod, "__file__", None)
        if f is None:
            foreign[name] = "<no __file__>"
            continue
        f = os.path.abspath(f)
        if os.path.dirname(f) == _PKG_DIR or f == bridge:
            files[name] = f
        else:
            foreign[name] = f
    return files, foreign


def verify_loaded_contract(stream=None):
    files, foreign = loaded_contract_files()
    if foreign:
        raise RuntimeError("plant_data_contract loaded from more than one place: %r (pinned dir %s)"
                           % (sorted(foreign.items())[:5], _PKG_DIR))
    with open(_LOCK, encoding="utf-8") as fh:
        expected = json.load(fh)["contract_digest"]
    got = contract_digest()
    if got != expected:
        raise RuntimeError("contract digest %s does not match CONTRACT_LOCK.json %s (%s)" % (got, expected, _PKG_DIR))
    print("[contract-pin] dir=%s sha256=%s modules=%d" % (_PKG_DIR, got, len(files)),
          file=stream if stream is not None else sys.stderr, flush=True)
    return {"dir": _PKG_DIR, "digest": got, "modules": len(files)}
