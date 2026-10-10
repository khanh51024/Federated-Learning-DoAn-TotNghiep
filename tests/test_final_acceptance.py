"""scripts/final_acceptance.run_command phải chịu được output UTF-8 trên Windows (cp1258) và timeout."""
from __future__ import annotations

import sys

from scripts import final_acceptance as fa


def test_run_command_survives_utf8_child_output():
    # Con in UTF-8 thô (kể cả byte mà cp1258 không giải mã được) — trước đây làm proc.stdout thành None
    code = "import sys; sys.stdout.buffer.write('Việt Nam: ạ ơ ư — ok\\n'.encode('utf-8'))"
    result = fa.run_command("utf8", [sys.executable, "-c", code], 60)
    assert result["status"] == "pass"
    assert "Việt Nam" in result["stdout_tail"]


def test_run_command_reports_timeout_instead_of_crashing():
    result = fa.run_command("slow", [sys.executable, "-c", "import time; time.sleep(30)"], 1)
    assert result["status"] == "fail"
    assert result["returncode"] is None
    assert "TIMEOUT" in result["stdout_tail"]
