"""Cơ chế của scripts/update_status.py: checklist do máy chấm, idempotent, bảo toàn nhật ký."""
from __future__ import annotations

from datetime import datetime

import pytest

from scripts import update_status as us
from src import registry as reg


@pytest.fixture(scope="module")
def evaluated():
    registry = reg.load_registry()
    ctx = us.Ctx(registry)
    return registry, ctx, us.evaluate(ctx)


def test_checklist_items_are_unique_and_consistent():
    keys = [item.key for item in us.ITEMS]
    assert len(keys) == len(set(keys))
    assert set(us.TECH_KEYS) <= set(keys)


def test_every_item_has_a_machine_verdict(evaluated):
    _registry, _ctx, results = evaluated
    assert len(results) == len(us.ITEMS)
    for item, result in results:
        assert result.status in (us.DONE, us.TODO, us.BLOCKED), item.key
        assert result.detail, f"{item.key}: thiếu bằng chứng"


def test_blocked_items_say_what_is_needed_and_how_to_unlock(evaluated):
    _registry, _ctx, results = evaluated
    blocked = [(i, r) for i, r in results if r.status == us.BLOCKED]
    assert blocked, "máy này không có dataset nên phải còn mục BLOCKED trung thực"
    for item, result in blocked:
        assert result.needs, item.key
        assert result.unlock, item.key


def test_render_is_deterministic_and_stamp_keeps_old_timestamp(evaluated):
    registry, ctx, results = evaluated
    journal = us.DEFAULT_JOURNAL
    first = us.render(registry, journal, ctx, results)
    assert first == us.render(registry, journal, ctx, results)

    stamped = us.stamp(first, None, datetime(2026, 1, 2, 3, 4))
    assert "2026-01-02 03:04" in stamped
    # nội dung không đổi -> giữ nguyên dấu thời gian cũ dù giờ chạy khác
    assert us.stamp(first, stamped, datetime(2030, 5, 6, 7, 8)) == stamped
    # nội dung đổi -> cập nhật dấu thời gian
    changed = us.stamp(first + "\nthêm dòng", stamped, datetime(2030, 5, 6, 7, 8))
    assert "2030-05-06 07:08" in changed


def test_journal_block_is_preserved_across_regeneration(evaluated, tmp_path, monkeypatch):
    registry, ctx, results = evaluated
    note = "### Nhật ký\n\n- 2099-01-01: ghi chú viết tay phải còn nguyên.\n"
    text = us.stamp(us.render(registry, note, ctx, results), None, datetime(2026, 1, 1, 0, 0))
    status_file = tmp_path / "PROJECT_STATUS.md"
    status_file.write_text(text, encoding="utf-8")
    monkeypatch.setattr(us, "STATUS_MD", status_file)
    assert "ghi chú viết tay phải còn nguyên" in us.read_journal()


def test_old_hand_ticked_checklist_is_dropped_but_journal_kept(tmp_path, monkeypatch):
    legacy = (
        f"{us.JOURNAL_BEGIN}\n### Checklist nghiệm thu\n\n- [x] tick tay cũ\n\n### Nhật ký\n\n- mục cũ còn giữ\n{us.JOURNAL_END}\n"
    )
    status_file = tmp_path / "PROJECT_STATUS.md"
    status_file.write_text(legacy, encoding="utf-8")
    monkeypatch.setattr(us, "STATUS_MD", status_file)
    journal = us.read_journal()
    assert "mục cũ còn giữ" in journal
    assert "tick tay cũ" not in journal


def test_full_tests_item_goes_stale_when_code_changes(evaluated, monkeypatch):
    registry, ctx, _results = evaluated
    ctx.final = {"technical_acceptance": "pass", "code_fingerprint": "khac-voi-ma-hien-tai", "generated_at": "x"}
    result = us.check_full_tests(ctx)
    assert result.status == us.TODO
    assert "không còn khớp" in result.detail
    ctx.final = {"technical_acceptance": "pass", "code_fingerprint": ctx.fingerprint, "generated_at": "x",
                 "skip_network": False, "checks": [{"name": "network_tests", "status": "pass"}]}
    assert us.check_full_tests(ctx).status == us.DONE


def test_hardcode_scanner_uses_ast_not_quotes(tmp_path):
    source = tmp_path / "m.py"
    source.write_text(
        '"""Docstring nhắc artifacts/checkpoints/x.pt không tính."""\n'
        "from pathlib import Path\n"
        'A = Path("artifacts")\n'
        'B = ROOT / "artifacts" / "metrics"\n'
        'C = "artifacts/branch_demo/x.json"\n'
        'D = "Chưa có artifacts/registry.json — chạy lại"  # câu thông báo, không phải đường dẫn\n'
        'E = {"artifacts": 1}  # khóa dict, không phải đường dẫn\n'
        'F = dict(help="xem artifacts/foo")\n',
        encoding="utf-8",
    )
    monkey_root = us.ROOT
    try:
        us.ROOT = tmp_path
        hits = us.path_literals(source)
    finally:
        us.ROOT = monkey_root
    assert sorted(h.split(":")[1] for h in hits) == ["3", "4", "5"]


def test_hand_written_journal_survives_even_with_different_heading(tmp_path, monkeypatch):
    block = "## Nhat ky (đổi tiêu đề)\n\n- 2099-12-31: viết tay, không được mất.\n"
    status_file = tmp_path / "PROJECT_STATUS.md"
    status_file.write_text(f"{us.JOURNAL_BEGIN}\n{block}{us.JOURNAL_END}\n", encoding="utf-8")
    monkeypatch.setattr(us, "STATUS_MD", status_file)
    assert "viết tay, không được mất" in us.read_journal()


def test_verdict_requires_every_item_done(evaluated):
    registry, _ctx, results = evaluated
    all_done = [(item, us.Result(us.DONE, "ok")) for item, _ in results]
    registry = {**registry, "gates": {**registry["gates"], "provenance": {"status": "verified"}, "final_acceptance": {"scientific": "pass"}}}
    assert us.readiness_verdict(registry, all_done) == "SẴN SÀNG NGHIỆM THU KHOA HỌC"
    one_todo = [*all_done[:-1], (all_done[-1][0], us.Result(us.TODO, "chưa"))]
    assert us.readiness_verdict(registry, one_todo) != "SẴN SÀNG NGHIỆM THU KHOA HỌC"


def test_full_tests_does_not_count_runs_that_skipped_the_network(evaluated):
    _registry, ctx, _results = evaluated
    checks = [{"name": "core_tests", "status": "pass"}, {"name": "other_tests", "status": "pass"}]
    ctx.final = {"technical_acceptance": "pass", "code_fingerprint": ctx.fingerprint, "generated_at": "x",
                 "skip_network": True, "checks": checks}
    assert us.check_full_tests(ctx).status == us.TODO
    ctx.final = {"technical_acceptance": "pass", "code_fingerprint": ctx.fingerprint, "generated_at": "x",
                 "skip_network": False, "checks": [*checks, {"name": "network_tests", "status": "pass"}], "test_files": ["a"]}
    assert us.check_full_tests(ctx).status == us.DONE


def test_provenance_unlock_points_at_the_slots_the_validator_reads(evaluated):
    _registry, ctx, _results = evaluated
    result = us.check_provenance(ctx)
    if result.status == us.BLOCKED:
        joined = "\n".join(result.unlock)
        assert reg.FEDAVG_BEST_PT in joined and reg.CENTRALIZED_BEST_PT in joined
        assert "Centralized" in result.needs


def test_static_test_results_are_flagged_in_docs():
    assert us.STATIC_TEST_RESULT_RE.search("```text\n60 passed\n2 warnings\n```")
    assert not us.STATIC_TEST_RESULT_RE.search("Chạy lại: python scripts/update_status.py --full-gates")


def test_dependency_declaration_is_single_sourced(evaluated):
    _registry, ctx, _results = evaluated
    assert us.check_deps(ctx).status == us.DONE


def test_vietnamese_number_formatting_touches_only_numbers():
    assert us.vn(37238) == "37.238"
    assert us.vn(999) == "999"
    assert us.vn(1234567) == "1.234.567"
    # câu có dấu phẩy phải giữ nguyên dấu phẩy (lỗi cũ: .replace(",", ".") áp lên cả câu)
    sentence = f"T=1.15, fit trên {us.vn(4225)} ảnh"
    assert sentence == "T=1.15, fit trên 4.225 ảnh"
