"""save_report path safety. Writes only to a temporary folder."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.conftest import make_tool_context
from tools import report_tools
from tools.report_tools import MAX_CONTENT_CHARS, STATE_SAVED_REPORT, save_report, slugify, write_report

NOW = datetime(2026, 11, 19, 12, 30, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "title, slug",
    [
        ("Q3 Return Rates", "q3-return-rates"),
        ("../../etc/passwd", "etc-passwd"),
        ("/absolute/path", "absolute-path"),
        ("..\\..\\windows\\system32", "windows-system32"),
        ("Tops & Tees: 11.0%!", "tops-tees-11-0"),
        ("...", "report"),
        ("", "report"),
        ("ü日本", "report"),
        ("a" * 200, "a" * 60),
    ],
)
def test_slugify_allows_only_safe_characters(title, slug):
    assert slugify(title) == slug


@pytest.mark.parametrize("title", ["../../outside", "/etc/cron.d/evil", "..", "sub/dir/name", "~/.ssh/id"])
def test_report_always_lands_directly_inside_reports_dir(tmp_path, title):
    reports = tmp_path / "reports"

    name = write_report(reports, title, "body", now=NOW)

    assert (reports / name).is_file()
    assert "/" not in name and name.endswith(".md")
    assert list(tmp_path.iterdir()) == [reports]  # nothing written next to the folder


def test_same_title_same_second_never_overwrites(tmp_path):
    first = write_report(tmp_path, "Weekly", "first", now=NOW)
    second = write_report(tmp_path, "Weekly", "second", now=NOW)

    assert first == "weekly-20261119-123000.md"
    assert second == "weekly-20261119-123000-2.md"
    assert "first" in (tmp_path / first).read_text() and "second" in (tmp_path / second).read_text()


def test_existing_file_is_not_overwritten(tmp_path):
    existing = tmp_path / "weekly-20261119-123000.md"
    existing.write_text("precious")

    name = write_report(tmp_path, "Weekly", "new", now=NOW)

    assert existing.read_text() == "precious"
    assert name != existing.name


def test_report_content_has_title_heading(tmp_path):
    name = write_report(tmp_path, "Return breaches", "Underwear is at 10.9%.", now=NOW)
    assert (tmp_path / name).read_text() == "# Return breaches\n\nUnderwear is at 10.9%.\n"


def test_no_duplicate_heading_when_content_has_one(tmp_path):
    name = write_report(tmp_path, "Return breaches", "# Underwear Check\n\nBody.", now=NOW)
    assert (tmp_path / name).read_text() == "# Underwear Check\n\nBody.\n"


def test_too_long_content_rejected(tmp_path):
    with pytest.raises(ValueError, match="too long"):
        write_report(tmp_path, "Big", "x" * (MAX_CONTENT_CHARS + 1), now=NOW)


def test_save_report_tool_returns_path_and_records_state(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": tmp_path}))
    ctx = make_tool_context()

    result = save_report("Breaches", "Five categories breach.", ctx)

    assert result["ok"] is True
    assert result["path"] == str(tmp_path / Path(result["path"]).name)  # relative to REPORTS_DIR
    assert ctx.state[STATE_SAVED_REPORT] == result["path"]


def test_save_report_tool_returns_error_instead_of_raising(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": tmp_path}))
    ctx = make_tool_context()

    result = save_report("Big", "x" * (MAX_CONTENT_CHARS + 1), ctx)

    assert result["ok"] is False and "too long" in result["error"]
    assert STATE_SAVED_REPORT not in ctx.state


def test_second_save_in_same_session_updates_same_file(tmp_path, monkeypatch, settings):
    # e.g. the orchestrator saves, the critic rejects, the revision saves again
    monkeypatch.setattr(report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": tmp_path}))
    ctx = make_tool_context()

    first = save_report("Breaches", "draft with a wrong number", ctx)
    second = save_report("Breaches", "corrected content", ctx)

    assert second["path"] == first["path"]
    assert len(list(tmp_path.iterdir())) == 1  # no orphaned draft
    assert "corrected content" in (tmp_path / Path(first["path"]).name).read_text()


def test_new_session_never_overwrites_another_sessions_report(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": tmp_path}))

    a = save_report("Same title", "A", make_tool_context())
    b = save_report("Same title", "B", make_tool_context())

    assert a["path"] != b["path"]
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("bad", ["../x.md", "/etc/passwd", "a/b.md", "x.txt", "UPPER.md", ""])
def test_replace_name_must_be_a_plain_report_name(tmp_path, bad):
    with pytest.raises(ValueError):
        write_report(tmp_path, "t", "c", replace_name=bad)


def test_save_report_caps_calls_per_session(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": tmp_path}))
    ctx = make_tool_context()

    for i in range(report_tools.MAX_REPORTS_PER_SESSION):
        assert save_report("Breaches", f"version {i}", ctx)["ok"] is True

    capped = save_report("Breaches", "too many", ctx)
    assert capped["ok"] is False and "limit reached" in capped["error"]
