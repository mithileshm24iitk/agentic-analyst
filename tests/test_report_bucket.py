"""save_report with REPORTS_BUCKET set, using a fake Cloud Storage client (no GCP access)."""

from datetime import datetime, timezone

import pytest
from google.api_core import exceptions as gcp

from tests.conftest import make_tool_context
from tools import report_tools
from tools.report_tools import MAX_CONTENT_CHARS, STATE_SAVED_REPORT, save_report, write_report_to_bucket

NOW = datetime(2026, 11, 19, 12, 30, 0, tzinfo=timezone.utc)
BUCKET = "test-reports-bucket"


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str) -> None:
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self.bucket.uploads.append((self.name, if_generation_match))
        if self.bucket.error:
            raise self.bucket.error
        # Like Cloud Storage: generation 0 means "only if no live object has this name" (else 412).
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise gcp.PreconditionFailed("object exists")
        self.bucket.objects[self.name] = (data, content_type)


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[str, str]] = {}
        self.uploads: list[tuple[str, int | None]] = []
        self.error: Exception | None = None

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        return self.buckets.setdefault(name, FakeBucket())


@pytest.fixture
def storage(monkeypatch, settings, tmp_path) -> FakeBucket:
    client = FakeStorageClient()
    local = tmp_path / "local-reports"
    monkeypatch.setattr(
        report_tools,
        "get_settings",
        lambda: settings.model_copy(update={"reports_bucket": BUCKET, "reports_dir": local}),
    )
    monkeypatch.setattr(report_tools, "get_storage_client", lambda: client)
    yield client.bucket(BUCKET)
    assert not local.exists()  # bucket mode never touches the local folder


def test_save_returns_object_name_only(storage):
    ctx = make_tool_context()

    result = save_report("Q3 Returns", "Underwear is at 10.9%.", ctx)

    assert result["ok"] is True
    name = result["path"]
    assert name.startswith("q3-returns-") and name.endswith(".md")
    assert "/" not in name and BUCKET not in name and "gs:" not in name and "http" not in name
    assert ctx.state[STATE_SAVED_REPORT] == name
    data, content_type = storage.objects[name]
    assert data == "# Q3 Returns\n\nUnderwear is at 10.9%.\n"
    assert content_type.startswith("text/markdown")


def test_every_upload_is_create_only(storage):
    ctx = make_tool_context()
    save_report("A", "one", ctx)
    save_report("A", "two", ctx)
    assert storage.uploads and all(generation == 0 for _, generation in storage.uploads)


def test_revisions_write_new_objects_and_return_the_newest(storage):
    ctx = make_tool_context()

    first = save_report("Breaches", "draft with a wrong number", ctx)["path"]
    second = save_report("Breaches", "corrected", ctx)["path"]
    third = save_report("Breaches", "corrected again", ctx)["path"]

    assert second == first.removesuffix(".md") + "-r2.md"
    assert third == first.removesuffix(".md") + "-r3.md"
    assert ctx.state[STATE_SAVED_REPORT] == third
    assert "corrected again" in storage.objects[third][0]
    assert "wrong number" in storage.objects[first][0]  # the draft stays: we cannot delete it


def test_revision_name_clash_moves_to_the_next_number(storage):
    ctx = make_tool_context()
    first = save_report("Breaches", "v1", ctx)["path"]
    clash = first.removesuffix(".md") + "-r2.md"
    storage.objects[clash] = ("someone else's", "text/markdown")

    second = save_report("Breaches", "v2", ctx)["path"]

    assert second == first.removesuffix(".md") + "-r3.md"
    assert storage.objects[clash][0] == "someone else's"


def test_other_sessions_never_overwrite(storage, monkeypatch):
    a = save_report("Same title", "A", make_tool_context())["path"]
    b = save_report("Same title", "B", make_tool_context())["path"]
    assert a != b
    assert storage.objects[a][0].endswith("A\n") and storage.objects[b][0].endswith("B\n")


def test_same_name_same_second_gets_a_suffix():
    client = FakeStorageClient()
    first = write_report_to_bucket(client, BUCKET, "Weekly", "1", now=NOW)
    second = write_report_to_bucket(client, BUCKET, "Weekly", "2", now=NOW)
    assert (first, second) == ("weekly-20261119-123000.md", "weekly-20261119-123000-2.md")


@pytest.mark.parametrize("title", ["../../outside", "/etc/passwd", "gs://other-bucket/x", "a/b/c"])
def test_title_can_never_pick_the_object_path(title):
    client = FakeStorageClient()
    name = write_report_to_bucket(client, BUCKET, title, "body", now=NOW)
    assert "/" not in name and ".." not in name
    assert list(client.buckets) == [BUCKET]


@pytest.mark.parametrize("previous", ["../x.md", "reports/x.md", "gs://b/x.md", "x.txt"])
def test_unsafe_previous_name_starts_a_new_report(previous):
    client = FakeStorageClient()
    name = write_report_to_bucket(client, BUCKET, "Fresh", "body", previous=previous, now=NOW)
    assert name == "fresh-20261119-123000.md"


def test_too_long_content_is_rejected_without_upload(storage):
    ctx = make_tool_context()
    result = save_report("Big", "x" * (MAX_CONTENT_CHARS + 1), ctx)
    assert result["ok"] is False and "too long" in result["error"]
    assert storage.uploads == [] and STATE_SAVED_REPORT not in ctx.state


def test_storage_error_returns_generic_message(storage):
    storage.error = gcp.Forbidden(f"caller lacks storage.objects.create on {BUCKET}")
    ctx = make_tool_context()

    result = save_report("Title", "body", ctx)

    assert result == {"ok": False, "path": None, "error": "The report could not be saved. Try again later."}
    assert STATE_SAVED_REPORT not in ctx.state


def test_without_bucket_the_local_folder_is_used(monkeypatch, settings, tmp_path):
    def fail():
        raise AssertionError("no storage client in local mode")

    monkeypatch.setattr(report_tools, "get_settings", lambda: settings.model_copy(update={"reports_dir": tmp_path}))
    monkeypatch.setattr(report_tools, "get_storage_client", fail)

    result = save_report("Local", "body", make_tool_context())

    assert result["ok"] is True and (tmp_path / result["path"].split("/")[-1]).is_file()


def test_bucket_save_caps_calls_per_session(storage):
    ctx = make_tool_context()
    for i in range(report_tools.MAX_REPORTS_PER_SESSION):
        assert save_report("Breaches", f"v{i}", ctx)["ok"] is True

    uploads_before = len(storage.uploads)
    capped = save_report("Breaches", "overflow", ctx)
    assert capped["ok"] is False and "limit reached" in capped["error"]
    assert len(storage.uploads) == uploads_before
