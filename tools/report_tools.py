"""save_report(): the orchestrator's one action tool.

Two destinations:
  - REPORTS_BUCKET set (Cloud Run): a Cloud Storage object; see the bucket section below.
  - otherwise: a markdown file in REPORTS_DIR (local development).

Local folder safety:
  - the file name is built by us from the title (letters, digits, dashes only),
    so the model can never choose a path;
  - the final path is checked to be directly inside REPORTS_DIR;
  - new files are created with mode "x", so another request's file is never overwritten.
    Only a second save in the same session (e.g. after a critic revision) replaces
    that session's own file, so no unreviewed draft is left behind;
  - the path returned to the API is relative to REPORTS_DIR's configured value,
    so the server's absolute directory layout is not exposed.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from google.adk.tools import ToolContext
from google.api_core import exceptions as gcp

from app.config import get_settings

STATE_SAVED_REPORT = "saved_report"
MAX_TITLE_SLUG = 60
MAX_CONTENT_CHARS = 100_000
MAX_NAME_CLASHES = 100
SAFE_NAME = re.compile(r"[a-z0-9-]+\.md")
REVISION_SUFFIX = re.compile(r"-r(\d+)$")  # "...-r2" = second version saved in one session

logger = logging.getLogger(__name__)


def slugify(title: str) -> str:
    """'Q3 Returns / ../../etc' -> 'q3-returns-etc'. Falls back to 'report'."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:MAX_TITLE_SLUG].strip("-") or "report"


def _inside(folder: Path, name: str) -> Path:
    path = (folder / name).resolve()
    if path.parent != folder:  # defence in depth; names are already [a-z0-9-].md
        raise ValueError("Report path escapes the reports folder.")
    return path


def render(title: str, content: str) -> str:
    """Markdown body with a title heading (unless the model already wrote one). Checks the size."""
    if len(content) > MAX_CONTENT_CHARS:
        raise ValueError(f"Report is too long ({len(content)} characters, max {MAX_CONTENT_CHARS}).")
    content = content.strip()
    return f"{content}\n" if content.startswith("# ") else f"# {title.strip() or 'Report'}\n\n{content}\n"


def new_base_name(title: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(title)}-{stamp}"


def write_report(
    reports_dir: Path,
    title: str,
    content: str,
    now: datetime | None = None,
    replace_name: str | None = None,
) -> str:
    """Write the report and return its file name. Raises ValueError on unsafe input.

    replace_name: a file this session already saved; it is updated in place.
    """
    body = render(title, content)
    reports_dir.mkdir(parents=True, exist_ok=True)
    folder = reports_dir.resolve()

    if replace_name is not None:
        if not SAFE_NAME.fullmatch(replace_name):
            raise ValueError("Invalid report name.")
        _inside(folder, replace_name).write_text(body, encoding="utf-8")
        return replace_name

    base = new_base_name(title, now)
    for n in range(1, MAX_NAME_CLASHES + 1):
        name = f"{base}.md" if n == 1 else f"{base}-{n}.md"
        try:
            with _inside(folder, name).open("x", encoding="utf-8") as f:  # "x" = fail if it exists
                f.write(body)
            return name
        except FileExistsError:
            continue
    raise ValueError("Too many reports with the same name; try a different title.")


# ---------------------------------------------------------------- Cloud Storage (REPORTS_BUCKET)
#
# The Cloud Run service account only has roles/storage.objectCreator on the bucket:
# it can create objects but not read, overwrite or delete them. So:
#   - every upload uses if_generation_match=0 ("only if no such object exists"); a clash
#     returns 412 and we try the next name, so no report is ever overwritten;
#   - a second save in the same session (e.g. after a critic revision) cannot replace
#     the first, so it writes "<name>-r2.md", then "-r3.md", and the response returns the
#     newest. The earlier draft stays in the bucket.
#   - the API only ever sees the object name: no gs:// path, no URL, no bucket name.


def get_storage_client():
    """Real Cloud Storage client. Tests replace this function with a fake."""
    from google.cloud import storage  # imported here: local development never needs it

    return storage.Client(project=get_settings().google_cloud_project)


def bucket_candidate_names(title: str, previous: str | None, now: datetime | None = None):
    """Object names to try, in order. previous = this session's last saved name, if any."""
    if previous and SAFE_NAME.fullmatch(previous):
        stem = previous.removesuffix(".md")
        match = REVISION_SUFFIX.search(stem)
        version = int(match.group(1)) if match else 1
        base = REVISION_SUFFIX.sub("", stem)
        for n in range(version + 1, version + 1 + MAX_NAME_CLASHES):
            yield f"{base}-r{n}.md"
        return
    base = new_base_name(title, now)
    for n in range(1, MAX_NAME_CLASHES + 1):
        yield f"{base}.md" if n == 1 else f"{base}-{n}.md"


def write_report_to_bucket(
    client, bucket_name: str, title: str, content: str, previous: str | None = None, now: datetime | None = None
) -> str:
    """Upload the report as a new object and return its name. Never overwrites."""
    body = render(title, content)
    bucket = client.bucket(bucket_name)  # no API call; needs no bucket-level read permission
    for name in bucket_candidate_names(title, previous, now):
        try:
            bucket.blob(name).upload_from_string(
                body, content_type="text/markdown; charset=utf-8", if_generation_match=0
            )
            return name
        except gcp.PreconditionFailed:  # 412: that name exists already
            continue
    raise ValueError("Too many reports with the same name; try a different title.")


# ---------------------------------------------------------------- the tool


def save_report(title: str, content: str, tool_context: ToolContext) -> dict:
    """Saves a markdown report. Use ONLY when the user explicitly asks to save a report.

    Calling it again in the same conversation saves the corrected version of the same report.

    Args:
        title: Short report title, e.g. "Return rate breaches November 2026".
        content: The full report body in markdown.

    Returns {ok, path} on success or {ok: false, error}.
    """
    settings = get_settings()
    previous = tool_context.state.get(STATE_SAVED_REPORT)
    try:
        if settings.reports_bucket:
            path = write_report_to_bucket(
                get_storage_client(), settings.reports_bucket, title, content, previous=previous
            )  # e.g. "q3-returns-20261119-123000.md"
        else:
            name = write_report(
                settings.reports_dir, title, content, replace_name=Path(previous).name if previous else None
            )
            path = str(settings.reports_dir / name)  # e.g. "reports/q3-returns-20261119-123000.md"
    except (ValueError, OSError) as e:  # never raise to the agent; these messages are our own
        return {"ok": False, "path": None, "error": str(e)}
    except Exception:  # Cloud Storage or auth errors: details go to the logs, not to the model
        logger.exception("save_report failed")
        return {"ok": False, "path": None, "error": "The report could not be saved. Try again later."}
    tool_context.state[STATE_SAVED_REPORT] = path
    return {"ok": True, "path": path, "error": None}
