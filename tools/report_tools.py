"""save_report(): the orchestrator's one action tool. Writes a markdown file into REPORTS_DIR.

Safety:
  - the file name is built by us from the title (letters, digits, dashes only),
    so the model can never choose a path;
  - the final path is checked to be directly inside REPORTS_DIR;
  - new files are created with mode "x", so another request's file is never overwritten.
    Only a second save in the same session (e.g. after a critic revision) replaces
    that session's own file, so no unreviewed draft is left behind;
  - the path returned to the API is relative to REPORTS_DIR's configured value,
    so the server's absolute directory layout is not exposed.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

from google.adk.tools import ToolContext

from app.config import get_settings

STATE_SAVED_REPORT = "saved_report"
MAX_TITLE_SLUG = 60
MAX_CONTENT_CHARS = 100_000
MAX_NAME_CLASHES = 100
SAFE_NAME = re.compile(r"[a-z0-9-]+\.md")


def slugify(title: str) -> str:
    """'Q3 Returns / ../../etc' -> 'q3-returns-etc'. Falls back to 'report'."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:MAX_TITLE_SLUG].strip("-") or "report"


def _inside(folder: Path, name: str) -> Path:
    path = (folder / name).resolve()
    if path.parent != folder:  # defence in depth; names are already [a-z0-9-].md
        raise ValueError("Report path escapes the reports folder.")
    return path


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
    if len(content) > MAX_CONTENT_CHARS:
        raise ValueError(f"Report is too long ({len(content)} characters, max {MAX_CONTENT_CHARS}).")

    reports_dir.mkdir(parents=True, exist_ok=True)
    folder = reports_dir.resolve()
    content = content.strip()
    # Add a title heading unless the model already wrote one.
    body = f"{content}\n" if content.startswith("# ") else f"# {title.strip() or 'Report'}\n\n{content}\n"

    if replace_name is not None:
        if not SAFE_NAME.fullmatch(replace_name):
            raise ValueError("Invalid report name.")
        _inside(folder, replace_name).write_text(body, encoding="utf-8")
        return replace_name

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(title)}-{stamp}"
    for n in range(1, MAX_NAME_CLASHES + 1):
        name = f"{base}.md" if n == 1 else f"{base}-{n}.md"
        try:
            with _inside(folder, name).open("x", encoding="utf-8") as f:  # "x" = fail if it exists
                f.write(body)
            return name
        except FileExistsError:
            continue
    raise ValueError("Too many reports with the same name; try a different title.")


def save_report(title: str, content: str, tool_context: ToolContext) -> dict:
    """Saves a markdown report file. Use ONLY when the user explicitly asks to save a report.

    Calling it again in the same conversation updates the same report file.

    Args:
        title: Short report title, e.g. "Return rate breaches November 2026".
        content: The full report body in markdown.

    Returns {ok, path} on success or {ok: false, error}.
    """
    reports_dir = get_settings().reports_dir
    previous = tool_context.state.get(STATE_SAVED_REPORT)
    try:
        name = write_report(
            reports_dir, title, content, replace_name=Path(previous).name if previous else None
        )
    except (ValueError, OSError) as e:  # never raise to the agent
        return {"ok": False, "path": None, "error": str(e)}
    path = str(reports_dir / name)  # e.g. "reports/q3-returns-20261119-123000.md"
    tool_context.state[STATE_SAVED_REPORT] = path
    return {"ok": True, "path": path, "error": None}
