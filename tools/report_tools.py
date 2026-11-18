"""save_report(): the orchestrator's one action tool. Writes a markdown file into REPORTS_DIR.

Safety:
  - the file name is built by us from the title (letters, digits, dashes only),
    so the model can never choose a path;
  - the final path is checked to be directly inside REPORTS_DIR;
  - files are created with mode "x", so an existing file is never overwritten.
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


def slugify(title: str) -> str:
    """'Q3 Returns / ../../etc' -> 'q3-returns-etc'. Falls back to 'report'."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:MAX_TITLE_SLUG].strip("-") or "report"


def write_report(reports_dir: Path, title: str, content: str, now: datetime | None = None) -> Path:
    """Write the report and return its path. Raises ValueError on unsafe input."""
    if len(content) > MAX_CONTENT_CHARS:
        raise ValueError(f"Report is too long ({len(content)} characters, max {MAX_CONTENT_CHARS}).")

    reports_dir.mkdir(parents=True, exist_ok=True)
    folder = reports_dir.resolve()
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    base = f"{slugify(title)}-{stamp}"
    content = content.strip()
    # Add a title heading unless the model already wrote one.
    body = f"{content}\n" if content.startswith("# ") else f"# {title.strip() or 'Report'}\n\n{content}\n"

    for n in range(1, MAX_NAME_CLASHES + 1):
        name = f"{base}.md" if n == 1 else f"{base}-{n}.md"
        path = (folder / name).resolve()
        if path.parent != folder:  # defence in depth; slugify already prevents this
            raise ValueError("Report path escapes the reports folder.")
        try:
            with path.open("x", encoding="utf-8") as f:  # "x" = fail if it already exists
                f.write(body)
            return path
        except FileExistsError:
            continue
    raise ValueError("Too many reports with the same name; try a different title.")


def save_report(title: str, content: str, tool_context: ToolContext) -> dict:
    """Saves a markdown report file. Use ONLY when the user explicitly asks to save a report.

    Args:
        title: Short report title, e.g. "Return rate breaches November 2026".
        content: The full report body in markdown.

    Returns {ok, path} on success or {ok: false, error}.
    """
    try:
        path = write_report(get_settings().reports_dir, title, content)
    except (ValueError, OSError) as e:  # never raise to the agent
        return {"ok": False, "path": None, "error": str(e)}
    tool_context.state[STATE_SAVED_REPORT] = str(path)
    return {"ok": True, "path": str(path), "error": None}
