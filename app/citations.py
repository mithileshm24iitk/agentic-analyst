"""Extract [file.md] citations from an answer."""

import re

# A bracketed group that contains at least one .md name, e.g. [a.md] or [a.md, b.md].
_BRACKETS = re.compile(r"\[([^\[\]]*?\.md[^\[\]]*)\]")
_FILE = re.compile(r"[A-Za-z0-9_.\-]+\.md")


def extract_citations(answer: str) -> list[str]:
    """All distinct file names cited in square brackets, in order of first appearance."""
    found: list[str] = []
    for group in _BRACKETS.findall(answer):
        for name in _FILE.findall(group):
            if name not in found:
                found.append(name)
    return found


def cited_sources(answer: str, retrieved_sources: list[str]) -> list[str]:
    """Citations that point at files that were actually retrieved (drops invented ones)."""
    retrieved = set(retrieved_sources)
    return [name for name in extract_citations(answer) if name in retrieved]
