"""Split markdown documents into chunks by heading, then by paragraph, with overlap.

Each chunk keeps its heading path (e.g. "ShopCo Returns Policy > Return shipping costs")
and repeats it at the top of its text, so a chunk makes sense on its own and both
retrievers can match on heading words.
"""

import re
from dataclasses import asdict, dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    chunk_id: str  # stable: "<file name>:c<n>", so re-ingesting overwrites instead of duplicating
    source: str  # file name, e.g. "returns_policy.md"
    heading: str  # heading path, "" for text before the first heading
    text: str  # heading line + body; this is what gets embedded and searched

    def to_dict(self) -> dict:
        return asdict(self)


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading path, body text) for each section, in order."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []  # heading titles, index = level - 1
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(path), text))
        body.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            # Text before the first heading (e.g. the demo disclaimer) joins the first section.
            if path:
                flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2)]
        else:
            body.append(line)
    flush()
    return sections


def _units(body: str, chunk_size: int) -> list[str]:
    """Break a section body into pieces that each fit in chunk_size.

    Paragraphs first; a paragraph that is too long is split into lines;
    a line that is still too long is split into fixed-size pieces.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for line in paragraph.splitlines():
            line = line.strip()
            while len(line) > chunk_size:
                units.append(line[:chunk_size])
                line = line[chunk_size:]
            if line:
                units.append(line)
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    """Last ~overlap characters of text, starting at a line or word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    if "\n" in tail:
        return tail.split("\n", 1)[1]  # start at the next full line
    if " " in tail:
        return tail.split(" ", 1)[1]  # start at the next full word
    return tail


def chunk_markdown(source: str, markdown: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk one markdown file. Chunks never cross a heading boundary."""
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[Chunk] = []

    def emit(heading: str, body: str) -> None:
        text = f"{heading}\n{body}" if heading else body
        chunks.append(Chunk(f"{source}:c{len(chunks)}", source, heading, text))

    for heading, body in split_sections(markdown):
        current = ""
        for unit in _units(body, chunk_size):
            candidate = f"{current}\n{unit}" if current else unit
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            emit(heading, current)
            # Start the next chunk with the end of the previous one (overlap).
            tail = _overlap_tail(current, overlap)
            current = f"{tail}\n{unit}" if tail and len(tail) + 1 + len(unit) <= chunk_size else unit
        if current:
            emit(heading, current)
    return chunks
