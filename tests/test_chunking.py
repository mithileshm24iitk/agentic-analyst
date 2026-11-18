"""Chunking: headings, paragraphs, overlap, stable IDs."""

import pytest

from tools.retrieval.chunking import chunk_markdown, split_sections

DOC = """> Demo disclaimer.

# Policy

Intro paragraph.

## Returns

Returns are accepted within 30 days.

Refunds take 5 business days.

## Shipping

Standard shipping takes 5 days.
"""


def test_sections_follow_heading_path_and_keep_preamble_with_title():
    sections = split_sections(DOC)
    assert [heading for heading, _ in sections] == ["Policy", "Policy > Returns", "Policy > Shipping"]
    assert sections[0][1].startswith("> Demo disclaimer.")  # text before first heading joins it


def test_chunks_never_cross_headings_and_carry_heading_text():
    chunks = chunk_markdown("policy.md", DOC, chunk_size=200, overlap=20)
    assert [c.heading for c in chunks] == ["Policy", "Policy > Returns", "Policy > Shipping"]
    returns = chunks[1]
    assert returns.text.startswith("Policy > Returns\n")
    assert "30 days" in returns.text and "Refunds" in returns.text  # paragraphs packed together
    assert "Standard shipping" not in returns.text


def test_chunk_ids_are_stable_and_sequential():
    first = chunk_markdown("policy.md", DOC, chunk_size=200, overlap=20)
    second = chunk_markdown("policy.md", DOC, chunk_size=200, overlap=20)
    assert [c.chunk_id for c in first] == ["policy.md:c0", "policy.md:c1", "policy.md:c2"]
    assert first == second
    assert all(c.source == "policy.md" for c in first)


def test_long_section_split_with_overlap_and_size_limit():
    lines = [f"Category {i}: maximum acceptable return rate is {10 + i / 10:.1f}%." for i in range(20)]
    doc = "# Thresholds\n\n" + "\n".join(lines)

    chunks = chunk_markdown("t.md", doc, chunk_size=200, overlap=60)

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[1]
        assert len(body) <= 200
    # Overlap: each chunk starts with the last full line of the previous chunk.
    for prev, nxt in zip(chunks, chunks[1:]):
        last_line = prev.text.splitlines()[-1]
        assert nxt.text.splitlines()[1] == last_line
    # Every line survives in some chunk, whole.
    for line in lines:
        assert any(line in c.text.splitlines() for c in chunks)


def test_very_long_line_is_hard_split():
    chunks = chunk_markdown("x.md", "# H\n\n" + "a" * 450, chunk_size=200, overlap=0)
    assert [len(c.text.split("\n", 1)[1]) for c in chunks] == [200, 200, 50]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_markdown("x.md", DOC, chunk_size=100, overlap=100)


def test_empty_document_gives_no_chunks():
    assert chunk_markdown("empty.md", "", chunk_size=100, overlap=10) == []
