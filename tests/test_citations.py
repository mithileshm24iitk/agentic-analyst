"""cited_sources: extract [file.md] citations and keep only retrieved files."""

from app.citations import cited_sources, extract_citations


def test_extracts_single_and_grouped_citations_in_order():
    answer = (
        "Jeans max is 11.0% [returns_policy.md]. Owner is Rahul [category_owners.md]. "
        "Escalate to the owner [category_owners.md, escalation_process.md]."
    )
    assert extract_citations(answer) == ["returns_policy.md", "category_owners.md", "escalation_process.md"]


def test_ignores_brackets_without_md_files_and_markdown_links():
    answer = "See [this table] and [1] and [link](http://x.md) but cite [kpi_definitions.md]."
    assert extract_citations(answer) == ["kpi_definitions.md"]


def test_handles_semicolons_and_spaces_inside_brackets():
    assert extract_citations("Fact [ a.md ; b.md ].") == ["a.md", "b.md"]


def test_no_citations():
    assert extract_citations("There were 124,780 orders.") == []


def test_cited_sources_drops_files_that_were_not_retrieved():
    answer = "Rule [returns_policy.md]. Invented [made_up.md]."
    assert cited_sources(answer, ["returns_policy.md", "shipping_sla.md"]) == ["returns_policy.md"]


def test_cited_sources_empty_when_nothing_retrieved():
    assert cited_sources("Rule [returns_policy.md].", []) == []
