"""Tracing config: 'none' installs nothing; 'console' exports spans with our attributes."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from opentelemetry import trace

from app.tracing import request_span, setup_tracing

ROOT = Path(__file__).resolve().parent.parent


def test_none_installs_no_tracer_provider_and_spans_are_no_ops():
    before = trace.get_tracer_provider()

    setup_tracing("none")

    assert trace.get_tracer_provider() is before
    assert type(before).__name__ == "ProxyTracerProvider"  # OpenTelemetry's built-in no-op default
    with request_span("req-1", "/ask") as span:
        span.set_attribute("route", "docs")  # accepted and ignored
        assert not span.is_recording()


def test_unknown_exporter_is_rejected():
    with pytest.raises(ValueError, match="TRACING_EXPORTER"):
        setup_tracing("jaeger")


def test_console_exporter_records_root_span_with_request_id_and_route():
    """Run in a subprocess: setting a global tracer provider must not leak into other tests."""
    code = textwrap.dedent(
        """
        import asyncio
        from opentelemetry import trace
        from app.tracing import request_span, setup_tracing
        from agents.orchestrator import build_orchestrator
        from app.agent_service import AgentService
        from tests.scripted_llm import ScriptedLlm

        setup_tracing("console")
        orchestrator = build_orchestrator(ScriptedLlm(script=["Out of scope."]), ScriptedLlm(script=[]),
                                          "p.d", 2)
        with request_span("req-trace-1", "/ask") as span:
            asyncio.run(AgentService(orchestrator).ask("weather?"))
            span.set_attribute("route", "none")
        trace.get_tracer_provider().force_flush()
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]

    # ConsoleSpanExporter prints one JSON object per span.
    spans, depth, buf = [], 0, ""
    for line in out.stdout.splitlines():
        depth += line.count("{") - line.count("}")
        buf += line + "\n"
        if depth == 0 and buf.strip():
            spans.append(json.loads(buf))
            buf = ""
    root = next(s for s in spans if s["name"] == "POST /ask")
    assert root["attributes"]["request_id"] == "req-trace-1"
    assert root["attributes"]["route"] == "none"
    # ADK's built-in spans are children in the same trace.
    adk_spans = [s for s in spans if s is not root and s["context"]["trace_id"] == root["context"]["trace_id"]]
    assert adk_spans, [s["name"] for s in spans]
