"""OpenTelemetry setup. ADK already creates spans for agents, model calls and tools;
we only choose where they go and add one root span per API request.

TRACING_EXPORTER:
  none    (default) no tracer provider is set; OpenTelemetry stays a no-op.
  console print spans to stdout (local debugging).
  gcp     send spans to Cloud Trace with ADK's own exporter helper.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("agentic_analyst")


def setup_tracing(exporter: str, project_id: str = "") -> None:
    if exporter == "none":
        return  # nothing installed: the app behaves exactly as without tracing

    from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers

    resource = None
    if exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        hooks = OTelHooks(span_processors=[SimpleSpanProcessor(ConsoleSpanExporter())])
    elif exporter == "gcp":
        from google.adk.telemetry._gcp_resource import get_gcp_resource
        from google.adk.telemetry.google_cloud import get_gcp_exporters

        hooks = get_gcp_exporters(enable_cloud_tracing=True)
        # Cloud Trace (telemetry.googleapis.com) rejects spans that do not name a
        # project (HTTP 400), so put gcp.project_id on the resource.
        resource = get_gcp_resource(project_id or None)
    else:
        raise ValueError(f"Unknown TRACING_EXPORTER: {exporter!r}")
    maybe_set_otel_providers([hooks], otel_resource=resource)
    logger.info("Tracing enabled: exporter=%s", exporter)


@contextmanager
def request_span(request_id: str, endpoint: str) -> Iterator[trace.Span]:
    """Root span for one API request. ADK's agent/model/tool spans nest under it.

    Without a tracer provider this is a no-op span (attributes are ignored).
    """
    with tracer.start_as_current_span(
        f"POST {endpoint}", attributes={"request_id": request_id, "endpoint": endpoint}
    ) as span:
        yield span
