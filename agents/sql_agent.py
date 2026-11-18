"""Phase 2 agent: answers questions with BigQuery SQL, retrying on errors."""

from google.adk.agents import Agent

from tools.bigquery_tools import get_schema, run_query


def build_sql_agent(model: str, dataset: str, max_attempts: int) -> Agent:
    """Create the SQL agent. Model, dataset and attempt cap come from config."""
    return Agent(
        name="sql_agent",
        model=model,
        description=f"Answers data questions by querying the BigQuery dataset {dataset}.",
        instruction=f"""You answer questions about the BigQuery dataset `{dataset}`.

Steps:
1. Call get_schema first to see the real tables and columns. Never guess names.
2. Write one BigQuery Standard SQL SELECT query. Use fully qualified table names
   in backticks, e.g. `{dataset}.orders`. Select only the columns you need.
3. Call run_query with your SQL.
4. If run_query returns ok=false, read `error`, fix the SQL and call run_query again.
   If it returns ok=true but zero rows, check your filters and values against the
   schema and try again if a fix is likely.
5. You have at most {max_attempts} run_query calls. If you get error_type
   "attempt_limit", stop and explain what you tried and why it did not work.

Answer from the returned rows only. Be concise and include the key numbers.
Never invent data that was not returned by run_query.""",
        tools=[get_schema, run_query],
    )
