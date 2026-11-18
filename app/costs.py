"""Estimated cost of one model call, from the price table in config."""

from app.config import ModelPrice


def call_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    cached_tokens: int,
    prices: dict[str, ModelPrice | None],
) -> float | None:
    """USD for one call, or None when the model has no known price (never a guess).

    input_tokens includes cached_tokens (that is how Gemini reports prompt_token_count);
    cached tokens are charged at the cached rate when the table has one.
    Thinking tokens are charged at the output rate.
    """
    price = prices.get(model)
    if price is None:
        return None
    cached = min(cached_tokens, input_tokens)
    cached_rate = price.cached_input_per_1m if price.cached_input_per_1m is not None else price.input_per_1m
    usd = (
        (input_tokens - cached) * price.input_per_1m
        + cached * cached_rate
        + (output_tokens + thinking_tokens) * price.output_per_1m
    ) / 1_000_000
    return round(usd, 8)


def total_cost_usd(costs: list[float | None]) -> float | None:
    """Sum of call costs; None if any call's cost is unknown (a partial sum would mislead)."""
    if any(c is None for c in costs):
        return None
    return round(sum(costs), 8)  # type: ignore[arg-type]
