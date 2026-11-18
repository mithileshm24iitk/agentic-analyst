"""Cost calculation from the price table."""

import pytest

from app.config import DEFAULT_MODEL_PRICES, ModelPrice
from app.costs import call_cost_usd, total_cost_usd

PRICES = {
    "priced": ModelPrice(input_per_1m=0.75, output_per_1m=3.75, cached_input_per_1m=0.075),
    "no-cache-rate": ModelPrice(input_per_1m=1.0, output_per_1m=2.0),
    "explicitly-unknown": None,
}


def test_input_output_and_thinking_tokens_priced():
    # 1M input at $0.75 + (1M output + 1M thinking) at $3.75 = $8.25
    assert call_cost_usd("priced", 1_000_000, 1_000_000, 1_000_000, 0, PRICES) == pytest.approx(8.25)


def test_thinking_tokens_cost_the_same_as_output_tokens():
    as_output = call_cost_usd("priced", 0, 1000, 0, 0, PRICES)
    as_thinking = call_cost_usd("priced", 0, 0, 1000, 0, PRICES)
    assert as_output == as_thinking == pytest.approx(0.00375)


def test_cached_input_tokens_use_the_cached_rate():
    # 1000 input of which 400 cached: 600 * 0.75 + 400 * 0.075, per 1M
    assert call_cost_usd("priced", 1000, 0, 0, 400, PRICES) == pytest.approx((600 * 0.75 + 400 * 0.075) / 1e6)


def test_cached_tokens_fall_back_to_input_rate_without_cached_price():
    assert call_cost_usd("no-cache-rate", 1000, 0, 0, 400, PRICES) == pytest.approx(1000 * 1.0 / 1e6)


@pytest.mark.parametrize("model", ["not-in-table", "explicitly-unknown", ""])
def test_unknown_model_cost_is_null_never_zero(model):
    assert call_cost_usd(model, 1000, 1000, 1000, 0, PRICES) is None


def test_total_is_null_if_any_call_is_unknown():
    assert total_cost_usd([0.01, None, 0.02]) is None
    assert total_cost_usd([0.01, 0.02]) == pytest.approx(0.03)
    assert total_cost_usd([]) == 0


def test_default_table_has_the_worker_model_from_the_pricing_page():
    price = DEFAULT_MODEL_PRICES["gemini-3.8-flash"]
    assert (price.input_per_1m, price.output_per_1m, price.cached_input_per_1m) == (0.75, 3.75, 0.075)
