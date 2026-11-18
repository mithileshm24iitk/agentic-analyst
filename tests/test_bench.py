"""The benchmark's 'stalled call' rule (no GCP)."""

from eval.bench import is_stalled


def call(latency_ms: float, output: int, thinking: int = 0) -> dict:
    return {"latency_ms": latency_ms, "output_tokens": output, "thinking_tokens": thinking}


def test_slow_call_with_few_tokens_is_stalled():
    assert is_stalled(call(42_000, 22))  # 0.5 tokens/s


def test_slow_call_that_generates_a_lot_is_not_stalled():
    assert not is_stalled(call(17_000, 300, 2000))  # busy thinking, not waiting


def test_fast_call_is_never_stalled():
    assert not is_stalled(call(2_000, 1))
