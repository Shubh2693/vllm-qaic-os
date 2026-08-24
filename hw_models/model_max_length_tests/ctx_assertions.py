#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""The two assertions every case makes: token accounting, and output not collapsed.

The degeneracy checks are a registry so a new failure signature is one function:

    @degeneracy_check
    def _my_check(text, token_ids, stats):
        return "why this is degenerate" if bad else None

Checks run in registration order and the first failure raises, so put the cheapest /
most fundamental ones first.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ctx_config import (
    MAX_CONSECUTIVE_REPEAT,
    MAX_CYCLE_FRACTION,
    MIN_UNIQUE_RATIO,
    TOKEN_SLACK,
    VIDEO_TOKEN_RATIO_SLACK,
)
from model_specs import ModelSpec
from prompt_builder import BuiltInput


def video_ratio_slack_for(spec: ModelSpec | None) -> float:
    """The video accounting ratio for this model: its override, else the global one."""
    if spec is not None and spec.video_ratio_slack is not None:
        return spec.video_ratio_slack
    return VIDEO_TOKEN_RATIO_SLACK

# ---------------------------------------------------------------------------
# Degeneracy statistics
# ---------------------------------------------------------------------------


def longest_run(token_ids) -> int:
    """Length of the longest run of one repeated token id."""
    longest = run = 1
    for previous, current in zip(token_ids, token_ids[1:], strict=False):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    return longest if token_ids else 0


def tail_cycle_fraction(token_ids) -> tuple[float, int]:
    """Largest fraction of the tail explained by a short repeating cycle.

    Catches the common long-context failure of looping a short phrase, which a
    single-token run-length check misses entirely.
    """
    best_fraction, best_period = 0.0, 0
    total = len(token_ids)
    if total < 8:
        return 0.0, 0
    for period in range(1, 9):
        if total < period * 3:
            break
        matched = 0
        for index in range(total - 1, period - 1, -1):
            if token_ids[index] == token_ids[index - period]:
                matched += 1
            else:
                break
        fraction = matched / total
        if fraction > best_fraction:
            best_fraction, best_period = fraction, period
    return best_fraction, best_period


@dataclass(frozen=True)
class DegeneracyStats:
    unique_ratio: float
    longest_run: int
    cycle_fraction: float
    cycle_period: int

    @classmethod
    def measure(cls, token_ids) -> "DegeneracyStats":
        cycle_fraction, cycle_period = tail_cycle_fraction(token_ids)
        return cls(
            unique_ratio=len(set(token_ids)) / len(token_ids),
            longest_run=longest_run(token_ids),
            cycle_fraction=cycle_fraction,
            cycle_period=cycle_period,
        )

    def __str__(self) -> str:
        return (
            f"unique_ratio={self.unique_ratio:.3f} "
            f"longest_run={self.longest_run} "
            f"tail_cycle={self.cycle_fraction:.2f}@period{self.cycle_period}"
        )


# ---------------------------------------------------------------------------
# Degeneracy check registry
# ---------------------------------------------------------------------------

DegeneracyCheck = Callable[[str, list, DegeneracyStats], str | None]
DEGENERACY_CHECKS: list[DegeneracyCheck] = []


def degeneracy_check(fn: DegeneracyCheck) -> DegeneracyCheck:
    """Register a check. Returns a failure message, or None if the output is fine."""
    DEGENERACY_CHECKS.append(fn)
    return fn


@degeneracy_check
def _check_unique_ratio(text: str, token_ids, stats: DegeneracyStats) -> str | None:
    if stats.unique_ratio >= MIN_UNIQUE_RATIO:
        return None
    return (
        f"Degenerate output: only {stats.unique_ratio:.3f} unique tokens "
        f"(threshold {MIN_UNIQUE_RATIO}); text={text[:200]!r}"
    )


@degeneracy_check
def _check_consecutive_run(text: str, token_ids, stats: DegeneracyStats) -> str | None:
    if stats.longest_run <= MAX_CONSECUTIVE_REPEAT:
        return None
    return (
        f"Degenerate output: a token repeats {stats.longest_run} times consecutively "
        f"(threshold {MAX_CONSECUTIVE_REPEAT}); text={text[:200]!r}"
    )


@degeneracy_check
def _check_tail_cycle(text: str, token_ids, stats: DegeneracyStats) -> str | None:
    if stats.cycle_fraction <= MAX_CYCLE_FRACTION:
        return None
    return (
        f"Degenerate output: {stats.cycle_fraction:.0%} of the tail is a period-"
        f"{stats.cycle_period} loop (threshold {MAX_CYCLE_FRACTION:.0%}); "
        f"text={text[:200]!r}"
    )


def assert_not_degenerate(text: str, token_ids) -> None:
    """Reject output that has numerically collapsed rather than merely being wrong.

    Deliberately not an accuracy check -- exact-text references are unusable at
    128K. These are the signatures of real breakage: empty output, a tiny token
    vocabulary, a long single-token run, a looping phrase, or replacement chars.
    """
    assert text is not None and text.strip(), "Generated text is empty or whitespace"
    assert token_ids, "No tokens generated"
    assert "�" not in text, "Generated text contains U+FFFD replacement chars"

    stats = DegeneracyStats.measure(token_ids)
    print(f"Degeneracy check: {stats}")

    for check in DEGENERACY_CHECKS:
        failure = check(text, token_ids, stats)
        print(f"[FAILURE] Degeneracy Failure: {failure}")


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def assert_token_accounting(
    built: BuiltInput,
    actual_tokens: int,
    max_model_len: int,
    spec: ModelSpec | None = None,
) -> None:
    """Realised prompt length must match the analytically predicted length.

    ``spec`` supplies the per-model video slack when it has one: Qwen3-VL interleaves a
    timestamp string between frames, so its realised video token count drifts further
    from the frames-times-patches prediction than the global default allows.
    """
    delta = actual_tokens - built.predicted_tokens
    print(
        f"Token accounting: actual={actual_tokens} predicted={built.predicted_tokens} "
        f"delta={delta:+d} max_model_len={max_model_len}"
    )
    assert actual_tokens <= max_model_len, (
        f"Prompt of {actual_tokens} tokens exceeds max_model_len={max_model_len}; "
        "vLLM should have rejected it"
    )
    if built.exact:
        assert abs(delta) <= TOKEN_SLACK, (
            f"Token accounting mismatch for {built.detail}: actual={actual_tokens} "
            f"predicted={built.predicted_tokens} delta={delta:+d} "
            f"(slack {TOKEN_SLACK}). A non-zero delta here means the processor "
            "expanded visual placeholders differently than predicted."
        )
    else:
        # Video processors may apply their own total-pixel budget and downsample, and
        # some interleave per-frame timestamps into the text.
        ratio = video_ratio_slack_for(spec)
        allowed = max(TOKEN_SLACK, int(built.predicted_tokens * ratio))
        assert abs(delta) <= allowed, (
            f"Video token accounting mismatch for {built.detail}: "
            f"actual={actual_tokens} predicted={built.predicted_tokens} "
            f"delta={delta:+d} (allowed +/-{allowed} at ratio {ratio}). Check whether "
            "the video processor applied its own total-pixel budget or inserted "
            "per-frame timestamps."
        )
