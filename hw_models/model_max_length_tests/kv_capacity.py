#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""KV cache sizing, the per-device capacity precheck, and engine-length sizing.

The other VLM suites hardcode ``kv_cache_memory_bytes`` to 2 GiB. For a 64-layer /
8-KV-head / 128-head-dim model that is 256 KiB per token across all ranks, so at TP=4
a 2 GiB budget holds only ~32K tokens -- silently short of anything above 32K no
matter what max_model_len claims. Compute it instead.

The budget covers ``num_seqs`` concurrent full-length sequences. With prefix caching
off, a batch of N requests at one context point needs N times the KV of a single one;
sizing for one and issuing N means vLLM preempts and re-prefills instead of running the
batch, which measures something other than what the case claims to measure.

``required_model_len()`` at the bottom answers the companion question -- how long the
engine for one case has to be -- so both halves of "how much does this case cost" live
in one place.
"""

import math

import pytest

from ctx_config import (
    ACTIVATION_HEADROOM_GB,
    DEVICE_MEM_GB,
    ENGINE_LEN_BLOCK,
    ENGINE_LEN_MARGIN_RATIO,
    KV_MIN_GB,
    TOKEN_SLACK,
)
from model_geometry import ModelGeometry
from model_specs import ModelSpec

KV_DTYPE_BYTES = 2  # fp16
KV_BLOCK_TOKENS = 16  # vLLM pages KV in 16-token blocks
KV_SAFETY_FACTOR = 1.05

# The sibling VLM suites hardcode a 2 GiB KV budget, and this file used to carry that
# as an unconditional floor. With engines right-sized per case that floor is the only
# thing binding below ~64K -- a 9216-token engine needs 0.30 GiB and would still have
# reserved 2 GiB -- so it is now opt-in via QAIC_KV_MIN_GB. The computed budget
# already covers num_seqs full max_model_len sequences plus KV_SAFETY_FACTOR. Set
# QAIC_KV_MIN_GB=2 to restore the old behaviour if a device turns out to want the
# headroom.
KV_MIN_BYTES = int(KV_MIN_GB * 1024**3)


def kv_bytes_per_token_per_worker(geom: ModelGeometry, tp_size: int) -> int:
    """KV bytes one token occupies on a single worker.

    vLLM replicates KV heads when tp_size exceeds num_kv_heads, so the per-worker
    head count floors at 1 -- TP=16 on an 8-KV-head model costs the same per worker
    as TP=8.

    Every layer is assumed to hold a full-length cache. A model that alternates in
    sliding-window layers (GPT-OSS) needs less than this, so the estimate is high
    rather than low, which is the direction that fails safely.
    """
    kv_heads = max(1, geom.num_kv_heads // tp_size)
    return 2 * geom.num_layers * kv_heads * geom.head_dim * KV_DTYPE_BYTES


def kv_cache_bytes_for(
    geom: ModelGeometry, tp_size: int, num_tokens: int, num_seqs: int = 1
) -> int:
    """Per-worker KV budget for num_seqs sequences of num_tokens, block-aligned."""
    per_token = kv_bytes_per_token_per_worker(geom, tp_size)
    blocks_per_seq = math.ceil(num_tokens * KV_SAFETY_FACTOR / KV_BLOCK_TOKENS)
    blocks = blocks_per_seq * max(1, num_seqs)
    return max(KV_MIN_BYTES, blocks * KV_BLOCK_TOKENS * per_token)


def weights_gb(spec: ModelSpec, geom: ModelGeometry, tp_size: int) -> float:
    """Per-device weight footprint, accounting for the checkpoint's quantization.

    MoE is already covered by approx_total_params_b: every expert is resident whether
    or not a given token routes to it, so total -- not active -- parameters is the
    right count.
    """
    return spec.approx_total_params_b * geom.weight_bytes_per_param(spec) / tp_size


def estimated_device_gb(
    spec: ModelSpec, geom: ModelGeometry, tp_size: int, kv_bytes: int
) -> float:
    """Rough per-device footprint: sharded weights + replicated ViT + KV + workspace."""
    return (
        weights_gb(spec, geom, tp_size)
        + spec.replicated_vision_gb
        + kv_bytes / 1024**3
        + ACTIVATION_HEADROOM_GB
    )


def skip_if_insufficient_capacity(
    spec: ModelSpec,
    geom: ModelGeometry,
    tp_size: int,
    kv_bytes: int,
    batch_size: int = 1,
) -> None:
    """Skip instead of letting the device OOM mid-run."""
    needed = estimated_device_gb(spec, geom, tp_size, kv_bytes)
    if needed > DEVICE_MEM_GB:
        pytest.skip(
            f"Estimated {needed:.1f} GB/device for {spec.model} at TP={tp_size} "
            f"batch={batch_size} (weights "
            f"{weights_gb(spec, geom, tp_size):.1f} GB at "
            f"{geom.weight_bytes_per_param(spec):.1f} B/param, KV "
            f"{kv_bytes / 1024**3:.1f} GiB) exceeds QAIC_DEVICE_MEM_GB="
            f"{DEVICE_MEM_GB:.0f} GB. Raise TP, lower the batch size, or raise "
            "QAIC_DEVICE_MEM_GB."
        )


# ---------------------------------------------------------------------------
# Engine-length sizing
# ---------------------------------------------------------------------------
# Sizing the engine per case instead of at the model ceiling is what keeps an 8K case
# from paying for 128K of KV cache -- at 32 KiB/token/worker on Qwen2.5-VL-32B that is
# 4.10 GiB/worker reserved to hold ~9K tokens.
#
# Two properties matter and are asserted by test_token_math_selftest.py:
#
#   * the result is never below prompt + generation, or the case would be rejected by
#     its own engine;
#   * every modality at one context point rounds to the same length, so the sweep
#     builds one engine per context point rather than one per case. Engine builds are
#     the dominant cost in this suite; surplus KV is not.


def _round_up(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block


def required_model_len(geom: ModelGeometry, prompt_tokens: int, gen_len: int) -> int:
    """max_model_len to build the engine at for a case of this PL and GL.

    ``prompt_tokens`` is the *predicted* prompt length, so the margin has to cover
    the difference between prediction and what the processor actually emits -- which
    for video is a ratio, not a constant. ENGINE_LEN_MARGIN_RATIO is floored at
    VIDEO_TOKEN_RATIO_SLACK for exactly that reason.

    Capped at the model's own ceiling: a case that wants more than the model can do
    should be rejected by vLLM, which is what test_ctx_boundary.py checks.
    """
    base = prompt_tokens + gen_len
    margin = max(TOKEN_SLACK, math.ceil(base * ENGINE_LEN_MARGIN_RATIO))
    return min(geom.max_model_len, _round_up(base + margin, ENGINE_LEN_BLOCK))
