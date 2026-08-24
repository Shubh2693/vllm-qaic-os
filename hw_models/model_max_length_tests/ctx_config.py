#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Every knob the max-context suite reads, in one place -- sourced from suite_config.json.

suite_config.json (next to this file, or wherever MAX_CTX_CONFIG_FILE points) is the
only file a user needs to edit to change models, TP sizes, context targets, batch
sizes, generation length, or any assertion/device threshold. Every other module in
this directory imports named constants from here and never touches the config file
itself, which is what keeps them a black box.

MAX_CTX_CONFIG_FILE is the one surviving environment variable, and it is a file
*locator*, not a knob: it lets run_tests.sh point a subprocess at a patched copy of
the config (for --tier/--tp/--models/--batch and the per-run summary path) without
touching the checked-in suite_config.json. No other env var is read here; a missing
or malformed config raises immediately rather than falling back to a Python default,
so a typo in the JSON is caught at import time instead of silently doing nothing.

Nothing here touches a model, a tokenizer or a device, so this module is safe to
import from anywhere in the suite -- including a plain Python shell when you want to
check what a given suite_config.json resolves to.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(
    os.environ.get("MAX_CTX_CONFIG_FILE") or Path(__file__).parent / "suite_config.json"
)

try:
    _CONFIG = json.loads(CONFIG_PATH.read_text())
except FileNotFoundError:
    raise RuntimeError(
        f"Max-context suite config not found: {CONFIG_PATH}. Set MAX_CTX_CONFIG_FILE "
        "or restore suite_config.json next to ctx_config.py."
    ) from None
except json.JSONDecodeError as exc:
    raise RuntimeError(f"Max-context suite config {CONFIG_PATH} is not valid JSON: {exc}") from exc

_MISSING = object()


def _get(*path, default=_MISSING):
    """Walk suite_config.json by dotted path; raise unless a default is given.

    A key that is present but explicitly ``null`` (e.g. engine_len_margin_ratio's
    "derive it" sentinel) returns None successfully -- only an absent key raises or
    falls back to ``default``.
    """
    node = _CONFIG
    for key in path:
        if not isinstance(node, dict) or key not in node:
            if default is _MISSING:
                raise RuntimeError(
                    f"{CONFIG_PATH} is missing required key {'.'.join(path)!r}"
                )
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
# Seeded once per session in conftest.py, and reused for synthetic image/video
# content, the engine seed and the sampling seed, so that token counts and
# generated text stay deterministic across runs. Not a suite_config.json knob --
# nobody tunes this per-run, it just has to be fixed.
RNG_SEED = 42


# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------
# "smoke" keeps the matrix cheap; "full" walks every context point up to the max;
# "custom" walks exactly sweep.custom_targets and nothing else -- no smoke/full
# auto-ceiling point gets added (see case_matrix.ctx_targets_for). run_tests.sh's
# --pl/--gl set this tier and custom_targets for one invocation, for "run exactly
# this (prompt_len, gen_len) point" without editing the file.
TIER = _get("sweep", "tier").lower()


def _targets(key: str) -> tuple[tuple[int, int], ...]:
    """sweep.<key> as (prompt_len, gen_len) pairs.

    Each pair is explicit and independent -- prompt length and generation length are
    both first-class knobs, not one derived from the other. ctx_len (prompt_len +
    gen_len) is computed downstream, for reporting only; it is never the input.
    """
    return tuple((entry["prompt_len"], entry["gen_len"]) for entry in _get("sweep", key))


SMOKE_TARGETS = _targets("smoke_targets")
FULL_TARGETS = _targets("full_targets")
CUSTOM_TARGETS = _targets("custom_targets")

# Video at the full ceiling needs ~250 frames / ~600MB of raw uint8 and several GB
# during preprocessing, so it gets its own (lower) default ceiling.
VIDEO_CTX_MAX = _get("sweep", "video_ctx_max")

SELECTED_MODELS = _get("selection", "models") or _get("default_models")
SELECTED_TPS = list(_get("selection", "tp_sizes"))

# null -> every modality a model's capabilities allow (today's behavior, unchanged).
# Else a list of modality names (see prompt_builder.MODALITIES) to restrict the sweep
# to. Modalities a selected model does not support are simply absent from the
# intersection, the same "yields nothing rather than erroring" behavior SELECTED_TPS
# already has for a TP size a model does not declare. An explicit empty list is
# distinct from null -- it means "no modalities," not "no restriction."
_selected_modalities = _get("selection", "modalities", default=None)
SELECTED_MODALITIES = (
    None if _selected_modalities is None else list(_selected_modalities)
)

# Requests issued per case. Each distinct batch size is a distinct engine (KV budget
# and max_num_seqs both scale with it), so this axis costs engine builds -- hence the
# default of 1. Batch > 1 needs the paged-attention decode path, i.e. do not also set
# QAIC_SDPA_DECODE=1; engine_pool skips the case if you do.
BATCH_SIZES = list(_get("selection", "batch_sizes"))

IMAGE_MANY_COUNT = _get("sweep", "image_many_count")
VIDEO_MANY_COUNT = _get("sweep", "video_many_count")


# ---------------------------------------------------------------------------
# Engine limits
# ---------------------------------------------------------------------------
# Chunked prefill must not split a single multimodal item, so this has to be >= the
# largest single image/video the suite builds (capped by MAX_SINGLE_ITEM_TOKENS).
MAX_SINGLE_ITEM_TOKENS = _get("engine_limits", "max_single_item_tokens")

# Extra LLM(...) kwargs applied to every model, before a model's own ModelSpec.engine_kwargs
# (which takes precedence key-by-key). See engine_pool.py's default_engine_kwargs for
# which keys this can and cannot override.
ENGINE_DEFAULT_KWARGS: dict = dict(_get("engine_defaults", "engine_kwargs", default={}))

# Per-device memory budget for the capacity precheck (skip rather than OOM).
DEVICE_MEM_GB = float(_get("device", "device_mem_gb"))
# Rough allowance for activations / workspace / fragmentation per device.
ACTIVATION_HEADROOM_GB = float(_get("device", "activation_headroom_gb"))


# ---------------------------------------------------------------------------
# Vision-encoder attention ceiling
# ---------------------------------------------------------------------------
# A single image is one ViT sequence, and the vision blocks that attend over all of it
# materialise a [heads_per_rank, patches, patches] score matrix. torch_qaic prechecks
# the total operand size of an op against the per-NSP virtual address space (~4 GB,
# less reserved headroom) and returns QAIC_ERROR_MMAP_FAILURE above it, so an image
# large enough to blow that budget cannot run at all: a 16256-visual-token image is
# 65024 patches, which is 16.9 GiB of scores per rank at TP=8.
#
# visual_inputs.max_visual_tokens_per_item() inverts that bound, and the image/video
# builders clamp their per-item budget to it. Set item_autocap off in the config to
# build the item the context point asks for regardless -- useful for reproducing the
# failure on purpose.
VIT_VA_BUDGET_GB = float(_get("vision", "vit_va_budget_gb"))
ITEM_AUTOCAP = bool(_get("vision", "item_autocap"))


# ---------------------------------------------------------------------------
# Assertion thresholds
# ---------------------------------------------------------------------------
# Token-accounting slack. Text/image expansion is exactly predictable; the video
# processor may apply its own total-pixel budget, so video gets a ratio tolerance.
TOKEN_SLACK = _get("assertions", "token_slack")
VIDEO_TOKEN_RATIO_SLACK = float(_get("assertions", "video_token_ratio_slack"))

# Degeneracy thresholds -- ctx_assertions.py documents what each one catches.
MIN_UNIQUE_RATIO = float(_get("assertions", "min_unique_ratio"))
MAX_CONSECUTIVE_REPEAT = _get("assertions", "max_consecutive_repeat")
MAX_CYCLE_FRACTION = float(_get("assertions", "max_cycle_fraction"))

PERF_TOLERANCE = float(_get("assertions", "perf_tolerance"))


# ---------------------------------------------------------------------------
# Engine right-sizing
# ---------------------------------------------------------------------------
# Each case builds its engine at the max_model_len that case actually needs (prompt
# length + generation length + drift allowance) rather than at the model's ceiling,
# so an 8K case does not pay for 128K of KV cache. kv_capacity.py does the
# arithmetic; the boundary tests opt out and ask for the true ceiling.
#
# Engine lengths are rounded up to this block so that every modality at one context
# point lands on the same length and therefore shares one engine. Rebuilding a 32B
# model's engine costs minutes, while a few thousand tokens of surplus KV costs
# ~100 MB; the block trades the cheap resource for the expensive one. Keep it a
# multiple of kv_capacity.KV_BLOCK_TOKENS.
ENGINE_LEN_BLOCK = _get("engine_limits", "engine_len_block")

# The engine has to tolerate at least as much token drift as the accounting
# assertion does, otherwise a video case that passes assert_token_accounting could
# still be rejected by the very engine it was sized for. Hence the floor.
# engine_limits.engine_len_margin_ratio may be null in the config, meaning "derive it
# from video_token_ratio_slack" -- the same value the floor uses.
_configured_margin_ratio = _get("engine_limits", "engine_len_margin_ratio")
ENGINE_LEN_MARGIN_RATIO = max(
    VIDEO_TOKEN_RATIO_SLACK if _configured_margin_ratio is None else _configured_margin_ratio,
    VIDEO_TOKEN_RATIO_SLACK,
)

# Optional floor under the computed KV budget, in GiB. 0 means "trust the
# arithmetic" -- see the note in kv_capacity.py. Set to 2 to reproduce what the
# sibling VLM suites hardcode.
KV_MIN_GB = float(_get("device", "kv_min_gb"))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# Defaults reproduce the suite's original hardcoded greedy decoding exactly
# (temperature=0.0, top_k disabled, no repetition penalty). Overriding these is an
# explicit opt-in: the degeneracy and token-accounting assertions were written and
# tuned against greedy output, not against sampled generation.
SAMPLING_TEMPERATURE = float(_get("sampling", "temperature"))
SAMPLING_TOP_P = float(_get("sampling", "top_p"))
SAMPLING_TOP_K = int(_get("sampling", "top_k"))
SAMPLING_REPETITION_PENALTY = float(_get("sampling", "repetition_penalty"))
SAMPLING_IGNORE_EOS = bool(_get("sampling", "ignore_eos"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SUMMARY_FILE = Path(_get("reporting", "summary_file"))
