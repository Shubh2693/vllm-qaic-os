#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Engine construction and the single-slot engine cache.

Engines are built at the max_model_len a case actually needs (see kv_capacity.py),
so the cache key is (model, tp_size, engine_len, batch_size): context length and batch
size both require a distinct engine, modality mix still does not. Because every modality
at one (context, batch) point rounds to the same engine length, the sweep builds one
engine per such point and reuses it across all input shapes -- which is why
case_matrix.py orders its cases with the modality innermost.

Batch size is part of the identity because two things scale with it: the KV budget,
which must hold that many full-length sequences with prefix caching off, and
max_num_seqs, which otherwise caps concurrency below what the case asked for.

A single-slot cache keeps at most one engine (and therefore one claim on the devices)
alive at a time, whatever order pytest runs the cases in.
"""

import gc
import os

import pytest
from vllm import LLM

from ctx_config import (
    ENGINE_DEFAULT_KWARGS,
    IMAGE_MANY_COUNT,
    MAX_SINGLE_ITEM_TOKENS,
    RNG_SEED,
    VIDEO_MANY_COUNT,
)
from kv_capacity import (
    estimated_device_gb,
    kv_bytes_per_token_per_worker,
    kv_cache_bytes_for,
    skip_if_insufficient_capacity,
    weights_gb,
)
from model_geometry import ModelGeometry
from model_specs import ModelSpec

_ENGINE_SLOT: dict = {"key": None, "llm": None}


def release_engine() -> None:
    """Shut down and drop the cached engine, if any. Safe to call repeatedly."""
    llm = _ENGINE_SLOT.get("llm")
    if llm is None:
        return
    print(f"Releasing engine {_ENGINE_SLOT['key']}")
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
    except Exception as exc:  # teardown must not mask a real test failure
        print(f"Engine shutdown raised (continuing): {exc}")
    _ENGINE_SLOT["key"] = None
    _ENGINE_SLOT["llm"] = None
    del llm
    gc.collect()


def _skip_if_batch_unsupported(batch_size: int) -> None:
    """The SDPA decode path is batch-1 only; say so rather than failing obscurely."""
    if batch_size > 1 and os.environ.get("QAIC_SDPA_DECODE") == "1":
        pytest.skip(
            f"batch={batch_size} needs the paged-attention decode path, but "
            "QAIC_SDPA_DECODE=1 selects the SDPA decode path, which supports batch "
            "size 1 only. Unset it (the default, 0) to run batched cases."
        )


def _multimodal_kwargs(spec: ModelSpec, geom: ModelGeometry, engine_len: int) -> dict:
    """The mm_* engine kwargs, or nothing at all for a text-only model.

    A text-only model has no processor to hand max_pixels to and no mm limits to set,
    so these must be omitted rather than passed with empty values.
    """
    if not (spec.supports_images or spec.supports_video):
        return {}

    # max_pixels has to admit the largest single image the suite builds. The margin
    # keeps that image strictly below the limit rather than exactly on it, so it
    # cannot matter whether smart_resize compares with > or >=. Bounding it by
    # engine_len as well keeps a small engine from advertising a ViT budget it can
    # never use: no builder makes a single item larger than 3/4 of the context. On
    # the PyTorch-eager path this is fully under test control: vllm_qaic's
    # _configure_multimodal_model (which would otherwise clamp it) is gated on
    # cls.is_aot.
    admissible_item_tokens = min(MAX_SINGLE_ITEM_TOKENS, engine_len)
    max_pixels = (admissible_item_tokens + 64) * geom.pixels_per_visual_token
    min_pixels = 4 * geom.pixels_per_visual_token

    # max_pixels/min_pixels/fps are Qwen2-VL-family processor kwarg names specifically
    # (see README's "Model coverage" section) -- a different VLM processor would take
    # different kwargs here, not just different values.
    limits: dict = {
        "image": {
            "count": max(16, IMAGE_MANY_COUNT * 2),
            "width": 4096,
            "height": 4096,
        }
    }
    processor_kwargs: dict = {"max_pixels": max_pixels, "min_pixels": min_pixels}
    if spec.supports_video:
        limits["video"] = {
            "count": max(4, VIDEO_MANY_COUNT * 2),
            "num_frames": 512,
            "width": 1024,
            "height": 1024,
        }
        processor_kwargs["fps"] = 1.0

    return {
        "mm_processor_kwargs": processor_kwargs,
        "limit_mm_per_prompt": limits,
        # Safe because kv_cache_memory_bytes is set explicitly, and it avoids dummy
        # multimodal encoder passes that dominate VLM startup.
        "skip_mm_profiling": True,
    }


# Kwargs the suite computes from the case under test and derives its sizing math
# from -- letting engine_defaults.engine_kwargs or a model's own engine_kwargs
# override any of these would silently desync the engine from what kv_capacity.py
# and case_matrix.py assumed it was built at.
PROTECTED_ENGINE_KWARGS = frozenset(
    {
        "model",
        "dtype",
        "seed",
        "tensor_parallel_size",
        "max_model_len",
        "kv_cache_memory_bytes",
        "trust_remote_code",
        "max_num_seqs",
    }
)


def _default_engine_kwargs(batch_size: int) -> dict:
    """Suite-level LLM(...) defaults, layered under config so they stay overridable."""
    return {
        # Off so that no case is served from another case's prefix, and so that the
        # requests of one batch cannot be served from each other.
        "enforce_eager": True,
        "enable_prefix_caching": False,
        # Required for RequestOutput.metrics (TTFT / prefill / decode timings).
        "disable_log_stats": False,
        "long_prefill_token_threshold": 8192 / batch_size,
    }


def _resolve_engine_kwargs(spec: ModelSpec, batch_size: int) -> dict:
    """Suite defaults, then engine_defaults.engine_kwargs, then the model's own
    engine_kwargs -- each layer overriding the previous key-by-key.

    Raises if any config layer tries to set a kwarg the suite's own sizing/identity
    logic depends on, instead of failing later with Python's cryptic "got multiple
    values for keyword argument" TypeError.
    """
    merged = {
        **_default_engine_kwargs(batch_size),
        **ENGINE_DEFAULT_KWARGS,
        **spec.engine_kwargs,
    }
    offending = PROTECTED_ENGINE_KWARGS & merged.keys()
    if offending:
        raise ValueError(
            f"engine_kwargs may not set {sorted(offending)} -- the suite derives "
            "these from the case under test. Remove them from "
            "engine_defaults.engine_kwargs or this model's own engine_kwargs in "
            "suite_config.json."
        )
    return merged


def _build_engine(
    spec: ModelSpec,
    geom: ModelGeometry,
    tp_size: int,
    engine_len: int,
    batch_size: int,
) -> LLM:
    _skip_if_batch_unsupported(batch_size)
    kv_bytes = kv_cache_bytes_for(geom, tp_size, engine_len, num_seqs=batch_size)
    skip_if_insufficient_capacity(spec, geom, tp_size, kv_bytes, batch_size)

    mm_kwargs = _multimodal_kwargs(spec, geom, engine_len)
    engine_kwargs = _resolve_engine_kwargs(spec, batch_size)

    print(
        f"\nBuilding engine: {spec.model} TP={tp_size} batch={batch_size} "
        f"dtype={spec.dtype} "
        f"max_model_len={engine_len} (model ceiling {geom.max_model_len}) "
        f"kv_cache={kv_bytes / 1024**3:.2f} GiB/worker "
        f"({kv_bytes} bytes, {kv_bytes_per_token_per_worker(geom, tp_size)} B/token) "
        f"weights={weights_gb(spec, geom, tp_size):.1f} GB at "
        f"{geom.weight_bytes_per_param(spec):.1f} B/param "
        f"est_device={estimated_device_gb(spec, geom, tp_size, kv_bytes):.1f} GB"
        f"\n  geometry: {geom.describe()}"
        f"\n  engine_kwargs: {engine_kwargs}"
        + (
            f"\n  multimodal: {mm_kwargs['mm_processor_kwargs']}"
            if mm_kwargs
            else "\n  multimodal: none (text-only)"
        )
    )
    gc.collect()
    return LLM(
        model=spec.model,
        dtype=spec.dtype,
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=engine_len,
        kv_cache_memory_bytes=kv_bytes,
        trust_remote_code=spec.trust_remote_code,
        # The case issues exactly batch_size requests at once; anything lower would
        # silently serialise them.
        max_num_seqs=batch_size,
        **mm_kwargs,
        **engine_kwargs,
    )


def get_engine(
    spec: ModelSpec,
    geom: ModelGeometry,
    tp_size: int,
    engine_len: int,
    batch_size: int = 1,
) -> LLM:
    """The engine for (model, tp, engine_len, batch), evicting any differently-shaped one."""
    key = (spec.model, tp_size, engine_len, batch_size)
    if _ENGINE_SLOT["key"] != key:
        release_engine()
        _ENGINE_SLOT["llm"] = _build_engine(
            spec, geom, tp_size, engine_len, batch_size
        )
        _ENGINE_SLOT["key"] = key
    return _ENGINE_SLOT["llm"]
