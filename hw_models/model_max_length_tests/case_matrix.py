#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Parametrisation: which (model, TP, modality, prompt len, gen len, batch) cases exist.

Collection must never fail. The sweep needs the HF config to know which (prompt_len,
gen_len) points are legal, so a model whose config cannot be read (offline, gated repo)
contributes no sweep cases. The boundary cases need no config at collection time, so
they are always emitted and ``model_geometry.resolve()`` reports the real reason as a
skip when the test runs.
"""

import pytest

from ctx_config import (
    BATCH_SIZES,
    CUSTOM_TARGETS,
    FULL_TARGETS,
    SELECTED_MODALITIES,
    SELECTED_MODELS,
    SELECTED_TPS,
    SMOKE_TARGETS,
    TIER,
    VIDEO_CTX_MAX,
)
from model_geometry import ModelGeometry, geometry_for
from model_specs import MODEL_SPECS
from prompt_builder import modalities_for


def ctx_targets_for(geom: ModelGeometry, modality_name: str) -> list[tuple[int, int]]:
    """(prompt_len, gen_len) pairs to sweep, clamped to the model max and the video ceiling.

    prompt_len and gen_len are independent, explicit knobs (suite_config.json's
    sweep.smoke_targets/full_targets/custom_targets), not one derived from the other
    -- ctx_len (prompt_len + gen_len) is a downstream, reported quantity, never the
    input. "custom" (run_tests.sh's --pl/--gl) walks exactly custom_targets and never
    gets the "full" tier's auto-ceiling point below -- it means "run exactly this,"
    not "and also the ceiling."

    video/video_many/mixed clamp on prompt_len alone against VIDEO_CTX_MAX: decode
    tokens don't add visual content, so a long generation shouldn't count against the
    video-specific ceiling. Every modality still drops a pair whose prompt_len +
    gen_len would exceed the model's true ceiling outright, regardless of modality.
    """
    if TIER == "full":
        targets = FULL_TARGETS
    elif TIER == "custom":
        targets = CUSTOM_TARGETS
    else:
        targets = SMOKE_TARGETS
    prompt_ceiling = (
        VIDEO_CTX_MAX
        if modality_name in ("video", "video_many", "mixed")
        else geom.max_model_len
    )

    pairs = {
        (prompt_len, gen_len)
        for prompt_len, gen_len in targets
        if prompt_len <= prompt_ceiling and prompt_len + gen_len <= geom.max_model_len
    }
    # Always include the modality's own ceiling in the full tier, using the largest
    # configured gen_len as its generation length -- there is no single global default
    # to fall back on anymore.
    if TIER == "full" and targets:
        ceiling_gen_len = max(gen_len for _, gen_len in targets)
        ceiling_prompt_len = min(prompt_ceiling, geom.max_model_len - ceiling_gen_len)
        pairs.add((ceiling_prompt_len, ceiling_gen_len))
    return sorted(pairs)


def _sweep_modalities_for(spec) -> tuple:
    """Modalities to sweep for this model: its capabilities, narrowed by
    selection.modalities when set.

    Mirrors _selected_model_tps()'s behavior for tp_sizes -- a configured modality
    the model does not support (or a typo) just yields nothing for that model rather
    than raising, since collection must never fail.
    """
    supported = modalities_for(spec)
    if SELECTED_MODALITIES is None:
        return supported
    return tuple(name for name in supported if name in SELECTED_MODALITIES)


def _selected_model_tps():
    """(model_name, spec, tp_size) for every selected combination. No config read."""
    for model_name in SELECTED_MODELS:
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            continue
        for tp_size in sorted(SELECTED_TPS):
            if tp_size in spec.tp_sizes:
                yield model_name, spec, tp_size


def build_sweep_cases() -> list:
    """Cases ordered so that consecutive cases share an engine.

    Engines are right-sized per case, and the (prompt_len, gen_len) pair and the batch
    size are both part of the engine identity: the KV budget covers batch_size
    full-length sequences and max_num_seqs is set from it. So the emitted order is
    model -> TP -> (prompt_len, gen_len) -> batch -> modality, with modality innermost
    because it is the only one of the five that does *not* change the engine -- all
    modalities at one (prompt_len, gen_len, batch) point round to the same engine
    length (see kv_capacity.py) and reuse a single engine. Ordering modality-first
    would rebuild the engine for every case.
    """
    cases = []
    for model_name, spec, tp_size in _selected_model_tps():
        try:
            geom = geometry_for(spec)
        except Exception:
            # Config unavailable (offline / gated). Collection must still succeed;
            # without it there is no way to know which (prompt_len, gen_len) points
            # are legal.
            continue
        # Invert modality -> (prompt_len, gen_len) points into point -> modalities, so
        # the emitted order is target-major. Video, video_many and mixed drop out of
        # the high prompt-length points because ctx_targets_for caps them at
        # VIDEO_CTX_MAX; a text-only model contributes only "text" because
        # modalities_for filters on its capabilities.
        by_target: dict[tuple[int, int], list[str]] = {}
        for modality_name in _sweep_modalities_for(spec):
            for target in ctx_targets_for(geom, modality_name):
                by_target.setdefault(target, []).append(modality_name)

        for prompt_len, gen_len in sorted(by_target):
            for batch_size in BATCH_SIZES:
                for modality_name in by_target[(prompt_len, gen_len)]:
                    cases.append(
                        pytest.param(
                            model_name,
                            tp_size,
                            modality_name,
                            prompt_len,
                            gen_len,
                            batch_size,
                            id=f"{model_name.split('/')[-1]}-tp{tp_size}-"
                            f"{modality_name}-pl{prompt_len}-gl{gen_len}-bs{batch_size}",
                        )
                    )
    return cases


def build_boundary_cases() -> list:
    """One case per (model, tp) -- the boundary tests always use text-only prompts."""
    return [
        pytest.param(
            model_name, tp_size, id=f"{model_name.split('/')[-1]}-tp{tp_size}"
        )
        for model_name, _spec, tp_size in _selected_model_tps()
    ]


SWEEP_CASES = build_sweep_cases()
BOUNDARY_CASES = build_boundary_cases()
