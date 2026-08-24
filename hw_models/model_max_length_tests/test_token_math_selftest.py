#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Token-math self-check: validates the builders with no device and no engine.

This is the cheap guard on the machinery the on-device accounting assertion relies
on, and on the registry itself -- a model declared with the wrong capabilities, or a
config whose geometry does not derive, fails here in seconds instead of after a
multi-minute engine build. Run it first.

    pytest test_token_math_selftest.py -s
"""

import pytest

from case_matrix import ctx_targets_for
from ctx_config import (
    BATCH_SIZES,
    SELECTED_MODELS,
    TOKEN_SLACK,
)
from ctx_assertions import video_ratio_slack_for
from filler_text import text_of_exact_len
from kv_capacity import (
    estimated_device_gb,
    kv_bytes_per_token_per_worker,
    kv_cache_bytes_for,
    required_model_len,
    weights_gb,
)
from model_geometry import resolve, tokenizer_for
from prompt_builder import build_batch, build_input, modalities_for
from visual_inputs import (
    make_image_of_tokens,
    make_video_of_tokens,
    max_visual_tokens_per_item,
)

# A generation length for the probes below that don't sweep suite_config.json's
# actual (prompt_len, gen_len) pairs -- they exercise the token math at one
# self-contained point, so they need *a* value, not the real sweep's.
_PROBE_GEN_LEN = 64


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_geometry_and_kv_sizing(model_name: str):
    """Print the derived geometry and the KV budget it implies at each TP size."""
    spec, geom = resolve(model_name)

    print(f"\n{spec.model}: {geom.describe()}")
    print(f"  modalities: {', '.join(modalities_for(spec))}")
    for tp_size in spec.tp_sizes:
        kv_bytes = kv_cache_bytes_for(geom, tp_size, geom.max_model_len)
        print(
            f"  TP={tp_size}: {kv_bytes_per_token_per_worker(geom, tp_size)} B/token, "
            f"KV@max={kv_bytes / 1024**3:.2f} GiB/worker, "
            f"weights={weights_gb(spec, geom, tp_size):.1f} GB, "
            f"est_device={estimated_device_gb(spec, geom, tp_size, kv_bytes):.1f} GB"
        )
        assert kv_bytes > 0
        assert weights_gb(spec, geom, tp_size) > 0


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_capabilities_match_config(model_name: str):
    """The registry's declared capabilities must agree with the HF config.

    Catches the two ways a new entry goes wrong: declaring images on a text-only model
    (which would fail deep inside the processor on a device) and forgetting to declare
    them on a VLM (which would silently never exercise the vision path).
    """
    spec, geom = resolve(model_name)

    mismatch = geom.capability_mismatch(spec)
    assert mismatch is None, mismatch

    # A model with no vision tower must contribute exactly the text modality, and one
    # with a vision tower must contribute more than that.
    modalities = modalities_for(spec)
    assert "text" in modalities
    if geom.has_vision:
        assert len(modalities) > 1, (
            f"{spec.model} has a vision tower but only produced {modalities}"
        )
    else:
        assert modalities == ("text",), (
            f"{spec.model} is text-only but produced {modalities}"
        )

    # MoE and quantization are derived, never declared, so just report them: an
    # unexpected value here means the config moved, not that the suite is wrong.
    print(
        f"\n  {spec.model}: moe={geom.is_moe} experts={geom.num_experts}"
        f"/{geom.num_experts_per_tok} quant={geom.quantization} "
        f"weight_bytes={geom.weight_bytes_per_param(spec):.1f} "
        f"dtype={spec.dtype} vision={geom.has_vision}"
    )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_filler_hits_target_length(model_name: str):
    """Filler text re-encodes to (within slack of) the requested token count."""
    spec, _geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    for target in (128, 1024, 8192):
        text = text_of_exact_len(tokenizer, target)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        print(f"  text target={target} actual={actual}")
        assert abs(actual - target) <= TOKEN_SLACK, (
            f"text_of_exact_len({target}) produced {actual} tokens"
        )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_filler_variants_differ_at_equal_length(model_name: str):
    """Batch variants must be distinct text of the same length.

    Both halves matter: equal length keeps every request in a batch at the context
    point under test, and distinctness is what stops vLLM serving request N from
    request 0's prefix.
    """
    spec, _geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    target = 1024
    texts = [text_of_exact_len(tokenizer, target, variant=v) for v in range(4)]
    lengths = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    print(f"  variants at target={target}: lengths={lengths}")

    assert len(set(texts)) == len(texts), "Filler variants are not distinct"
    for index, actual in enumerate(lengths):
        assert abs(actual - target) <= TOKEN_SLACK, (
            f"variant {index} produced {actual} tokens for a target of {target}"
        )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_image_token_math(model_name: str):
    """Image dimensions are block-aligned and the token count is rows*cols."""
    spec, geom = resolve(model_name)
    if not spec.supports_images:
        pytest.skip(f"{spec.model} is text-only")

    block = geom.visual_token_px
    for target in (64, 256, 1024, 4096):
        image, tokens = make_image_of_tokens(geom, target)
        print(
            f"  image target={target} -> {tokens} tokens, {image.width}x{image.height}"
        )
        assert image.width % block == 0 and image.height % block == 0
        assert tokens == (image.height // block) * (image.width // block)
        assert tokens <= target


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_image_variants_differ(model_name: str):
    """Same token count, different pixels -- or the encoder cache dedupes the batch."""
    spec, geom = resolve(model_name)
    if not spec.supports_images:
        pytest.skip(f"{spec.model} is text-only")

    target = 256
    payloads, counts = [], []
    for variant in range(4):
        image, tokens = make_image_of_tokens(geom, target, variant=variant)
        payloads.append(image.tobytes())
        counts.append(tokens)

    print(f"  image variants at target={target}: tokens={counts}")
    assert len(set(counts)) == 1, "Variants must not change the token count"
    assert len(set(payloads)) == len(payloads), (
        "Image variants are byte-identical; vLLM would hash them to one encoder "
        "cache entry and the batch would run the ViT once"
    )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_video_token_math(model_name: str):
    """Frame count is a multiple of temporal_patch_size and tokens follow the patching.

    tokens = (F / temporal_patch_size) * (H / block) * (W / block)
    """
    spec, geom = resolve(model_name)
    if not spec.supports_video:
        pytest.skip(f"{spec.model} does not support video")

    block = geom.visual_token_px
    for target in (256, 1024, 8192):
        frames, tokens, num_frames = make_video_of_tokens(geom, target)
        height, width = frames.shape[1], frames.shape[2]
        print(
            f"  video target={target} -> {tokens} tokens, {num_frames} frames "
            f"{width}x{height} ({frames.nbytes / 1024**2:.1f} MiB)"
        )
        assert num_frames % geom.temporal_patch_size == 0
        expected = (
            (num_frames // geom.temporal_patch_size)
            * (height // block)
            * (width // block)
        )
        assert tokens == expected
        assert tokens <= target


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_video_variants_differ(model_name: str):
    """Same token count, different pixels -- mirrors test_selftest_image_variants_differ.

    video_many puts several clips in one prompt, so distinctness matters there the
    same way it does for image_many: identical clips would hash to one entry in
    vLLM's multimodal encoder cache and the ViT would only run once.
    """
    spec, geom = resolve(model_name)
    if not spec.supports_video:
        pytest.skip(f"{spec.model} does not support video")

    target = 256
    payloads, counts = [], []
    for variant in range(4):
        frames, tokens, _num_frames = make_video_of_tokens(geom, target, variant=variant)
        payloads.append(frames.tobytes())
        counts.append(tokens)

    print(f"  video variants at target={target}: tokens={counts}")
    assert len(set(counts)) == 1, "Variants must not change the token count"
    assert len(set(payloads)) == len(payloads), (
        "Video variants are byte-identical; vLLM would hash them to one encoder "
        "cache entry and the batch would run the ViT once"
    )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_visual_items_fit_the_vit_ceiling(model_name: str):
    """No built item may exceed what the ViT's attention can actually map.

    One image (or one clip) is one ViT sequence, and the vision blocks that attend over
    all of it materialise a [heads_on_rank, patches, patches] score matrix. Above the
    per-NSP address space that operator cannot be mapped at all, so an over-large item
    is a hard device failure rather than a slow path. The builders clamp to this bound;
    this asserts they actually did.
    """
    spec, geom = resolve(model_name)
    if not (spec.supports_images or spec.supports_video):
        pytest.skip(f"{spec.model} is text-only")
    tokenizer = tokenizer_for(spec)

    for tp_size in spec.tp_sizes:
        ceiling = max_visual_tokens_per_item(geom, tp_size)
        if ceiling is None:
            pytest.skip("No ViT attention ceiling derivable for this model")
        print(f"\n  TP={tp_size}: ceiling {ceiling} visual tokens per item")

        for modality in modalities_for(spec):
            if modality == "text":
                continue
            # 32K is the point where an unclamped single image would be 16256 tokens.
            target = min(32 * 1024, geom.max_model_len)
            built = build_input(spec, geom, tokenizer, modality, target, tp_size=tp_size)
            # image_many splits its budget across IMAGE_MANY_COUNT items and
            # video_many/mixed similarly split theirs, so bound the largest single
            # item rather than the total.
            largest_item = max(
                built.image_tokens // max(1, built.num_images),
                built.video_tokens // max(1, built.num_videos),
            )
            print(f"    {modality}: largest item {largest_item} tok | {built.detail}")
            assert largest_item <= ceiling, (
                f"{modality}@{target} built a {largest_item}-token item at TP="
                f"{tp_size}, above the {ceiling}-token ViT ceiling"
            )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_batch_inputs_are_distinct(model_name: str):
    """A batch is N distinct prompts of the same predicted length."""
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    batch_size = max(2, max(BATCH_SIZES))
    for modality in modalities_for(spec):
        batch = build_batch(
            spec, geom, tokenizer, modality, 4096, batch_size, tp_size=8
        )
        lengths = [built.predicted_tokens for built in batch]
        print(f"  {modality} x{batch_size}: predicted={lengths}")

        assert len(batch) == batch_size
        assert max(lengths) - min(lengths) <= TOKEN_SLACK, (
            f"{modality}: batch members differ in length by more than the slack: "
            f"{lengths}"
        )
        prompts = [built.prompt for built in batch]
        assert len(set(prompts)) == batch_size, (
            f"{modality}: batch members are not distinct prompts"
        )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_kv_scales_with_batch(model_name: str):
    """The KV budget must cover batch_size full-length sequences, not just one.

    With prefix caching off there is no sharing between the requests of a batch, so a
    budget sized for one sequence would make vLLM preempt and re-prefill instead of
    running the batch concurrently.
    """
    spec, geom = resolve(model_name)

    tokens = min(8192, geom.max_model_len)
    single = kv_cache_bytes_for(geom, 8, tokens, num_seqs=1)
    for batch_size in (2, 4):
        batched = kv_cache_bytes_for(geom, 8, tokens, num_seqs=batch_size)
        print(
            f"  batch={batch_size}: {batched / 1024**3:.2f} GiB vs "
            f"{single / 1024**3:.2f} GiB at batch=1"
        )
        assert batched == single * batch_size, (
            f"KV for batch={batch_size} is {batched}, expected {single * batch_size}"
        )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_engine_sizing(model_name: str):
    """Right-sized engines must fit their case, tolerate drift, and be shared.

    Three properties, each of which would cost real device time to discover:

      * engine_len >= PL + GL, or the case is rejected by its own engine;
      * engine_len leaves room for the drift assert_token_accounting tolerates, so a
        video case that passes accounting cannot be refused by the engine;
      * all modalities at one (prompt_len, gen_len) point land on the *same*
        engine_len, so the sweep builds one engine per point instead of one per case.
    """
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    saving = kv_cache_bytes_for(geom, 8, geom.max_model_len) - kv_cache_bytes_for(
        geom, 8, required_model_len(geom, 8192 - _PROBE_GEN_LEN, _PROBE_GEN_LEN)
    )
    print(
        f"\n  KV saved at the 8K point vs a ceiling-sized engine (TP=8): "
        f"{saving / 1024**3:.2f} GiB/worker"
    )

    modalities = modalities_for(spec)
    targets = sorted({t for m in modalities for t in ctx_targets_for(geom, m)})

    for prompt_len, gen_len in targets:
        lengths = {}
        for modality in modalities:
            if (prompt_len, gen_len) not in ctx_targets_for(geom, modality):
                continue
            built = build_input(spec, geom, tokenizer, modality, prompt_len, tp_size=8)
            engine_len = required_model_len(geom, built.predicted_tokens, gen_len)
            lengths[modality] = engine_len

            needed = built.predicted_tokens + gen_len
            assert engine_len >= min(needed, geom.max_model_len), (
                f"{modality}@pl{prompt_len}/gl{gen_len}: engine_len={engine_len} "
                f"cannot hold PL={built.predicted_tokens} + GL={gen_len}"
            )
            assert engine_len <= geom.max_model_len

            # Drift the accounting assertion would accept must still fit.
            drift = (
                TOKEN_SLACK
                if built.exact
                else max(
                    TOKEN_SLACK,
                    int(built.predicted_tokens * video_ratio_slack_for(spec)),
                )
            )
            if needed + drift <= geom.max_model_len:
                assert engine_len >= needed + drift, (
                    f"{modality}@pl{prompt_len}/gl{gen_len}: engine_len={engine_len} "
                    f"would reject a prompt drifting by the tolerated {drift} tokens"
                )

        print(f"  pl={prompt_len} gl={gen_len}: engine lengths {lengths}")
        assert len(set(lengths.values())) == 1, (
            f"pl={prompt_len}/gl={gen_len} needs {len(set(lengths.values()))} engines "
            f"instead of 1: {lengths}. Raise engine_limits.engine_len_block in "
            "suite_config.json to re-share them."
        )


@pytest.mark.parametrize("model_name", SELECTED_MODELS)
def test_selftest_assembled_cases_predict_their_target(model_name: str):
    """Every modality's assembled prompt predicts a count at or below its target."""
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    for modality in modalities_for(spec):
        target = min(8192, geom.max_model_len)
        built = build_input(spec, geom, tokenizer, modality, target, tp_size=8)
        print(
            f"  {modality}: predicted={built.predicted_tokens} target={target} "
            f"| {built.detail}"
        )
        assert built.predicted_tokens <= geom.max_model_len
        assert abs(built.predicted_tokens - target) <= max(TOKEN_SLACK, target // 20), (
            f"{modality} predicted {built.predicted_tokens} for a target of {target}"
        )
