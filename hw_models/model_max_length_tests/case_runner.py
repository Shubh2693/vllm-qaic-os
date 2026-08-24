#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""The shared per-case runner: generate once, then assert the three properties.

  1. every request completes and the engine survives;
  2. each realised prompt token count matches the analytically predicted count;
  3. no output has numerically collapsed into repetition.

A case is a batch of one or more requests, issued in a single ``llm.generate()`` call so
they are actually resident together rather than run back to back. Every assertion is
per-request: a batch that fails on its fourth sequence only is exactly the kind of
failure this suite exists to catch.
"""

import gc
import time

from vllm import SamplingParams

from ctx_assertions import assert_not_degenerate, assert_token_accounting
from ctx_config import (
    RNG_SEED,
    SAMPLING_IGNORE_EOS,
    SAMPLING_REPETITION_PENALTY,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
)
from engine_pool import get_engine
from model_geometry import ModelGeometry
from model_specs import ModelSpec
from prompt_builder import BuiltInput
from run_metrics import record_metrics


def sampling_params_for(max_tokens: int) -> SamplingParams:
    """Deterministic by default (temperature=0.0) -- suite_config.json's sampling
    section overrides it. seed=RNG_SEED keeps results reproducible run-to-run even
    at a non-zero temperature.
    """
    return SamplingParams(
        seed=RNG_SEED,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        top_k=SAMPLING_TOP_K,
        repetition_penalty=SAMPLING_REPETITION_PENALTY,
        ignore_eos=SAMPLING_IGNORE_EOS,
        max_tokens=max_tokens,
    )


def generate_once(llm, batch: list[BuiltInput] | BuiltInput, max_tokens: int):
    """One greedy generation over the whole batch. Returns (results, wall_time).

    Accepts a single BuiltInput for the callers that only ever issue one request
    (test_ctx_boundary), and normalises it to a one-element batch.
    """
    if isinstance(batch, BuiltInput):
        batch = [batch]

    gc.collect()
    start = time.perf_counter()
    results = llm.generate(
        [built.as_request() for built in batch],
        sampling_params=sampling_params_for(max_tokens),
    )
    wall_time = time.perf_counter() - start

    assert results, "llm.generate returned no results"
    assert len(results) == len(batch), (
        f"Asked for {len(batch)} sequences, got {len(results)} results back"
    )
    return results, wall_time


def run_case(
    spec: ModelSpec,
    geom: ModelGeometry,
    tp_size: int,
    modality_name: str,
    ctx_len: int,
    batch: list[BuiltInput] | BuiltInput,
    gen_len: int,
    engine_len: int,
) -> tuple[int, int, float | None]:
    """Generate once, print the accounting, and assert liveness/accounting/quality.

    ``engine_len`` is the max_model_len this case's engine was built at, which is the
    limit the requests are actually held to -- a tighter and more meaningful bound than
    the model's ceiling now that engines are right-sized per case.

    Returns (prompt_tokens, decode_tokens, prefill) for the *first* request, so a
    batch-1 case reports exactly what it used to.
    """
    if isinstance(batch, BuiltInput):
        batch = [batch]
    batch_size = len(batch)

    llm = get_engine(spec, geom, tp_size, engine_len, batch_size)

    print(
        f"\n--- {spec.model} TP={tp_size} modality={modality_name} ctx={ctx_len} "
        f"batch={batch_size} gen={gen_len} engine_len={engine_len} ---\n"
        f"{batch[0].detail}\n"
        f"predicted_prompt_tokens={batch[0].predicted_tokens}"
        + (
            f" (x{batch_size} requests, distinct content)"
            if batch_size > 1
            else ""
        )
    )

    results, wall_time = generate_once(llm, batch, gen_len)

    per_request: list[tuple[int, int]] = []
    for index, (built, result) in enumerate(zip(batch, results, strict=True)):
        output = result.outputs[0]
        prompt_tokens = len(result.prompt_token_ids)
        decode_tokens = len(output.token_ids)
        per_request.append((prompt_tokens, decode_tokens))

        print(
            f"[{index}] prompt_tokens={prompt_tokens} decode_tokens={decode_tokens} "
            f"finish_reason={output.finish_reason}"
        )
        if index == 0:
            print(f"    wall={wall_time:.2f}s output={output.text[:400]!r}")
        else:
            print(f"    output={output.text[:120]!r}")

        # 1. The request completed.
        assert output.finish_reason in ("length", "stop"), (
            f"Unexpected finish_reason={output.finish_reason!r} for request {index} of "
            f"{modality_name} at ctx={ctx_len} batch={batch_size}"
        )
        # 2. Token accounting matches prediction and respects the engine's limit.
        assert_token_accounting(built, prompt_tokens, engine_len, spec=spec)
        assert prompt_tokens + decode_tokens <= engine_len, (
            f"Request {index}: prompt({prompt_tokens}) + decode({decode_tokens}) "
            f"exceeds the engine's max_model_len={engine_len}"
        )
        # 3. Output has not collapsed.
        assert_not_degenerate(output.text, list(output.token_ids))

    # Distinct inputs must not produce identical outputs -- that would mean the batch
    # was served from one sequence's state rather than being genuinely concurrent.
    if batch_size > 1:
        texts = [result.outputs[0].text for result in results]
        assert len(set(texts)) > 1, (
            f"All {batch_size} requests returned byte-identical text despite distinct "
            "prompts, which suggests the batch was not actually run as one"
        )

    prefill = record_metrics(
        spec.model,
        tp_size,
        modality_name,
        ctx_len,
        engine_len,
        results,
        wall_time,
        per_request,
        batch,
    )
    return per_request[0][0], per_request[0][1], prefill
