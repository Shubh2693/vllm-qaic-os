#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""The ceiling itself: landing on max_model_len, decoding up to it, and exceeding it.

All three use text-only prompts -- the point here is the position limit, not the
vision path, and text is the only modality whose length is exactly controllable.

These are the one place that deliberately opts out of engine right-sizing: the
subject under test *is* the model's declared ceiling, so the engine is built at
geom.max_model_len. All three share that engine, so the file costs one engine build.

They also stay at batch 1 whatever selection.batch_sizes says: the subject is the
position limit of one sequence, and a ceiling-length engine times a batch would need
more KV than any single device has.

    pytest test_ctx_boundary.py -s
"""

import pytest

from case_matrix import BOUNDARY_CASES
from case_runner import generate_once, run_case, sampling_params_for
from ctx_assertions import assert_not_degenerate
from ctx_config import TOKEN_SLACK
from engine_pool import get_engine
from model_geometry import resolve, tokenizer_for
from prompt_builder import build_input


def _generate_single(llm, built, max_tokens):
    """generate_once for a one-request batch, unpacked to (result, output, wall)."""
    results, wall_time = generate_once(llm, built, max_tokens)
    return results[0], results[0].outputs[0], wall_time


@pytest.mark.skipif(not BOUNDARY_CASES, reason="No max-context boundary cases selected")
@pytest.mark.parametrize("model_name,tp_size", BOUNDARY_CASES)
def test_boundary_exact_max_model_len(model_name: str, tp_size: int):
    """A prompt of max_model_len - 1 plus one token lands exactly on the ceiling."""
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    built = build_input(spec, geom, tokenizer, "text", geom.max_model_len - 1)
    prompt_tokens, decode_tokens, _ = run_case(
        spec, geom, tp_size, "text", geom.max_model_len, built, 1, geom.max_model_len
    )
    assert prompt_tokens + decode_tokens <= geom.max_model_len
    assert geom.max_model_len - (prompt_tokens + decode_tokens) <= TOKEN_SLACK, (
        f"Expected to land on the {geom.max_model_len}-token ceiling, got "
        f"{prompt_tokens} + {decode_tokens}"
    )


@pytest.mark.skipif(not BOUNDARY_CASES, reason="No max-context boundary cases selected")
@pytest.mark.parametrize("model_name,tp_size", BOUNDARY_CASES)
def test_boundary_decode_to_ceiling(model_name: str, tp_size: int):
    """Decoding must stop at the ceiling rather than running past it.

    Asks for exactly the number of tokens that fit. Requesting *more* than fits
    (prompt + max_tokens > max_model_len) is a different thing entirely: vLLM may
    reject such a request upfront instead of truncating it, which would make this
    test fail for a reason unrelated to long-context behaviour. Over-length handling
    is covered by test_boundary_over_max_is_rejected_and_engine_survives.
    """
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    room = 256
    built = build_input(spec, geom, tokenizer, "text", geom.max_model_len - room)
    llm = get_engine(spec, geom, tp_size, geom.max_model_len)

    # The prompt may land a little under target, so derive the ask from what the
    # engine will actually see rather than from the nominal room.
    available = geom.max_model_len - built.predicted_tokens
    assert available > 0, (
        f"No room left to decode: predicted prompt {built.predicted_tokens} vs "
        f"max_model_len {geom.max_model_len}"
    )
    print(f"Asking for {available} tokens with {built.predicted_tokens} of prompt")

    result, output, _ = _generate_single(llm, built, available)
    prompt_tokens = len(result.prompt_token_ids)
    decode_tokens = len(output.token_ids)
    total = prompt_tokens + decode_tokens
    print(
        f"prompt_tokens={prompt_tokens} decode_tokens={decode_tokens} total={total} "
        f"finish_reason={output.finish_reason} ceiling={geom.max_model_len}"
    )

    assert total <= geom.max_model_len, (
        f"Decode ran past the ceiling: {prompt_tokens} + {decode_tokens} = {total} > "
        f"{geom.max_model_len}"
    )
    # "length" means it used the whole budget; "stop" means it chose to end early,
    # which is legitimate and not a long-context failure.
    assert output.finish_reason in (
        "length",
        "stop",
    ), f"Unexpected finish_reason={output.finish_reason!r} decoding at the ceiling"
    assert decode_tokens > 0, "No tokens generated at the ceiling"
    assert_not_degenerate(output.text, list(output.token_ids))


@pytest.mark.skipif(not BOUNDARY_CASES, reason="No max-context boundary cases selected")
@pytest.mark.parametrize("model_name,tp_size", BOUNDARY_CASES)
def test_boundary_over_max_is_rejected_and_engine_survives(
    model_name: str, tp_size: int
):
    """An over-length prompt must be refused cleanly, leaving the engine usable.

    This is the real upper-edge test: it is not enough that 128000 works, an
    over-length request must not take the engine down with it.
    """
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)
    llm = get_engine(spec, geom, tp_size, geom.max_model_len)

    over = build_input(spec, geom, tokenizer, "text", geom.max_model_len + 1024)
    rejected_cleanly = False
    try:
        results = llm.generate(over.as_request(), sampling_params=sampling_params_for(1))
        # Some versions return an empty/aborted result instead of raising.
        if not results or not results[0].outputs[0].token_ids:
            rejected_cleanly = True
            print("Over-length prompt returned an empty result (accepted as rejection)")
        else:
            prompt_tokens = len(results[0].prompt_token_ids)
            pytest.fail(
                f"Over-length prompt of {prompt_tokens} tokens was served despite "
                f"max_model_len={geom.max_model_len}"
            )
    except Exception as exc:
        rejected_cleanly = True
        print(f"Over-length prompt rejected with {type(exc).__name__}: {exc}")

    assert rejected_cleanly

    # The engine must still serve a normal request afterwards.
    follow_up = build_input(spec, geom, tokenizer, "text", 1024)
    results = llm.generate(follow_up.as_request(), sampling_params=sampling_params_for(16))
    assert results and results[0].outputs[0].token_ids, (
        "Engine failed to serve a normal request after rejecting an over-length "
        "prompt -- the rejection was not clean"
    )
    print("Engine healthy after the over-length rejection")
