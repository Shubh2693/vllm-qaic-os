#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

import gc
import os

import pytest
import torch
from vllm import LLM, SamplingParams, platforms

assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

RNG_SEED = 42
torch.manual_seed(RNG_SEED)


@pytest.fixture(scope="function")
def env_sdpa_decode(request):
    flag = str(request.param)
    os.environ["QAIC_SDPA_DECODE"] = flag
    yield
    os.environ.pop("QAIC_SDPA_DECODE", None)


@pytest.mark.parametrize("env_sdpa_decode", ["1"], indirect=True)
def test_vllm_llama_ngram(env_sdpa_decode):
    """Smoke test n-gram speculative decoding through vLLM.

    NgramProposer is CPU-only and exercises the verify-path on QAIC without
    requiring a model with native MTP heads. A repeating prompt biases the
    n-gram lookup toward producing accepted draft tokens, exercising the
    accept path in addition to the proposer.
    """

    gc.collect()

    llm = LLM(
        model="meta-llama/Llama-3.1-8B",
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=1,
        max_model_len=2048,
        enforce_eager=True,
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 3,
        },
    )

    prompt = (
        "The cat sat on the mat. The cat sat on the mat. "
        "The cat sat on the mat. The cat sat on the"
    )

    results = llm.generate(
        [prompt],
        sampling_params=SamplingParams(
            seed=RNG_SEED,
            temperature=0.0,
            max_tokens=16,
        ),
    )

    assert len(results) == 1
    output = results[0].outputs[0]
    assert len(output.token_ids) > 0
    assert output.text != ""

    print(
        "Llama n-gram smoke test completed: "
        f"prompt_tokens={len(results[0].prompt_token_ids)}, "
        f"output_tokens={len(output.token_ids)}, "
        f"output={output.text!r}"
    )
