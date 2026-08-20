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
def test_vllm_llama_eagle3(env_sdpa_decode):
    """Smoke test EAGLE3 speculative decoding through vLLM.

    Loads a Llama 3 target model paired with an EAGLE3 draft model. Both
    models share embed_tokens and lm_head with the target as detected by
    EagleProposer.load_model. The QAIC platform routes ``method="eagle3"``
    through the same EagleProposer path used by MTP, with
    ``disable_padded_drafter_batch=True`` flipped on to bypass the Triton
    padded-batch helpers.
    """

    gc.collect()

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=1,
        max_model_len=2048,
        enforce_eager=True,
        speculative_config={
            "method": "eagle3",
            "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 3,
        },
    )

    results = llm.generate(
        ["The future of speculative decoding on QAIC is"],
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
        "Llama 3.1 + EAGLE3 smoke test completed: "
        f"prompt_tokens={len(results[0].prompt_token_ids)}, "
        f"output_tokens={len(output.token_ids)}, "
        f"output={output.text!r}"
    )
