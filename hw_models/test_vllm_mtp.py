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

# For reproducibility.
RNG_SEED = 42
torch.manual_seed(RNG_SEED)


@pytest.fixture(scope="function")
def env_sdpa_decode(request):
    flag = str(request.param)
    os.environ["QAIC_SDPA_DECODE"] = flag
    yield
    os.environ.pop("QAIC_SDPA_DECODE", None)


@pytest.mark.parametrize("env_sdpa_decode", ["0"], indirect=True)
def test_vllm_mimo_mtp(env_sdpa_decode):
    """Smoke test MiMo through vLLM's native MTP path.

    XiaomiMiMo/MiMo-7B-Base is the smallest/simple native MTP model among the
    vLLM 0.16.0 MTP model types targeted by this CI smoke test.
    """

    # Garbage collect so that RAM is freed for RAM limited host.
    gc.collect()

    llm = LLM(
        model="XiaomiMiMo/MiMo-7B-Base",
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=1,
        max_model_len=8192,
        enforce_eager=True,
        trust_remote_code=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
    )

    results = llm.generate(
        ["The future of AI is"],
        sampling_params=SamplingParams(
            seed=RNG_SEED,
            temperature=0.0,
            max_tokens=10,
        ),
    )

    assert len(results) == 1
    output = results[0].outputs[0]
    assert 1 <= len(output.token_ids) <= 10

    print(
        "MiMo MTP smoke test completed: "
        f"prompt_tokens={len(results[0].prompt_token_ids)}, "
        f"output_tokens={len(output.token_ids)}, "
        f"output={output.text!r}"
    )
