#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

import gc
import os
import time

import pytest
import torch
from vllm import LLM, SamplingParams, platforms
from vllm.config import CompilationConfig, CompilationMode

assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

RNG_SEED = 42
torch.manual_seed(RNG_SEED)

# GPT-OSS 20B variants, parametrized at TP=4.
MODEL_IDS = ["openai/gpt-oss-20b", "unsloth/gpt-oss-20b-BF16"]

# Model used by the torch.compile test (paged attention path).
COMPILE_MODEL_ID = "openai/gpt-oss-20b"

# Default TP size used by non-parametrized tests (e.g. torch.compile).
TP_SIZE = 4

from enum import Enum


class QaicDecodePath(Enum):
    """Decode-path selector for the parametrized inference test.

    Each value maps to the ``QAIC_SDPA_DECODE`` env var setting and chooses
    between the two mutually exclusive decode implementations:
      QAIC_SDPA            → QAIC_SDPA_DECODE=1 (_run_sdpa_decode_forward)
      QAIC_PAGED_ATTENTION → QAIC_SDPA_DECODE=0 (QAicAttnWithKvCache)
    """

    QAIC_SDPA = "1"
    QAIC_PAGED_ATTENTION = "0"


# Parametrize over all decode paths with clean test IDs (no class prefix).
QAIC_SDPA_DECODE_PARAMS = [pytest.param(p, id=p.name) for p in QaicDecodePath]

DTYPE = "float16"
KV_CACHE_BYTES = int(2.0 * 1024**3)  # 2 GB per device
MAX_PROMPT_LEN = 512
MAX_NEW_TOKENS = 128
MAX_MODEL_LEN = MAX_PROMPT_LEN + MAX_NEW_TOKENS

DEFAULT_PROMPTS = ["The future of AI is"]
REF_OUTPUTS = {
    "The future of AI is": " a topic of much debate and speculation. Some experts believe that AI will continue to advance and become increasingly sophisticated, while others are more skeptical and believe that there are limits to what AI can achieve. Ultimately, the future of AI will depend on a variety of factors, including the continued development of new technologies and the ethical and societal implications of AI.\n\nThe future of AI is a topic of much debate and speculation. Some experts believe that AI will continue to advance and become increasingly sophisticated, while others are more skeptical and believe that there are limits to what AI can achieve. Ultimately, the future of AI will depend on a variety of factors,",
}

# Accuracy references for torch.compile mode (enforce_eager=False).
# Keyed separately from REF_OUTPUTS because torch.compile with the inductor
# backend may produce different floating-point results than eager execution,
# leading to different token outputs even with temperature=0.0.
REF_OUTPUTS_COMPILE = {
    "The future of AI is": " a topic of much debate and speculation. Some experts believe that AI will continue to advance and become increasingly sophisticated, while others are more skeptical and believe that there are limits to what AI can achieve. Ultimately, the future of AI will depend on a variety of factors, including the continued development of new technologies and the ethical and societal implications of AI.\n\nThe future of AI is a topic of much debate and speculation. Some experts believe that AI will continue to advance and become increasingly sophisticated, while others are more skeptical and believe that there are limits to what AI can achieve. Ultimately, the future of AI will depend on a variety of factors,",
}


@pytest.fixture(scope="function")
def env_decode_path(request):
    """Set the decode-path env vars from the parametrized switch and isolate it.

    The test parameter (``request.param``) is a single switch — the desired
    ``QAIC_SDPA_DECODE`` value ("1" or "0") — which chooses between the two
    mutually exclusive decode implementations, removing the need to juggle two
    separate flags:

      * ``"1"`` → SDPA decode path (``_run_sdpa_decode_forward``).
        ``QAIC_PAGED_ATTENTION`` is forced to ``0``.
      * ``"0"`` → QAIC native paged attention decode path
        (``QAicAttnWithKvCache``). ``QAIC_PAGED_ATTENTION`` is forced to ``1``.

    Both variables are always set explicitly from the parameter (never left to
    a value inherited from the caller's environment) and restored afterwards so
    each parametrized test runs in isolation. Because the two are derived from
    one switch they can never be both-on or both-off.

    Yields the resolved ``(sdpa_decode, paged_attention)`` int pair for logging.
    """
    prev_sdpa = os.environ.get("QAIC_SDPA_DECODE")
    prev_paged = os.environ.get("QAIC_PAGED_ATTENTION")

    use_sdpa = request.param == QaicDecodePath.QAIC_SDPA
    if use_sdpa:
        os.environ["QAIC_SDPA_DECODE"] = "1"
        os.environ["QAIC_PAGED_ATTENTION"] = "0"
    else:
        os.environ["QAIC_SDPA_DECODE"] = "0"
        os.environ["QAIC_PAGED_ATTENTION"] = "1"

    yield (
        int(os.environ["QAIC_SDPA_DECODE"]),
        int(os.environ["QAIC_PAGED_ATTENTION"]),
    )

    # Restore prior values (or remove if they were unset).
    for name, prev in (
        ("QAIC_SDPA_DECODE", prev_sdpa),
        ("QAIC_PAGED_ATTENTION", prev_paged),
    ):
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("env_decode_path", QAIC_SDPA_DECODE_PARAMS, indirect=True)
def test_gpt_oss_20b_inference(env_decode_path, model_id, tp_size):
    """Smoke test: GPT-OSS 20B prefill + decode on QAIC at TP=4.

    Verifies that the model loads, generates output for a short prompt, and
    reports TTFT and decode throughput.  For prompts present in REF_OUTPUTS,
    the generated text must match exactly (greedy decode, temperature=0).
    No strict perf assertion is applied.

    The decode implementation is a parametrized test input — a single
    ``QAIC_SDPA_DECODE`` switch (see ``QAIC_SDPA_DECODE_PARAMS`` and the
    ``env_decode_path`` fixture, which documents the switch semantics), so each
    value is a distinct test id:
      - ``[1-<model>]``: SDPA decode path.
      - ``[0-<model>]``: QAIC native paged attention.

    Which ids actually run in CI is chosen in scripts/run_tools_tests.sh.
    """
    sdpa_decode, paged_attention = env_decode_path
    qccl_enabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))

    gc.collect()

    llm = LLM(
        model=model_id,
        dtype=DTYPE,
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        trust_remote_code=False,
    )

    # First pass: measure TTFT using a single generated token.
    ttft_wall_start = time.perf_counter()
    llm.generate(
        DEFAULT_PROMPTS,
        sampling_params=SamplingParams(
            seed=RNG_SEED,
            temperature=0.0,
            max_tokens=1,
        ),
    )
    ttft_wall_time = time.perf_counter() - ttft_wall_start

    # Second pass: full generation for output validation and decode throughput.
    decode_wall_start = time.perf_counter()
    results = llm.generate(
        DEFAULT_PROMPTS,
        sampling_params=SamplingParams(
            seed=RNG_SEED,
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
        ),
    )
    decode_wall_time = time.perf_counter() - decode_wall_start

    for i, r in enumerate(results):
        output = r.outputs[0]
        decode_tokens = len(output.token_ids)
        decode_tps = (
            (decode_tokens - 1) / (decode_wall_time - ttft_wall_time)
            if decode_wall_time > ttft_wall_time
            else 0.0
        )

        print(
            f"Model:{model_id}  TP={tp_size}  dtype={DTYPE}  "
            f"QCCL:{qccl_enabled}  SDPA_DECODE:{sdpa_decode}  "
            f"PAGED_ATTENTION:{paged_attention}"
        )
        print(f"  Prompt tokens: {len(r.prompt_token_ids)}")
        print(f"  Output tokens: {decode_tokens}")
        print(f"  TTFT wall time: {ttft_wall_time:.3f}s")
        print(f"  Decode wall time: {decode_wall_time:.3f}s")
        print(f"  Decode throughput: {decode_tps:.2f} tokens/s")
        print(f"  Output: {DEFAULT_PROMPTS[i]!r} -> {output.text!r}")

        prompt = DEFAULT_PROMPTS[i]
        if prompt in REF_OUTPUTS:
            assert output.text == REF_OUTPUTS[prompt], (
                f"Output mismatch for model {model_id!r}, prompt {prompt!r}:\n"
                f"  expected: {REF_OUTPUTS[prompt]!r}\n"
                f"  got:      {output.text!r}"
            )
            print(f"  Output matches reference for prompt {prompt!r}")

    del llm
    gc.collect()


def test_gpt_oss_20b_compile():
    """Exercise the torch.compile path for GPT-OSS 20B on QAIC (enforce_eager=False).

    Mirrors test_vlm_vllm_compile: builds the engine with a STOCK_TORCH_COMPILE
    CompilationConfig (inductor backend, all vLLM custom ops enabled), runs a
    prefill + decode pass, and validates the generated text against
    REF_OUTPUTS_COMPILE when a reference is present for the configuration.

    Notes:
      - QAIC_SDPA_DECODE=0 with QAIC_PAGED_ATTENTION=1 must be set for the
        inductor backend to avoid a FakeTensor error for
        ops._C.cpu_attention_with_kv_cache during graph tracing (that
        operator's decode path lacks a meta kernel; the paged CPU-attention
        custom op provides the traceable fake instead). This is a hard
        requirement of the compile path rather than a configuration choice, so
        it is set here instead of being parametrized.
      - REF_OUTPUTS_COMPILE is keyed separately from REF_OUTPUTS because
        torch.compile (inductor) may produce different floating-point results
        than eager execution, yielding different tokens even at temperature=0.
    """
    enforce_eager = False
    enable_prefix_caching = True

    # The compile path REQUIRES paged attention (see the docstring above), so
    # SDPA decode is turned off here rather than left to whatever the
    # environment holds.
    os.environ["QAIC_SDPA_DECODE"] = "0"
    os.environ["QAIC_PAGED_ATTENTION"] = "1"
    sdpa_decode = int(os.getenv("QAIC_SDPA_DECODE", 0))
    paged_attention = int(os.getenv("QAIC_PAGED_ATTENTION", 0))
    qccl_enabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))

    gc.collect()

    cc = CompilationConfig(
        mode=CompilationMode.STOCK_TORCH_COMPILE,  # standard PT2 compile
        backend="inductor",  # PyTorch inductor backend
        custom_ops=["all"],  # enable all vLLM custom ops (matches qaic-inference)
    )

    try:
        llm = LLM(
            model=COMPILE_MODEL_ID,
            dtype=DTYPE,
            seed=RNG_SEED,
            tensor_parallel_size=TP_SIZE,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            kv_cache_memory_bytes=KV_CACHE_BYTES,
            trust_remote_code=False,
            compilation_config=cc,
        )

        # First pass: measure approximate TTFT using a single generated token.
        ttft_wall_start = time.perf_counter()
        llm.generate(
            DEFAULT_PROMPTS,
            sampling_params=SamplingParams(
                seed=RNG_SEED,
                temperature=0.0,
                max_tokens=1,
            ),
        )
        ttft_wall_time = time.perf_counter() - ttft_wall_start

        # Second pass: full generation for output validation and throughput.
        decode_wall_start = time.perf_counter()
        results = llm.generate(
            DEFAULT_PROMPTS,
            sampling_params=SamplingParams(
                seed=RNG_SEED,
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
            ),
        )
        decode_wall_time = time.perf_counter() - decode_wall_start

        for i, r in enumerate(results):
            output = r.outputs[0]
            decode_tokens = len(output.token_ids)
            decode_tps = (
                (decode_tokens - 1) / (decode_wall_time - ttft_wall_time)
                if decode_wall_time > ttft_wall_time
                else 0.0
            )

            print(
                f"Model:{COMPILE_MODEL_ID}  TP={TP_SIZE}  dtype={DTYPE}  "
                f"QCCL:{qccl_enabled}  SDPA_DECODE:{sdpa_decode}  "
                f"PAGED_ATTENTION:{paged_attention}  "
                f"ENFORCE_EAGER:{enforce_eager}  "
                f"PREFIX_CACHING:{enable_prefix_caching}"
            )
            print(f"  Prompt tokens: {len(r.prompt_token_ids)}")
            print(f"  Output tokens: {decode_tokens}")
            print(f"  TTFT wall time: {ttft_wall_time:.3f}s")
            print(f"  Decode wall time: {decode_wall_time:.3f}s")
            print(f"  Decode throughput: {decode_tps:.2f} tokens/s")
            print(f"  Output: {DEFAULT_PROMPTS[i]!r} -> {output.text!r}")

            # Accuracy check against the compile-mode reference.
            prompt = DEFAULT_PROMPTS[i]
            ref_output = REF_OUTPUTS_COMPILE.get(prompt)
            config = (
                COMPILE_MODEL_ID,
                TP_SIZE,
                enable_prefix_caching,
                qccl_enabled,
                sdpa_decode,
                paged_attention,
            )
            config_desc = (
                "(model, tp, prefix_caching, qccl, sdpa_decode, "
                f"paged_attention): {config}"
            )
            if ref_output is None:
                print(f"  Missing compile accuracy reference for prompt {prompt!r}")
            else:
                assert output.text == ref_output, (
                    f"Output mismatch for compile config {config_desc}\n"
                    f"  prompt:   {prompt!r}\n"
                    f"  expected: {ref_output!r}\n"
                    f"  got:      {output.text!r}"
                )
                print(f"  Output matches compile reference for prompt {prompt!r}")

        del llm
        gc.collect()
    finally:
        os.environ.pop("QAIC_SDPA_DECODE", None)
        os.environ.pop("QAIC_PAGED_ATTENTION", None)
