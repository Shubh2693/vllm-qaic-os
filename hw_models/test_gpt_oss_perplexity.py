#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

import gc

import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, platforms

from conftest import retry

assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

RNG_SEED = 42
torch.manual_seed(RNG_SEED)

DTYPE = "float16"
KV_CACHE_BYTES = int(2.0 * 1024**3)  # 2 GB per device
CTX_LEN = 1024
STRIDE = 512
NSAMPLES = 500
# 5 Tokens extra to accomodate the generated tokens
MAX_MODEL_LEN = CTX_LEN + 5


# GPU reference perplexity on wikitext-2-raw-v1 (ctx_len=1024, stride=512),
PERPLEXITY_SCORE_REF_MAP = {
    "openai/gpt-oss-20b": 171.5,
    "openai/gpt-oss-120b": 226.5,
    "unsloth/gpt-oss-20b-BF16": 171.5,
    "unsloth/gpt-oss-120b-BF16": 226.5,
}

# (model_name, tensor_parallel_size)
GPT_OSS_MODELS = [
    ("openai/gpt-oss-20b", 4),
    ("openai/gpt-oss-120b", 8),
    ("unsloth/gpt-oss-20b-BF16", 4),
    ("unsloth/gpt-oss-120b-BF16", 8),
]


@retry()
def _load_wikitext2_with_retry():
    return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")


def _build_windows(tokenizer, dataset_text, ctx_len, stride, nsamples):
    """Slices the concatenated dataset into fixed-length, strided token windows."""
    encodings = tokenizer(dataset_text, return_tensors="pt")
    seq_len = encodings.input_ids.shape[1]
    starts = [idx for idx in range(0, seq_len, stride) if (idx + ctx_len) < seq_len]
    if nsamples != -1:
        starts = starts[:nsamples]
    return [encodings.input_ids[0, s : s + ctx_len].tolist() for s in starts]


def compute_wikitext2_perplexity(
    model_name: str,
    tp_size: int,
    ctx_len: int = CTX_LEN,
    stride: int = STRIDE,
    nsamples: int = NSAMPLES,
) -> float:
    """
    Computes wikitext-2-raw-v1 perplexity for `model_name` on QAIC via vLLM.

    Mirrors the sliding-window methodology of
    tests/models/single_soc/functional/test_causal_lms_perplexity.py, but
    sources per-token log-probabilities from vLLM's `prompt_logprobs`
    instead of a raw HF forward pass.
    """
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = _load_wikitext2_with_retry()
    windows = _build_windows(
        tokenizer, "\n\n".join(dataset["text"]), ctx_len, stride, nsamples
    )

    llm = LLM(
        model=model_name,
        dtype=DTYPE,
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        trust_remote_code=False,
    )

    sampling_params = SamplingParams(
        seed=RNG_SEED,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    # QAIC_SDPA_DECODE only supports batch size 1, so generate one window
    # at a time rather than batching all prompts into a single call.
    prompts = [{"prompt_token_ids": window} for window in windows]
    results = [
        llm.generate(prompt, sampling_params=sampling_params)[0] for prompt in prompts
    ]

    # Position 0 has no logprob (nothing precedes it); sum -logprob of the
    # actual token at every later position, matching the shifted
    # next-token cross-entropy computed by the HF-based perplexity test.
    nll_sum = 0.0
    token_count = 0
    for window, result in zip(windows, results):
        for pos in range(1, len(window)):
            nll_sum += -result.prompt_logprobs[pos][window[pos]].logprob
            token_count += 1

    del llm
    gc.collect()

    mean_nll = nll_sum / token_count
    return torch.exp(torch.tensor(mean_nll)).item()


@pytest.mark.parametrize("model_name,tp_size", GPT_OSS_MODELS)
def test_gpt_oss_perplexity(model_name: str, tp_size: int):
    """
    Wikitext-2 perplexity regression test for GPT-OSS models on QAIC via vLLM.

    Compares against a GPU reference in PERPLEXITY_SCORE_REF_MAP using the
    same rtol/atol=0.02 tolerance as the single-SoC HF-based perplexity
    suite. Models without a populated reference are reported but not
    asserted.
    """
    perplexity = compute_wikitext2_perplexity(model_name, tp_size)
    print(f"Model:{model_name}  TP={tp_size}  Perplexity: {perplexity:.4f}")

    gpu_ref = PERPLEXITY_SCORE_REF_MAP.get(model_name)
    if gpu_ref is None:
        print(
            f"No GPU reference for {model_name}; skipping accuracy assertion. "
            "Populate PERPLEXITY_SCORE_REF_MAP to enable it."
        )
        return

    qaic_perplexity = torch.tensor(perplexity)
    gpu_perplexity = torch.tensor(gpu_ref)
    assert torch.allclose(qaic_perplexity, gpu_perplexity, rtol=0.05, atol=0.05), (
        f"Perplexity Score is not within tolerant range. "
        f"QAIC: {qaic_perplexity}, Ref: {gpu_perplexity}"
    )
