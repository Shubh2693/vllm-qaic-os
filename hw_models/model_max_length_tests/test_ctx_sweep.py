#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""The context sweep: fill the window from ~8K to the model ceiling, every input shape.

    pytest test_ctx_sweep.py -s      # sweep.tier in suite_config.json: smoke or full
    ./run_tests.sh --tier full --batch 1,4   # or override tier/batch for one run

Prompt length and generation length are independent knobs -- suite_config.json's
sweep.smoke_targets/full_targets are explicit {prompt_len, gen_len} pairs, not a
single context total with a generation length subtracted from it. ctx_len
(prompt_len + gen_len) is computed here purely for labeling/reporting.

Which modalities a case can use follows from the model's declared capabilities, so a
text-only model contributes one case per (prompt_len, gen_len, batch) point and a VLM
contributes up to six (see README.md's "Input shapes" section).

See README.md's "Configuration" section for the full suite_config.json field table.
"""

import pytest

from case_matrix import SWEEP_CASES
from case_runner import run_case
from kv_capacity import required_model_len
from model_geometry import resolve, tokenizer_for
from prompt_builder import build_batch
from run_metrics import check_perf_reference


@pytest.mark.skipif(not SWEEP_CASES, reason="No max-context sweep cases selected")
@pytest.mark.parametrize(
    "model_name,tp_size,modality,prompt_len,gen_len,batch_size", SWEEP_CASES
)
def test_ctx_sweep(
    model_name: str,
    tp_size: int,
    modality: str,
    prompt_len: int,
    gen_len: int,
    batch_size: int,
):
    """Build a prompt_len-token prompt, decode gen_len tokens, for one modality mix.

    The engine is built at what this case needs -- prompt_len plus gen_len -- rather
    than at the model's ceiling, so the KV cache reserved scales with the (prompt,
    generation) point under test and with the batch size.
    """
    spec, geom = resolve(model_name)
    tokenizer = tokenizer_for(spec)

    batch = build_batch(
        spec,
        geom,
        tokenizer,
        modality,
        prompt_len,
        batch_size,
        tp_size=tp_size,
    )
    # Every request in the batch is the same length by construction, but size the
    # engine off the longest so a one-token wobble in BPE re-encoding cannot make the
    # engine too short for one member of the batch.
    longest = max(built.predicted_tokens for built in batch)
    engine_len = required_model_len(geom, longest, gen_len)

    # ctx_len is prompt + generation -- reported/labeled, never the input.
    ctx_len = prompt_len + gen_len
    _, _, prefill = run_case(
        spec, geom, tp_size, modality, ctx_len, batch, gen_len, engine_len
    )
    check_perf_reference(model_name, tp_size, ctx_len, modality, prefill, batch_size)
