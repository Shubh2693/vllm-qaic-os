# --------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# -------------------------------------------------------------------------------------
#
# MIT License
#
# Copyright (c) 2023 OpenGVLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# LICENCE: https://github.com/OpenGVLab/InternVL/blob/main/LICENSE
# SOURCE:  https://github.com/OpenGVLab/InternVL
# --------------------------------------------------------------------------------------

from __future__ import annotations

import os
import ast
import json
import logging
import pytest
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset
from conftest import retry

# External project utilities (expected to exist in PYTHONPATH)
from data_utils import CAT_SHORT2LONG, process_single_sample, str2bool
from eval_utils import (
    evaluate,
    replace_image_placeholder,
    replace_image_placeholder_qwen,
    mmmu_post_process,
)

# vLLM imports
from vllm import LLM, SamplingParams, platforms

ACCURACY_REF = {
    "OpenGVLab/InternVL3_5-8B-Instruct": {
        100: 0.49,
    },
    "Qwen/Qwen2.5-VL-32B-Instruct": {
        100: 0.45,
    },
}


def _require_qaic():
    """Raise RuntimeError if not running on a QAIC platform."""
    if platforms.current_platform.device_type != "qaic":
        raise RuntimeError("vLLM could not detect qaic plugin")


# --------------------------------------------------------------------------------------
# HuggingFace dataset loading with retry
# --------------------------------------------------------------------------------------


@retry()
def _load_dataset_with_retry(root, subject, split, cache_dir):
    return load_dataset(root, subject, split=split, cache_dir=cache_dir)


# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------


def _setup_logging(log_file: str, debug: bool) -> logging.Logger:
    logger = logging.getLogger("mmmu_eval")
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# --------------------------------------------------------------------------------------
# Data utilities
# --------------------------------------------------------------------------------------


def collate_fn(batches: List[Dict[str, Any]]) -> Tuple:
    images = [b["images"] for b in batches]
    questions = [b["question"] for b in batches]
    answers = [b["answer"] for b in batches]
    data_ids = [b["data_id"] for b in batches]
    options = [b["option"] for b in batches]
    question_type = [b["question_type"] for b in batches]
    return images, questions, answers, data_ids, options, question_type


class MMMUDataset(Dataset):
    """
    Dataset for MMMU evaluation. Loads each subject split from HF, concatenates,
    and yields processed multimodal samples.
    """

    def __init__(
        self, root: str, split: str, nsamples: int = -1, cache_dir: str | None = None
    ):
        cache_dir = cache_dir or os.path.join(os.getcwd(), "data", "MMMU")
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values(), desc="Loading subjects"):
            sub_dataset = _load_dataset_with_retry(
                root, subject, split=split, cache_dir=cache_dir
            )
            sub_dataset_list.append(sub_dataset)

        self.data = concatenate_datasets(sub_dataset_list)

        if nsamples != -1:
            self.data = self.data.select(range(min(nsamples, len(self.data))))

        self.prompt = {
            "multiple-choice": "Answer with the option's letter from the given choices directly.",
            "open": "Answer the question using a single word or phrase.",
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        datum = process_single_sample(self.data[idx])

        data_id = datum["id"]
        question = datum["question"].strip()
        pil_images = datum["image"]
        question_type = datum["question_type"]

        # Safe parse options
        raw_opts = datum["options"]
        choices = raw_opts if isinstance(raw_opts, list) else ast.literal_eval(raw_opts)
        answer = datum.get("answer", None)

        letters = list("ABCDEFGHIJKLM")
        choice_list: List[str] = []
        options: Dict[str, str] = {}
        for i, c in enumerate(choices):
            txt = f"({letters[i]}) {c.strip()}"
            choice_list.append(txt)
            options[letters[i]] = c.strip()

        choice_txt = "\n".join(choice_list)
        if len(choice_txt) > 0:
            question += "\n" + choice_txt
        question += "\n\n" + self.prompt[question_type]
        question = question.strip()

        # Only keep images whose slot is actually referenced in the question text.
        # pil_images has 7 slots (image_1..image_7); the question uses <image N> tags.
        # Mismatches (placeholder count != image count) cause vLLM assertion failures.
        valid_images = [
            img
            for i, img in enumerate(pil_images, start=1)
            if img is not None and f"<image {i}>" in question
        ]

        return {
            "question": question,
            "images": valid_images,  # list of PIL images, one per <image N> placeholder
            "answer": answer,
            "option": options,
            "data_id": data_id,
            "question_type": question_type,
        }


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


@torch.inference_mode()
def run_eager_mode_inference(
    vllm_obj: LLM,
    generation_len: int,
    dataloader: DataLoader,
    logger: logging.Logger,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Runs inference sample-by-sample and returns a list of per-sample predictions.
    """
    outputs: List[Dict[str, Any]] = []
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=generation_len,
    )

    # Select the appropriate image placeholder replacement for this model family.
    if "Qwen" in model_name:
        _replace_placeholder = replace_image_placeholder_qwen
    else:
        _replace_placeholder = replace_image_placeholder

    for sno, (images, questions, answers, data_ids, options, question_type) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluating Samples"
    ):
        cur_prompt = {
            "prompt": _replace_placeholder(questions[0]),
            "multi_modal_data": {"image": images[0]},
        }

        vllm_output = vllm_obj.generate([cur_prompt], sampling_params=sampling_params)

        for image in images[0]:
            if image is not None:
                image.close()  # Close PIL images to free memory

        out = vllm_output[0]
        gen_text = out.outputs[0].text.lstrip()

        # If answer is not predicted will treat it as failure
        if len(gen_text) == 0:
            gen_text = " "

        logger.debug(
            f"Sample {sno} | prompt_tokens={len(out.prompt_token_ids)} | gen_tokens={len(out.outputs[0].token_ids)}"
        )
        logger.debug("Prompt: %s", cur_prompt["prompt"])
        parsed_pred = mmmu_post_process([gen_text], options, question_type)
        logger.info(
            f"ID: {data_ids[0]} | Ans(raw): {repr(gen_text)} | Parsed prediction: {parsed_pred} | Correct answer: {answers[0]}"
        )

        outputs.append(
            {
                "id": data_ids[0],
                "question_type": question_type[0],
                "answer": answers[0],
                "parsed_pred": parsed_pred,
            }
        )
    return outputs


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def evaluate_mmmu(
    model_name: str,
    dataset_name: str,
    output_file: str,
    log_file: str,
    nsamples: int,
    ctx_len: int,
    generation_len: int,
    debug: bool = False,
    tp_size: int = 1,
    kv_cache_bytes: int = 2 * 1024**3,
    qaic_sdpa_decode: bool = True,
    hf_cache_dir: str = "data/MMMU",
) -> float:
    """
    Runs MMMU evaluation with vLLM and returns the overall accuracy.
    """
    _require_qaic()
    logger = _setup_logging(log_file, debug)

    # Pin all random seeds here, not at module level, so they take effect at eval time.
    np.random.seed(42)
    torch.manual_seed(42)

    hf_cache_dir = os.getenv("HF_HOME", hf_cache_dir)

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Context Length: {ctx_len}")
    logger.info(f"Number of samples: {nsamples}")
    logger.info(f"HF Cache Directory: {hf_cache_dir}")

    # If qaic sdpa decode is enable set it to 1 else 0
    os.environ["QAIC_SDPA_DECODE"] = str(int(qaic_sdpa_decode))

    # Dataset loader
    dataset = MMMUDataset(
        root=dataset_name,
        split="validation",
        nsamples=nsamples,
        cache_dir=os.path.abspath(hf_cache_dir),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # vLLM configuration
    TRUST_REMOTE_CODE = False
    mm_processor_args = None

    if "Qwen" in model_name:
        mm_processor_args = {"max_pixels": 100 * 28 * 28}
    elif "InternVL" in model_name:
        TRUST_REMOTE_CODE = True
    else:
        raise ValueError(f"Unsupported model family: {model_name}")

    logger.info(
        f"Initializing vLLM | tp={tp_size} | kv_cache_bytes={kv_cache_bytes} | trust_remote_code={TRUST_REMOTE_CODE}"
    )

    vllm_obj = LLM(
        model=model_name,
        dtype="float16",
        tensor_parallel_size=tp_size,
        max_model_len=ctx_len,
        enforce_eager=True,
        kv_cache_memory_bytes=kv_cache_bytes,
        trust_remote_code=TRUST_REMOTE_CODE,
        mm_processor_kwargs=mm_processor_args,
        enable_prefix_caching=False,
        seed=42,
    )

    # Inference
    preds = run_eager_mode_inference(
        vllm_obj, generation_len, dataloader, logger, model_name
    )

    # Evaluate
    acc_dict, acc, preds_dict = evaluate(preds)
    logger.info(f"Accuracy: {acc}, Total Samples: {len(preds)}")
    print(f"Accuracy: {acc}, Total Samples: {len(preds)}")

    # Save per-sample results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(preds_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved per-sample predictions to: {output_file}")

    if model_name in ACCURACY_REF:
        ref_map = ACCURACY_REF[model_name]
        if nsamples in ref_map:
            ref_accuracy = ref_map[nsamples]
            assert (
                ref_accuracy <= acc
            ), f"Calculated accuracy lesser than reference: GPU: {ref_accuracy} vs QAIC: {acc}"
            print(f"Accuracy validated with GPU for {model_name}")
        else:
            print(
                f"No GPU reference for {model_name} at nsamples={nsamples}; "
                "skipping accuracy assertion. Populate ACCURACY_REF to enable it."
            )

    # Resetting process specific env variable
    os.environ.pop("QAIC_SDPA_DECODE", None)

    return acc


# --------------------------------------------------------------------------------------
# pytest
# --------------------------------------------------------------------------------------

# Models and TP sizes to evaluate on MMMU.
MMMU_MODELS = [
    "OpenGVLab/InternVL3_5-8B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
]


@pytest.mark.parametrize("model_name", MMMU_MODELS)
@pytest.mark.parametrize("tp_size", [4])
def test_vlm_mmmu_accuracy(model_name: str, tp_size: int):
    """Run MMMU validation-set evaluation and assert accuracy is within 5% of GPU reference."""

    evaluate_mmmu(
        model_name=model_name,
        dataset_name="MMMU/MMMU",
        output_file=f"mmmu_preds_{model_name.replace('/', '_')}_tp{tp_size}.json",
        log_file=f"mmmu_{model_name.replace('/', '_')}_tp{tp_size}.log",
        nsamples=100,
        ctx_len=4096,
        generation_len=2,
        tp_size=tp_size,
        debug=True,
        kv_cache_bytes=2 * 1024**3,
        qaic_sdpa_decode=True,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(cfg):
    evaluate_mmmu(**vars(cfg))


if __name__ == "__main__":
    parser = ArgumentParser(description="MMMU Dataset QAIC Accuracy Script (vLLM)")

    parser.add_argument(
        "--model-name", "--model_name", required=True, help="HF model id or local path"
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        type=str,
        default="MMMU/MMMU",
        help="HF dataset path for MMMU",
    )

    parser.add_argument(
        "--ctx-len",
        "--ctx_len",
        required=True,
        type=int,
        help="Max model context length",
    )
    parser.add_argument(
        "--generation-len",
        "--generation_len",
        required=True,
        type=int,
        help="Max tokens to generate per prompt",
    )

    parser.add_argument(
        "--output-file",
        "--output_file",
        default="output_preds_dict.json",
        help="Where to write per-sample predictions",
    )
    parser.add_argument(
        "--log-file",
        "--log_file",
        type=str,
        default="output.log",
        help="Where to write logs",
    )

    parser.add_argument(
        "--nsamples",
        default=-1,
        type=int,
        help="Number of validation samples to evaluate (-1 = all)",
    )
    parser.add_argument(
        "--debug", default=False, type=str2bool, help="Enable verbose debug logs"
    )

    # Tuning knobs
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument(
        "--kv-cache-bytes",
        type=int,
        default=2 * 1024**3,
        help="KV cache memory in bytes (per worker)",
    )

    parser.add_argument(
        "--qaic-sdpa-decode",
        default=True,
        type=str2bool,
        help="Enable SDPA Decode path in vLLM via QAIC",
    )

    cfg = parser.parse_args()
    main(cfg)
