#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Synthetic images and video clips whose visual-token counts are exactly predictable.

Everything here is derived from ModelGeometry, so the arithmetic follows the model's
own patch/merge/temporal-pair configuration rather than hardcoded Qwen constants --
it works for any model in the Qwen2-VL/2.5-VL/3-VL family, with whatever patch size
or merge factor that particular model declares. It is not generic across VLM
*architectures*: sizing an image in pixels to hit an exact token count only makes
sense under this family's resolution-dependent tokenisation. A VLM with a fixed
token-per-image encoder or AnyRes-style tiling needs different builders here, not
different ModelGeometry values (see README's "Model coverage" section).

``variant`` on the two makers produces different pixel content at the same token count.
That matters for batched cases: vLLM keys its multimodal encoder cache on a hash of the
item, so N identical images would be encoded once and the batch would not exercise the
ViT N times at all.
"""

import math

import numpy as np
from PIL import Image

from ctx_config import ITEM_AUTOCAP, RNG_SEED, VIT_VA_BUDGET_GB
from model_geometry import ModelGeometry

# Bytes per element of the ViT attention score matrix (fp16).
_SCORE_DTYPE_BYTES = 2


def max_visual_tokens_per_item(geom: ModelGeometry, tp_size: int) -> int | None:
    """Largest single-item visual-token count whose ViT attention can be mapped.

    One image is one ViT sequence, and the vision blocks that attend over the whole of
    it produce a ``[heads_on_this_rank, patches, patches]`` score matrix. torch_qaic
    prechecks an operator's total operand size against the per-NSP virtual address
    space and returns QAIC_ERROR_MMAP_FAILURE above it, so beyond this bound the
    encoder cannot run at all -- it is not a slow path, it is a hard failure.

    Inverting ``heads * patches**2 * 2 bytes <= budget`` and converting pre-merge
    patches to post-merge tokens gives the cap. At TP=8 on Qwen2.5-VL (16 ViT heads,
    merge 2) that is ~8.0K tokens, which is why a 16256-token image fails and the
    ~6.5K-token images in the ``mixed`` case do not.

    Returns None when there is nothing to bound -- a text-only model, or a config that
    did not expose the ViT head count.
    """
    if not geom.has_vision or not geom.vision_num_heads:
        return None
    if not ITEM_AUTOCAP:
        return None

    heads_on_rank = max(1, geom.vision_num_heads // max(1, tp_size))
    budget_bytes = VIT_VA_BUDGET_GB * 1024**3
    max_patches = math.isqrt(int(budget_bytes / (heads_on_rank * _SCORE_DTYPE_BYTES)))
    return max(1, max_patches // geom.patches_per_visual_token)


def clamp_item_tokens(
    geom: ModelGeometry, tp_size: int, requested_tokens: int
) -> tuple[int, str | None]:
    """Requested per-item token budget, reduced to what the ViT can map.

    Returns (tokens, note) where note explains the reduction, or None if there was
    none, so the case can report why its image is smaller than the context point
    would suggest.
    """
    ceiling = max_visual_tokens_per_item(geom, tp_size)
    if ceiling is None or requested_tokens <= ceiling:
        return requested_tokens, None
    return ceiling, (
        f"item clamped {requested_tokens}->{ceiling} tok by the ViT attention "
        f"budget at TP={tp_size}"
    )


def _near_square_grid(target_tokens: int) -> tuple[int, int]:
    """Grid of visual-token blocks (rows, cols) with rows*cols <= target_tokens.

    Deliberately does not require target_tokens to factorise -- a prime target would
    otherwise force a 1 x N strip. The shortfall is topped up with filler text by
    the caller, which keeps image dimensions sane for every target.
    """
    rows = max(1, math.isqrt(target_tokens))
    cols = max(1, target_tokens // rows)
    return rows, cols


def make_image_of_tokens(
    geom: ModelGeometry, target_tokens: int, variant: int = 0
) -> tuple[Image.Image, int]:
    """Synthesise an RGB image consuming a known number of visual tokens.

    Dimensions are exact multiples of the visual-token block size so that HF's
    smart_resize leaves them untouched (given a max_pixels wide enough to admit
    them), making the resulting token count exactly rows*cols.
    """
    block = geom.visual_token_px
    rows, cols = _near_square_grid(target_tokens)
    height, width = rows * block, cols * block

    # Deterministic non-uniform content: flat colour can be degenerate for a ViT,
    # and pure noise compresses poorly in host memory terms for no benefit. The
    # variant shifts the gradients and re-seeds the noise so two items of the same
    # size are not the same bytes.
    rng = np.random.default_rng(RNG_SEED + target_tokens + 7919 * variant)
    phase = 37 * variant
    ys = np.linspace(0, 255, height, dtype=np.uint16)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.uint16)[None, :]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = (ys + xs + phase) % 256
    base[..., 1] = (ys * 2 + xs // 2 + phase) % 256
    base[..., 2] = rng.integers(0, 256, size=(height, width), dtype=np.uint8)

    return Image.fromarray(base, mode="RGB"), rows * cols


def _video_frame_side(geom: ModelGeometry, target_tokens: int) -> int:
    """Frame side length in visual-token blocks.

    Total pixels for a video of N tokens is invariant
    (frames * H * W = temporal_patch_size * pixels_per_visual_token * tokens), so
    larger frames mean fewer of them and less per-frame Python/host overhead. Small
    targets need small frames to remain expressible at all.
    """
    if target_tokens >= 4096:
        return 32  # 896 x 896 at a 28px block -> 1024 tokens per frame pair
    if target_tokens >= 1024:
        return 16  # 448 x 448 -> 256 tokens per frame pair
    return 8  # 224 x 224 -> 64 tokens per frame pair


def make_video_of_tokens(
    geom: ModelGeometry, target_tokens: int, variant: int = 0
) -> tuple[np.ndarray, int, int]:
    """Synthesise an (F, H, W, 3) uint8 clip consuming a known number of visual tokens.

    Qwen-style video tokenisation groups frames in pairs (temporal_patch_size), so
    tokens = (F / temporal_patch_size) * (H / block) * (W / block).

    Returns (frames, achieved_tokens, num_frames).
    """
    block = geom.visual_token_px
    side_blocks = _video_frame_side(geom, target_tokens)
    height = width = side_blocks * block
    tokens_per_group = side_blocks * side_blocks

    groups = max(1, target_tokens // tokens_per_group)
    num_frames = groups * geom.temporal_patch_size

    # A moving gradient gives the temporal axis something real to encode without
    # allocating a second buffer per frame. The variant offsets the whole sweep.
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    phase = 53 * variant
    ys = np.linspace(0, 255, height, dtype=np.uint16)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.uint16)[None, :]
    for index in range(num_frames):
        shift = (index * 256) // max(1, num_frames) + phase
        frames[index, ..., 0] = (ys + xs + shift) % 256
        frames[index, ..., 1] = (ys * 2 + shift) % 256
        frames[index, ..., 2] = (xs * 2 + 255 - shift) % 256

    return frames, groups * tokens_per_group, num_frames
