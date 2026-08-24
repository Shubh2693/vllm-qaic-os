#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Assembles one test case: a prompt, its multimodal payload, and the predicted length.

The input shapes live in a registry rather than an if/elif chain, so adding another is a
single decorated function:

    @modality("two_videos", requires_video=True)
    def _build_two_videos(spec, geom, tokenizer, target_tokens, *, tp_size, variant):
        ...
        return assemble(spec, tokenizer, target_tokens, [], [], videos, tokens, frames,
                         "two videos", variant=variant)

``MODALITIES`` and ``modalities_for()`` pick the new entry up automatically, which
means the sweep matrix and the self-test cover it with no further edits. A modality
whose requirements the model does not declare is dropped rather than degraded, so a
text-only model contributes exactly one case per context point.

Images and videos are both plural throughout (``images``/``videos``, both lists), so
one case can hold any count of either -- ``image_many``/``video_many`` are exactly
"the multi-item builder with count > 1", not a separate code path from
``image_single``/``video``.

``variant`` makes one request of a batch differ from the next at the same token length;
``build_batch()`` is what the sweep calls.

The pad-token-expands-1:1-into-N-visual-tokens model this file assumes (one
placeholder per image/video, replaced by exactly the built-time-predicted token
count) is the Qwen2-VL/2.5-VL/3-VL family's processor behaviour, not a universal VLM
property -- see README's "Model coverage" section before pointing this at a
structurally different VLM.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ctx_config import IMAGE_MANY_COUNT, MAX_SINGLE_ITEM_TOKENS, VIDEO_MANY_COUNT
from filler_text import text_of_exact_len
from model_geometry import ModelGeometry
from model_specs import ModelSpec
from visual_inputs import clamp_item_tokens, make_image_of_tokens, make_video_of_tokens

QUESTION = "Summarise the material above in two sentences."


@dataclass
class BuiltInput:
    """A prompt, its multimodal payload, and the token count we expect vLLM to see."""

    prompt: str
    multi_modal_data: dict
    predicted_tokens: int
    detail: str
    # Where predicted_tokens comes from. text_tokens is exact -- the tokenizer has
    # already encoded that text -- while the visual counts are the expansion the
    # processor is expected to perform on the placeholders, so
    # text_tokens + image_tokens + video_tokens == predicted_tokens by construction.
    text_tokens: int = 0
    filler_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0
    num_images: int = 0
    num_videos: int = 0
    # (width, height) of one item. Every image (and every video clip) in one case is
    # the same size, so one pair describes all of them.
    image_size: tuple[int, int] | None = None
    video_size: tuple[int, int] | None = None
    # Frame count of one clip, not the sum across clips -- mirrors image_size above.
    num_frames: int = 0
    # Video processors may impose their own total-pixel budget, so video cases are
    # asserted against a ratio rather than an exact count.
    exact: bool = True
    # Which request of a batch this is. 0 for a single-request case.
    variant: int = 0

    @property
    def visual_tokens(self) -> int:
        return self.image_tokens + self.video_tokens

    def as_request(self) -> dict:
        request: dict = {"prompt": self.prompt}
        if self.multi_modal_data:
            request["multi_modal_data"] = self.multi_modal_data
        return request


# ---------------------------------------------------------------------------
# The shared assembler
# ---------------------------------------------------------------------------


def assemble(
    spec: ModelSpec,
    tokenizer,
    target_tokens: int,
    images: list[Image.Image],
    image_tokens: list[int],
    videos: list[np.ndarray],
    video_tokens: list[int],
    num_frames: list[int],
    label: str,
    variant: int = 0,
    notes: list[str] | None = None,
) -> BuiltInput:
    """Glue placeholders, filler text and the question into one exact-length prompt.

    ``images``/``videos`` are both lists so this one function covers "no items", "one
    item" and "many items" uniformly -- image_single/image_many and video/video_many
    differ only in what they pass in here, not in how assemble() works.

    The predicted count is the encoded length of the literal prompt, minus the pad
    token each placeholder contributes, plus the visual tokens the processor expands
    those pads into. ``label`` names the shape; the item dimensions and the token
    breakdown in ``detail`` are derived here so every modality reports them alike.
    ``notes`` carries anything the builder wants surfaced, e.g. that the item budget
    was clamped by the ViT attention ceiling.
    """
    placeholders = spec.image_placeholder * len(images) + spec.video_placeholder * len(
        videos
    )

    total_image_tokens = sum(image_tokens)
    total_video_tokens = sum(video_tokens)
    visual_tokens = total_image_tokens + total_video_tokens
    num_items = len(images) + len(videos)

    # Tokens the scaffolding costs us before any filler text is added.
    scaffold = f"USER: {placeholders}\n\n\n{QUESTION}\nASSISTANT:"
    scaffold_tokens = len(tokenizer.encode(scaffold, add_special_tokens=False))
    fixed_tokens = (
        scaffold_tokens - num_items * spec.placeholder_pad_tokens + visual_tokens
    )

    filler_budget = max(0, target_tokens - fixed_tokens)
    filler = text_of_exact_len(tokenizer, filler_budget, variant=variant)

    prompt = f"USER: {placeholders}\n{filler}\n\n{QUESTION}\nASSISTANT:"
    # The placeholder pads are the only part of the encoded prompt that is not text,
    # so dropping them leaves exactly the text tokens: scaffold + filler + question.
    text_tokens = (
        len(tokenizer.encode(prompt, add_special_tokens=False))
        - num_items * spec.placeholder_pad_tokens
    )
    predicted = text_tokens + visual_tokens

    multi_modal_data: dict = {}
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    image_size = (images[0].width, images[0].height) if images else None
    # A clip is (F, H, W, 3); report it width-first like the images. Every clip in
    # one case is the same size by construction, so the first one describes them all.
    video_size = (int(videos[0].shape[2]), int(videos[0].shape[1])) if videos else None
    one_clip_frames = num_frames[0] if num_frames else 0

    shapes = []
    if image_size is not None:
        shapes.append(
            f"{len(images)} image{'s' if len(images) > 1 else ''} "
            f"{image_size[0]}x{image_size[1]} ({total_image_tokens} tok)"
        )
    if video_size is not None:
        shapes.append(
            f"{len(videos)} video{'s' if len(videos) > 1 else ''} "
            f"{one_clip_frames}f {video_size[0]}x{video_size[1]} "
            f"({total_video_tokens} tok)"
        )

    detail = (
        f"{label} [{', '.join(shapes) if shapes else 'no visual input'}] "
        f"prompt={predicted} = text {text_tokens} (filler={filler_budget}) "
        f"+ visual {visual_tokens} (image={total_image_tokens}, "
        f"video={total_video_tokens}), items={num_items}"
    )
    if notes:
        detail += " | " + "; ".join(notes)

    return BuiltInput(
        prompt=prompt,
        multi_modal_data=multi_modal_data,
        predicted_tokens=predicted,
        detail=detail,
        text_tokens=text_tokens,
        filler_tokens=filler_budget,
        image_tokens=total_image_tokens,
        video_tokens=total_video_tokens,
        num_images=len(images),
        num_videos=len(videos),
        image_size=image_size,
        video_size=video_size,
        num_frames=one_clip_frames,
        exact=not videos,
        variant=variant,
    )


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Modality:
    """One input shape: how to build it, and what the model must declare to get it."""

    name: str
    build: Callable[..., BuiltInput] = field(repr=False)
    requires_images: bool = False
    requires_video: bool = False


MODALITY_BUILDERS: dict[str, Modality] = {}


def modality(name: str, *, requires_images: bool = False, requires_video: bool = False):
    """Register a builder under a modality name. Registration order is run order."""

    def register(fn: Callable[..., BuiltInput]) -> Callable[..., BuiltInput]:
        MODALITY_BUILDERS[name] = Modality(
            name=name,
            build=fn,
            requires_images=requires_images,
            requires_video=requires_video,
        )
        return fn

    return register


@modality("text")
def _build_text(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    target_tokens: int,
    *,
    tp_size: int,
    variant: int = 0,
) -> BuiltInput:
    return assemble(
        spec, tokenizer, target_tokens, [], [], [], [], [], "text-only", variant=variant
    )


@modality("image_single", requires_images=True)
def _build_image_single(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    target_tokens: int,
    *,
    tp_size: int,
    variant: int = 0,
) -> BuiltInput:
    # One image, capped so chunked prefill can still admit it in a single chunk, and
    # again so its ViT attention fits the per-NSP address space.
    budget = min(target_tokens // 2, MAX_SINGLE_ITEM_TOKENS)
    budget, note = clamp_item_tokens(geom, tp_size, budget)
    image, tokens = make_image_of_tokens(geom, budget, variant=variant)
    return assemble(
        spec,
        tokenizer,
        target_tokens,
        [image],
        [tokens],
        [],
        [],
        [],
        "single image",
        variant=variant,
        notes=[note] if note else None,
    )


@modality("image_many", requires_images=True)
def _build_image_many(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    target_tokens: int,
    *,
    tp_size: int,
    variant: int = 0,
) -> BuiltInput:
    count = IMAGE_MANY_COUNT
    per_image = min(max(1, (target_tokens * 3 // 4) // count), MAX_SINGLE_ITEM_TOKENS)
    per_image, note = clamp_item_tokens(geom, tp_size, per_image)
    images, tokens = [], []
    for index in range(count):
        # Distinct content per image within the case as well as across the batch, so
        # the encoder cache cannot collapse the eight of them into one.
        image, actual = make_image_of_tokens(
            geom, per_image, variant=variant * count + index
        )
        images.append(image)
        tokens.append(actual)
    return assemble(
        spec,
        tokenizer,
        target_tokens,
        images,
        tokens,
        [],
        [],
        [],
        f"{count} images",
        variant=variant,
        notes=[note] if note else None,
    )


@modality("video", requires_video=True)
def _build_video(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    target_tokens: int,
    *,
    tp_size: int,
    variant: int = 0,
) -> BuiltInput:
    # A clip is one ViT sequence over all of its frames, so the same attention
    # ceiling that bounds a single image bounds the whole video.
    budget = min(target_tokens * 3 // 4, MAX_SINGLE_ITEM_TOKENS)
    budget, note = clamp_item_tokens(geom, tp_size, budget)
    frames, tokens, num_frames = make_video_of_tokens(geom, budget, variant=variant)
    return assemble(
        spec,
        tokenizer,
        target_tokens,
        [],
        [],
        [frames],
        [tokens],
        [num_frames],
        "single video",
        variant=variant,
        notes=[note] if note else None,
    )


@modality("video_many", requires_video=True)
def _build_video_many(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    target_tokens: int,
    *,
    tp_size: int,
    variant: int = 0,
) -> BuiltInput:
    """Several distinct clips in one prompt -- the video count stress point.

    Directly mirrors _build_image_many: split the visual budget across
    VIDEO_MANY_COUNT clips (default 2, versus images' default 8 -- a clip is far more
    expensive per item than an image), clamp each to the ViT ceiling independently,
    and offset every clip's variant so the encoder cache can't collapse them into one.
    """
    count = VIDEO_MANY_COUNT
    per_video = min(max(1, (target_tokens * 3 // 4) // count), MAX_SINGLE_ITEM_TOKENS)
    per_video, note = clamp_item_tokens(geom, tp_size, per_video)
    videos, tokens, frame_counts = [], [], []
    for index in range(count):
        frames, actual, num_frames = make_video_of_tokens(
            geom, per_video, variant=variant * count + index
        )
        videos.append(frames)
        tokens.append(actual)
        frame_counts.append(num_frames)
    return assemble(
        spec,
        tokenizer,
        target_tokens,
        [],
        [],
        videos,
        tokens,
        frame_counts,
        f"{count} videos",
        variant=variant,
        notes=[note] if note else None,
    )


@modality("mixed", requires_images=True)
def _build_mixed(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    target_tokens: int,
    *,
    tp_size: int,
    variant: int = 0,
) -> BuiltInput:
    # ~40% images, ~20% video, remainder text -- interleaved so the mRoPE
    # section boundaries between T/H/W axes are crossed within one prompt.
    # Degrades to images + text on a model without video support rather than
    # dropping the case entirely.
    notes = []
    image_budget = min(target_tokens * 4 // 10, MAX_SINGLE_ITEM_TOKENS * 2)
    per_image = max(1, min(image_budget // 2, MAX_SINGLE_ITEM_TOKENS))
    per_image, note = clamp_item_tokens(geom, tp_size, per_image)
    if note:
        notes.append(f"images: {note}")
    images, tokens = [], []
    for index in range(2):
        image, actual = make_image_of_tokens(
            geom, per_image, variant=variant * 2 + index
        )
        images.append(image)
        tokens.append(actual)

    videos, video_tokens, video_frames = [], [], []
    if spec.supports_video:
        video_budget = min(target_tokens * 2 // 10, MAX_SINGLE_ITEM_TOKENS)
        video_budget, video_note = clamp_item_tokens(geom, tp_size, video_budget)
        if video_note:
            notes.append(f"video: {video_note}")
        frames, actual, num_frames = make_video_of_tokens(
            geom, video_budget, variant=variant
        )
        videos, video_tokens, video_frames = [frames], [actual], [num_frames]

    return assemble(
        spec,
        tokenizer,
        target_tokens,
        images,
        tokens,
        videos,
        video_tokens,
        video_frames,
        f"mixed: 2 images + {'video' if videos else 'no video'}",
        variant=variant,
        notes=notes or None,
    )


# Registration order, so the sweep runs text -> images -> video -> video_many -> mixed.
MODALITIES: tuple[str, ...] = tuple(MODALITY_BUILDERS)


def modalities_for(spec: ModelSpec) -> tuple[str, ...]:
    """Modalities this model can actually be asked for.

    A text-only model yields ("text",): every other shape needs images, video, or
    both, and building one for a model with no vision tower would fail in the
    processor rather than test anything.
    """
    return tuple(
        name
        for name, entry in MODALITY_BUILDERS.items()
        if (spec.supports_images or not entry.requires_images)
        and (spec.supports_video or not entry.requires_video)
    )


def build_input(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    modality_name: str,
    target_tokens: int,
    tp_size: int = 1,
    variant: int = 0,
) -> BuiltInput:
    """Build one case at (approximately) target_tokens prompt tokens.

    ``tp_size`` only affects modalities with visual items, where it sets how much of
    the ViT score matrix lands on one rank and therefore how large a single item may
    be. Text-only callers can leave it at the default.
    """
    entry = MODALITY_BUILDERS.get(modality_name)
    if entry is None:
        raise ValueError(
            f"Unknown modality: {modality_name} (known: {', '.join(MODALITIES)})"
        )
    return entry.build(
        spec, geom, tokenizer, target_tokens, tp_size=tp_size, variant=variant
    )


def build_batch(
    spec: ModelSpec,
    geom: ModelGeometry,
    tokenizer,
    modality_name: str,
    target_tokens: int,
    batch_size: int,
    tp_size: int = 1,
) -> list[BuiltInput]:
    """One BuiltInput per request in the batch, distinct but of equal length.

    Each request gets its own filler rotation and its own synthetic pixel content, so
    no two requests are byte-identical. That is deliberate: with identical inputs
    vLLM's multimodal encoder cache would serve N-1 of them from the first request's
    embeddings, and the batch would not exercise the ViT N times.
    """
    return [
        build_input(
            spec,
            geom,
            tokenizer,
            modality_name,
            target_tokens,
            tp_size=tp_size,
            variant=index,
        )
        for index in range(batch_size)
    ]
