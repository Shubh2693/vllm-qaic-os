#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Model registry for the max-context suite, loaded from suite_config.json.

To add a model, add an entry under "models" in suite_config.json -- not here. This
file only defines the shape (``ModelSpec``) and loads the registry into it. An entry
carries *declarations* only -- prompt-format facts, capability flags, and the two
hints the capacity precheck needs. Everything numeric (layer counts, KV heads, head
dim, max_model_len, vision patch geometry, MoE expert counts, the quantization
method) is derived from the HF config at runtime by
``model_geometry.ModelGeometry.from_hf()``.

The suite covers text-only LLMs, MoE models and quantized checkpoints alongside VLMs,
so ``supports_images`` / ``supports_video`` default to False: a new entry is text-only
until it says otherwise, which fails safe. A registry entry that disagrees with the HF
config -- images declared on a model with no vision tower, or vice versa -- is caught
by ``test_selftest_capabilities_match_config`` without touching a device.

A text-only model, of any architecture, MoE topology or quantization, is always just
a registry entry -- model_geometry.py derives everything else from AutoConfig. A VLM
is only a registry entry if it shares the Qwen2-VL/2.5-VL/3-VL processor scheme this
suite's vision builders assume (see README's "Model coverage" section); a
structurally different VLM family needs new code in visual_inputs.py/
prompt_builder.py, not just different image_placeholder/video_placeholder strings
here.
"""

from dataclasses import dataclass, field

from ctx_config import _get

# Qwen-family vision placeholders. The processor expands the inner pad token into the
# correct number of visual tokens; the outer start/end tokens are literal. Every
# model currently in suite_config.json is Qwen-family, so entries there only need to
# set image_placeholder/video_placeholder when a future model's format differs.
QWEN_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one model under test. Everything numeric is derived."""

    model: str

    # --- what to run it at ---------------------------------------------------
    # TP sizes this model is known to run at. suite_config.json's selection.tp_sizes
    # filters this further.
    tp_sizes: tuple[int, ...] = (8, 4)
    # Engine dtype. Quantized checkpoints still load under float16 on QAIC -- the
    # quantization method comes from the config, not from here.
    dtype: str = "float16"
    # Extra LLM(...) kwargs for models that need them (e.g. enable_expert_parallel).
    # Kept as an escape hatch so a one-model quirk does not become an engine_pool if.
    engine_kwargs: dict = field(default_factory=dict)

    # --- capabilities --------------------------------------------------------
    # Text-only unless declared otherwise. modalities_for() reads both of these, so a
    # text-only model contributes exactly one modality and never builds an image.
    supports_images: bool = False
    supports_video: bool = False

    # --- prompt format (only meaningful when supports_images/video) ----------
    image_placeholder: str = QWEN_IMAGE_PLACEHOLDER
    video_placeholder: str = QWEN_VIDEO_PLACEHOLDER
    # Number of tokens the placeholder itself encodes to, and how many of those are
    # the expandable pad token. For Qwen the placeholder is 3 tokens of which 1
    # (the pad) is replaced by N visual tokens.
    placeholder_tokens: int = 3
    placeholder_pad_tokens: int = 1

    # --- loading -------------------------------------------------------------
    trust_remote_code: bool = False
    # Only needed when the HF config does not expose max_position_embeddings, or to
    # hold a model below a ceiling that is impractical to build an engine at.
    max_model_len_override: int | None = None

    # --- capacity precheck hints (never affect correctness, only whether we skip) --
    approx_total_params_b: float = 33.5
    # Bytes of device memory per parameter. None derives it from the config's
    # quantization method (see ModelGeometry.weight_bytes_per_param), which is what
    # you want for anything quantized; set it only to override that.
    weight_bytes_per_param: float | None = None
    # The ViT is generally replicated per rank rather than TP-sharded. 0 for a
    # text-only model.
    replicated_vision_gb: float = 0.0

    # --- assertion tuning ----------------------------------------------------
    # Per-model override of VIDEO_TOKEN_RATIO_SLACK. Qwen3-VL interleaves a timestamp
    # string between video frames, so its realised video token count drifts further
    # from the frames-times-patches prediction than Qwen2.5-VL's does.
    video_ratio_slack: float | None = None


def _load_model_specs() -> dict[str, "ModelSpec"]:
    """Build the registry from suite_config.json's "models" section.

    Each entry only needs the fields that differ from ModelSpec's defaults, same as
    the Python literals this replaced. ``tp_sizes`` comes back from JSON as a list;
    ModelSpec wants a tuple.
    """
    specs = {}
    for name, entry in _get("models").items():
        kwargs = dict(entry)
        if "tp_sizes" in kwargs:
            kwargs["tp_sizes"] = tuple(kwargs["tp_sizes"])
        specs[name] = ModelSpec(model=name, **kwargs)
    return specs


MODEL_SPECS: dict[str, ModelSpec] = _load_model_specs()

# What runs when selection.models is null in suite_config.json: one model per shape
# under test -- text-only MoE/quantized, text-only dense, and the two VLM generations.
DEFAULT_MODELS: tuple[str, ...] = tuple(_get("default_models"))
