#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

"""Numeric model properties derived from the HF config, plus the cached accessors.

Keeping the derivation here is what lets model_specs.py stay purely declarative: a new
model needs one registry entry, and layer counts / KV heads / head dim /
max_model_len / vision patch geometry / MoE expert counts / the quantization method all
come from ``AutoConfig`` at runtime.

The non-vision fields (layers, KV heads, head dim, max_model_len, MoE, quantization)
are architecture-agnostic -- any HF causal LM populates them. The vision fields
(patch_size, spatial_merge_size, temporal_patch_size, pixels_per_visual_token) assume
the Qwen2-VL/2.5-VL/3-VL processor scheme specifically; see README's "Model coverage"
section before relying on them for a different VLM family.
"""

from dataclasses import dataclass

import pytest
from transformers import AutoConfig, AutoTokenizer

from model_specs import MODEL_SPECS, ModelSpec

# Device bytes per parameter, by the quantization method the config declares.
#
# These are deliberate over-estimates. Only part of a "4-bit" checkpoint is actually
# 4-bit -- attention projections, embeddings, norms and the quantization scales stay
# 16-bit, and a runtime without the matching kernels dequantizes on load -- so 1.0
# rather than 0.5. The consumer is a precheck whose failure mode is skipping a case
# that would have fit; under-estimating would instead let the device OOM mid-run.
WEIGHT_BYTES_BY_QUANT: dict[str, float] = {
    "mxfp4": 1.0,
    "nvfp4": 1.0,
    "fp4": 1.0,
    "int4": 1.0,
    "awq": 1.0,
    "gptq": 1.0,
    "gptq_marlin": 1.0,
    "compressed-tensors": 1.1,
    "fp8": 1.1,
    "int8": 1.1,
}
UNQUANTIZED_WEIGHT_BYTES = 2.0  # fp16 / bf16


@dataclass(frozen=True)
class ModelGeometry:
    """Numeric model properties needed to size KV cache and predict token counts."""

    model: str
    max_model_len: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    # Pixels represented by one post-merge visual token: (patch_size * merge_size)**2.
    pixels_per_visual_token: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    # --- derived capability / architecture facts ---------------------------------
    # True when the config carries a vision_config, i.e. the model has a ViT.
    has_vision: bool = False
    # Attention heads in the ViT. Sets how much of the vision score matrix lands on
    # one rank, which is what bounds the largest single image (see visual_inputs).
    vision_num_heads: int = 0
    vision_depth: int = 0
    # MoE facts. num_experts is 0 for a dense model.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    # Quantization method the checkpoint declares, lowercased, or None.
    quantization: str | None = None
    # Sliding-window span when the model uses one. Informational: the KV estimate
    # assumes full attention on every layer, which over-estimates for a model that
    # alternates in sliding-window layers, and over-estimating is the safe direction.
    sliding_window: int | None = None

    @property
    def visual_token_px(self) -> int:
        """Side length in pixels of the square block one visual token covers."""
        return self.patch_size * self.spatial_merge_size

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def patches_per_visual_token(self) -> int:
        """Pre-merge ViT patches behind one post-merge visual token."""
        return self.spatial_merge_size**2

    def weight_bytes_per_param(self, spec: ModelSpec) -> float:
        """Device bytes per parameter: the spec's override, else from the quant method."""
        if spec.weight_bytes_per_param is not None:
            return spec.weight_bytes_per_param
        if self.quantization is None:
            return UNQUANTIZED_WEIGHT_BYTES
        return WEIGHT_BYTES_BY_QUANT.get(self.quantization, UNQUANTIZED_WEIGHT_BYTES)

    def capability_mismatch(self, spec: ModelSpec) -> str | None:
        """Why the spec's declared capabilities disagree with the config, if they do.

        A registry entry claiming images on a model with no vision tower would
        otherwise fail deep inside the processor on a device; this turns it into a
        device-free assertion.
        """
        if spec.supports_images and not self.has_vision:
            return (
                f"{spec.model} declares supports_images but its config has no "
                "vision_config"
            )
        if spec.supports_video and not self.has_vision:
            return (
                f"{spec.model} declares supports_video but its config has no "
                "vision_config"
            )
        if self.has_vision and not spec.supports_images:
            return (
                f"{spec.model} has a vision_config but declares supports_images=False, "
                "so the vision path is never exercised"
            )
        return None

    def describe(self) -> str:
        """One-line architecture summary for the self-test and the engine build log."""
        parts = [
            f"max_model_len={self.max_model_len}",
            f"layers={self.num_layers}",
            f"kv_heads={self.num_kv_heads}",
            f"head_dim={self.head_dim}",
        ]
        if self.is_moe:
            parts.append(f"moe experts={self.num_experts}/{self.num_experts_per_tok}")
        else:
            parts.append("dense")
        parts.append(f"quant={self.quantization or 'none'}")
        if self.sliding_window:
            parts.append(f"sliding_window={self.sliding_window}")
        if self.has_vision:
            parts.append(
                f"vit heads={self.vision_num_heads} depth={self.vision_depth} "
                f"block={self.visual_token_px}px"
            )
        else:
            parts.append("text-only")
        return " ".join(parts)

    @classmethod
    def from_hf(cls, spec: ModelSpec) -> "ModelGeometry":
        cfg = AutoConfig.from_pretrained(
            spec.model, trust_remote_code=spec.trust_remote_code
        )
        # Multimodal configs nest the language model under text_config on some
        # architectures and inline it on others.
        text_cfg = getattr(cfg, "text_config", None) or cfg

        def _pick(*names, default=None):
            for source in (text_cfg, cfg):
                for name in names:
                    value = getattr(source, name, None)
                    if value is not None:
                        return value
            if default is None:
                raise ValueError(
                    f"Could not derive {names[0]!r} from the config of {spec.model}"
                )
            return default

        def _pick_optional(*names):
            for source in (text_cfg, cfg):
                for name in names:
                    value = getattr(source, name, None)
                    if value is not None:
                        return value
            return None

        num_layers = int(_pick("num_hidden_layers"))
        num_heads = int(_pick("num_attention_heads"))
        num_kv_heads = int(_pick("num_key_value_heads", default=num_heads))
        hidden_size = int(_pick("hidden_size"))
        head_dim = int(_pick("head_dim", default=hidden_size // num_heads))

        max_model_len = spec.max_model_len_override or int(
            _pick("max_position_embeddings")
        )

        # MoE. Every family spells the expert count differently, and a dense model
        # simply has none of them.
        num_experts = int(
            _pick_optional("num_local_experts", "num_experts", "n_routed_experts") or 0
        )
        num_experts_per_tok = int(
            _pick_optional("num_experts_per_tok", "moe_topk", "moe_k") or 0
        )

        # Quantization. quantization_config is a dict on some configs and an object
        # with a quant_method attribute on others.
        quantization = None
        quant_cfg = getattr(cfg, "quantization_config", None)
        if quant_cfg is not None:
            method = (
                quant_cfg.get("quant_method")
                if isinstance(quant_cfg, dict)
                else getattr(quant_cfg, "quant_method", None)
            )
            if method is not None:
                quantization = str(method).lower()

        sliding_window = _pick_optional("sliding_window")
        sliding_window = int(sliding_window) if sliding_window else None

        # Vision geometry. spatial_merge_size / temporal_patch_size are frequently
        # absent from config.json and fall back to their HF defaults of 2 -- the
        # same getattr-with-default approach vllm_qaic's
        # _apply_dynamic_resolution_config uses.
        #
        # These three fields assume the Qwen2-VL/2.5-VL/3-VL dynamic-resolution
        # scheme specifically (patch grid, spatial merge, frame pairing) -- see
        # README's "Model coverage" section. A VLM with a different vision
        # architecture (fixed-token-per-image, AnyRes tiling, ...) would still
        # populate these with *some* value via the fallbacks below, but
        # pixels_per_visual_token would not describe how its processor actually
        # expands visual placeholders.
        vision_cfg = getattr(cfg, "vision_config", None)
        patch_size = int(
            getattr(
                vision_cfg, "patch_size", getattr(vision_cfg, "spatial_patch_size", 14)
            )
            if vision_cfg is not None
            else 14
        )
        spatial_merge_size = int(
            getattr(vision_cfg, "spatial_merge_size", 2) if vision_cfg else 2
        )
        temporal_patch_size = int(
            getattr(vision_cfg, "temporal_patch_size", 2) if vision_cfg else 2
        )
        vision_num_heads = int(getattr(vision_cfg, "num_heads", 0) if vision_cfg else 0)
        vision_depth = int(getattr(vision_cfg, "depth", 0) if vision_cfg else 0)
        if vision_cfg is not None and not vision_num_heads:
            # Some vision configs spell it num_attention_heads instead.
            vision_num_heads = int(getattr(vision_cfg, "num_attention_heads", 0) or 0)

        return cls(
            model=spec.model,
            max_model_len=max_model_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            pixels_per_visual_token=(patch_size * spatial_merge_size) ** 2,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            has_vision=vision_cfg is not None,
            vision_num_heads=vision_num_heads,
            vision_depth=vision_depth,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            quantization=quantization,
            sliding_window=sliding_window,
        )


_GEOMETRY_CACHE: dict[str, ModelGeometry] = {}
_TOKENIZER_CACHE: dict[str, object] = {}


def geometry_for(spec: ModelSpec) -> ModelGeometry:
    if spec.model not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[spec.model] = ModelGeometry.from_hf(spec)
    return _GEOMETRY_CACHE[spec.model]


def tokenizer_for(spec: ModelSpec):
    if spec.model not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[spec.model] = AutoTokenizer.from_pretrained(
            spec.model, trust_remote_code=spec.trust_remote_code
        )
    return _TOKENIZER_CACHE[spec.model]


def resolve(model_name: str) -> tuple[ModelSpec, ModelGeometry]:
    """Spec + geometry for a parametrised model name, skipping with a real reason.

    Case collection deliberately swallows config-read failures so that collection
    always succeeds; this is where an offline or gated model turns into a visible
    skip.
    """
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        pytest.skip(f"{model_name} is not in MODEL_SPECS")
    try:
        return spec, geometry_for(spec)
    except Exception as exc:
        pytest.skip(f"Could not read the HF config for {model_name}: {exc}")
