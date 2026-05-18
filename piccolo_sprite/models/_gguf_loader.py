"""
Custom GGUF loader for WanTransformer3DModel.

diffusers from_single_file fails for WAN2.2 because the GGUF key names
(e.g. blocks.0.self_attn.q) differ from diffusers key names (blocks.0.attn1.to_q).
This module loads the GGUF tensors directly, remaps keys, then passes the corrected
dict back into from_single_file (which accepts a pre-built dict as its first argument)
so that GGUFQuantizationConfig still sets up the quantized-layer forward methods.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import gguf


# ── key remapping tables ────────────────────────────────────────────────────

_TOPLEVEL_MAP: dict[str, str] = {
    "head.head":         "proj_out",
    "head.modulation":   "scale_shift_table",
    "text_embedding.0":  "condition_embedder.text_embedder.linear_1",
    "text_embedding.2":  "condition_embedder.text_embedder.linear_2",
    "time_embedding.0":  "condition_embedder.time_embedder.linear_1",
    "time_embedding.2":  "condition_embedder.time_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
}

_BLOCK_MAP: dict[str, str] = {
    "self_attn.q":       "attn1.to_q",
    "self_attn.k":       "attn1.to_k",
    "self_attn.v":       "attn1.to_v",
    "self_attn.o":       "attn1.to_out.0",
    "self_attn.norm_q":  "attn1.norm_q",
    "self_attn.norm_k":  "attn1.norm_k",
    "cross_attn.q":      "attn2.to_q",
    "cross_attn.k":      "attn2.to_k",
    "cross_attn.v":      "attn2.to_v",
    "cross_attn.o":      "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "ffn.0":             "ffn.net.0.proj",
    "ffn.2":             "ffn.net.2",
    "modulation":        "scale_shift_table",
    "norm3":             "norm2",
}

_BLOCK_RE = re.compile(r"^(blocks\.\d+)\.(.*)")

# Sorted by descending key length so longer prefixes match first.
_TOPLEVEL_SORTED = sorted(_TOPLEVEL_MAP.items(), key=lambda x: -len(x[0]))
_BLOCK_SORTED    = sorted(_BLOCK_MAP.items(),    key=lambda x: -len(x[0]))


def _remap(key: str) -> str:
    for old, new in _TOPLEVEL_SORTED:
        if key == old or key.startswith(old + "."):
            return new + key[len(old):]
    m = _BLOCK_RE.match(key)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        for old, new in _BLOCK_SORTED:
            if suffix == old or suffix.startswith(old + "."):
                return f"{prefix}.{new}{suffix[len(old):]}"
    return key  # unchanged (e.g. patch_embedding.weight)


# ── tensor conversion ────────────────────────────────────────────────────────

_FLOAT_TYPES = {
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.BF16,
}
_DTYPE_MAP = {
    gguf.GGMLQuantizationType.F16:  torch.float16,
    gguf.GGMLQuantizationType.F32:  torch.float32,
    gguf.GGMLQuantizationType.BF16: torch.bfloat16,
}


def _to_tensor(t: gguf.ReaderTensor) -> torch.Tensor:
    """Convert a GGUFReader tensor to a torch tensor (GGUFParameter if quantized)."""
    if t.tensor_type in _FLOAT_TYPES:
        return torch.from_numpy(t.data.copy()).to(_DTYPE_MAP[t.tensor_type])

    from diffusers.quantizers.gguf.utils import GGUFParameter
    raw = torch.from_numpy(t.data.view(np.uint8).copy())
    return GGUFParameter(raw, quant_type=t.tensor_type)


# ── public API ───────────────────────────────────────────────────────────────

def load_wan_transformer_gguf(
    gguf_path: str | Path,
    model_id: str,
    subfolder: str,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> "WanTransformer3DModel":
    """
    Load a WanTransformer3DModel from a GGUF file.

    Bypasses from_single_file (which mis-detects config when given a dict and
    errors on WAN2.2 key mismatch when given a path). Instead:
      1. Loads GGUF tensors and remaps keys to diffusers convention
      2. Creates model architecture on meta device from the HF config
      3. Replaces nn.Linear with GGUF-aware layers via _replace_with_gguf_linear
      4. Loads state dict with assign=True (replaces meta tensors in-place)
    """
    from diffusers import GGUFQuantizationConfig, WanTransformer3DModel
    from diffusers.quantizers import DiffusersAutoQuantizer
    from diffusers.quantizers.gguf.gguf_quantizer import _replace_with_gguf_linear

    print(f"  Loading GGUF tensors from {Path(gguf_path).name} ...")
    reader = gguf.GGUFReader(str(gguf_path), "r")
    state_dict: dict[str, torch.Tensor] = {}
    for t in reader.tensors:
        state_dict[_remap(t.name)] = _to_tensor(t)

    print(f"  Creating WanTransformer3DModel ({subfolder}) on meta device ...")
    cfg = WanTransformer3DModel.load_config(model_id, subfolder=subfolder)
    with torch.device("meta"):
        model = WanTransformer3DModel(**cfg)

    # Replace nn.Linear with GGUF-aware layers that dequantize on forward.
    _replace_with_gguf_linear(model, compute_dtype, state_dict)

    # load_state_dict(assign=True) still checks shapes, which breaks for Q4 tensors
    # (shape [out, compressed_in] ≠ [out, in]).  Assign directly to bypass the check.
    model_params = dict(model.named_parameters())
    missing = [k for k in model_params if k not in state_dict]
    unexpected = [k for k in state_dict if k not in model_params]
    for name, param in state_dict.items():
        if name not in model_params:
            continue
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        p = param if isinstance(param, torch.nn.Parameter) else torch.nn.Parameter(param, requires_grad=False)
        setattr(module, parts[-1], p)

    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.is_quantized = True
    q_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
    model.hf_quantizer = DiffusersAutoQuantizer.from_config(q_config)

    return model
