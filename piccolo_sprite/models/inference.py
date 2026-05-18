"""
WAN2.2 I2V A14B inference for pixel sprite animation.
LoRA: https://civitai.com/models/2172038/wan22-lora-walk-animation-side-view-sprite-animation-pixel-style

Uses GGUF Q4 quantized experts (bullerwins/Wan2.2-I2V-A14B-GGUF) to run on 4090 (24 GB VRAM).
The pipeline routes between high-noise and low-noise transformer experts automatically based on
denoising timestep (MoE design). Each animation type stacks both LoRAs — one per expert.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files, upload_file
from PIL import Image

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
BASE_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
GGUF_REPO = "bullerwins/Wan2.2-I2V-A14B-GGUF"
GGUF_HIGH = "wan22-i2v-a14b-high-q4-k-s.gguf"
GGUF_LOW = "wan22-i2v-a14b-low-q4-k-s.gguf"
LORA_REPO = "soasme/piccolo-sprite"

# Add new animation types here when LoRAs become available.
LORA_REGISTRY: dict[str, dict[str, str]] = {
    "walk": {
        "high": "wan2.2_i2v_a14b_pixel_walk_lora_v1_high_noise.safetensors",
        "low": "wan2.2_i2v_a14b_pixel_walk_lora_v1_low_noise.safetensors",
    },
    # "attack": { "high": "...", "low": "..." },
    # "jump":   { "high": "...", "low": "..." },
}

# Source repo and file names for each animation type.
# "local" is checked first (e.g. a file already downloaded to /tmp).
_SRC_REPO = "theamusing/wan2.2_pixel_walk_lora"
_SRC_FILES: dict[str, dict[str, tuple[str, str | None]]] = {
    "walk": {
        "high": ("pixel_walk_lora_v1_high_noise.safetensors", "/tmp/pixel_walk_lora_v1_high_noise.safetensors"),
        "low":  ("pixel_walk_lora_v1_low_noise.safetensors",  None),
    },
}


def ensure_loras(animation_type: str) -> tuple[Path, Path]:
    """
    Ensure both LoRA files for *animation_type* exist in soasme/piccolo-sprite,
    uploading from the source repo or a local file if needed.
    Returns (high_noise_local_path, low_noise_local_path).
    """
    lora = LORA_REGISTRY[animation_type]
    src = _SRC_FILES.get(animation_type, {})

    try:
        existing = set(list_repo_files(LORA_REPO, token=HF_TOKEN))
    except Exception:
        existing = set()

    def _ensure(dest_name: str, src_filename: str, local_fallback: str | None) -> None:
        if dest_name in existing:
            return
        if local_fallback and Path(local_fallback).exists():
            local_path: str | Path = local_fallback
            print(f"Using local file {local_fallback} for {dest_name}")
        else:
            print(f"Downloading {src_filename} from {_SRC_REPO} ...")
            local_path = hf_hub_download(_SRC_REPO, src_filename, token=HF_TOKEN)
        print(f"Uploading {dest_name} → {LORA_REPO} ...")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=dest_name,
            repo_id=LORA_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )

    if src:
        _ensure(lora["high"], *src["high"])
        _ensure(lora["low"],  *src["low"])

    high_path = hf_hub_download(LORA_REPO, lora["high"], token=HF_TOKEN)
    low_path  = hf_hub_download(LORA_REPO, lora["low"],  token=HF_TOKEN)
    return Path(high_path), Path(low_path)


def load_pipeline(lora_weight: float = 0.8, animation_type: str = "walk"):
    """
    Build and return a WanImageToVideoPipeline.

    Both transformer experts are loaded as GGUF Q4 (≈8 GB each) and the rest of
    the pipeline is in bfloat16.  enable_model_cpu_offload() keeps peak VRAM under
    24 GB on a 4090.  Both LoRAs are fused at the requested weight.
    """
    from diffusers import (
        GGUFQuantizationConfig,
        WanImageToVideoPipeline,
        WanTransformer3DModel,
    )

    high_lora, low_lora = ensure_loras(animation_type)

    q_cfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    print("Downloading GGUF experts ...")
    high_gguf = hf_hub_download(GGUF_REPO, GGUF_HIGH, token=HF_TOKEN)
    low_gguf  = hf_hub_download(GGUF_REPO, GGUF_LOW,  token=HF_TOKEN)

    print("Loading transformer (high-noise expert, GGUF Q4) ...")
    transformer = WanTransformer3DModel.from_single_file(
        high_gguf,
        quantization_config=q_cfg,
        torch_dtype=torch.bfloat16,
    )

    print("Loading transformer_2 (low-noise expert, GGUF Q4) ...")
    transformer_2 = WanTransformer3DModel.from_single_file(
        low_gguf,
        quantization_config=q_cfg,
        torch_dtype=torch.bfloat16,
    )

    print("Loading pipeline base components ...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        BASE_MODEL_ID,
        transformer=transformer,
        transformer_2=transformer_2,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    pipe.enable_model_cpu_offload()

    # High-noise LoRA targets transformer; low-noise LoRA targets transformer_2.
    # Both LoRA files must use diffusers key prefixes ("transformer.*" / "transformer_2.*").
    # If they were exported from a non-diffusers trainer, keys may need remapping.
    print(f"Loading high-noise LoRA ({high_lora.name}) ...")
    pipe.load_lora_weights(str(high_lora), adapter_name="pixel_high")

    print(f"Loading low-noise LoRA ({low_lora.name}) ...")
    pipe.load_lora_weights(str(low_lora), adapter_name="pixel_low")

    pipe.set_adapters(["pixel_high", "pixel_low"], adapter_weights=[lora_weight, lora_weight])

    return pipe


def run_inference(
    image: Image.Image | str,
    prompt: str = "pixel style looping walk animation",
    animation_type: str = "walk",
    num_frames: int = 24,
    fps: int = 16,
    lora_weight: float = 0.8,
    seed: int | None = None,
    output_path: str | None = None,
    pipeline=None,
) -> str:
    """Run inference and save to an MP4. Returns the output path."""
    from diffusers.utils import export_to_video

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if pipeline is None:
        pipeline = load_pipeline(lora_weight=lora_weight, animation_type=animation_type)

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    result = pipeline(
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        generator=generator,
    )

    if output_path is None:
        output_path = f"output_{animation_type}.mp4"

    export_to_video(result.frames[0], output_path, fps=fps)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAN2.2 I2V pixel sprite animation")
    parser.add_argument("--image", required=True, help="Path to reference image")
    parser.add_argument("--prompt", default="pixel style looping walk animation")
    parser.add_argument(
        "--type",
        dest="animation_type",
        default="walk",
        choices=list(LORA_REGISTRY),
        help="Animation type (selects LoRA pair)",
    )
    parser.add_argument("--frames",      type=int,   default=24,  dest="num_frames")
    parser.add_argument("--fps",         type=int,   default=16)
    parser.add_argument("--lora-weight", type=float, default=0.8)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    run_inference(
        image=args.image,
        prompt=args.prompt,
        animation_type=args.animation_type,
        num_frames=args.num_frames,
        fps=args.fps,
        lora_weight=args.lora_weight,
        seed=args.seed,
        output_path=args.output,
    )
