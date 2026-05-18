"""
Gradio web UI and API for piccolo-sprite pixel animation generation.
Wraps inference.py — the pipeline is cached in memory across requests.
"""

from __future__ import annotations

import gradio as gr

from piccolo_sprite.models.inference import LORA_REGISTRY, load_pipeline, run_inference

_pipeline = None
_pipeline_key: tuple[str, float] | None = None


def generate(
    image,
    prompt: str,
    animation_type: str,
    lora_weight: float,
    num_frames: int,
    fps: int,
    seed: float,
) -> str:
    global _pipeline, _pipeline_key

    seed_val = int(seed) if seed >= 0 else None
    key = (animation_type, lora_weight)

    if _pipeline is None or _pipeline_key != key:
        _pipeline = load_pipeline(lora_weight=lora_weight, animation_type=animation_type)
        _pipeline_key = key

    return run_inference(
        image=image,
        prompt=prompt,
        animation_type=animation_type,
        num_frames=int(num_frames),
        fps=int(fps),
        lora_weight=lora_weight,
        seed=seed_val,
        pipeline=_pipeline,
    )


with gr.Blocks(title="Piccolo Sprite") as demo:
    gr.Markdown("# Piccolo Sprite — Pixel Animation Generator")
    with gr.Row():
        with gr.Column():
            image_input  = gr.Image(type="pil", label="Reference Image")
            prompt_input = gr.Textbox(value="pixel style looping walk animation", label="Prompt")
            type_input   = gr.Dropdown(choices=list(LORA_REGISTRY), value="walk", label="Animation Type")
            lora_weight  = gr.Slider(0.7, 0.9, value=0.8, step=0.05, label="LoRA Weight")
            num_frames   = gr.Slider(8, 49, value=24, step=1, label="Frames")
            fps_input    = gr.Slider(8, 30, value=16, step=1, label="FPS")
            seed_input   = gr.Number(value=-1, label="Seed (−1 for random)", precision=0)
            run_btn      = gr.Button("Generate", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Output")

    run_btn.click(
        fn=generate,
        inputs=[image_input, prompt_input, type_input, lora_weight, num_frames, fps_input, seed_input],
        outputs=video_output,
        api_name="generate",
    )

if __name__ == "__main__":
    demo.launch()
