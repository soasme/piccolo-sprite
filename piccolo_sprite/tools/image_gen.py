import base64
from pathlib import Path
from langchain_core.tools import tool
from openai import OpenAI


@tool
def generate_sprite_strip(
    action: str,
    direction: str,
    cell: int,
    run_dir: str,
    prompt: str,
    key_color: str = "#00ff00",
) -> dict:
    """Generate a sprite animation strip using gpt-image-2 and save to run/source/.

    prompt: full image generation prompt describing character, action, and direction.
    key_color: chroma-key color for the background (default #00ff00).
    Returns {ok, path} on success or {ok, error} on failure.
    The returned path is passed to assemble_action_sheet as generated input.
    """
    client = OpenAI()
    try:
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        out_path = Path(run_dir) / "source" / f"{cell}-{action}-{direction}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image_bytes)
        return {"ok": True, "path": str(out_path)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
