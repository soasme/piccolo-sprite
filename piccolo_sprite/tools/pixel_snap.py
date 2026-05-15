from langchain_core.tools import tool
from ._base import run_script


@tool
def clean_sheet(
    input: str,
    output: str,
    cell: int = 64,
    palette: int = 96,
    chroma_key: str = "#00ff00",
    alpha_threshold: int = 12,
    edge_flood_threshold: float = 105.0,
    residue_threshold: float = 72.0,
) -> dict:
    """Remove chroma key and quantize palette on a sprite sheet.

    Must be called before validate_sheet. Use palette=96-128 for attack/VFX,
    palette=32-64 for simple idle/walk. Returns {ok, stdout, stderr}.
    """
    args = [
        "--input", input,
        "--output", output,
        "--cell", str(cell),
        "--palette", str(palette),
        "--chroma-key", chroma_key,
        "--alpha-threshold", str(alpha_threshold),
        "--edge-flood-threshold", str(edge_flood_threshold),
        "--residue-threshold", str(residue_threshold),
    ]
    return run_script("pixel_snap.py", args)
