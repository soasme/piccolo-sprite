from langchain_core.tools import tool
from ._base import run_script


@tool
def export_previews(
    atlas: str,
    out_dir: str,
    rows: int,
    columns: int,
    cell: int = 64,
    scale: int = 4,
    row_names: str = "",
    prefix: str = "row",
    duration: int = 90,
) -> dict:
    """Export GIF and WebP preview animations from a sprite atlas.

    Returns {ok, stdout, stderr}. Preview files are written to out_dir.
    """
    args = [
        "--atlas", atlas,
        "--out-dir", out_dir,
        "--rows", str(rows),
        "--columns", str(columns),
        "--cell", str(cell),
        "--scale", str(scale),
        "--prefix", prefix,
        "--duration", str(duration),
    ]
    if row_names:
        args += ["--row-names", row_names]
    return run_script("export_animation_previews.py", args)
