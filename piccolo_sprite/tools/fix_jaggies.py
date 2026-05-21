from langchain_core.tools import tool
from ._base import run_script


@tool
def fix_jaggies(
    input: str,
    output: str = "",
    cell: int = 64,
    alpha_threshold: int = 8,
    fix: bool = False,
    soft: bool = False,
) -> dict:
    """Detect (and optionally fix) jaggies in a pixel-art sprite sheet.

    Jaggies are orphan edge pixels connected to the body only diagonally,
    creating staircase artifacts on diagonal outlines.

    Detection reports per-frame counts of:
      - orphan_0: pixel with zero orthogonal opaque neighbours (definite jaggy)
      - orphan_1: pixel with one orthogonal opaque neighbour (likely jaggy corner)

    Args:
        input: Path to the sprite-sheet PNG to inspect.
        output: Output path for the cleaned sheet. Required when fix=True.
        cell: Cell size in pixels (default 64).
        alpha_threshold: Pixels with alpha <= this value are treated as
            transparent (default 8).
        fix: When True, write a corrected copy to `output`.
        soft: When True (and fix=True), only remove orphan_0 pixels;
            orphan_1 pixels are left untouched.

    Returns:
        {ok, stdout, stderr} — stdout contains a human-readable report.
        Exit code 1 (ok=False) when jaggies are found; 0 when clean.
    """
    args = [
        "--input", input,
        "--cell", str(cell),
        "--alpha-threshold", str(alpha_threshold),
    ]
    if fix:
        if not output:
            return {"ok": False, "stdout": "", "stderr": "output is required when fix=True"}
        args += ["--fix", "--output", output]
    if soft:
        args.append("--soft")
    return run_script("fix_jaggies.py", args)
