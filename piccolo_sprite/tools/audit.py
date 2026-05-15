from pathlib import Path
from langchain_core.tools import tool
from ._base import run_script


@tool
def audit_motion(
    input: str,
    rows: int,
    columns: int,
    cell: int = 64,
    row_names: str = "",
    json_out: str = "",
    fail_on_warnings: bool = False,
) -> dict:
    """Audit sprite motion for near-duplicate frames, chroma residue, and clipping.

    If json_out is omitted, a path is auto-derived from the input path.
    Returns {ok, stdout, stderr}.
    """
    if not json_out:
        json_out = str(Path(input).with_name(Path(input).stem + "-audit.json"))
    args = [
        "--input", input,
        "--rows", str(rows),
        "--columns", str(columns),
        "--cell", str(cell),
        "--json-out", json_out,
    ]
    if row_names:
        args += ["--row-names", row_names]
    if fail_on_warnings:
        args.append("--fail-on-warnings")
    return run_script("audit_sprite_motion.py", args)
