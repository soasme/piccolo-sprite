from langchain_core.tools import tool
from ._base import run_script


@tool
def assemble_action_sheet(
    input_dir: str,
    output: str,
    action: str,
    directions: str,
    columns: int = 6,
    cell: int = 64,
    key_color: str = "#00ff00",
    frames_dir: str = "",
) -> dict:
    """Assemble per-direction strips into a single action atlas PNG.

    directions: comma-separated e.g. "south,east,north,west"
    Returns {ok, stdout, stderr}.
    """
    args = [
        "--input-dir", input_dir,
        "--output", output,
        "--action", action,
        "--directions", directions,
        "--columns", str(columns),
        "--cell", str(cell),
        "--key-color", key_color,
    ]
    if frames_dir:
        args += ["--frames-dir", frames_dir]
    return run_script("assemble_action_sheet.py", args)
