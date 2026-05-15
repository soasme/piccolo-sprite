from pathlib import Path
from langchain_core.tools import tool
from ._base import run_script


@tool
def validate_sheet(
    input: str,
    rows: int,
    columns: int,
    cell: int = 64,
    row_names: str = "",
    json_out: str = "",
    contact_sheet: str = "",
    fail_on_warnings: bool = True,
) -> dict:
    """Validate sprite atlas geometry, chroma residue, and frame coverage.

    clean_sheet must be called before validate_sheet. If json_out is omitted,
    a path is auto-derived from the input path. Returns {ok, stdout, stderr}.
    """
    if not json_out:
        json_out = str(Path(input).with_name(Path(input).stem + "-validation.json"))
    args = [
        "--input", input,
        "--rows", str(rows),
        "--columns", str(columns),
        "--cell", str(cell),
        "--json-out", json_out,
    ]
    if row_names:
        args += ["--row-names", row_names]
    if contact_sheet:
        args += ["--contact-sheet", contact_sheet]
    if fail_on_warnings:
        args.append("--fail-on-warnings")
    return run_script("validate_sheet.py", args)


@tool
def validate_manifest(
    manifest: str,
    required_sizes: str = "",
    required_actions: str = "",
    required_directions: str = "",
    require_visual_review: bool = False,
    json_out: str = "",
) -> dict:
    """Validate run-manifest.json for imagegen provenance and scope completeness.

    Returns {ok, stdout, stderr}.
    """
    args = ["--manifest", manifest]
    if required_sizes:
        args += ["--required-sizes", required_sizes]
    if required_actions:
        args += ["--required-actions", required_actions]
    if required_directions:
        args += ["--required-directions", required_directions]
    if require_visual_review:
        args.append("--require-visual-review")
    if json_out:
        args += ["--json-out", json_out]
    return run_script("validate_run_manifest.py", args)


@tool
def validate_hierarchy(
    base32: str,
    rows: int,
    columns: int,
    json_out: str,
    sheet64: str = "",
    sheet128: str = "",
    fail_on_warnings: bool = True,
) -> dict:
    """Validate 32/64/128 resolution hierarchy by comparing thumbnail primary structure.

    Only call for multi-size jobs. This is QA only — never use output images as assets.
    Returns {ok, stdout, stderr}.
    """
    args = [
        "--base32", base32,
        "--rows", str(rows),
        "--columns", str(columns),
        "--json-out", json_out,
    ]
    if sheet64:
        args += ["--sheet64", sheet64]
    if sheet128:
        args += ["--sheet128", sheet128]
    if fail_on_warnings:
        args.append("--fail-on-warnings")
    return run_script("validate_resolution_hierarchy.py", args)
