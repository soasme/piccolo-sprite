#!/usr/bin/env python3
"""Detect and optionally fix jaggies (staircase artifacts) in pixel-art sprite sheets.

A "jaggy" in pixel art is an orphan edge pixel that is connected to the body
only diagonally — it has zero or one orthogonal opaque neighbours.  These
single-pixel protrusions create the jagged staircase look on diagonals.

Algorithm per cell / frame:
  1. Collect all opaque pixels (alpha > alpha_threshold).
  2. Identify edge pixels: opaque pixels that have at least one transparent
     orthogonal neighbour.
  3. For each edge pixel count its orthogonal opaque neighbours.
     - orphan_0: 0 orthogonal opaque neighbours → definite jaggy (lone dot)
     - orphan_1: exactly 1 orthogonal opaque neighbour  → likely jaggy corner
  4. Report counts per frame and total.

With --fix:
  - orphan_0 pixels are made fully transparent.
  - orphan_1 pixels are made fully transparent (strict) or left (with --soft).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG  = [(-1, -1), (1, -1), (-1, 1), (1, 1)]


def detect_frame(pixels: list[list[bool]], w: int, h: int) -> list[tuple[int, int, int]]:
    """Return list of (x, y, ortho_opaque_count) for jaggy candidates."""
    jaggies = []
    for y in range(h):
        for x in range(w):
            if not pixels[y][x]:
                continue
            ortho_opaque = sum(
                1 for dx, dy in ORTHO
                if 0 <= x + dx < w and 0 <= y + dy < h and pixels[y + dy][x + dx]
            )
            ortho_transparent = 4 - ortho_opaque - sum(
                1 for dx, dy in ORTHO
                if not (0 <= x + dx < w and 0 <= y + dy < h)
            )
            has_transparent_ortho_neighbour = any(
                (0 <= x + dx < w and 0 <= y + dy < h and not pixels[y + dy][x + dx])
                or not (0 <= x + dx < w and 0 <= y + dy < h)
                for dx, dy in ORTHO
            )
            if has_transparent_ortho_neighbour and ortho_opaque <= 1:
                jaggies.append((x, y, ortho_opaque))
    return jaggies


def process_sheet(
    input_path: Path,
    output_path: Path | None,
    cell: int,
    alpha_threshold: int,
    soft: bool,
    fix: bool,
) -> dict:
    with Image.open(input_path) as opened:
        sheet = opened.convert("RGBA")

    sw, sh = sheet.size
    cols = sw // cell
    rows = sh // cell
    total_cells = cols * rows

    report: list[dict] = []
    total_jaggies = 0

    out_sheet = sheet.copy() if fix else None

    for row in range(rows):
        for col in range(cols):
            frame_idx = row * cols + col
            x0, y0 = col * cell, row * cell
            frame = sheet.crop((x0, y0, x0 + cell, y0 + cell))
            pix = frame.load()

            # Build boolean opaque map
            opaque = [[False] * cell for _ in range(cell)]
            for fy in range(cell):
                for fx in range(cell):
                    r, g, b, a = pix[fx, fy]
                    opaque[fy][fx] = a > alpha_threshold

            jaggies = detect_frame(opaque, cell, cell)
            orphan_0 = [(x, y) for x, y, n in jaggies if n == 0]
            orphan_1 = [(x, y) for x, y, n in jaggies if n == 1]

            count = len(orphan_0) + len(orphan_1)
            total_jaggies += count

            report.append({
                "frame": frame_idx,
                "col": col,
                "row": row,
                "orphan_0": len(orphan_0),
                "orphan_1": len(orphan_1),
                "total": count,
            })

            if fix and out_sheet is not None:
                out_pix = out_sheet.load()
                for fx, fy in orphan_0:
                    r, g, b, a = out_pix[x0 + fx, y0 + fy]
                    out_pix[x0 + fx, y0 + fy] = (r, g, b, 0)
                if not soft:
                    for fx, fy in orphan_1:
                        r, g, b, a = out_pix[x0 + fx, y0 + fy]
                        out_pix[x0 + fx, y0 + fy] = (r, g, b, 0)

    if fix and output_path is not None and out_sheet is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_sheet.save(output_path)

    return {
        "input": str(input_path),
        "output": str(output_path) if output_path else None,
        "cell": cell,
        "total_cells": total_cells,
        "total_jaggies": total_jaggies,
        "fixed": fix,
        "frames": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to sprite sheet PNG")
    parser.add_argument("--output", help="Output path for fixed sheet (required with --fix)")
    parser.add_argument("--cell", type=int, default=64, help="Cell size in pixels (default 64)")
    parser.add_argument("--alpha-threshold", type=int, default=8,
                        help="Pixels with alpha <= this are transparent (default 8)")
    parser.add_argument("--fix", action="store_true", help="Write a fixed copy to --output")
    parser.add_argument("--soft", action="store_true",
                        help="With --fix: only remove orphan_0; leave orphan_1 pixels")
    parser.add_argument("--json", action="store_true", help="Output report as JSON")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = None
    if args.fix:
        if not args.output:
            print("ERROR: --fix requires --output", file=sys.stderr)
            sys.exit(1)
        output_path = Path(args.output).expanduser().resolve()

    result = process_sheet(
        input_path=input_path,
        output_path=output_path,
        cell=args.cell,
        alpha_threshold=args.alpha_threshold,
        soft=args.soft,
        fix=args.fix,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Sheet : {result['input']}")
        print(f"Cells : {result['total_cells']}  ({result['cell']}px each)")
        print(f"Jaggies found : {result['total_jaggies']}")
        if result["fixed"]:
            print(f"Fixed → {result['output']}")
        print()
        for f in result["frames"]:
            if f["total"] > 0:
                print(
                    f"  frame {f['frame']:>3} (col={f['col']}, row={f['row']}): "
                    f"orphan_0={f['orphan_0']}  orphan_1={f['orphan_1']}  total={f['total']}"
                )
        if result["total_jaggies"] == 0:
            print("  ✓ No jaggies detected.")

    sys.exit(0 if result["total_jaggies"] == 0 else 1)


if __name__ == "__main__":
    main()
