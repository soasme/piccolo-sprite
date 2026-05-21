"""Unit tests for scripts/fix_jaggies.py — tests the detection algorithm directly."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from fix_jaggies import detect_frame, process_sheet  # noqa: E402


def _make_opaque(coords: list[tuple[int, int]], size: int = 8) -> list[list[bool]]:
    grid = [[False] * size for _ in range(size)]
    for x, y in coords:
        grid[y][x] = True
    return grid


def test_detect_isolated_pixel():
    """A lone pixel with no opaque ortho neighbours is orphan_0."""
    grid = _make_opaque([(4, 4)])
    jaggies = detect_frame(grid, 8, 8)
    assert any(x == 4 and y == 4 and n == 0 for x, y, n in jaggies)


def test_detect_no_jaggies_solid_block():
    """Interior pixels of a solid 4×4 block have 4 ortho neighbours — not jaggies."""
    coords = [(x, y) for x in range(1, 5) for y in range(1, 5)]
    grid = _make_opaque(coords)
    jaggies = detect_frame(grid, 8, 8)
    # Only edge pixels of the block should be candidates, not interior ones.
    interior = {(x, y, n) for x, y, n in jaggies if 2 <= x <= 3 and 2 <= y <= 3}
    assert not interior


def test_process_sheet_no_jaggies(tmp_path):
    """A sheet with one solid 8×8 frame reports zero jaggies."""
    img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    path = tmp_path / "sheet.png"
    img.save(path)
    result = process_sheet(path, None, cell=8, alpha_threshold=8, soft=False, fix=False)
    assert result["total_jaggies"] == 0
    assert result["fixed"] is False


def test_process_sheet_detects_lone_pixel(tmp_path):
    """A transparent 8×8 frame with one opaque pixel has at least one jaggy."""
    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    pix = img.load()
    pix[4, 4] = (255, 0, 0, 255)
    path = tmp_path / "sheet.png"
    img.save(path)
    result = process_sheet(path, None, cell=8, alpha_threshold=8, soft=False, fix=False)
    assert result["total_jaggies"] >= 1


def test_process_sheet_fix_removes_lone_pixel(tmp_path):
    """With fix=True, the orphan pixel should be removed in the output."""
    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    pix = img.load()
    pix[4, 4] = (255, 0, 0, 255)
    in_path = tmp_path / "sheet.png"
    out_path = tmp_path / "sheet-fixed.png"
    img.save(in_path)
    result = process_sheet(in_path, out_path, cell=8, alpha_threshold=8, soft=False, fix=True)
    assert result["fixed"] is True
    assert out_path.exists()
    with Image.open(out_path) as fixed:
        fixed_pix = fixed.convert("RGBA").load()
        _, _, _, a = fixed_pix[4, 4]
    assert a == 0
