from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.validate import validate_sheet, validate_manifest, validate_hierarchy


def test_validate_sheet_passes_required_args():
    mock_result = MagicMock(returncode=0, stdout="passed", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = validate_sheet.invoke({
            "input": "/tmp/walk-clean.png",
            "rows": 1,
            "columns": 6,
            "cell": 64,
            "json_out": "/tmp/qa/validation.json",
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--input" in cmd
    assert "--rows" in cmd and "1" in cmd
    assert "--columns" in cmd and "6" in cmd
    assert "--json-out" in cmd


def test_validate_sheet_auto_generates_json_out_when_omitted():
    mock_result = MagicMock(returncode=0, stdout="passed", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        validate_sheet.invoke({
            "input": "/tmp/run/64/final/walk-clean.png",
            "rows": 1,
            "columns": 6,
        })
    cmd = mock_run.call_args[0][0]
    assert "--json-out" in cmd


def test_validate_manifest_passes_manifest_arg():
    mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = validate_manifest.invoke({
            "manifest": "/tmp/run/run-manifest.json",
            "required_sizes": "64",
            "required_actions": "walk",
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--manifest" in cmd
    assert "/tmp/run/run-manifest.json" in cmd
    assert "--required-sizes" in cmd
    assert "64" in cmd


def test_validate_hierarchy_passes_base32():
    mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = validate_hierarchy.invoke({
            "base32": "/tmp/run/32/final/walk-clean.png",
            "rows": 1,
            "columns": 6,
            "json_out": "/tmp/run/qa/hierarchy.json",
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--base32" in cmd
    assert "--json-out" in cmd
