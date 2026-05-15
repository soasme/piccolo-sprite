import json
from pathlib import Path
import pytest
from piccolo_sprite.tools.manifest import read_manifest, write_manifest


def test_write_manifest_creates_file(tmp_path):
    result = write_manifest.invoke({
        "run_dir": str(tmp_path),
        "data": {"scope": {"sizes": [64]}, "strips": []},
    })
    assert result["ok"] is True
    manifest_path = tmp_path / "run-manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["scope"]["sizes"] == [64]


def test_read_manifest_returns_data(tmp_path):
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(json.dumps({"scope": {"sizes": [32, 64]}}))
    result = read_manifest.invoke({"run_dir": str(tmp_path)})
    assert result["ok"] is True
    assert result["manifest"]["scope"]["sizes"] == [32, 64]


def test_read_manifest_missing_file(tmp_path):
    result = read_manifest.invoke({"run_dir": str(tmp_path)})
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_write_manifest_creates_run_dir(tmp_path):
    nested = tmp_path / "newrun"
    result = write_manifest.invoke({"run_dir": str(nested), "data": {}})
    assert result["ok"] is True
    assert (nested / "run-manifest.json").exists()
