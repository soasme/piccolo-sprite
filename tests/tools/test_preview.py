from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.preview import export_previews


def test_export_previews_builds_correct_command():
    mock_result = MagicMock(returncode=0, stdout="wrote previews", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = export_previews.invoke({
            "atlas": "/tmp/walk-clean.png",
            "out_dir": "/tmp/qa/previews",
            "rows": 1,
            "columns": 6,
            "cell": 64,
            "row_names": "south",
            "prefix": "walk",
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--atlas" in cmd
    assert "--out-dir" in cmd
    assert "--row-names" in cmd
    assert "south" in cmd
    assert "--prefix" in cmd
    assert "walk" in cmd
