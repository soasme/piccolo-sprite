from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.pixel_snap import clean_sheet


def test_clean_sheet_builds_correct_command():
    mock_result = MagicMock(returncode=0, stdout="done", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = clean_sheet.invoke({
            "input": "/tmp/walk-sheet.png",
            "output": "/tmp/walk-sheet-clean.png",
            "cell": 64,
            "palette": 96,
            "chroma_key": "#00ff00",
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--input" in cmd
    assert "/tmp/walk-sheet.png" in cmd
    assert "--output" in cmd
    assert "--cell" in cmd
    assert "64" in cmd
    assert "--palette" in cmd
    assert "96" in cmd
    assert "--chroma-key" in cmd
    assert "#00ff00" in cmd


def test_clean_sheet_propagates_failure():
    mock_result = MagicMock(returncode=1, stdout="", stderr="bad input")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result):
        result = clean_sheet.invoke({"input": "/bad", "output": "/out"})
    assert result["ok"] is False
