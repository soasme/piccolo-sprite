from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.assemble import assemble_action_sheet


def test_assemble_builds_correct_command():
    mock_result = MagicMock(returncode=0, stdout="wrote output.png", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = assemble_action_sheet.invoke({
            "input_dir": "/tmp/run/source",
            "output": "/tmp/run/64/final/walk-sheet.png",
            "action": "walk",
            "directions": "south,east",
            "columns": 6,
            "cell": 64,
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--input-dir" in cmd
    assert "/tmp/run/source" in cmd
    assert "--action" in cmd
    assert "walk" in cmd
    assert "--directions" in cmd
    assert "south,east" in cmd
    assert "--columns" in cmd
    assert "6" in cmd
    assert "--cell" in cmd
    assert "64" in cmd


def test_assemble_returns_ok_false_on_nonzero_exit():
    mock_result = MagicMock(returncode=1, stdout="", stderr="frame seed error")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result):
        result = assemble_action_sheet.invoke({
            "input_dir": "/tmp/in",
            "output": "/tmp/out.png",
            "action": "walk",
            "directions": "south",
        })
    assert result["ok"] is False
    assert "frame seed error" in result["stderr"]
