from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.fix_jaggies import fix_jaggies


def test_fix_jaggies_detect_only():
    mock_result = MagicMock(returncode=0, stdout="No jaggies detected.", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = fix_jaggies.invoke({"input": "/tmp/sheet.png", "cell": 64})
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--input" in cmd
    assert "/tmp/sheet.png" in cmd
    assert "--fix" not in cmd


def test_fix_jaggies_fix_mode():
    mock_result = MagicMock(returncode=0, stdout="Fixed → /tmp/sheet-fixed.png", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = fix_jaggies.invoke({
            "input": "/tmp/sheet.png",
            "output": "/tmp/sheet-fixed.png",
            "fix": True,
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--fix" in cmd
    assert "--output" in cmd
    assert "/tmp/sheet-fixed.png" in cmd


def test_fix_jaggies_fix_requires_output():
    result = fix_jaggies.invoke({"input": "/tmp/sheet.png", "fix": True})
    assert result["ok"] is False
    assert "output" in result["stderr"]


def test_fix_jaggies_propagates_failure():
    mock_result = MagicMock(returncode=1, stdout="frame 0: orphan_0=3", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result):
        result = fix_jaggies.invoke({"input": "/tmp/sheet.png"})
    assert result["ok"] is False
