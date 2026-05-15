from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.audit import audit_motion


def test_audit_motion_builds_correct_command():
    mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        result = audit_motion.invoke({
            "input": "/tmp/walk-clean.png",
            "rows": 1,
            "columns": 6,
            "cell": 64,
            "json_out": "/tmp/qa/audit.json",
        })
    assert result["ok"] is True
    cmd = mock_run.call_args[0][0]
    assert "--input" in cmd
    assert "--rows" in cmd
    assert "--json-out" in cmd


def test_audit_motion_auto_derives_json_out():
    mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("piccolo_sprite.tools._base.subprocess.run", return_value=mock_result) as mock_run:
        audit_motion.invoke({
            "input": "/tmp/run/64/final/walk-clean.png",
            "rows": 1,
            "columns": 6,
        })
    cmd = mock_run.call_args[0][0]
    assert "--json-out" in cmd
