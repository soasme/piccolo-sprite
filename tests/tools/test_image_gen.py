import base64
from pathlib import Path
from unittest.mock import patch, MagicMock
from piccolo_sprite.tools.image_gen import generate_sprite_strip


def _make_mock_response(b64_data: str):
    mock_image = MagicMock()
    mock_image.b64_json = b64_data
    mock_response = MagicMock()
    mock_response.data = [mock_image]
    return mock_response


def test_generate_saves_image_to_run_source(tmp_path):
    fake_png = base64.b64encode(b"PNG_BYTES").decode()
    mock_response = _make_mock_response(fake_png)
    with patch("piccolo_sprite.tools.image_gen.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.images.generate.return_value = mock_response
        result = generate_sprite_strip.invoke({
            "action": "walk",
            "direction": "south",
            "cell": 64,
            "run_dir": str(tmp_path),
            "prompt": "64x64 knight walk south strip",
        })
    assert result["ok"] is True
    expected_path = tmp_path / "source" / "64-walk-south.png"
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"PNG_BYTES"
    assert result["path"] == str(expected_path)


def test_generate_calls_correct_model():
    fake_png = base64.b64encode(b"PNG").decode()
    mock_response = _make_mock_response(fake_png)
    with patch("piccolo_sprite.tools.image_gen.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.images.generate.return_value = mock_response
        generate_sprite_strip.invoke({
            "action": "idle",
            "direction": "east",
            "cell": 32,
            "run_dir": "/tmp/run",
            "prompt": "test",
        })
    call_kwargs = mock_client.images.generate.call_args[1]
    assert call_kwargs["model"] == "gpt-image-2"
    assert call_kwargs["response_format"] == "b64_json"


def test_generate_returns_ok_false_on_api_error(tmp_path):
    with patch("piccolo_sprite.tools.image_gen.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.images.generate.side_effect = Exception("rate limit")
        result = generate_sprite_strip.invoke({
            "action": "walk",
            "direction": "north",
            "cell": 64,
            "run_dir": str(tmp_path),
            "prompt": "test",
        })
    assert result["ok"] is False
    assert "rate limit" in result["error"]
