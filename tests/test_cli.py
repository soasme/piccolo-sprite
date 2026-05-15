import argparse
from unittest.mock import patch, AsyncMock
import pytest
from piccolo_sprite.cli import parse_args, TRUNCATE_LINES, truncate


def test_parse_args_prompt_flag():
    args = parse_args(["-p", "make a knight"])
    assert args.prompt == "make a knight"


def test_parse_args_long_prompt_flag():
    args = parse_args(["--prompt", "make a knight"])
    assert args.prompt == "make a knight"


def test_parse_args_no_flags_gives_none_prompt():
    args = parse_args([])
    assert args.prompt is None


def test_truncate_short_output_unchanged():
    text = "line1\nline2\nline3"
    assert truncate(text) == text.strip()


def test_truncate_long_output_adds_ellipsis():
    lines = [f"line{i}" for i in range(TRUNCATE_LINES + 5)]
    result = truncate("\n".join(lines))
    assert "… +" in result
    assert "lines (ctrl+o to expand)" in result
