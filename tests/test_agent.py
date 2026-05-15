from unittest.mock import patch
from piccolo_sprite.agent import build_agent
from piccolo_sprite.tools import TOOLS


def test_agent_builds_without_error():
    with patch("piccolo_sprite.agent.ChatOpenAI"):
        agent = build_agent()
    assert agent is not None


def test_agent_has_all_tools():
    with patch("piccolo_sprite.agent.ChatOpenAI"):
        agent = build_agent()
    expected = {t.name for t in TOOLS}
    # CompiledStateGraph exposes bound tools via the tool node
    bound_tools = set(agent.nodes["tools"].bound.tools_by_name.keys())
    assert expected == bound_tools
