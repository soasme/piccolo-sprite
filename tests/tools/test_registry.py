from piccolo_sprite.tools import TOOLS


EXPECTED_TOOL_NAMES = {
    "generate_sprite_strip",
    "assemble_action_sheet",
    "clean_sheet",
    "validate_sheet",
    "audit_motion",
    "validate_hierarchy",
    "export_previews",
    "validate_manifest",
    "read_manifest",
    "write_manifest",
}


def test_tools_list_contains_all_expected_tools():
    names = {t.name for t in TOOLS}
    assert names == EXPECTED_TOOL_NAMES


def test_tools_list_has_no_duplicates():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))


def test_all_tools_are_callable():
    for tool in TOOLS:
        # Tools are StructuredTool instances with a func attribute
        assert hasattr(tool, 'func') and callable(tool.func)
