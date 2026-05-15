from piccolo_sprite.system_prompt import SYSTEM_PROMPT


def test_system_prompt_contains_pipeline_keywords():
    for keyword in [
        "generate_sprite_strip",
        "assemble_action_sheet",
        "clean_sheet",
        "validate_sheet",
        "audit_motion",
        "export_previews",
        "validate_manifest",
        "ok=false",
        "south",
        "contact",
    ]:
        assert keyword in SYSTEM_PROMPT, f"Missing keyword: {keyword}"


def test_system_prompt_is_nonempty_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 500
