from pose_lift.prompt import OPTIMIZER_SYSTEM_PROMPT, enforce_prompt_hygiene


def test_hygiene_clause_is_appended():
    out = enforce_prompt_hygiene("Spider-Man on a rooftop at dusk")
    assert "Spider-Man on a rooftop at dusk" in out
    assert "watermark" in out.lower()
    assert "stock photo" in out.lower()


def test_hygiene_is_idempotent():
    once = enforce_prompt_hygiene("Hulk smashing a car")
    twice = enforce_prompt_hygiene(once)
    assert once == twice


def test_blank_prompt_is_rejected():
    try:
        enforce_prompt_hygiene("   ")
    except ValueError:
        return
    raise AssertionError("blank prompt should raise ValueError")


def test_system_prompt_names_the_two_validated_rules():
    lowered = OPTIMIZER_SYSTEM_PROMPT.lower()
    assert "watermark" in lowered
    assert "concrete" in lowered
