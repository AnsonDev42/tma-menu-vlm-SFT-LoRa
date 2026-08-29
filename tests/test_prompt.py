import hashlib

from menu_vlm.prompt import system_prompt


def test_system_prompt_preserves_authoritative_tma_bytes() -> None:
    prompt = system_prompt()

    assert len(prompt.encode("utf-8")) == 4284
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "c4c4466bbdf49eb066bab6486bd9c9a0bf9230aeafb2da60b0ab02cd617fa476"
    )
    assert prompt.endswith("\n")
    assert not prompt.endswith("\n\n")
