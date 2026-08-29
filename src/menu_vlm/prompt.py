import hashlib

from .constants import PROMPT_PATH, PROMPT_VERSION


def system_prompt() -> str:
    return PROMPT_PATH.read_bytes().decode("utf-8")


def prompt_sha256() -> str:
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def render_user_prompt(*, page_index: int, ocr_lines: list[str]) -> str:
    numbered = "\n".join(f"{index}. {line}" for index, line in enumerate(ocr_lines, 1))
    return (
        f"Page index: {page_index}\n\n"
        "<ocr_lines>\n"
        f"{numbered or '(no OCR lines)'}\n"
        "</ocr_lines>\n\n"
        "Use the image to recover page layout and column order. "
        "Use OCR line numbers to prove coverage. Return original "
        "menu items, printed sections and their notes. Include "
        "shared sides/sauces and supplier provenance under their "
        "section, plus dish-specific serving instructions."
    )


def prompt_metadata() -> dict[str, str]:
    return {"version": PROMPT_VERSION, "sha256": prompt_sha256()}
