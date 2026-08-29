import json
from pathlib import Path

import pytest

from menu_vlm.training import preflight_config


def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "qwen3-vl-4b-lora.json"


def test_no_download_preflight_pins_model_and_covers_target_families() -> None:
    report = preflight_config(config_path(), no_download=True)
    assert report["valid"] is True
    assert report["model_revision"] == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert report["image_token_safe"] is True
    assert report["vision_tower_frozen"] is True
    assert set(report["matched_families"]) == {"language", "projector"}


def test_preflight_rejects_image_truncation_and_missing_projector(tmp_path: Path) -> None:
    payload = json.loads(config_path().read_text(encoding="utf-8"))
    payload["max_length"] = 1024
    payload["target_modules"] = ["q_proj"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_length"):
        preflight_config(path, no_download=True)
