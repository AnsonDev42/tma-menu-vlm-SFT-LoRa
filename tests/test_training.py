import json
from pathlib import Path

import pytest

from menu_vlm.compiler import CompileOptions, compile_release
from menu_vlm.synthetic import create_synthetic_release
from menu_vlm.training import (
    preflight_config,
    trainer_rows,
    validate_peft_target_modules,
)


def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "qwen3-vl-4b-lora.json"


def test_no_download_preflight_pins_model_and_covers_target_families() -> None:
    report = preflight_config(config_path(), no_download=True)
    assert report["valid"] is True
    assert report["model_revision"] == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert report["image_token_safe"] is True
    assert report["vision_tower_frozen"] is True
    assert set(report["matched_families"]) == {"language", "projector"}
    assert report["dataset_format"] == "conversational_prompt_completion"
    assert report["loss_scope"] == "completion_only"
    assert report["sft_config"]["completion_only_loss"] is True
    assert report["sft_config"]["assistant_only_loss"] is False


def test_preflight_rejects_image_truncation_and_missing_projector(tmp_path: Path) -> None:
    payload = json.loads(config_path().read_text(encoding="utf-8"))
    payload["max_length"] = 1024
    path = tmp_path / "truncated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_length"):
        preflight_config(path, no_download=True)

    payload = json.loads(config_path().read_text(encoding="utf-8"))
    payload["target_module_selectors"]["projector"]["exact"] = []
    path = tmp_path / "missing-projector.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Projector targets"):
        preflight_config(path, no_download=True)


def test_trainer_rows_use_supported_vlm_prompt_completion_shape(tmp_path: Path) -> None:
    release = tmp_path / "release"
    dataset = tmp_path / "dataset"
    create_synthetic_release(release)
    compile_release(CompileOptions(release=release, output=dataset))

    row = trainer_rows(dataset, "train.jsonl")[0]

    assert set(row) == {"image", "prompt", "completion"}
    assert [message["role"] for message in row["prompt"]] == ["system", "user"]
    assert [message["role"] for message in row["completion"]] == ["assistant"]
    assert "messages" not in row


def test_target_resolution_excludes_vision_suffix_collisions_and_peft_is_exact() -> None:
    modules = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.self_attn.q_proj",
    ]

    report = preflight_config(config_path(), no_download=True, module_names=modules)

    assert report["matched_module_count"] == 9
    assert "model.visual.blocks.0.mlp.linear_fc1" not in report["matched_modules"]
    assert "model.visual.blocks.0.self_attn.q_proj" not in report["matched_modules"]
    injected = [f"base_model.model.{name}" for name in report["matched_modules"]]
    validate_peft_target_modules(report["matched_modules"], injected)
    with pytest.raises(ValueError, match="unexpected"):
        validate_peft_target_modules(
            report["matched_modules"],
            [*injected, "base_model.model.model.visual.blocks.0.mlp.linear_fc1"],
        )
