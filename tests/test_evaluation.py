import json
from pathlib import Path

import pytest

from menu_vlm.evaluation import evaluate_files, evaluate_test_once, select_checkpoint

REFERENCE = {
    "s": [
        {"id": "s1", "h": {"t": "Grill", "l": [1]}, "notes": []},
        {"id": "s2", "h": {"t": "Sides", "l": [5]}, "notes": []},
    ],
    "i": [
        {"l": [2, 3], "n": "Steak", "a": 2, "c": 1.0, "s": "s1", "notes": []},
        {"l": [6], "n": "Fries", "a": 6, "c": 1.0, "s": "s2", "notes": []},
    ],
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluation_is_order_and_section_id_insensitive(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    reordered = {
        "s": [
            {"id": "other2", "h": {"t": "Sides", "l": [5]}, "notes": []},
            {"id": "other1", "h": {"t": "Grill", "l": [1]}, "notes": []},
        ],
        "i": [
            {"l": [6], "n": "Fries", "a": 6, "c": 0.7, "s": "other2", "notes": []},
            {"l": [3, 2], "n": "STEAK", "a": 2, "c": 0.8, "s": "other1", "notes": []},
        ],
    }
    write_jsonl(
        references,
        [{"example_id": "a", "ocr_line_count": 6, "target": REFERENCE, "kind": "primary"}],
    )
    write_jsonl(
        predictions,
        [
            {
                "example_id": "a",
                "prediction": reordered,
                "latency_seconds": 0.5,
                "tokens_per_second": 20,
                "peak_memory_bytes": 100,
            }
        ],
    )

    metrics = evaluate_files(references, predictions)

    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["item_f1"] == 1.0
    assert metrics["section_accuracy"] == 1.0
    assert metrics["dish_name_accuracy"] == 1.0
    assert metrics["structural_f1"] == 1.0
    assert metrics["merge_errors"] == 0
    assert metrics["split_errors"] == 0
    assert metrics["performance"]["latency_seconds_mean"] == 0.5


def test_evaluation_reports_invalid_schema_and_ocr_reference(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        references,
        [{"example_id": "a", "ocr_line_count": 6, "target": REFERENCE, "kind": "primary"}],
    )
    write_jsonl(predictions, [{"example_id": "a", "prediction": {"s": [], "i": [{"l": [9]}]}}])
    metrics = evaluate_files(references, predictions)
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["ocr_reference_valid_rate"] == 0.0


def test_evaluation_requires_exact_prediction_accounting(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        references, [{"example_id": "a", "ocr_line_count": 1, "target": {"s": [], "i": []}}]
    )
    predictions.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Prediction IDs"):
        evaluate_files(references, predictions)


def test_checkpoint_selection_uses_declared_lexicographic_policy(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    write_jsonl(
        metrics,
        [
            {"checkpoint": "checkpoint-1", "structural_f1": 0.8, "item_f1": 0.9, "eval_loss": 0.2},
            {"checkpoint": "checkpoint-2", "structural_f1": 0.9, "item_f1": 0.7, "eval_loss": 0.1},
            {"checkpoint": "checkpoint-3", "structural_f1": 0.9, "item_f1": 0.8, "eval_loss": 0.3},
        ],
    )
    selection = select_checkpoint(metrics)
    assert selection["selected_checkpoint"] == "checkpoint-3"
    assert selection["policy"] == ["structural_f1:max", "item_f1:max", "eval_loss:min"]


def test_printed_notes_are_order_insensitive_but_text_and_ownership_affect_structure(
    tmp_path: Path,
) -> None:
    reference = json.loads(json.dumps(REFERENCE))
    reference["s"][0]["notes"] = [
        {"t": "Dinner only", "l": [4]},
        {"t": "Ask about allergens", "l": [7]},
    ]
    reference["i"][0]["notes"] = [{"t": "Best served rare", "l": [3]}]
    reference["i"][1]["notes"] = [{"t": "Gluten free", "l": [8]}]
    references = tmp_path / "references.jsonl"
    write_jsonl(
        references,
        [{"example_id": "a", "ocr_line_count": 8, "target": reference, "kind": "primary"}],
    )

    reordered = json.loads(json.dumps(reference))
    reordered["s"].reverse()
    reordered["s"][1]["notes"].reverse()
    reordered["i"].reverse()
    predictions = tmp_path / "reordered.jsonl"
    write_jsonl(predictions, [{"example_id": "a", "prediction": reordered}])
    exact = evaluate_files(references, predictions)
    assert exact["note_f1"] == 1.0
    assert exact["note_text_accuracy"] == 1.0
    assert exact["note_ownership_accuracy"] == 1.0
    assert exact["structural_f1"] == 1.0

    wrong = json.loads(json.dumps(reference))
    wrong["s"][0]["notes"][0]["t"] = "Lunch only"
    moved = wrong["i"][0]["notes"].pop()
    wrong["i"][1]["notes"].append(moved)
    wrong_predictions = tmp_path / "wrong.jsonl"
    write_jsonl(wrong_predictions, [{"example_id": "a", "prediction": wrong}])
    degraded = evaluate_files(references, wrong_predictions)
    assert degraded["note_f1"] == 1.0
    assert degraded["note_text_accuracy"] < 1.0
    assert degraded["note_ownership_accuracy"] < 1.0
    assert degraded["structural_f1"] < 1.0

    missing = json.loads(json.dumps(reference))
    missing["s"][0]["notes"].pop()
    missing_predictions = tmp_path / "missing.jsonl"
    write_jsonl(missing_predictions, [{"example_id": "a", "prediction": missing}])
    missing_metrics = evaluate_files(references, missing_predictions)
    assert missing_metrics["note_recall"] < 1.0
    assert missing_metrics["structural_f1"] < 1.0


def test_frozen_test_gate_is_bound_to_dataset_model_and_adapter_not_output_path(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        references,
        [{"example_id": "a", "ocr_line_count": 6, "target": REFERENCE, "kind": "primary"}],
    )
    write_jsonl(predictions, [{"example_id": "a", "prediction": REFERENCE}])
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter A")
    gate_store = tmp_path / "fixed-test-gates"
    luna_predictions = tmp_path / "luna-predictions.jsonl"
    write_jsonl(luna_predictions, [{"example_id": "a", "prediction": REFERENCE}])
    dataset_manifest = tmp_path / "manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_sha256": "a" * 64}), encoding="utf-8")

    evaluate_test_once(
        references,
        predictions,
        tmp_path / "first-metrics.json",
        dataset_sha256="a" * 64,
        dataset_manifest=dataset_manifest,
        checkpoint=adapter,
        gate_store=gate_store,
        luna_predictions=luna_predictions,
    )

    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_test_once(
            references,
            predictions,
            tmp_path / "different-output.json",
            dataset_sha256="a" * 64,
            dataset_manifest=dataset_manifest,
            checkpoint=adapter,
            gate_store=gate_store,
            luna_predictions=luna_predictions,
        )

    other_adapter = tmp_path / "other-adapter"
    other_adapter.mkdir()
    (other_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (other_adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter B")
    evaluate_test_once(
        references,
        predictions,
        tmp_path / "other-checkpoint-metrics.json",
        dataset_sha256="a" * 64,
        dataset_manifest=dataset_manifest,
        checkpoint=other_adapter,
        gate_store=gate_store,
        luna_predictions=luna_predictions,
    )
    assert len(list(gate_store.glob("*.json"))) == 2
    gates = [json.loads(path.read_text(encoding="utf-8")) for path in gate_store.glob("*.json")]
    assert all(gate["luna_prediction_sha256"] for gate in gates)
    assert all(gate["identity"]["dataset_manifest_sha256"] for gate in gates)

    dataset_manifest.write_text(json.dumps({"dataset_sha256": "b" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest does not match"):
        evaluate_test_once(
            references,
            predictions,
            tmp_path / "mismatched-manifest-metrics.json",
            dataset_sha256="a" * 64,
            dataset_manifest=dataset_manifest,
            checkpoint=adapter,
            gate_store=gate_store,
            luna_predictions=luna_predictions,
        )
