import json
from pathlib import Path

import pytest

from menu_vlm.evaluation import evaluate_files, select_checkpoint

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
