import json
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .constants import MODEL_ID, MODEL_REVISION
from .jsonio import canonical_json, read_jsonl, sha256_file, sha256_json, write_json
from .schemas import CompactItem, CompactOutput, CompactSection, validate_compact_output


@dataclass
class _Totals:
    documents: int = 0
    parseable: int = 0
    schema_valid: int = 0
    ocr_valid: int = 0
    reference_items: int = 0
    predicted_items: int = 0
    matched_items: int = 0
    reference_sections: int = 0
    predicted_sections: int = 0
    matched_sections: int = 0
    matching_names: int = 0
    matching_sections: int = 0
    merge_errors: int = 0
    split_errors: int = 0
    reference_ocr_lines: int = 0
    predicted_ocr_lines: int = 0
    matching_ocr_lines: int = 0
    reference_notes: int = 0
    predicted_notes: int = 0
    matched_notes: int = 0
    matching_note_text: int = 0
    matching_note_ownership: int = 0


@dataclass(frozen=True)
class _NoteEntity:
    text: str
    line_indices: tuple[int, ...]
    owner_type: str
    owner_index: int


def evaluate_files(
    references_path: Path,
    predictions_path: Path,
    *,
    luna_predictions_path: Path | None = None,
) -> dict[str, Any]:
    references = read_jsonl(references_path)
    predictions = read_jsonl(predictions_path)
    reference_by_id = _index(references, "reference")
    prediction_by_id = _index(predictions, "prediction")
    if set(reference_by_id) != set(prediction_by_id):
        missing = sorted(set(reference_by_id) - set(prediction_by_id))
        extra = sorted(set(prediction_by_id) - set(reference_by_id))
        raise ValueError(
            f"Prediction IDs do not exactly match references; missing={missing}, extra={extra}"
        )

    totals = _Totals()
    documents: list[dict[str, Any]] = []
    latencies: list[float] = []
    throughputs: list[float] = []
    peak_memories: list[int] = []
    kind_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for example_id in sorted(reference_by_id):
        reference_row = reference_by_id[example_id]
        prediction_row = prediction_by_id[example_id]
        audit = _evaluate_one(reference_row, prediction_row, totals)
        documents.append({"example_id": example_id, **audit})
        latency = prediction_row.get("latency_seconds")
        throughput = prediction_row.get("tokens_per_second")
        peak_memory = prediction_row.get("peak_memory_bytes")
        if isinstance(latency, int | float) and latency >= 0:
            latencies.append(float(latency))
        if isinstance(throughput, int | float) and throughput >= 0:
            throughputs.append(float(throughput))
        if isinstance(peak_memory, int) and peak_memory >= 0:
            peak_memories.append(peak_memory)
        kind = str(reference_row.get("kind", "primary"))
        pair = kind_rows.setdefault(kind, ([], []))
        pair[0].append(reference_row)
        pair[1].append(prediction_row)

    result = _summarize(totals)
    result["performance"] = {
        "latency_seconds_mean": _mean(latencies),
        "tokens_per_second_mean": _mean(throughputs),
        "peak_memory_bytes_max": max(peak_memories, default=None),
    }
    result["documents"] = documents
    if len(kind_rows) > 1:
        result["by_kind"] = {
            kind: _evaluate_rows(reference_rows, prediction_rows)
            for kind, (reference_rows, prediction_rows) in sorted(kind_rows.items())
        }
    if luna_predictions_path is not None:
        result["luna_baseline"] = evaluate_files(references_path, luna_predictions_path)
    return result


def select_checkpoint(metrics_path: Path) -> dict[str, Any]:
    rows = read_jsonl(metrics_path)
    if not rows:
        raise ValueError("Checkpoint metrics are empty")
    required = {"checkpoint", "structural_f1", "item_f1", "eval_loss"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"Checkpoint metrics missing keys: {sorted(required - set(row))}")
        if any(not isinstance(row[key], int | float) for key in required - {"checkpoint"}):
            raise ValueError("Checkpoint selection metrics must be numeric")
    selected = max(
        rows,
        key=lambda row: (
            float(row["structural_f1"]),
            float(row["item_f1"]),
            -float(row["eval_loss"]),
            str(row["checkpoint"]),
        ),
    )
    return {
        "schema_version": "1.0",
        "policy": ["structural_f1:max", "item_f1:max", "eval_loss:min"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_metrics": selected,
        "candidate_count": len(rows),
    }


def evaluate_test_once(
    references: Path,
    predictions: Path,
    output: Path,
    *,
    dataset_sha256: str,
    checkpoint: Path,
    gate_store: Path,
    luna_predictions: Path | None = None,
) -> dict[str, Any]:
    if len(dataset_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_sha256
    ):
        raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
    checkpoint_sha256 = _directory_sha256(checkpoint)
    identity = {
        "dataset_sha256": dataset_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
    }
    identity_sha256 = sha256_json(identity)
    gates = gate_store.resolve()
    gates.mkdir(parents=True, exist_ok=True)
    gate = gates / f"{identity_sha256}.json"
    if output.exists():
        raise FileExistsError(f"Frozen test output already exists: {output}")
    metrics = evaluate_files(references, predictions, luna_predictions_path=luna_predictions)
    gate_payload = {
        "schema_version": "1.0",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "reference_sha256": sha256_file(references),
        "prediction_sha256": sha256_file(predictions),
        "luna_prediction_sha256": (
            sha256_file(luna_predictions) if luna_predictions is not None else None
        ),
        "status": "reserved",
    }
    try:
        with gate.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(gate_payload) + "\n")
    except FileExistsError as exc:
        raise FileExistsError(
            "Frozen test identity was already consumed for this dataset, model, and adapter"
        ) from exc
    write_json(output, metrics)
    gate_payload["status"] = "completed"
    gate_payload["metrics_sha256"] = sha256_file(output)
    write_json(gate, gate_payload)
    metrics["frozen_test_gate"] = {
        "identity_sha256": identity_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "path": str(gate),
    }
    return metrics


def _directory_sha256(path: Path) -> str:
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"Checkpoint is not a directory: {root}")
    required = ("adapter_config.json", "adapter_model.safetensors")
    hashes = {}
    for name in required:
        candidate = root / name
        if not candidate.is_file():
            raise ValueError(f"Checkpoint is missing identity file: {name}")
        hashes[name] = sha256_file(candidate)
    return sha256_json(hashes)


def _evaluate_rows(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    totals = _Totals()
    for reference, prediction in zip(references, predictions, strict=True):
        _evaluate_one(reference, prediction, totals)
    return _summarize(totals)


def _evaluate_one(
    reference_row: dict[str, Any], prediction_row: dict[str, Any], totals: _Totals
) -> dict[str, Any]:
    count = int(reference_row.get("ocr_line_count", 0))
    reference = validate_compact_output(reference_row.get("target"), ocr_line_count=count)
    totals.documents += 1
    totals.reference_items += len(reference.items)
    totals.reference_sections += len(reference.sections)
    totals.reference_ocr_lines += len(_claimed_lines(reference))
    reference_notes = _notes(reference)
    totals.reference_notes += len(reference_notes)
    raw = prediction_row.get("prediction")
    parseable = True
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            parseable = False
    elif not isinstance(raw, dict):
        parseable = False
    if parseable:
        totals.parseable += 1
    prediction: CompactOutput | None = None
    schema_valid = False
    ocr_valid = False
    if parseable:
        try:
            prediction = CompactOutput.model_validate(raw)
            schema_valid = True
            totals.schema_valid += 1
            validate_compact_output(raw, ocr_line_count=count)
            ocr_valid = True
            totals.ocr_valid += 1
        except (ValidationError, ValueError):
            pass
    if prediction is None:
        return {
            "parseable": parseable,
            "schema_valid": schema_valid,
            "ocr_references_valid": ocr_valid,
            "matched_items": 0,
            "matched_sections": 0,
        }

    totals.predicted_items += len(prediction.items)
    totals.predicted_sections += len(prediction.sections)
    reference_lines = _claimed_lines(reference)
    predicted_lines = _claimed_lines(prediction)
    totals.predicted_ocr_lines += len(predicted_lines)
    totals.matching_ocr_lines += len(reference_lines & predicted_lines)
    section_pairs = _match_sections(reference.sections, prediction.sections)
    item_pairs = _match_items(reference.items, prediction.items)
    totals.matched_sections += len(section_pairs)
    totals.matched_items += len(item_pairs)
    section_map = {
        prediction_index: reference_index for reference_index, prediction_index in section_pairs
    }
    reference_section_by_id = {
        section.id: index for index, section in enumerate(reference.sections)
    }
    prediction_section_by_id = {
        section.id: index for index, section in enumerate(prediction.sections)
    }
    matching_names = 0
    matching_sections = 0
    for reference_index, prediction_index in item_pairs:
        expected = reference.items[reference_index]
        actual = prediction.items[prediction_index]
        matching_names += _normalize(expected.name) == _normalize(actual.name)
        if expected.section_ref is None and actual.section_ref is None:
            matching_sections += 1
        elif expected.section_ref is not None and actual.section_ref is not None:
            expected_section = reference_section_by_id.get(expected.section_ref)
            actual_section = prediction_section_by_id.get(actual.section_ref)
            matching_sections += (
                expected_section is not None
                and actual_section is not None
                and section_map.get(actual_section) == expected_section
            )
    totals.matching_names += matching_names
    totals.matching_sections += matching_sections
    predicted_notes = _notes(prediction)
    note_pairs = _match_notes(reference_notes, predicted_notes)
    item_map = {
        prediction_index: reference_index for reference_index, prediction_index in item_pairs
    }
    matching_note_text = 0
    matching_note_ownership = 0
    for reference_index, prediction_index in note_pairs:
        expected_note = reference_notes[reference_index]
        actual_note = predicted_notes[prediction_index]
        matching_note_text += _normalize(expected_note.text) == _normalize(actual_note.text)
        owner_map = section_map if actual_note.owner_type == "section" else item_map
        matching_note_ownership += (
            expected_note.owner_type == actual_note.owner_type
            and owner_map.get(actual_note.owner_index) == expected_note.owner_index
        )
    totals.predicted_notes += len(predicted_notes)
    totals.matched_notes += len(note_pairs)
    totals.matching_note_text += matching_note_text
    totals.matching_note_ownership += matching_note_ownership
    merge_errors, split_errors = _structural_errors(reference.items, prediction.items)
    totals.merge_errors += merge_errors
    totals.split_errors += split_errors
    return {
        "parseable": parseable,
        "schema_valid": schema_valid,
        "ocr_references_valid": ocr_valid,
        "matched_items": len(item_pairs),
        "matched_sections": len(section_pairs),
        "matching_names": matching_names,
        "matching_section_assignments": matching_sections,
        "matched_notes": len(note_pairs),
        "matching_note_text": matching_note_text,
        "matching_note_ownership": matching_note_ownership,
        "merge_errors": merge_errors,
        "split_errors": split_errors,
    }


def _summarize(totals: _Totals) -> dict[str, Any]:
    item_precision, item_recall, item_f1 = _prf(
        totals.matched_items, totals.predicted_items, totals.reference_items
    )
    section_precision, section_recall, section_f1 = _prf(
        totals.matched_sections, totals.predicted_sections, totals.reference_sections
    )
    ocr_precision, ocr_recall, ocr_f1 = _prf(
        totals.matching_ocr_lines, totals.predicted_ocr_lines, totals.reference_ocr_lines
    )
    section_accuracy = _ratio(totals.matching_sections, totals.matched_items)
    name_accuracy = _ratio(totals.matching_names, totals.matched_items)
    note_precision, note_recall, note_f1 = _prf(
        totals.matched_notes, totals.predicted_notes, totals.reference_notes
    )
    note_text_accuracy = _ratio(totals.matching_note_text, totals.matched_notes)
    note_ownership_accuracy = _ratio(totals.matching_note_ownership, totals.matched_notes)
    structural_f1 = statistics.fmean(
        (
            item_f1,
            section_f1,
            section_accuracy,
            note_f1,
            note_text_accuracy,
            note_ownership_accuracy,
        )
    )
    return {
        "document_count": totals.documents,
        "parseable_rate": _ratio(totals.parseable, totals.documents),
        "schema_valid_rate": _ratio(totals.schema_valid, totals.documents),
        "ocr_reference_valid_rate": _ratio(totals.ocr_valid, totals.documents),
        "ocr_line_precision": ocr_precision,
        "ocr_line_recall": ocr_recall,
        "ocr_line_f1": ocr_f1,
        "item_precision": item_precision,
        "item_recall": item_recall,
        "item_f1": item_f1,
        "section_precision": section_precision,
        "section_recall": section_recall,
        "section_f1": section_f1,
        "section_accuracy": section_accuracy,
        "dish_name_accuracy": name_accuracy,
        "note_precision": note_precision,
        "note_recall": note_recall,
        "note_f1": note_f1,
        "note_text_accuracy": note_text_accuracy,
        "note_ownership_accuracy": note_ownership_accuracy,
        "structural_f1": structural_f1,
        "merge_errors": totals.merge_errors,
        "split_errors": totals.split_errors,
    }


def _match_sections(
    references: list[CompactSection], predictions: list[CompactSection]
) -> list[tuple[int, int]]:
    candidates = []
    for left, reference in enumerate(references):
        for right, prediction in enumerate(predictions):
            name = _normalize(reference.heading.text) == _normalize(prediction.heading.text)
            overlap = _jaccard(
                set(reference.heading.line_indices), set(prediction.heading.line_indices)
            )
            score = (0.7 if name else 0.0) + 0.3 * overlap
            if name or overlap > 0:
                candidates.append((score, left, right))
    return _greedy_pairs(candidates)


def _match_items(
    references: list[CompactItem], predictions: list[CompactItem]
) -> list[tuple[int, int]]:
    candidates = []
    for left, reference in enumerate(references):
        for right, prediction in enumerate(predictions):
            name = _normalize(reference.name) == _normalize(prediction.name)
            overlap = _jaccard(set(reference.line_indices), set(prediction.line_indices))
            score = 0.65 * overlap + (0.35 if name else 0.0)
            if name or overlap > 0:
                candidates.append((score, left, right))
    return _greedy_pairs(candidates)


def _notes(output: CompactOutput) -> list[_NoteEntity]:
    section_notes = [
        _NoteEntity(
            text=note.text,
            line_indices=tuple(note.line_indices),
            owner_type="section",
            owner_index=section_index,
        )
        for section_index, section in enumerate(output.sections)
        for note in section.notes
    ]
    item_notes = [
        _NoteEntity(
            text=note.text,
            line_indices=tuple(note.line_indices),
            owner_type="item",
            owner_index=item_index,
        )
        for item_index, item in enumerate(output.items)
        for note in item.notes
    ]
    return section_notes + item_notes


def _match_notes(
    references: list[_NoteEntity], predictions: list[_NoteEntity]
) -> list[tuple[int, int]]:
    candidates = []
    for left, reference in enumerate(references):
        for right, prediction in enumerate(predictions):
            text_matches = _normalize(reference.text) == _normalize(prediction.text)
            overlap = _jaccard(set(reference.line_indices), set(prediction.line_indices))
            score = 0.6 * overlap + (0.4 if text_matches else 0.0)
            if text_matches or overlap > 0:
                candidates.append((score, left, right))
    return _greedy_pairs(candidates)


def _greedy_pairs(candidates: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    left_used: set[int] = set()
    right_used: set[int] = set()
    result: list[tuple[int, int]] = []
    for _, left, right in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
        if left not in left_used and right not in right_used:
            left_used.add(left)
            right_used.add(right)
            result.append((left, right))
    return result


def _structural_errors(
    references: list[CompactItem], predictions: list[CompactItem]
) -> tuple[int, int]:
    merge_errors = sum(
        sum(
            bool(set(prediction.line_indices) & set(reference.line_indices))
            for reference in references
        )
        > 1
        for prediction in predictions
    )
    split_errors = sum(
        sum(
            bool(set(reference.line_indices) & set(prediction.line_indices))
            for prediction in predictions
        )
        > 1
        for reference in references
    )
    return merge_errors, split_errors


def _claimed_lines(output: CompactOutput) -> set[int]:
    return (
        {
            line
            for section in output.sections
            for text in [section.heading, *section.notes]
            for line in text.line_indices
        }
        | {line for item in output.items for line in item.line_indices}
        | {line for item in output.items for note in item.notes for line in note.line_indices}
    )


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in result:
            raise ValueError(f"{label.title()} IDs are missing or duplicated")
        result[example_id] = row
    return result


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _prf(matches: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = _ratio(matches, predicted)
    recall = _ratio(matches, reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else (1.0 if numerator == 0 else 0.0)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
