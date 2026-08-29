import json
from pathlib import Path

import pytest

from menu_vlm.compiler import CompileOptions, compile_release, validate_compiled_dataset
from menu_vlm.synthetic import create_synthetic_release


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_release_compiler_is_stable_and_preserves_grouped_lineage(tmp_path: Path) -> None:
    release = tmp_path / "release"
    create_synthetic_release(release)
    first = tmp_path / "first"
    second = tmp_path / "second"

    left = compile_release(CompileOptions(release=release, output=first))
    right = compile_release(CompileOptions(release=release, output=second))

    assert left["dataset_sha256"] == right["dataset_sha256"]
    assert left["files_sha256"] == right["files_sha256"]
    assert validate_compiled_dataset(first)["valid"] is True
    assert read_json(first / "manifest.json")["source_release_splits_preserved"] is True
    train = (first / "train.jsonl").read_text(encoding="utf-8")
    assert "menu_00001_aug_01" in train
    assert "Page index: 0" in train
    assert "<ocr_lines>\\n1. Grill\\n2. Steak $12" in train
    assert '"s":[{"h":{"l":[1],"t":"Grill"},"id":"s1"' in train


def test_compiler_routes_non_train_derivatives_to_robustness(tmp_path: Path) -> None:
    release = tmp_path / "release"
    create_synthetic_release(release)
    output = tmp_path / "compiled"
    manifest = compile_release(CompileOptions(release=release, output=output))

    assert manifest["counts"] == {
        "train": 4,
        "validation": 1,
        "test": 1,
        "robustness_validation": 1,
        "robustness_test": 1,
        "excluded": 1,
        "projection_failures": 0,
    }
    assert "aug" not in (output / "validation.jsonl").read_text(encoding="utf-8")
    assert "aug" in (output / "robustness_validation.jsonl").read_text(encoding="utf-8")


def test_compiler_rejects_checksum_drift_and_lineage_leakage(tmp_path: Path) -> None:
    drift = tmp_path / "drift"
    create_synthetic_release(drift)
    (drift / "records.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        compile_release(CompileOptions(release=drift, output=tmp_path / "out-drift"))

    leaked = tmp_path / "leaked"
    create_synthetic_release(leaked)
    lineage = read_json(leaked / "lineage.json")
    lineage[0]["split"] = "test"
    (leaked / "lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    _rehash_manifest(leaked)
    with pytest.raises(ValueError, match="crosses dataset splits"):
        compile_release(CompileOptions(release=leaked, output=tmp_path / "out-leaked"))


def test_release_validator_rejects_unhashed_extra_inventory(tmp_path: Path) -> None:
    release = tmp_path / "release"
    create_synthetic_release(release)
    (release / "unhashed-extra.txt").write_text("not in immutable inventory\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        compile_release(CompileOptions(release=release, output=tmp_path / "compiled"))


def test_compiled_validator_rejects_checksum_drift(tmp_path: Path) -> None:
    release = tmp_path / "release"
    create_synthetic_release(release)
    compiled = tmp_path / "compiled"
    compile_release(CompileOptions(release=release, output=compiled))
    with (compiled / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    with pytest.raises(ValueError, match="checksum drift"):
        validate_compiled_dataset(compiled)


def test_compiler_rejects_invalid_ocr_reference_without_silent_loss(tmp_path: Path) -> None:
    release = tmp_path / "release"
    create_synthetic_release(release)
    records = read_json(release / "records.json")
    records[0]["annotation"]["sections"][0]["items"][0]["name"]["source_ids"] = ["ocr_missing"]
    records[1]["annotation"] = records[0]["annotation"]
    (release / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rehash_manifest(release)

    manifest = compile_release(CompileOptions(release=release, output=tmp_path / "compiled"))

    assert manifest["counts"]["projection_failures"] == 2
    failures = read_json(tmp_path / "compiled" / "projection-failures.json")
    assert failures[0]["document_id"] == records[0]["document_id"]
    assert "unknown OCR source" in failures[0]["reason"]
    assert manifest["accounting"]["input_records"] == (
        manifest["accounting"]["compiled_records"] + manifest["counts"]["projection_failures"]
    )


def test_compiler_rejects_traversal_document_id_before_creating_output(tmp_path: Path) -> None:
    release = tmp_path / "release"
    create_synthetic_release(release)
    records = read_json(release / "records.json")
    records[0]["document_id"] = "../../escaped"
    (release / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rehash_manifest(release)
    output = tmp_path / "compiled"

    with pytest.raises(ValueError, match="document_id"):
        compile_release(CompileOptions(release=release, output=output))

    assert not output.exists()
    assert not (tmp_path / "escaped.svg").exists()


def test_unsplit_mode_is_deterministic_and_groups_derivatives(tmp_path: Path) -> None:
    release = tmp_path / "unsplit"
    create_synthetic_release(release, include_splits=False)
    first = compile_release(
        CompileOptions(
            release=release,
            output=tmp_path / "out1",
            allow_unsplit=True,
            split_seed=99,
        )
    )
    second = compile_release(
        CompileOptions(
            release=release,
            output=tmp_path / "out2",
            allow_unsplit=True,
            split_seed=99,
        )
    )
    assert first["split_assignments"] == second["split_assignments"]
    assignments = first["split_assignments"]
    assert assignments["menu_00001"] == assignments["menu_00001_aug_01"]
    assert first["source_release_splits_preserved"] is False


def _rehash_manifest(root: Path) -> None:
    import hashlib

    manifest = read_json(root / "manifest.json")
    manifest["files_sha256"] = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in manifest["files_sha256"]
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
