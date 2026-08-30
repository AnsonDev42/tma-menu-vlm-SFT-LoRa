import json
from pathlib import Path

import pytest

from menu_vlm.compiler import CompileOptions, compile_release
from menu_vlm.jsonio import canonical_json, read_jsonl, sha256_file, verify_sha256_sidecar
from menu_vlm.luna_baseline import import_luna_baseline
from menu_vlm.synthetic import create_synthetic_release


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, str, Path, str]:
    release = tmp_path / "release"
    dataset = tmp_path / "dataset"
    create_synthetic_release(release)
    compile_release(CompileOptions(release=release, output=dataset))
    compiled = read_jsonl(dataset / "test.jsonl")[0]
    image_sha256 = sha256_file(dataset / compiled["image"])

    root = tmp_path / "tma-data"
    run_id = "synthetic-luna-run"
    document_id = "different-tma-document-id"
    evaluation = root / "evaluation" / run_id
    cache_key = "a" * 64
    call_path = f"baselines/tma-core/{cache_key}/call.json"
    report = {
        "run_id": run_id,
        "status": "complete",
        "model": "current-tma-core:gpt-5.6-luna",
        "document_count": 1,
        "completed_document_count": 1,
        "baseline": {"contract": {"model": "gpt-5.6-luna"}},
        "documents": [{"document_id": document_id}],
    }
    run = {
        "run_id": run_id,
        "status": "complete",
        "model": "current-tma-core:gpt-5.6-luna",
        "document_count": 1,
    }
    reference = {"document_id": document_id, "image": {"sha256": image_sha256}}
    prediction = {
        "_evaluation": {
            "state": "succeeded",
            "cache_key": cache_key,
            "call_path": call_path,
        }
    }
    _write_json(evaluation / "report.json", report)
    _write_json(evaluation / "run.json", run)
    _write_json(evaluation / "references" / f"{document_id}.json", reference)
    prediction_path = evaluation / "predictions" / f"{document_id}.json"
    _write_json(prediction_path, prediction)
    _write_json(
        root / call_path,
        {"state": "succeeded", "inputs": {"image_sha256": image_sha256}},
    )
    _write_json(
        root / "baselines" / "tma-core" / cache_key / "provider.raw.json",
        {
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": canonical_json(compiled["target"]),
                    }
                }
            ],
        },
    )
    return dataset, root, run_id, prediction_path, document_id


def test_import_luna_baseline_matches_images_and_is_byte_deterministic(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction, _document_id = _fixture(tmp_path)
    output = tmp_path / "luna.jsonl"

    report = import_luna_baseline(dataset, root, run_id, output)
    first = output.read_bytes()
    first_sidecar = output.with_suffix(".jsonl.sha256").read_bytes()
    row = read_jsonl(output)[0]
    reference = read_jsonl(dataset / "test.jsonl")[0]

    assert report["examples"] == 1
    assert row["example_id"] == reference["example_id"]
    assert row["prediction"] == reference["target"]
    assert row["raw_output"] == canonical_json(reference["target"])
    assert verify_sha256_sidecar(output, output.with_suffix(".jsonl.sha256"))["valid"]

    output.unlink()
    output.with_suffix(".jsonl.sha256").unlink()
    import_luna_baseline(dataset, root, run_id, output)
    assert output.read_bytes() == first
    assert output.with_suffix(".jsonl.sha256").read_bytes() == first_sidecar


@pytest.mark.parametrize(
    "call_path",
    [
        "/tmp/call.json",
        "../baselines/tma-core/" + "a" * 64 + "/call.json",
        "baselines/tma-core/" + "b" * 64 + "/call.json",
        "baselines/tma-core/" + "a" * 64 + "/provider.raw.json",
    ],
)
def test_import_rejects_unsafe_or_mismatched_cache_paths(
    tmp_path: Path, call_path: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["_evaluation"]["call_path"] = call_path
    _write_json(prediction_path, prediction)

    with pytest.raises(ValueError, match="cache path"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_cache_symlink_that_escapes_explicit_root(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    cache = root / "baselines" / "tma-core" / key
    outside = tmp_path / "outside"
    outside.mkdir()
    for path in cache.iterdir():
        path.unlink()
    cache.rmdir()
    cache.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes release root"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_image_identity_mismatch(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction, document_id = _fixture(tmp_path)
    reference = root / "evaluation" / run_id / "references" / f"{document_id}.json"
    value = json.loads(reference.read_text(encoding="utf-8"))
    value["image"]["sha256"] = "b" * 64
    _write_json(reference, value)

    with pytest.raises(ValueError, match="do not exactly match"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "not compact JSON"),
        ('{"s":[],"i":[{"l":[999]}]}', "validation error"),
        ('{"s":[],"i":[],"extra":true}', "validation error"),
    ],
)
def test_import_rejects_invalid_compact_provider_output(
    tmp_path: Path, content: str, message: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["choices"][0]["message"]["content"] = content
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match=message):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_missing_extra_and_duplicate_identities(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction, document_id = _fixture(tmp_path)
    predictions = root / "evaluation" / run_id / "predictions"
    source = predictions / f"{document_id}.json"
    _write_json(predictions / "unexpected.json", json.loads(source.read_text(encoding="utf-8")))
    with pytest.raises(ValueError, match="identities do not exactly match"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "extra.jsonl")

    (predictions / "unexpected.json").unlink()
    source.unlink()
    with pytest.raises(ValueError, match="identities do not exactly match"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "missing.jsonl")

    _dataset, root, run_id, _prediction, document_id = _fixture(tmp_path / "duplicate")
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"].append({"document_id": document_id})
    report["document_count"] = report["completed_document_count"] = 2
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="duplicate document identities"):
        import_luna_baseline(_dataset, root, run_id, tmp_path / "duplicate.jsonl")


def test_sidecar_rejects_checksum_drift_and_wrong_file_identity(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction, _document_id = _fixture(tmp_path)
    output = tmp_path / "luna.jsonl"
    import_luna_baseline(dataset, root, run_id, output)
    sidecar = output.with_suffix(".jsonl.sha256")

    output.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum does not match"):
        verify_sha256_sidecar(output, sidecar)

    sidecar.write_text(f"{'a' * 64}  other.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file identity"):
        verify_sha256_sidecar(output, sidecar)
