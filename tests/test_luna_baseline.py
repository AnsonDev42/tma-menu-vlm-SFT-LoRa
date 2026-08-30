import json
import shutil
from pathlib import Path

import pytest

from menu_vlm.compiler import CompileOptions, compile_release
from menu_vlm.jsonio import canonical_json, read_jsonl, verify_sha256_sidecar
from menu_vlm.luna_baseline import import_luna_baseline
from menu_vlm.synthetic import create_synthetic_luna_evaluation, create_synthetic_release


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, str, Path, str]:
    release = tmp_path / "release"
    dataset = tmp_path / "dataset"
    create_synthetic_release(release)
    compile_release(CompileOptions(release=release, output=dataset))
    root = tmp_path / "tma-data"
    summary = create_synthetic_luna_evaluation(dataset, root)
    run_id = str(summary["evaluation_run"])
    evaluation = root / "evaluation" / run_id
    report = json.loads((evaluation / "report.json").read_text(encoding="utf-8"))
    document_id = str(report["documents"][0]["document_id"])
    prediction_path = evaluation / "predictions" / f"{document_id}.json"
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


@pytest.mark.parametrize("damage", ["absolute", "traversal", "key", "filename"])
def test_import_rejects_unsafe_or_mismatched_cache_paths(
    tmp_path: Path, damage: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = {
        "absolute": "/tmp/call.json",
        "traversal": f"../baselines/tma-core/{key}/call.json",
        "key": f"baselines/tma-core/{'b' * 64}/call.json",
        "filename": f"baselines/tma-core/{key}/provider.raw.json",
    }[damage]
    prediction["_evaluation"]["call_path"] = call_path
    _write_json(prediction_path, prediction)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"]["call_path"] = call_path
    _write_json(report_path, report)

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


def test_import_rejects_cache_key_that_is_not_exact_digest_of_inputs(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = root / "baselines" / "tma-core" / key / "call.json"
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["inputs"]["contract"]["scope"] = "changed but consistently reported"
    _write_json(call_path, call)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["baseline"]["contract"] = call["inputs"]["contract"]
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="cache digest"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_call_contract_that_differs_from_report(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = root / "baselines" / "tma-core" / key / "call.json"
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["inputs"]["contract"]["scope"] = "mismatched"
    _write_json(call_path, call)

    with pytest.raises(ValueError, match="contract mismatch"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize("damage", ["content", "order"])
def test_import_rejects_reference_ocr_that_differs_from_compiled_prompt(
    tmp_path: Path, damage: str
) -> None:
    dataset, root, run_id, _prediction_path, document_id = _fixture(tmp_path)
    reference_path = root / "evaluation" / run_id / "references" / f"{document_id}.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if damage == "content":
        reference["ocr"]["spans"][0]["text"] = "changed OCR"
    else:
        reference["ocr"]["spans"][0], reference["ocr"]["spans"][1] = (
            reference["ocr"]["spans"][1],
            reference["ocr"]["spans"][0],
        )
    _write_json(reference_path, reference)

    with pytest.raises(ValueError, match="does not match compiled prompt order"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_call_ocr_digest_mismatch(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = root / "baselines" / "tma-core" / key / "call.json"
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["inputs"]["ocr_sha256"] = "b" * 64
    _write_json(call_path, call)

    with pytest.raises(ValueError, match="call/OCR identity mismatch"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_unexpected_call_result_path(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = root / "baselines" / "tma-core" / key / "call.json"
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["result_path"] = "other.json"
    _write_json(call_path, call)

    with pytest.raises(ValueError, match="completed extraction"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_report_execution_mismatch(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction_path, _document_id = _fixture(tmp_path)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"]["cache_key"] = "b" * 64
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="execution identity mismatch"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_non_null_provider_error(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["_evaluation"]["provider_error"] = "SyntheticProviderError"
    _write_json(prediction_path, prediction)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"]["provider_error"] = "SyntheticProviderError"
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="not terminal-success"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_non_stop_provider_finish_reason(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["choices"][0]["finish_reason"] = "length"
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="finish with stop"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_malformed_provider_message(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["choices"][0]["message"] = "not-an-object"
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="assistant text content"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("field", "value"),
    [("refusal", "refused"), ("tool_calls", [{}]), ("function_call", {})],
)
def test_import_rejects_refusal_or_tool_call_content(
    tmp_path: Path, field: str, value: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["choices"][0]["message"][field] = value
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="refusal or tool call"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize("kind", ["reference", "prediction"])
def test_import_rejects_internal_evaluation_file_symlink_alias(
    tmp_path: Path, kind: str
) -> None:
    dataset, root, run_id, _prediction_path, document_id = _fixture(tmp_path)
    directory = root / "evaluation" / run_id / f"{kind}s"
    source = directory / f"{document_id}.json"
    alias_target = directory / f".{kind}-internal-alias"
    shutil.copyfile(source, alias_target)
    source.unlink()
    source.symlink_to(alias_target)

    with pytest.raises(ValueError, match="filesystem alias"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_internal_cache_directory_symlink_alias(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    cache = root / "baselines" / "tma-core" / key
    alias_target = root / "baselines" / "tma-core" / "internal-alias"
    shutil.copytree(cache, alias_target)
    shutil.rmtree(cache)
    cache.symlink_to(alias_target, target_is_directory=True)

    with pytest.raises(ValueError, match="filesystem alias"):
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
    report["documents"].append(json.loads(json.dumps(report["documents"][0])))
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
