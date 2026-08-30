import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import pytest

import menu_vlm.luna_baseline as luna_baseline_module
from menu_vlm.compiler import CompileOptions, compile_release
from menu_vlm.jsonio import canonical_json, read_jsonl, verify_sha256_sidecar
from menu_vlm.luna_baseline import (
    LUNA_RESPONSE_PROVENANCE_NAME,
    LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME,
)
from menu_vlm.luna_baseline import (
    import_luna_baseline as _import_luna_baseline,
)
from menu_vlm.synthetic import create_synthetic_luna_evaluation, create_synthetic_release


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def import_luna_baseline(
    dataset: Path, tma_data_root: Path, evaluation_run: str, output: Path
) -> dict[str, object]:
    return _import_luna_baseline(
        dataset,
        tma_data_root,
        evaluation_run,
        tma_data_root / LUNA_RESPONSE_PROVENANCE_NAME,
        tma_data_root / LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME,
        output,
    )


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


def _cache_paths(root: Path, prediction_path: Path) -> tuple[dict[str, object], Path, Path]:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    evaluation = prediction["_evaluation"]
    assert isinstance(evaluation, dict)
    key = evaluation["cache_key"]
    assert isinstance(key, str)
    cache = root / "baselines" / "tma-core" / key
    return prediction, cache / "call.json", cache / "tma.json"


def _promote_cost_fixture_to_production(
    dataset: Path,
    root: Path,
    run_id: str,
    prediction_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Exercise production cost validation with public one-document evidence."""
    prediction, call_path, _result_path = _cache_paths(root, prediction_path)
    raw_path = call_path.with_name("provider.raw.json")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_usage = {
        "completion_tokens": 1,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens": 1,
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cache_write_tokens": None,
            "cached_tokens": 0,
        },
        "total_tokens": 2,
    }
    raw["usage"] = raw_usage
    _write_json(raw_path, raw)
    cost: dict[str, object] = {
        "cached_input_tokens": 0,
        "estimated_usd": "0.000001",
        "input_tokens": 1,
        "output_tokens": 1,
        "provider_reported_usd": None,
        "raw_usage": raw_usage,
        "reasoning_tokens": 0,
        "unknown_reason": None,
    }
    prediction["_evaluation"]["cost"] = cost
    _write_json(prediction_path, prediction)
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["cost"] = cost
    _write_json(call_path, call)

    evaluation = root / "evaluation" / run_id
    report_path = evaluation / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dataset_version"] = "public-production-cost-fixture-v1"
    report["metrics"] = {key: 0.0 for key in luna_baseline_module._METRIC_KEYS}
    report["counts"] = {key: 0 for key in luna_baseline_module._COUNT_KEYS}
    report["costs"] = {
        **{key: 0 for key in luna_baseline_module._COST_INTEGER_KEYS},
        "all_predictions_estimated_usd": "0.000001",
        "new_estimated_usd": "0.000001",
        "original_api_latency_median_seconds": 0.0,
        "provider_seconds_total": 0.0,
    }
    report["price_diagnostics"] = {
        key: 0 for key in luna_baseline_module._PRICE_DIAGNOSTIC_KEYS
    }
    report["baseline"]["rate"] = {
        key: (4096 if key == "max_input_tokens" else "public-test")
        for key in luna_baseline_module._RATE_KEYS
    }
    call["rate"] = report["baseline"]["rate"]
    _write_json(call_path, call)
    document = report["documents"][0]
    document["metrics"] = {key: 0.0 for key in luna_baseline_module._METRIC_KEYS}
    document["counts"] = {key: 0 for key in luna_baseline_module._COUNT_KEYS}
    document["execution"]["cost"] = cost
    report["worst_documents"] = [
        {
            "document_id": document["document_id"],
            "extra": 0,
            "missing": 0,
            "price_errors": 0,
        }
    ]
    run = {
        key: json.loads(json.dumps(report[key]))
        for key in luna_baseline_module._RUN_KEYS
    }
    _write_json(report_path, report)
    _write_json(evaluation / "run.json", run)

    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_release_version"] = "public-production-cost-fixture-v1"
    _write_json(manifest_path, manifest)

    provenance_path = root / LUNA_RESPONSE_PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["documents"][0]["provider_response_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    _write_json(provenance_path, provenance)
    luna_baseline_module.write_sha256_sidecar(
        provenance_path, root / LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME
    )
    approved_sha256 = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        luna_baseline_module,
        "approved_response_provenance_sha256",
        lambda _manifest: approved_sha256,
    )
    original_validate = luna_baseline_module._validate_response_provenance

    def validate_public_single_document(
        value: dict[str, object], dataset_manifest: dict[str, object], evaluation: str
    ) -> None:
        synthetic_manifest = dict(dataset_manifest)
        synthetic_manifest["source_release_version"] = "synthetic-v1"
        original_validate(value, synthetic_manifest, evaluation)

    monkeypatch.setattr(
        luna_baseline_module,
        "_validate_response_provenance",
        validate_public_single_document,
    )
    return cost


def _write_mirrored_cost(
    root: Path, run_id: str, prediction_path: Path, cost: dict[str, object]
) -> None:
    prediction, call_path, _result_path = _cache_paths(root, prediction_path)
    prediction["_evaluation"]["cost"] = cost
    _write_json(prediction_path, prediction)
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["cost"] = cost
    _write_json(call_path, call)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"]["cost"] = cost
    _write_json(report_path, report)


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


def test_import_rejects_output_not_bound_to_provenance_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, root, run_id, _prediction, _document_id = _fixture(tmp_path)
    output = tmp_path / "luna.jsonl"
    original_write = luna_baseline_module.write_jsonl

    def forged_write(path: Path, rows: list[dict[str, object]]) -> None:
        original_write(path, rows)
        path.write_bytes(path.read_bytes() + b'{"forged":true}\n')

    monkeypatch.setattr(luna_baseline_module, "write_jsonl", forged_write)
    with pytest.raises(ValueError, match="approved provenance identity"):
        import_luna_baseline(dataset, root, run_id, output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.sha256").exists()


def test_full_import_accepts_exact_public_production_cost_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    _promote_cost_fixture_to_production(
        dataset, root, run_id, prediction_path, monkeypatch
    )

    result = import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")

    assert result["valid"] is True


@pytest.mark.parametrize(
    "damage",
    [
        "forged-object",
        "bool-token",
        "string-token",
        "float-token",
        "extra-key",
        "missing-key",
        "estimated-numeric",
        "estimated-invalid-string",
        "raw-bool",
        "raw-string",
        "raw-float",
        "raw-extra",
        "raw-missing",
        "detail-extra",
        "detail-missing",
    ],
)
def test_full_import_rejects_mirrored_production_cost_schema_attacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    approved = _promote_cost_fixture_to_production(
        dataset, root, run_id, prediction_path, monkeypatch
    )
    cost = json.loads(json.dumps(approved))
    if damage == "forged-object":
        cost = {"forged": True}
    elif damage == "bool-token":
        cost["cached_input_tokens"] = True
    elif damage == "string-token":
        cost["input_tokens"] = "1"
    elif damage == "float-token":
        cost["output_tokens"] = 1.0
    elif damage == "extra-key":
        cost["forged"] = True
    elif damage == "missing-key":
        del cost["unknown_reason"]
    elif damage == "estimated-numeric":
        cost["estimated_usd"] = 0.000001
    elif damage == "estimated-invalid-string":
        cost["estimated_usd"] = "NaN"
    else:
        usage = cost["raw_usage"]
        assert isinstance(usage, dict)
        if damage == "raw-bool":
            usage["prompt_tokens"] = True
        elif damage == "raw-string":
            usage["completion_tokens"] = "1"
        elif damage == "raw-float":
            usage["total_tokens"] = 2.0
        elif damage == "raw-extra":
            usage["forged"] = True
        elif damage == "raw-missing":
            del usage["total_tokens"]
        else:
            details = usage["prompt_tokens_details"]
            assert isinstance(details, dict)
            if damage == "detail-extra":
                details["forged"] = 0
            elif damage == "detail-missing":
                del details["cache_write_tokens"]
            else:  # pragma: no cover - parametrization is exhaustive
                raise AssertionError(damage)
    _write_mirrored_cost(root, run_id, prediction_path, cost)

    with pytest.raises(ValueError, match=r"execution is malformed|call record is malformed"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_full_import_rejects_schema_valid_mirrored_cost_usage_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    approved = _promote_cost_fixture_to_production(
        dataset, root, run_id, prediction_path, monkeypatch
    )
    cost = json.loads(json.dumps(approved))
    usage = cost["raw_usage"]
    assert isinstance(usage, dict)
    cost["input_tokens"] += 1
    usage["prompt_tokens"] += 1
    usage["total_tokens"] += 1
    _write_mirrored_cost(root, run_id, prediction_path, cost)

    with pytest.raises(ValueError, match="cost usage does not match provider response"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_provenance_parent_directory_symlink_alias(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction, _document_id = _fixture(tmp_path)
    alias = tmp_path / "provenance-parent-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match=r"parent paths.*filesystem aliases"):
        _import_luna_baseline(
            dataset,
            root,
            run_id,
            alias / LUNA_RESPONSE_PROVENANCE_NAME,
            alias / LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME,
            tmp_path / "luna.jsonl",
        )


def test_runtime_uses_approved_pydantic_version() -> None:
    assert importlib.metadata.version("pydantic") == "2.13.4"


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
    old_key = prediction["_evaluation"]["cache_key"]
    new_key = "c" * 64
    old_cache = root / "baselines" / "tma-core" / old_key
    new_cache = root / "baselines" / "tma-core" / new_key
    old_cache.rename(new_cache)
    new_call_path = f"baselines/tma-core/{new_key}/call.json"
    prediction["_evaluation"]["cache_key"] = new_key
    prediction["_evaluation"]["call_path"] = new_call_path
    _write_json(prediction_path, prediction)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"]["cache_key"] = new_key
    report["documents"][0]["execution"]["call_path"] = new_call_path
    _write_json(report_path, report)

    with pytest.raises(ValueError, match=r"cache digest|response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("extra-key", "key inventory"),
        ("adapter-version", "critical settings"),
        ("endpoint", "critical settings"),
        ("image-detail", "critical settings"),
        ("numeric-type", "typed critical settings"),
        ("excluded-stages", "excluded stages"),
        ("scope-type", "source or scope provenance"),
        ("empty-sources", "no source hash inventory"),
        ("unsafe-source", "invalid source hash inventory"),
        ("uppercase-digest", "invalid source hash inventory"),
        ("empty-loaded", "loaded source provenance"),
        ("duplicate-loaded", "loaded source provenance"),
        ("unknown-loaded", "loaded source provenance"),
        ("adapter-digest", "adapter source hash"),
        ("runtime-keys", "runtime versions"),
        ("runtime-empty", "runtime versions"),
    ],
)
def test_import_rejects_non_runtime_contract(
    tmp_path: Path, damage: str, message: str
) -> None:
    dataset, root, run_id, _prediction_path, _document_id = _fixture(tmp_path)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contract = report["baseline"]["contract"]
    if damage == "extra-key":
        contract["unexpected"] = True
    elif damage == "adapter-version":
        contract["adapter_version"] = "other"
    elif damage == "endpoint":
        contract["endpoint"] = "https://example.invalid/v1"
    elif damage == "image-detail":
        contract["image_detail"] = "high"
    elif damage == "numeric-type":
        contract["timeout_seconds"] = 75.0
    elif damage == "excluded-stages":
        contract["excluded_stages"] = ["fresh OCR"]
    elif damage == "scope-type":
        contract["scope"] = ["wrong type"]
    elif damage == "empty-sources":
        contract["source_sha256"] = {}
    elif damage == "unsafe-source":
        contract["source_sha256"] = {"../private.py": "a" * 64}
    elif damage == "uppercase-digest":
        contract["source_sha256"]["src/core/config.py"] = "A" * 64
    elif damage == "empty-loaded":
        contract["loaded_source_paths"] = []
    elif damage == "duplicate-loaded":
        contract["loaded_source_paths"].append(contract["loaded_source_paths"][0])
    elif damage == "unknown-loaded":
        contract["loaded_source_paths"] = ["src/unknown.py"]
    elif damage == "adapter-digest":
        contract["adapter_source_sha256"] = "not-a-digest"
    elif damage == "runtime-keys":
        contract["runtime_versions"] = {"openai": "1.0"}
    elif damage == "runtime-empty":
        contract["runtime_versions"]["openai"] = ""
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(damage)
    _write_json(report_path, report)

    with pytest.raises(ValueError, match=message):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_self_consistent_forged_source_provenance(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    old_key = prediction["_evaluation"]["cache_key"]
    old_cache = root / "baselines" / "tma-core" / old_key
    call = json.loads((old_cache / "call.json").read_text(encoding="utf-8"))
    result = json.loads((old_cache / "tma.json").read_text(encoding="utf-8"))
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    forged = json.loads(json.dumps(report["baseline"]["contract"]))
    forged["source_sha256"] = {"src/forged.py": "d" * 64}
    forged["loaded_source_paths"] = ["src/forged.py"]
    forged["adapter_source_sha256"] = "e" * 64
    forged["runtime_versions"] = {
        "openai": "forged",
        "pydantic": "forged",
        "Pillow": "forged",
    }
    call["inputs"]["contract"] = forged
    new_key = hashlib.sha256(
        json.dumps(call["inputs"], sort_keys=True).encode()
    ).hexdigest()
    new_call_path = f"baselines/tma-core/{new_key}/call.json"
    new_cache = root / "baselines" / "tma-core" / new_key
    old_cache.rename(new_cache)
    _write_json(new_cache / "call.json", call)
    result["contract"] = forged
    _write_json(new_cache / "tma.json", result)
    prediction["tma"] = result
    prediction["_evaluation"]["cache_key"] = new_key
    prediction["_evaluation"]["call_path"] = new_call_path
    _write_json(prediction_path, prediction)
    report["baseline"]["contract"] = forged
    report["documents"][0]["execution"]["cache_key"] = new_key
    report["documents"][0]["execution"]["call_path"] = new_call_path
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="approved provenance anchor"):
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


def test_import_rejects_missing_materialized_result(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    (root / "baselines" / "tma-core" / key / "tma.json").unlink()

    with pytest.raises(ValueError, match="materialized result is missing"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_non_object_materialized_result(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    result = root / "baselines" / "tma-core" / key / "tma.json"
    _write_json(result, [])

    with pytest.raises(ValueError, match="invalid key inventory"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_materialized_result_symlink_alias(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    result = root / "baselines" / "tma-core" / key / "tma.json"
    alias = result.with_name(".internal-result-alias.json")
    shutil.copyfile(result, alias)
    result.unlink()
    result.symlink_to(alias)

    with pytest.raises(ValueError, match="filesystem alias"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("extra-key", "key inventory"),
        ("contract", "contract mismatch"),
        ("results-type", "collections are malformed"),
        ("results-entry", "collections are malformed"),
        ("items-entry", "collections are malformed"),
        ("empty-objects", "item has an invalid key inventory"),
        ("result-extra", "result row has an invalid inventory"),
        ("item-extra", "item has an invalid key inventory"),
        ("info-extra", "result info has an invalid inventory"),
        ("item-binding", "result/item binding is invalid"),
        ("note-location", "item has malformed field types"),
        ("count-type", "counts are malformed"),
        ("length-mismatch", "accounting is inconsistent"),
        ("count-sum", "accounting is inconsistent"),
        ("provider-received", "provider evidence is contradictory"),
        ("provider-error", "provider evidence is contradictory"),
        ("finish-reason", "provider evidence is contradictory"),
        ("timing", "timings are malformed"),
        ("processed-image", "processed image evidence is malformed"),
        ("processed-digest", "does not match source processing"),
        ("processed-dimensions", "does not match source processing"),
    ],
)
def test_import_rejects_invalid_materialized_result(
    tmp_path: Path, damage: str, message: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    result_path = root / "baselines" / "tma-core" / key / "tma.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if damage == "extra-key":
        result["unexpected"] = True
    elif damage == "contract":
        result["contract"]["adapter_source_sha256"] = "c" * 64
    elif damage == "results-type":
        result["results"] = {}
    elif damage == "results-entry":
        result["results"] = [None]
        result["items"] = [{}]
        result["vision_item_count"] = 1
    elif damage == "items-entry":
        result["results"] = [{}]
        result["items"] = [1]
        result["vision_item_count"] = 1
    elif damage == "empty-objects":
        result["results"] = [{}]
        result["items"] = [{}]
    elif damage == "result-extra":
        result["results"][0]["unexpected"] = True
    elif damage == "item-extra":
        result["items"][0]["unexpected"] = True
    elif damage == "info-extra":
        result["results"][0]["info"]["unexpected"] = True
    elif damage == "item-binding":
        result["results"][0]["info"]["text"] = "Different"
    elif damage == "note-location":
        result["items"][0]["notes"] = [
            {
                "id": "synthetic-note",
                "page_index": 0,
                "original_text": "Synthetic note",
                "ocr_line_indices": [2],
                "locations": [True],
                "translation": None,
                "translation_language": None,
                "translation_status": "pending",
            }
        ]
    elif damage == "count-type":
        result["vision_item_count"] = True
    elif damage == "length-mismatch":
        result["results"] = [{}]
        result["items"] = []
        result["vision_item_count"] = 1
    elif damage == "count-sum":
        result["vision_item_count"] = 2
    elif damage == "provider-received":
        result["provider_response_received"] = False
    elif damage == "provider-error":
        result["provider_error"] = "SyntheticProviderError"
    elif damage == "finish-reason":
        result["provider_finish_reason"] = "length"
    elif damage == "timing":
        result["provider_seconds"] = -1
    elif damage == "processed-image":
        result["processed_image"]["width"] = 0
    elif damage == "processed-digest":
        result["processed_image"]["sha256"] = "f" * 64
    elif damage == "processed-dimensions":
        result["processed_image"]["width"] += 1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(damage)
    _write_json(result_path, result)

    with pytest.raises(ValueError, match=message):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_prediction_not_bound_to_materialized_result(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["tma"]["provider_seconds"] = 0.03
    _write_json(prediction_path, prediction)

    with pytest.raises(ValueError, match="not bound to its materialized result"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_report_execution_mismatch(tmp_path: Path) -> None:
    dataset, root, run_id, _prediction_path, _document_id = _fixture(tmp_path)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"]["cache_key"] = "b" * 64
    _write_json(report_path, report)

    with pytest.raises(
        ValueError, match=r"execution identity mismatch|response provenance|cache path"
    ):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("report-extra", "report/run has an unexpected key inventory"),
        ("run-extra", "report/run has an unexpected key inventory"),
        ("document-extra", "report document has an unexpected key inventory"),
        ("execution-extra", "report execution is malformed"),
        ("document-count-bool", "document accounting is incomplete"),
        ("completed-count-bool", "document accounting is incomplete"),
        ("run-count-bool", "run/report document accounting is mismatched"),
        ("schema-error", "document metrics are malformed"),
        ("created-type", "malformed audited metadata"),
        ("dataset-version-type", "malformed audited metadata"),
        ("dataset-version-mismatch", "malformed audited metadata"),
        ("report-metrics-type", "malformed audited metadata"),
        ("report-metrics-extra", "malformed audited metadata"),
        ("report-metrics-bool", "malformed audited metadata"),
        ("document-metrics-type", "document metrics are malformed"),
        ("document-metrics-extra", "document metrics are malformed"),
        ("document-metrics-bool", "document metrics are malformed"),
        ("document-counts-bool", "document metrics are malformed"),
        ("git-commit-type", "malformed audited metadata"),
        ("reference-kind-type", "malformed audited metadata"),
        ("metric-version-type", "malformed audited metadata"),
        ("match-threshold-bool", "malformed audited metadata"),
        ("match-threshold-string", "malformed audited metadata"),
        ("release-type", "malformed audited metadata"),
        ("release-caveats-type", "malformed audited metadata"),
        ("release-caveat-item", "malformed audited metadata"),
        ("by-tag-type", "malformed audited metadata"),
        ("costs-type", "malformed audited metadata"),
        ("price-diagnostics-type", "malformed audited metadata"),
        ("rate-extra", "malformed rate provenance"),
        ("unsupported-field-type", "malformed rate provenance"),
        ("matches-type", "document metrics are malformed"),
        ("match-shape", "document metrics are malformed"),
        ("unmatched-gold-type", "document metrics are malformed"),
        ("unmatched-prediction-item", "document metrics are malformed"),
        ("worst-documents-type", "worst-document evidence is malformed"),
    ],
)
def test_import_rejects_malformed_report_shapes(
    tmp_path: Path, damage: str, message: str
) -> None:
    dataset, root, run_id, _prediction_path, _document_id = _fixture(tmp_path)
    evaluation = root / "evaluation" / run_id
    report_path = evaluation / "report.json"
    run_path = evaluation / "run.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if damage == "report-extra":
        report["unexpected"] = True
    elif damage == "run-extra":
        run["unexpected"] = True
    elif damage == "document-extra":
        report["documents"][0]["unexpected"] = True
    elif damage == "execution-extra":
        report["documents"][0]["execution"]["unexpected"] = True
    elif damage == "document-count-bool":
        report["document_count"] = True
    elif damage == "completed-count-bool":
        report["completed_document_count"] = True
    elif damage == "run-count-bool":
        run["document_count"] = True
    elif damage == "schema-error":
        report["documents"][0]["schema_error"] = "forged"
    elif damage == "created-type":
        report["created_at"] = 0
    elif damage == "dataset-version-type":
        report["dataset_version"] = []
    elif damage == "dataset-version-mismatch":
        report["dataset_version"] = run["dataset_version"] = "forged-production-v1"
    elif damage == "report-metrics-type":
        report["metrics"] = []
    elif damage == "report-metrics-extra":
        report["metrics"]["forged"] = 0.5
    elif damage == "report-metrics-bool":
        report["metrics"]["dish_f1"] = True
    elif damage == "document-metrics-type":
        report["documents"][0]["metrics"] = []
    elif damage == "document-metrics-extra":
        report["documents"][0]["metrics"]["forged"] = 0.5
    elif damage == "document-metrics-bool":
        report["documents"][0]["metrics"]["dish_f1"] = True
    elif damage == "document-counts-bool":
        report["documents"][0]["counts"]["documents"] = True
    elif damage == "git-commit-type":
        report["git_commit"] = run["git_commit"] = True
    elif damage == "reference-kind-type":
        report["reference_kind"] = run["reference_kind"] = []
    elif damage == "metric-version-type":
        report["metric_version"] = run["metric_version"] = 1
    elif damage == "match-threshold-bool":
        report["match_threshold"] = run["match_threshold"] = True
    elif damage == "match-threshold-string":
        report["match_threshold"] = run["match_threshold"] = "0.5"
    elif damage == "release-type":
        report["release"] = run["release"] = False
    elif damage == "release-caveats-type":
        report["release_caveats"] = run["release_caveats"] = "forged"
    elif damage == "release-caveat-item":
        report["release_caveats"] = run["release_caveats"] = [True]
    elif damage == "by-tag-type":
        report["by_tag"] = []
    elif damage == "costs-type":
        report["costs"] = []
    elif damage == "price-diagnostics-type":
        report["price_diagnostics"] = []
    elif damage == "rate-extra":
        report["baseline"]["rate"]["forged"] = True
        run["baseline"]["rate"]["forged"] = True
    elif damage == "unsupported-field-type":
        report["baseline"]["unsupported_fields"] = [True]
        run["baseline"]["unsupported_fields"] = [True]
    elif damage == "matches-type":
        report["documents"][0]["matches"] = {}
    elif damage == "match-shape":
        report["documents"][0]["matches"] = [{"forged": True}]
    elif damage == "unmatched-gold-type":
        report["documents"][0]["unmatched_gold"] = {}
    elif damage == "unmatched-prediction-item":
        report["documents"][0]["unmatched_predictions"] = [True]
    elif damage == "worst-documents-type":
        report["worst_documents"] = {}
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(damage)
    _write_json(report_path, report)
    _write_json(run_path, run)

    with pytest.raises(ValueError, match=message):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_prediction_execution_extra_key(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["_evaluation"]["unexpected"] = True
    _write_json(prediction_path, prediction)

    with pytest.raises(ValueError, match="prediction execution is malformed"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_call_extra_key(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = root / "baselines" / "tma-core" / key / "call.json"
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["unexpected"] = True
    _write_json(call_path, call)

    with pytest.raises(ValueError, match="call record is malformed"):
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

    with pytest.raises(ValueError, match="response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_top_level_provider_error(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["error"] = {"type": "synthetic_error"}
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize("damage", ["zero-bytes", "choice-count", "model-name"])
def test_import_rejects_raw_only_substitution_before_parse(
    tmp_path: Path, damage: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    _prediction, call_path, _result_path = _cache_paths(root, prediction_path)
    raw_path = call_path.with_name("provider.raw.json")
    if damage == "zero-bytes":
        raw_path.write_bytes(b"")
    else:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if damage == "choice-count":
            raw["choices"].append(json.loads(json.dumps(raw["choices"][0])))
        else:
            raw["model"] = "gpt-5.6-luna-substituted"
        _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_non_null_call_error(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    call_path = root / "baselines" / "tma-core" / key / "call.json"
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["error"] = "SyntheticProviderError"
    _write_json(call_path, call)

    with pytest.raises(ValueError, match="call record is malformed"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("object", "other", "invalid completion envelope"),
        ("id", "", "invalid completion envelope"),
        ("created", True, "invalid completion envelope"),
        ("usage", [], "malformed token usage"),
    ],
)
def test_import_rejects_invalid_provider_envelope(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw[field] = value
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("model-suffix", "model identity mismatch"),
        ("model-type", "model identity mismatch"),
        ("usage-negative", "malformed token usage"),
        ("usage-string", "malformed token usage"),
        ("usage-total", "inconsistent token usage"),
        ("usage-extra", "malformed token usage"),
        ("detail-string", "malformed token usage"),
        ("choice-index", "finish with stop"),
        ("choice-index-type", "finish with stop"),
        ("choice-logprobs", "finish with stop"),
        ("choice-extra", "finish with stop"),
        ("message-extra", "assistant text content"),
        ("message-missing", "assistant text content"),
        ("top-extra", "invalid completion envelope"),
        ("metadata-type", "malformed optional metadata"),
        ("service-forged", "malformed optional metadata"),
        ("moderation-forged", "malformed optional metadata"),
        ("annotation-bool", "assistant text content"),
        ("annotation-forged", "assistant text content"),
    ],
)
def test_import_rejects_malformed_provider_evidence(
    tmp_path: Path, damage: str, message: str
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    choice = raw["choices"][0]
    message_value = choice["message"]
    if damage == "model-suffix":
        raw["model"] = "gpt-5.6-luna-forged"
    elif damage == "model-type":
        raw["model"] = 56
    elif damage == "usage-negative":
        raw["usage"]["prompt_tokens"] = -1
    elif damage == "usage-string":
        raw["usage"]["completion_tokens"] = "1"
    elif damage == "usage-total":
        raw["usage"]["total_tokens"] = 99
    elif damage == "usage-extra":
        raw["usage"]["forged_tokens"] = 1
    elif damage == "detail-string":
        raw["usage"]["prompt_tokens_details"] = {"cached_tokens": "1"}
    elif damage == "choice-index":
        choice["index"] = 1
    elif damage == "choice-index-type":
        choice["index"] = 0.0
    elif damage == "choice-logprobs":
        choice["logprobs"] = "forged"
    elif damage == "choice-extra":
        choice["unexpected"] = True
    elif damage == "message-extra":
        message_value["unexpected"] = True
    elif damage == "message-missing":
        message_value.pop("annotations")
    elif damage == "top-extra":
        raw["unexpected"] = True
    elif damage == "metadata-type":
        raw["service_tier"] = []
    elif damage == "service-forged":
        raw["service_tier"] = "forged"
    elif damage == "moderation-forged":
        raw["moderation"] = {"error": True}
    elif damage == "annotation-bool":
        message_value["annotations"] = [True]
    elif damage == "annotation-forged":
        message_value["annotations"] = [
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 0,
                    "end_index": 1,
                    "title": "forged",
                    "url": "not-a-url",
                    "unexpected": True,
                },
            }
        ]
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(damage)
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_combined_malformed_provider_fixture(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["model"] = "gpt-5.6-luna-forged"
    raw["usage"]["prompt_tokens"] = "many"
    raw["choices"][0]["index"] = 1
    raw["choices"][0]["logprobs"] = "not-an-object"
    raw["unexpected"] = {"forged": True}
    raw["choices"][0]["message"]["unexpected"] = True
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="response provenance"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


def test_import_rejects_malformed_provider_message(tmp_path: Path) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    key = prediction["_evaluation"]["cache_key"]
    raw_path = root / "baselines" / "tma-core" / key / "provider.raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["choices"][0]["message"] = "not-an-object"
    _write_json(raw_path, raw)

    with pytest.raises(ValueError, match="response provenance"):
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

    with pytest.raises(ValueError, match="response provenance"):
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

    with pytest.raises(ValueError, match="response provenance"):
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


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("run_id", "other-run"),
        ("model", "current-tma-core:gpt-5.6-luna-forged"),
        ("created_at", "2026-08-30T00:00:01Z"),
        ("dataset_version", "forged"),
        ("git_commit", "forged"),
        ("document_count", 0),
        ("reference_kind", "forged"),
        ("metric_version", "forged"),
        ("match_threshold", 0.6),
        ("release", "forged"),
        ("release_caveats", ["forged"]),
        ("status", "incomplete"),
        (
            "baseline",
            {
                "contract": None,
                "rate": {"forged": True},
                "unsupported_fields": [],
            },
        ),
    ],
)
def test_import_rejects_every_run_report_shared_field_mismatch(
    tmp_path: Path, field: str, forged: object
) -> None:
    dataset, root, run_id, _prediction_path, _document_id = _fixture(tmp_path)
    run_path = root / "evaluation" / run_id / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if field == "baseline":
        forged_baseline = json.loads(json.dumps(forged))
        forged_baseline["contract"] = run["baseline"]["contract"]
        run[field] = forged_baseline
    else:
        run[field] = forged
    _write_json(run_path, run)

    with pytest.raises(ValueError):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("cache_key", "b" * 64),
        ("cached", False),
        ("state", "failed"),
        ("cost", {"forged": 1}),
        ("call_path", f"baselines/tma-core/{'b' * 64}/call.json"),
        ("provider_seconds", 0.02),
        ("fallback_item_count", 1),
        ("provider_error", "forged"),
        ("provider_finish_reason", "length"),
        ("original_call_seconds", 0.02),
    ],
)
def test_import_rejects_every_report_prediction_execution_mismatch(
    tmp_path: Path, field: str, forged: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["_evaluation"][field] = forged
    _write_json(prediction_path, prediction)

    with pytest.raises(ValueError):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("field", "forged"),
    [("cost", {"forged": 1}), ("rate", {"forged": 1}), ("elapsed_seconds", 0.02)],
)
def test_import_rejects_call_execution_binding_attack(
    tmp_path: Path, field: str, forged: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    _prediction, call_path, _result_path = _cache_paths(root, prediction_path)
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call[field] = forged
    _write_json(call_path, call)

    message = (
        r"call record is malformed|call execution evidence is mismatched"
        if field == "cost"
        else "call execution evidence is mismatched"
    )
    with pytest.raises(ValueError, match=message):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(
    ("field", "forged"), [("provider_seconds", 0.02), ("fallback_item_count", 1)]
)
def test_import_rejects_materialized_execution_binding_attack(
    tmp_path: Path, field: str, forged: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["_evaluation"][field] = forged
    _write_json(prediction_path, prediction)
    report_path = root / "evaluation" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["documents"][0]["execution"][field] = forged
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="materialized execution evidence is mismatched"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize("forged", [False, 0.0, "0"])
def test_import_rejects_non_integer_materialized_result_id(
    tmp_path: Path, forged: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    _prediction, _call_path, result_path = _cache_paths(root, prediction_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["results"][0]["id"] = forged
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="result row has an invalid inventory"):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize("owner", ["item", "info", "note", "location"])
@pytest.mark.parametrize("forged", [False, 0.0, "0"])
def test_import_rejects_non_integer_materialized_page_index(
    tmp_path: Path, owner: str, forged: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    _prediction, _call_path, result_path = _cache_paths(root, prediction_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    item = result["items"][0]
    info = result["results"][0]["info"]
    if owner in {"item", "info"}:
        (item if owner == "item" else info)["page_index"] = forged
    else:
        location = {
            "page_index": 0,
            "page_label": "Page 1",
            "text": "Synthetic note",
            "bounding_box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
            "score": 1.0,
            "source": "ocr_reference",
        }
        note = {
            "id": "synthetic-note",
            "page_index": 0,
            "original_text": "Synthetic note",
            "ocr_line_indices": [2],
            "locations": [location],
            "translation": None,
            "translation_language": None,
            "translation_status": "pending",
        }
        (note if owner == "note" else location)["page_index"] = forged
        item["notes"] = [note]
        serialized_location = {
            key: value for key, value in location.items() if key != "bounding_box"
        } | {"boundingBox": location["bounding_box"]}
        info["notes"] = [note | {"locations": [serialized_location]}]
    _write_json(result_path, result)

    with pytest.raises(ValueError):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(("field", "forged"), [("image_width", "320"), ("image_height", 240.0)])
def test_import_rejects_strict_ocr_coercion(
    tmp_path: Path, field: str, forged: object
) -> None:
    dataset, root, run_id, _prediction_path, document_id = _fixture(tmp_path)
    reference_path = root / "evaluation" / run_id / "references" / f"{document_id}.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["ocr"][field] = forged
    _write_json(reference_path, reference)

    with pytest.raises(ValueError):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")


@pytest.mark.parametrize(("field", "forged"), [("a", "2"), ("c", "1.0"), ("l", ["2"])])
def test_import_rejects_strict_compact_output_coercion(
    tmp_path: Path, field: str, forged: object
) -> None:
    dataset, root, run_id, prediction_path, _document_id = _fixture(tmp_path)
    _prediction, call_path, _result_path = _cache_paths(root, prediction_path)
    raw_path = call_path.with_name("provider.raw.json")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    compact = json.loads(raw["choices"][0]["message"]["content"])
    compact["i"][0][field] = forged
    raw["choices"][0]["message"]["content"] = json.dumps(compact)
    _write_json(raw_path, raw)

    with pytest.raises(ValueError):
        import_luna_baseline(dataset, root, run_id, tmp_path / "luna.jsonl")
