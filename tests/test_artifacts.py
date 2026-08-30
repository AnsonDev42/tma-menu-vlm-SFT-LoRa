import json
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from menu_vlm.artifacts import (
    PRIVATE_LUNA_TRUST_ROOT_ROLES,
    REQUIRED_ARTIFACT_ROLES,
    create_artifact_bundle,
    package_directory,
    verify_archive,
)
from menu_vlm.compiler import CompileOptions, compile_release
from menu_vlm.jsonio import sha256_file, sha256_json
from menu_vlm.luna_baseline import (
    LUNA_TRUST_ROOT_NAME,
    LUNA_TRUST_ROOT_SIDECAR_NAME,
    create_luna_trust_root,
    import_luna_baseline,
)
from menu_vlm.synthetic import create_synthetic_luna_evaluation, create_synthetic_release


@lru_cache(maxsize=1)
def _synthetic_evidence() -> tuple[bytes, bytes, bytes, bytes, bytes]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        release = root / "release"
        dataset = root / "dataset"
        tma = root / "tma"
        create_synthetic_release(release)
        compile_release(CompileOptions(release=release, output=dataset))
        summary = create_synthetic_luna_evaluation(dataset, tma)
        luna = root / "luna-test-predictions.jsonl"
        import_luna_baseline(
            dataset,
            tma,
            str(summary["evaluation_run"]),
            tma / "luna-response-provenance.json",
            tma / "luna-response-provenance.json.sha256",
            luna,
        )
        trust_root = root / LUNA_TRUST_ROOT_NAME
        create_luna_trust_root(
            dataset,
            tma,
            str(summary["evaluation_run"]),
            luna,
            sha256_file(luna),
            trust_root,
        )
        return (
            (dataset / "manifest.json").read_bytes(),
            (tma / "luna-response-provenance.json").read_bytes(),
            luna.read_bytes(),
            trust_root.read_bytes(),
            (root / LUNA_TRUST_ROOT_SIDECAR_NAME).read_bytes(),
        )


def _artifact_run(
    tmp_path: Path, *, private_trust: bool = False
) -> tuple[Path, Path, dict[str, str]]:
    run = tmp_path / "run"
    run.mkdir()
    roles = {}
    for role in sorted(REQUIRED_ARTIFACT_ROLES):
        path = run / f"{role}.txt"
        path.write_text(f"synthetic {role}\n", encoding="utf-8")
        roles[role] = path.name
    (
        dataset_manifest_bytes,
        response_provenance_bytes,
        luna_prediction_bytes,
        trust_root_bytes,
        trust_root_sidecar_bytes,
    ) = (
        _synthetic_evidence()
    )
    (run / roles["dataset_manifest"]).write_bytes(dataset_manifest_bytes)
    (run / roles["luna_test_predictions"]).write_bytes(luna_prediction_bytes)
    provenance_path = run / roles["luna_response_provenance"]
    provenance_path = provenance_path.with_name("luna-response-provenance.json")
    (run / roles["luna_response_provenance"]).unlink()
    provenance_path.write_bytes(response_provenance_bytes)
    roles["luna_response_provenance"] = provenance_path.name
    if private_trust:
        trust_root = run / LUNA_TRUST_ROOT_NAME
        trust_root.write_bytes(trust_root_bytes)
        trust_root_sidecar = run / LUNA_TRUST_ROOT_SIDECAR_NAME
        trust_root_sidecar.write_bytes(trust_root_sidecar_bytes)
        roles["luna_trust_root"] = trust_root.name
        roles["luna_trust_root_sidecar"] = trust_root_sidecar.name
    dataset_manifest = json.loads(dataset_manifest_bytes)
    test_reference_sha256 = dataset_manifest["files_sha256"]["test.jsonl"]
    dataset_sha256 = dataset_manifest["dataset_sha256"]
    checkpoint_sha256 = sha256_json(
        {
            "adapter_config.json": sha256_file(run / roles["adapter_config"]),
            "adapter_model.safetensors": sha256_file(run / roles["adapter_weights"]),
        }
    )
    identity = {
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": sha256_file(run / roles["dataset_manifest"]),
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "checkpoint_sha256": checkpoint_sha256,
    }
    (run / roles["test_gate"]).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "identity": identity,
                "identity_sha256": sha256_json(identity),
                "reference_sha256": test_reference_sha256,
                "prediction_sha256": sha256_file(run / roles["test_predictions"]),
                "luna_prediction_sha256": sha256_file(run / roles["luna_test_predictions"]),
                "status": "completed",
                "metrics_sha256": sha256_file(run / roles["test_metrics"]),
            }
        ),
        encoding="utf-8",
    )
    spec = run / "artifact-spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model_id": identity["model_id"],
                "model_revision": identity["model_revision"],
                "dataset_sha256": identity["dataset_sha256"],
                "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
                "seed": 20260829,
                "hardware": {"gpu": "synthetic"},
                "commands": ["synthetic command"],
                "files": roles,
            }
        ),
        encoding="utf-8",
    )
    return run, spec, roles


def test_artifact_bundle_and_transfer_archive_round_trip(tmp_path: Path) -> None:
    assert {
        "checkpoint_candidates",
        "selected_validation_predictions",
        "selected_validation_metrics",
        "robustness_validation_predictions",
        "robustness_validation_metrics",
        "luna_test_predictions",
        "luna_response_provenance",
        "test_predictions",
        "test_metrics",
        "robustness_test_predictions",
        "robustness_test_metrics",
        "test_gate",
    }.issubset(REQUIRED_ARTIFACT_ROLES)
    run, spec, _roles = _artifact_run(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = create_artifact_bundle(run, spec, bundle)
    assert set(manifest["files"]) == set(REQUIRED_ARTIFACT_ROLES)

    archive = tmp_path / "bundle.tar.gz"
    package_directory(bundle, archive)
    restored = tmp_path / "restored"
    report = verify_archive(archive, restored)
    assert report["valid"] is True
    assert (restored / "artifact-manifest.json").is_file()
    assert not (restored / "TRANSFER-MANIFEST.json").exists()


def test_private_artifact_bundle_binds_luna_trust_root(tmp_path: Path) -> None:
    run, spec, _roles = _artifact_run(tmp_path, private_trust=True)
    manifest = create_artifact_bundle(run, spec, tmp_path / "bundle")

    assert set(manifest["files"]) == set(REQUIRED_ARTIFACT_ROLES) | set(
        PRIVATE_LUNA_TRUST_ROOT_ROLES
    )


@pytest.mark.parametrize("role", ["luna_trust_root", "luna_trust_root_sidecar"])
def test_private_artifact_bundle_rejects_trust_root_mutation(
    tmp_path: Path, role: str
) -> None:
    run, spec, roles = _artifact_run(tmp_path, private_trust=True)
    (run / roles[role]).write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"trust root|checksum"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")


def test_artifact_bundle_rejects_missing_contract_role(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    spec = run / "spec.json"
    spec.write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="roles"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")


@pytest.mark.parametrize(
    "role",
    [
        "test_predictions",
        "test_metrics",
        "luna_test_predictions",
        "adapter_config",
        "adapter_weights",
    ],
)
def test_artifact_bundle_rejects_evidence_mutated_after_completed_gate(
    tmp_path: Path, role: str
) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    (run / roles[role]).write_text("mutated after completed gate\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completed test gate"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_artifact_bundle_rejects_mutated_luna_response_provenance(tmp_path: Path) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    (run / roles["luna_response_provenance"]).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="approved anchor"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_artifact_bundle_rejects_new_gate_for_changed_luna_with_old_provenance(
    tmp_path: Path,
) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    luna = run / roles["luna_test_predictions"]
    luna.write_text('{"forged":"new Luna baseline"}\n', encoding="utf-8")
    gate_path = run / roles["test_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["luna_prediction_sha256"] = sha256_file(luna)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="response provenance"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("damage", ["wrong-name", "symlink", "role-collision"])
def test_artifact_bundle_rejects_provenance_path_aliases_and_collisions(
    tmp_path: Path, damage: str
) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    provenance = run / roles["luna_response_provenance"]
    spec_value = json.loads(spec.read_text(encoding="utf-8"))
    if damage == "wrong-name":
        renamed = run / "caller-controlled-provenance.json"
        provenance.rename(renamed)
        spec_value["files"]["luna_response_provenance"] = renamed.name
    elif damage == "symlink":
        target = run / ".private-provenance-target"
        provenance.rename(target)
        provenance.symlink_to(target.name)
    else:
        spec_value["files"]["adapter_config"] = provenance.name
    spec.write_text(json.dumps(spec_value), encoding="utf-8")

    with pytest.raises(ValueError, match=r"reserved filename|filesystem alias|collide"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "field",
    [
        "dataset_sha256",
        "test.jsonl",
        "train.jsonl",
        "compiled_records",
        "release_exclusions",
        "assignment",
    ],
)
def test_artifact_bundle_rejects_dataset_manifest_mutated_after_completed_gate(
    tmp_path: Path, field: str
) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    manifest_path = run / roles["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "dataset_sha256":
        manifest[field] = "f" * 64
    elif field in {"compiled_records", "release_exclusions"}:
        manifest["accounting"][field] += 1
    elif field == "assignment":
        manifest["split_assignments"]["forged"] = "train"
    else:
        manifest["files_sha256"][field] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset manifest"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "damage", ["split-assignment", "prompt", "source-manifest", "count-allocation"]
)
def test_artifact_bundle_rejects_semantic_manifest_mutation_with_unchanged_file_aggregate(
    tmp_path: Path, damage: str
) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    manifest_path = run / roles["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if damage == "split-assignment":
        key = next(iter(manifest["split_assignments"]))
        manifest["split_assignments"][key] = (
            "validation" if manifest["split_assignments"][key] != "validation" else "train"
        )
    elif damage == "prompt":
        manifest["prompt"]["sha256"] = "f" * 64
    elif damage == "source-manifest":
        manifest["source_release_manifest_sha256"] = "f" * 64
    else:
        manifest["counts"]["train"] -= 1
        manifest["counts"]["validation"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest bytes"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("reserved", "not completed"),
        ("schema-bool", "not completed"),
        ("missing", "unexpected schema"),
        ("extra", "unexpected schema"),
        ("digest-bool", "invalid SHA-256"),
        ("digest-invalid", "invalid SHA-256"),
        ("manifest-digest-bool", "invalid SHA-256"),
        ("identity-missing", "identity has an unexpected schema"),
        ("identity-extra", "identity has an unexpected schema"),
        ("identity-digest", "identity does not match"),
        ("dataset", "identity does not match"),
        ("manifest-spec-mismatch", "identity does not match"),
    ],
)
def test_artifact_bundle_rejects_invalid_test_gate(
    tmp_path: Path, damage: str, message: str
) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    gate_path = run / roles["test_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if damage == "reserved":
        gate["status"] = "reserved"
    elif damage == "schema-bool":
        gate["schema_version"] = True
    elif damage == "missing":
        gate.pop("metrics_sha256")
    elif damage == "extra":
        gate["unexpected"] = True
    elif damage == "digest-bool":
        gate["luna_prediction_sha256"] = True
    elif damage == "digest-invalid":
        gate["metrics_sha256"] = "not-a-digest"
    elif damage == "manifest-digest-bool":
        gate["identity"]["dataset_manifest_sha256"] = True
    elif damage == "identity-missing":
        gate["identity"].pop("checkpoint_sha256")
    elif damage == "identity-extra":
        gate["identity"]["unexpected"] = True
    elif damage == "identity-digest":
        gate["identity_sha256"] = "d" * 64
    elif damage == "dataset":
        gate["identity"]["dataset_sha256"] = "e" * 64
        gate["identity_sha256"] = sha256_json(gate["identity"])
    elif damage == "manifest-spec-mismatch":
        gate["identity"]["dataset_manifest_sha256"] = "e" * 64
        gate["identity_sha256"] = sha256_json(gate["identity"])
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(damage)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
