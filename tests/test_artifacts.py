import json
from pathlib import Path

import pytest

from menu_vlm.artifacts import (
    REQUIRED_ARTIFACT_ROLES,
    create_artifact_bundle,
    package_directory,
    verify_archive,
)
from menu_vlm.jsonio import sha256_file, sha256_json


def _artifact_run(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    run = tmp_path / "run"
    run.mkdir()
    roles = {}
    for role in sorted(REQUIRED_ARTIFACT_ROLES):
        path = run / f"{role}.txt"
        path.write_text(f"synthetic {role}\n", encoding="utf-8")
        roles[role] = path.name
    identity = {
        "dataset_sha256": "a" * 64,
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "checkpoint_sha256": "b" * 64,
    }
    (run / roles["test_gate"]).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "identity": identity,
                "identity_sha256": sha256_json(identity),
                "reference_sha256": "c" * 64,
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


def test_artifact_bundle_rejects_missing_contract_role(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    spec = run / "spec.json"
    spec.write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="roles"):
        create_artifact_bundle(run, spec, tmp_path / "bundle")


def test_artifact_bundle_rejects_luna_mutated_after_completed_gate(tmp_path: Path) -> None:
    run, spec, roles = _artifact_run(tmp_path)
    (run / roles["luna_test_predictions"]).write_text(
        '{"example_id":"b","prediction":{"s":[],"i":[]}}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Luna predictions do not match"):
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
        ("identity-missing", "identity has an unexpected schema"),
        ("identity-extra", "identity has an unexpected schema"),
        ("identity-digest", "identity does not match"),
        ("dataset", "identity does not match"),
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
    elif damage == "identity-missing":
        gate["identity"].pop("checkpoint_sha256")
    elif damage == "identity-extra":
        gate["identity"]["unexpected"] = True
    elif damage == "identity-digest":
        gate["identity_sha256"] = "d" * 64
    elif damage == "dataset":
        gate["identity"]["dataset_sha256"] = "e" * 64
        gate["identity_sha256"] = sha256_json(gate["identity"])
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(damage)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        create_artifact_bundle(run, spec, tmp_path / "bundle")
