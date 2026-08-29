import json
from pathlib import Path

import pytest

from menu_vlm.artifacts import (
    REQUIRED_ARTIFACT_ROLES,
    create_artifact_bundle,
    package_directory,
    verify_archive,
)


def test_artifact_bundle_and_transfer_archive_round_trip(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    roles = {}
    for role in sorted(REQUIRED_ARTIFACT_ROLES):
        path = run / f"{role}.txt"
        path.write_text(f"synthetic {role}\n", encoding="utf-8")
        roles[role] = path.name
    spec = run / "artifact-spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "dataset_sha256": "a" * 64,
                "seed": 20260829,
                "hardware": {"gpu": "synthetic"},
                "commands": ["synthetic command"],
                "files": roles,
            }
        ),
        encoding="utf-8",
    )
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
