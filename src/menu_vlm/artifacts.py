import gzip
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

from .constants import (
    DATASET_FORMAT,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_VERSION,
    RELEASE_FORMAT,
    SPLITS,
)
from .jsonio import (
    canonical_json,
    read_json,
    safe_relative,
    sha256_file,
    sha256_json,
    verify_sha256_sidecar,
    write_json,
)
from .luna_baseline import (
    LUNA_RESPONSE_PROVENANCE_NAME,
    LUNA_TRUST_ROOT_NAME,
    LUNA_TRUST_ROOT_SIDECAR_NAME,
    approved_response_provenance_sha256,
    validate_luna_trust_root,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_TEST_GATE_KEYS = {
    "schema_version",
    "identity",
    "identity_sha256",
    "reference_sha256",
    "prediction_sha256",
    "luna_prediction_sha256",
    "status",
    "metrics_sha256",
}
_TEST_GATE_IDENTITY_KEYS = {
    "dataset_sha256",
    "dataset_manifest_sha256",
    "model_id",
    "model_revision",
    "checkpoint_sha256",
}

REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "adapter_config",
        "adapter_weights",
        "processor_provenance",
        "dependency_lock",
        "training_config",
        "dataset_manifest",
        "training_logs",
        "checkpoint_candidates",
        "checkpoint_selection",
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
        "hardware",
        "commands",
    }
)
PRIVATE_LUNA_TRUST_ROOT_ROLES = frozenset(
    {"luna_trust_root", "luna_trust_root_sidecar"}
)


def create_artifact_bundle(run_root: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    root = run_root.resolve()
    spec = read_json(spec_path)
    files = spec.get("files", {})
    supplied_roles = set(files) if isinstance(files, dict) else set()
    allowed_roles = set(REQUIRED_ARTIFACT_ROLES) | set(PRIVATE_LUNA_TRUST_ROOT_ROLES)
    private_roles = supplied_roles & set(PRIVATE_LUNA_TRUST_ROOT_ROLES)
    if (
        not isinstance(files, dict)
        or not set(REQUIRED_ARTIFACT_ROLES).issubset(supplied_roles)
        or not supplied_roles.issubset(allowed_roles)
        or private_roles not in (set(), set(PRIVATE_LUNA_TRUST_ROOT_ROLES))
    ):
        missing = sorted(set(REQUIRED_ARTIFACT_ROLES) - supplied_roles)
        if private_roles:
            missing.extend(sorted(set(PRIVATE_LUNA_TRUST_ROOT_ROLES) - private_roles))
        extra = sorted(supplied_roles - allowed_roles)
        raise ValueError(f"Artifact roles do not match contract; missing={missing}, extra={extra}")
    required_metadata = {
        "schema_version",
        "model_id",
        "model_revision",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "seed",
        "hardware",
        "commands",
    }
    if not required_metadata.issubset(spec):
        raise ValueError(
            f"Artifact spec metadata is incomplete: {sorted(required_metadata - set(spec))}"
        )
    if spec["model_id"] != MODEL_ID or spec["model_revision"] != MODEL_REVISION:
        raise ValueError("Artifact spec model provenance does not match the pinned base")
    destination = output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Artifact output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        manifest_files: dict[str, dict[str, str]] = {}
        copied_sources: set[Path] = set()
        for role, relative in sorted(files.items()):
            source = safe_relative(root, str(relative))
            expected_source = root / str(relative)
            if source != expected_source or source.is_symlink():
                raise ValueError(f"Artifact role uses a filesystem alias: {role}={relative}")
            if not source.is_file():
                raise ValueError(f"Artifact role is not a file: {role}={relative}")
            if source in copied_sources:
                raise ValueError(f"Artifact roles collide on one source file: {role}={relative}")
            copied_sources.add(source)
            if role == "luna_response_provenance" and source.name != LUNA_RESPONSE_PROVENANCE_NAME:
                raise ValueError("Luna response provenance role must use its reserved filename")
            if role == "luna_trust_root" and source.name != LUNA_TRUST_ROOT_NAME:
                raise ValueError("Luna trust root role must use its reserved filename")
            if (
                role == "luna_trust_root_sidecar"
                and source.name != LUNA_TRUST_ROOT_SIDECAR_NAME
            ):
                raise ValueError("Luna trust root sidecar role must use its reserved filename")
            bundled_relative = f"files/{role}/{source.name}"
            target = destination / bundled_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest_files[role] = {
                "path": bundled_relative,
                "sha256": sha256_file(target),
            }
        _validate_test_gate(
            destination / manifest_files["test_gate"]["path"], spec, manifest_files
        )
        manifest = {
            "format": "tma-menu-vlm-adapter-bundle-v1",
            "schema_version": "1.0",
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "dataset_sha256": spec["dataset_sha256"],
            "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
            "seed": spec["seed"],
            "hardware": spec["hardware"],
            "commands": spec["commands"],
            "files": manifest_files,
        }
        write_json(destination / "artifact-manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _validate_test_gate(
    gate_path: Path, spec: dict[str, Any], manifest_files: dict[str, dict[str, str]]
) -> None:
    gate = read_json(gate_path)
    if not isinstance(gate, dict) or set(gate) != _TEST_GATE_KEYS:
        raise ValueError("Artifact test gate has an unexpected schema")
    if gate.get("schema_version") != "1.0" or gate.get("status") != "completed":
        raise ValueError("Artifact test gate is not completed schema version 1.0")
    identity = gate.get("identity")
    if not isinstance(identity, dict) or set(identity) != _TEST_GATE_IDENTITY_KEYS:
        raise ValueError("Artifact test gate identity has an unexpected schema")
    digests = (
        gate.get("identity_sha256"),
        gate.get("reference_sha256"),
        gate.get("prediction_sha256"),
        gate.get("luna_prediction_sha256"),
        gate.get("metrics_sha256"),
        identity.get("dataset_sha256"),
        identity.get("dataset_manifest_sha256"),
        identity.get("checkpoint_sha256"),
    )
    if any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in digests):
        raise ValueError("Artifact test gate contains an invalid SHA-256 identity")
    if (
        identity.get("dataset_sha256") != spec.get("dataset_sha256")
        or identity.get("dataset_manifest_sha256") != spec.get("dataset_manifest_sha256")
        or identity.get("model_id") != spec.get("model_id")
        or identity.get("model_revision") != spec.get("model_revision")
        or gate.get("identity_sha256") != sha256_json(identity)
    ):
        raise ValueError("Artifact test gate identity does not match the artifact spec")
    bound_roles = {
        "prediction_sha256": "test_predictions",
        "metrics_sha256": "test_metrics",
        "luna_prediction_sha256": "luna_test_predictions",
    }
    for gate_key, role in bound_roles.items():
        if gate.get(gate_key) != manifest_files[role]["sha256"]:
            raise ValueError(f"Artifact {role} do not match the completed test gate")

    checkpoint_hashes = {
        "adapter_config.json": manifest_files["adapter_config"]["sha256"],
        "adapter_model.safetensors": manifest_files["adapter_weights"]["sha256"],
    }
    if identity.get("checkpoint_sha256") != sha256_json(checkpoint_hashes):
        raise ValueError("Artifact adapter does not match the completed test gate checkpoint")

    dataset_manifest_path = Path(manifest_files["dataset_manifest"]["path"])
    if (
        identity.get("dataset_manifest_sha256")
        != manifest_files["dataset_manifest"]["sha256"]
    ):
        raise ValueError("Artifact dataset manifest bytes do not match the completed test gate")
    dataset_manifest = read_json(gate_path.parent.parent.parent / dataset_manifest_path)
    _validate_compiled_dataset_manifest(dataset_manifest)
    files_sha256 = dataset_manifest["files_sha256"]
    if (
        not isinstance(dataset_manifest, dict)
        or dataset_manifest.get("dataset_sha256") != spec.get("dataset_sha256")
        or dataset_manifest.get("dataset_sha256") != identity.get("dataset_sha256")
        or not isinstance(files_sha256, dict)
        or files_sha256.get("test.jsonl") != gate.get("reference_sha256")
    ):
        raise ValueError("Artifact dataset manifest does not match the completed test gate")
    has_private_trust_root = PRIVATE_LUNA_TRUST_ROOT_ROLES.issubset(manifest_files)
    if (
        not has_private_trust_root
        and manifest_files["luna_response_provenance"]["sha256"]
        != approved_response_provenance_sha256(dataset_manifest)
    ):
        raise ValueError("Artifact Luna response provenance does not match the approved anchor")
    provenance_path = Path(manifest_files["luna_response_provenance"]["path"])
    provenance = read_json(gate_path.parent.parent.parent / provenance_path)
    if (
        not isinstance(provenance, dict)
        or provenance.get("dataset_sha256") != dataset_manifest.get("dataset_sha256")
        or provenance.get("baseline_prediction_sha256")
        != manifest_files["luna_test_predictions"]["sha256"]
    ):
        raise ValueError("Artifact Luna response provenance does not match the dataset identity")
    if has_private_trust_root:
        bundle_root = gate_path.parent.parent.parent
        trust_path = bundle_root / manifest_files["luna_trust_root"]["path"]
        trust_sidecar = bundle_root / manifest_files["luna_trust_root_sidecar"]["path"]
        try:
            verify_sha256_sidecar(trust_path, trust_sidecar)
            trust_value = read_json(trust_path)
            if trust_path.read_bytes() != (canonical_json(trust_value) + "\n").encode("utf-8"):
                raise ValueError("Luna trust root is not canonical exact JSON")
            validate_luna_trust_root(
                trust_value,
                dataset_manifest,
                manifest_files["dataset_manifest"]["sha256"],
                str(provenance.get("evaluation_run")),
                manifest_files["luna_test_predictions"]["sha256"],
            )
        except ValueError as exc:
            raise ValueError(f"Artifact Luna trust root verification failed: {exc}") from exc


def _validate_compiled_dataset_manifest(value: Any) -> None:
    keys = {
        "format",
        "schema_version",
        "source_release_format",
        "source_release_version",
        "source_release_manifest_sha256",
        "source_release_splits_preserved",
        "split_seed",
        "split_assignments",
        "prompt",
        "one_image_per_example",
        "counts",
        "accounting",
        "files",
        "files_sha256",
        "dataset_sha256",
    }
    expected_files = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
        "robustness_validation": "robustness_validation.jsonl",
        "robustness_test": "robustness_test.jsonl",
    }
    count_keys = set(expected_files) | {"excluded", "projection_failures"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Artifact dataset manifest has an unexpected schema")
    files_sha256 = value.get("files_sha256")
    dataset_sha256 = value.get("dataset_sha256")
    counts = value.get("counts")
    accounting = value.get("accounting")
    assignments = value.get("split_assignments")
    prompt = value.get("prompt")
    split_seed = value.get("split_seed")
    input_records = accounting.get("input_records") if isinstance(accounting, dict) else None
    safe_input_records: int = (
        input_records
        if isinstance(input_records, int) and not isinstance(input_records, bool)
        else -1
    )
    release_exclusions = (
        accounting.get("release_exclusions") if isinstance(accounting, dict) else None
    )
    safe_release_exclusions: int = (
        release_exclusions
        if isinstance(release_exclusions, int) and not isinstance(release_exclusions, bool)
        else -1
    )
    required_hashed_files = set(expected_files.values()) | {
        "excluded.json",
        "projection-failures.json",
        "split-manifest.json",
    }
    image_hashes = (
        {relative for relative in files_sha256 if relative.startswith("images/")}
        if isinstance(files_sha256, dict)
        else set()
    )
    if (
        value.get("format") != DATASET_FORMAT
        or value.get("schema_version") != "1.0"
        or value.get("source_release_format") != RELEASE_FORMAT
        or not isinstance(value.get("source_release_version"), str)
        or not value["source_release_version"]
        or not _is_digest(value.get("source_release_manifest_sha256"))
        or not isinstance(value.get("source_release_splits_preserved"), bool)
        or not (split_seed is None or _is_nonnegative_int(split_seed))
        or not isinstance(assignments, dict)
        or len(assignments) != safe_input_records + safe_release_exclusions
        or any(
            not isinstance(key, str) or not key or item not in SPLITS
            for key, item in assignments.items()
        )
        or not isinstance(prompt, dict)
        or set(prompt) != {"version", "sha256"}
        or prompt.get("version") != PROMPT_VERSION
        or not _is_digest(prompt.get("sha256"))
        or value.get("one_image_per_example") is not True
        or not isinstance(counts, dict)
        or set(counts) != count_keys
        or any(not _is_nonnegative_int(item) for item in counts.values())
        or not isinstance(accounting, dict)
        or set(accounting) != {"input_records", "compiled_records", "release_exclusions"}
        or any(not _is_nonnegative_int(item) for item in accounting.values())
        or accounting.get("compiled_records")
        != sum(counts[name] for name in expected_files)
        or accounting.get("input_records")
        != int(accounting["compiled_records"]) + int(counts["projection_failures"])
        or accounting.get("release_exclusions") != counts.get("excluded")
        or value.get("files") != expected_files
        or not isinstance(files_sha256, dict)
        or not files_sha256
        or not required_hashed_files.issubset(files_sha256)
        or len(image_hashes) != accounting.get("compiled_records")
        or set(files_sha256) != required_hashed_files | image_hashes
        or any(
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_digest(digest)
            for relative, digest in files_sha256.items()
        )
        or not _is_digest(dataset_sha256)
        or sha256_json(files_sha256) != dataset_sha256
        or (value.get("source_release_splits_preserved") is True and split_seed is not None)
        or (value.get("source_release_splits_preserved") is False and split_seed is None)
    ):
        raise ValueError("Artifact dataset manifest has malformed audited evidence")


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def package_directory(source: Path, archive: Path) -> dict[str, Any]:
    root = source.resolve()
    if not root.is_dir():
        raise ValueError(f"Package source is not a directory: {root}")
    destination = archive.resolve()
    if destination.exists():
        raise FileExistsError(f"Archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not source_files:
        raise ValueError("Refusing to package an empty directory")
    file_hashes = {str(path.relative_to(root)): sha256_file(path) for path in source_files}
    transfer_manifest = {
        "format": "tma-private-transfer-v1",
        "files_sha256": file_hashes,
    }
    manifest_bytes = (_canonical_transfer_json(transfer_manifest) + "\n").encode("utf-8")
    with (
        destination.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        _add_bytes(tar, "TRANSFER-MANIFEST.json", manifest_bytes)
        for path in source_files:
            _add_file(tar, path, str(path.relative_to(root)))
    digest = sha256_file(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return {
        "archive": str(destination),
        "archive_sha256": digest,
        "sidecar": str(sidecar),
        "file_count": len(source_files),
    }


def verify_archive(archive: Path, output: Path) -> dict[str, Any]:
    source = archive.resolve()
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if sha256_file(source) != expected:
            raise ValueError("Transfer archive checksum does not match sidecar")
    destination = output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Archive output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(source, mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"Transfer archive contains unsupported entry: {member.name}")
                target = safe_relative(destination, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read transfer member: {member.name}")
                with target.open("xb") as output_stream:
                    shutil.copyfileobj(stream, output_stream)
        manifest_path = destination / "TRANSFER-MANIFEST.json"
        manifest = read_json(manifest_path)
        hashes = manifest.get("files_sha256", {})
        actual = {
            str(path.relative_to(destination))
            for path in destination.rglob("*")
            if path.is_file() and path.name != "TRANSFER-MANIFEST.json"
        }
        if set(hashes) != actual:
            raise ValueError("Transfer manifest does not account for archive contents")
        for relative, expected in hashes.items():
            if sha256_file(safe_relative(destination, relative)) != expected:
                raise ValueError(f"Transferred file checksum drift: {relative}")
        manifest_path.unlink()
        return {
            "valid": True,
            "archive_sha256": sha256_file(source),
            "file_count": len(hashes),
        }
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _add_bytes(tar: tarfile.TarFile, name: str, value: bytes) -> None:
    import io

    info = tarfile.TarInfo(name)
    info.size = len(value)
    _normalize_tar_info(info)
    tar.addfile(info, io.BytesIO(value))


def _add_file(tar: tarfile.TarFile, path: Path, name: str) -> None:
    info = tar.gettarinfo(str(path), arcname=name)
    _normalize_tar_info(info)
    with path.open("rb") as stream:
        tar.addfile(info, stream)


def _normalize_tar_info(info: tarfile.TarInfo) -> None:
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644


def _canonical_transfer_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
