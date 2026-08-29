import gzip
import shutil
import tarfile
from pathlib import Path
from typing import Any

from .constants import MODEL_ID, MODEL_REVISION
from .jsonio import read_json, safe_relative, sha256_file, write_json

REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "adapter_config",
        "adapter_weights",
        "processor_provenance",
        "dependency_lock",
        "training_config",
        "dataset_manifest",
        "training_logs",
        "checkpoint_selection",
        "predictions",
        "metrics",
        "hardware",
        "commands",
    }
)


def create_artifact_bundle(run_root: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    root = run_root.resolve()
    spec = read_json(spec_path)
    files = spec.get("files", {})
    if not isinstance(files, dict) or set(files) != set(REQUIRED_ARTIFACT_ROLES):
        missing = sorted(REQUIRED_ARTIFACT_ROLES - set(files))
        extra = sorted(set(files) - REQUIRED_ARTIFACT_ROLES)
        raise ValueError(f"Artifact roles do not match contract; missing={missing}, extra={extra}")
    required_metadata = {
        "schema_version",
        "model_id",
        "model_revision",
        "dataset_sha256",
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
        for role, relative in sorted(files.items()):
            source = safe_relative(root, str(relative))
            if not source.is_file():
                raise ValueError(f"Artifact role is not a file: {role}={relative}")
            bundled_relative = f"files/{role}/{source.name}"
            target = destination / bundled_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest_files[role] = {
                "path": bundled_relative,
                "sha256": sha256_file(target),
            }
        manifest = {
            "format": "tma-menu-vlm-adapter-bundle-v1",
            "schema_version": "1.0",
            "model_id": spec["model_id"],
            "model_revision": spec["model_revision"],
            "dataset_sha256": spec["dataset_sha256"],
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
