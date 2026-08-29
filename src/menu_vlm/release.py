from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .constants import RELEASE_FORMAT, SPLITS
from .jsonio import read_json, safe_relative, sha256_file
from .schemas import ReleaseRecord


@dataclass(frozen=True)
class ValidatedRelease:
    root: Path
    manifest: dict[str, Any]
    records: tuple[ReleaseRecord, ...]
    exclusions: tuple[dict[str, Any], ...]
    mapping: tuple[dict[str, Any], ...]
    lineage: tuple[dict[str, Any], ...]
    splits: dict[str, Any] | None
    manifest_sha256: str


_RECORDS = TypeAdapter(list[ReleaseRecord])


def validate_release(path: Path, *, allow_unsplit: bool = False) -> ValidatedRelease:
    root = path.resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"Release root has no manifest.json: {root}")
    manifest = read_json(manifest_path)
    expected_format = "menu-canonical-unsplit-v1" if allow_unsplit else RELEASE_FORMAT
    if manifest.get("format") not in (
        {RELEASE_FORMAT, expected_format} if allow_unsplit else {RELEASE_FORMAT}
    ):
        raise ValueError(f"Unsupported release format: {manifest.get('format')!r}")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Release manifest must contain files_sha256")
    required = {"records.json", "excluded.json", "mapping.json", "lineage.json"}
    if not allow_unsplit or "splits.json" in hashes:
        required.add("splits.json")
    if not required.issubset(hashes):
        raise ValueError(
            f"Release manifest does not hash required files: {sorted(required - set(hashes))}"
        )
    for relative, expected in sorted(hashes.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("Release hashes must map paths to SHA-256 strings")
        target = safe_relative(root, relative)
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Release integrity check failed: {relative}")

    records = tuple(_RECORDS.validate_python(read_json(root / "records.json")))
    exclusions = tuple(_require_dict_list(read_json(root / "excluded.json"), "excluded.json"))
    mapping = tuple(_require_dict_list(read_json(root / "mapping.json"), "mapping.json"))
    lineage = tuple(_require_dict_list(read_json(root / "lineage.json"), "lineage.json"))
    splits = read_json(root / "splits.json") if (root / "splits.json").is_file() else None
    _validate_counts(manifest, records, exclusions, mapping, lineage)
    _validate_identities(records, exclusions, mapping, lineage)
    _validate_images(root, records, hashes)
    if splits is not None:
        _validate_splits(records, exclusions, mapping, lineage, splits)
    elif not allow_unsplit:
        raise ValueError("Immutable silver release is missing splits.json")

    return ValidatedRelease(
        root=root,
        manifest=manifest,
        records=records,
        exclusions=exclusions,
        mapping=mapping,
        lineage=lineage,
        splits=splits,
        manifest_sha256=sha256_file(manifest_path),
    )


def _require_dict_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must contain a JSON array of objects")
    return value


def _validate_counts(
    manifest: dict[str, Any],
    records: tuple[ReleaseRecord, ...],
    exclusions: tuple[dict[str, Any], ...],
    mapping: tuple[dict[str, Any], ...],
    lineage: tuple[dict[str, Any], ...],
) -> None:
    source_count = sum(record.kind == "source_reference" for record in records)
    derivative_count = sum(record.kind == "silver_augmented" for record in records)
    declared = {
        "source_count": source_count,
        "derivative_count": derivative_count,
        "excluded_count": len(exclusions),
        "document_count": len(records),
    }
    for key, actual in declared.items():
        if manifest.get(key) != actual:
            raise ValueError(f"Release {key} is {manifest.get(key)!r}; expected {actual}")
    if len(mapping) != source_count:
        raise ValueError("Mapping count does not account for every source")
    if len(lineage) != derivative_count + len(exclusions):
        raise ValueError("Lineage count does not account for included and excluded derivatives")


def _validate_identities(
    records: tuple[ReleaseRecord, ...],
    exclusions: tuple[dict[str, Any], ...],
    mapping: tuple[dict[str, Any], ...],
    lineage: tuple[dict[str, Any], ...],
) -> None:
    record_ids = [record.document_id for record in records]
    excluded_ids = [str(row.get("document_id", "")) for row in exclusions]
    source_ids = [str(row.get("canonical_id", "")) for row in mapping]
    lineage_ids = [str(row.get("document_id", "")) for row in lineage]
    for label, values in (
        ("record", record_ids),
        ("excluded", excluded_ids),
        ("canonical source", source_ids),
        ("lineage", lineage_ids),
    ):
        if not all(values) or len(values) != len(set(values)):
            raise ValueError(f"Release {label} identities are missing or duplicated")
    expected_sources = {
        record.document_id for record in records if record.kind == "source_reference"
    }
    expected_derivatives = {
        record.document_id for record in records if record.kind == "silver_augmented"
    } | set(excluded_ids)
    if set(source_ids) != expected_sources or set(lineage_ids) != expected_derivatives:
        raise ValueError("Records, exclusions, mapping, and lineage are not an exact partition")

    records_by_id = {record.document_id: record for record in records}
    lineage_by_id = {str(row["document_id"]): row for row in lineage}
    for record in records:
        if record.kind != "silver_augmented":
            continue
        row = lineage_by_id[record.document_id]
        source_id = str(row.get("source_id", ""))
        source = records_by_id.get(source_id)
        if source is None or source.kind != "source_reference":
            raise ValueError("Derivative lineage references an unknown canonical source")
        if row.get("content_preserving") is not True:
            raise ValueError("Derivative lineage is not content-preserving")
        if record.annotation != source.annotation:
            raise ValueError("Derivative annotation is not inherited from its canonical source")


def _validate_images(
    root: Path, records: tuple[ReleaseRecord, ...], hashes: dict[str, str]
) -> None:
    for record in records:
        image_path = safe_relative(root, record.image.path)
        if not image_path.is_file() or sha256_file(image_path) != record.image.sha256:
            raise ValueError(f"Record image integrity check failed: {record.document_id}")
        if hashes.get(record.image.path) != record.image.sha256:
            raise ValueError(f"Record image is not release-verified: {record.document_id}")


def _validate_splits(
    records: tuple[ReleaseRecord, ...],
    exclusions: tuple[dict[str, Any], ...],
    mapping: tuple[dict[str, Any], ...],
    lineage: tuple[dict[str, Any], ...],
    splits: dict[str, Any],
) -> None:
    documents = splits.get("documents")
    split_lists = splits.get("splits")
    if not isinstance(documents, dict) or not isinstance(split_lists, dict):
        raise ValueError("splits.json must contain documents and splits mappings")
    if set(split_lists) != set(SPLITS):
        raise ValueError("Release split names must be exactly train, validation, and test")
    all_ids = {record.document_id for record in records} | {
        str(entry["document_id"]) for entry in exclusions
    }
    if set(documents) != all_ids:
        raise ValueError("Release split assignments do not account for all records and exclusions")
    listed = [document_id for split in SPLITS for document_id in split_lists[split]]
    if len(listed) != len(set(listed)) or set(listed) != all_ids:
        raise ValueError("Release split lists are missing or duplicating documents")
    if any(
        documents.get(document_id) != split
        for split in SPLITS
        for document_id in split_lists[split]
    ):
        raise ValueError("Release split lists disagree with document assignments")
    if any(record.split != documents[record.document_id] for record in records):
        raise ValueError("Release record split disagrees with immutable assignment")

    source_ids = {str(row["canonical_id"]) for row in mapping}
    for row in lineage:
        document_id = str(row["document_id"])
        source_id = str(row.get("source_id", ""))
        if source_id not in source_ids:
            raise ValueError("Derivative lineage references an unknown source")
        if documents.get(document_id) != documents.get(source_id) or row.get(
            "split"
        ) != documents.get(source_id):
            raise ValueError("Derivative lineage crosses dataset splits")
    source_splits = {
        documents[record.document_id] for record in records if record.kind == "source_reference"
    }
    if source_splits != set(SPLITS):
        raise ValueError("Release has degenerate primary source splits")
