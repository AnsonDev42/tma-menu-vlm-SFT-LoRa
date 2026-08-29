import copy
import shutil
from pathlib import Path
from typing import Any

from .artifacts import REQUIRED_ARTIFACT_ROLES
from .constants import MODEL_ID, MODEL_REVISION, RELEASE_FORMAT, SPLITS
from .jsonio import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def create_synthetic_release(output: Path, *, include_splits: bool = True) -> dict[str, Any]:
    """Create a tiny, deterministic, obviously synthetic release for public proof."""
    root = output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Synthetic release output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        assignments = {
            "menu_00001": "train",
            "menu_00001_aug_01": "train",
            "menu_00002": "train",
            "menu_00002_aug_01": "train",
            "menu_00002_aug_02": "train",
            "menu_00003": "validation",
            "menu_00003_aug_01": "validation",
            "menu_00004": "test",
            "menu_00004_aug_01": "test",
        }
        records: list[dict[str, Any]] = []
        mapping: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        source_records: dict[str, dict[str, Any]] = {}
        for index in range(1, 5):
            document_id = f"menu_{index:05d}"
            restaurant_id = f"synthetic-restaurant-{index}"
            record = _record(
                root,
                document_id=document_id,
                kind="source_reference",
                split=assignments[document_id] if include_splits else None,
                restaurant_id=restaurant_id,
                source_id=None,
                annotation=None,
            )
            source_records[document_id] = record
            records.append(record)
            mapping.append(
                {
                    "canonical_id": document_id,
                    "original_id": f"synthetic-original-{index}",
                    "restaurant_id": restaurant_id,
                    **({"split": assignments[document_id]} if include_splits else {}),
                }
            )
            derivative_id = f"{document_id}_aug_01"
            derivative = _record(
                root,
                document_id=derivative_id,
                kind="silver_augmented",
                split=assignments[derivative_id] if include_splits else None,
                restaurant_id=restaurant_id,
                source_id=document_id,
                annotation=record["annotation"],
            )
            records.append(derivative)
            lineage.append(
                {
                    "document_id": derivative_id,
                    "source_id": document_id,
                    "content_preserving": True,
                    "transform": {"synthetic_brightness": index},
                    **({"split": assignments[derivative_id]} if include_splits else {}),
                }
            )
        excluded_id = "menu_00002_aug_02"
        lineage.append(
            {
                "document_id": excluded_id,
                "source_id": "menu_00002",
                "content_preserving": True,
                "transform": {"synthetic_brightness": 99},
                **({"split": assignments[excluded_id]} if include_splits else {}),
            }
        )
        exclusions = [{"document_id": excluded_id, "reason": "synthetic exclusion fixture"}]
        write_json(root / "records.json", records)
        write_json(root / "excluded.json", exclusions)
        write_json(root / "mapping.json", mapping)
        write_json(root / "lineage.json", lineage)
        if include_splits:
            write_json(
                root / "splits.json",
                {
                    "schema_version": "1.0",
                    "seed": 20260829,
                    "group_key": "restaurant_id",
                    "groups": {
                        f"synthetic-restaurant-{index}": assignments[f"menu_{index:05d}"]
                        for index in range(1, 5)
                    },
                    "documents": assignments,
                    "splits": {
                        split: sorted(
                            document_id
                            for document_id, assigned in assignments.items()
                            if assigned == split
                        )
                        for split in SPLITS
                    },
                },
            )
        hashes = {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "format": RELEASE_FORMAT if include_splits else "menu-canonical-unsplit-v1",
            "version": "synthetic-v1",
            "source_count": 4,
            "derivative_count": 4,
            "excluded_count": 1,
            "document_count": 8,
            "files_sha256": hashes,
        }
        write_json(root / "manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def create_reordered_predictions(references: Path, output: Path) -> dict[str, Any]:
    rows = []
    for row in read_jsonl(references):
        target = copy.deepcopy(row["target"])
        target["s"].reverse()
        old_to_new = {}
        for index, section in enumerate(target["s"], 1):
            old_to_new[section["id"]] = f"generated-section-{index}"
            section["id"] = old_to_new[section["id"]]
            section["h"]["l"].reverse()
        target["i"].reverse()
        for item in target["i"]:
            item["l"].reverse()
            item["n"] = item["n"].upper()
            if item["s"] is not None:
                item["s"] = old_to_new[item["s"]]
        rows.append(
            {
                "example_id": row["example_id"],
                "prediction": target,
                "latency_seconds": 0.01,
                "tokens_per_second": 100.0,
                "peak_memory_bytes": 1024,
            }
        )
    write_jsonl(output, rows)
    return {"examples": len(rows), "output": str(output)}


def create_synthetic_artifact_run(output: Path, *, dataset_manifest_path: Path) -> dict[str, Any]:
    root = output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Synthetic artifact run is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    dataset_manifest = read_json(dataset_manifest_path)
    files = {}
    for role in sorted(REQUIRED_ARTIFACT_ROLES):
        path = root / f"{role}.synthetic.txt"
        path.write_text(f"public synthetic {role}\n", encoding="utf-8")
        files[role] = path.name
    spec = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "seed": 20260829,
        "hardware": {"gpu": "public-synthetic"},
        "commands": ["public synthetic smoke"],
        "files": files,
    }
    write_json(root / "artifact-spec.json", spec)
    return {"run_root": str(root), "spec": str(root / "artifact-spec.json")}


def _record(
    root: Path,
    *,
    document_id: str,
    kind: str,
    split: str | None,
    restaurant_id: str,
    source_id: str | None,
    annotation: dict[str, Any] | None,
) -> dict[str, Any]:
    image_relative = f"images/{document_id}.svg"
    image_path = root / image_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label = document_id if source_id is None else f"{source_id} synthetic derivative"
    image_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240">'
        '<rect width="320" height="240" fill="white"/>'
        f'<text x="20" y="40">SYNTHETIC {label}</text>'
        '<text x="20" y="90">Grill</text>'
        '<text x="20" y="130">Steak $12</text>'
        "</svg>\n",
        encoding="utf-8",
    )
    annotation = annotation or {
        "schema_version": "1.0",
        "currency": "USD",
        "sections": [
            {
                "name": {"text": "Grill", "source_ids": ["ocr_0001"]},
                "items": [
                    {
                        "name": {"text": "Steak", "source_ids": ["ocr_0002"]},
                        "description": {
                            "text": "charred vegetables",
                            "source_ids": ["ocr_0003"],
                        },
                        "prices": [
                            {
                                "raw": "$12",
                                "amount": "12.00",
                                "currency": "USD",
                                "label": None,
                                "source_ids": ["ocr_0002"],
                            }
                        ],
                        "calories": None,
                        "variants": [],
                        "notes": [{"text": "Best served rare", "source_ids": ["ocr_0004"]}],
                    }
                ],
                "notes": [],
            }
        ],
        "unsectioned_items": [],
    }
    result: dict[str, Any] = {
        "document_id": document_id,
        "kind": kind,
        "image": {
            "path": image_relative,
            "sha256": sha256_file(image_path),
            "original_name": f"{document_id}.svg",
        },
        "ocr": {
            "schema_version": "1.0",
            "provider": "synthetic",
            "image_width": 320,
            "image_height": 240,
            "spans": [
                _span(1, "Grill", 0.1),
                _span(2, "Steak $12", 0.3),
                _span(3, "charred vegetables", 0.5),
                _span(4, "Best served rare", 0.7),
            ],
        },
        "metadata": {
            "restaurant_id": restaurant_id,
            "language": ["en"],
            "source": "synthetic-fixture",
            "synthetic": True,
            "tags": ["public-synthetic"],
        },
        "annotation": annotation,
        "provenance": {
            "synthetic_fixture": True,
            **({"source_id": source_id} if source_id else {}),
        },
    }
    if split is not None:
        result["split"] = split
    return result


def _span(index: int, text: str, y: float) -> dict[str, Any]:
    return {
        "id": f"ocr_{index:04d}",
        "text": text,
        "polygon": [0.1, y, 0.9, y, 0.9, min(0.99, y + 0.1), 0.1, min(0.99, y + 0.1)],
        "confidence": 1.0,
    }
