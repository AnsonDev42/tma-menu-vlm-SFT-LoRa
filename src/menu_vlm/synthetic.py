import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import REQUIRED_ARTIFACT_ROLES
from .compiler import validate_compiled_dataset
from .constants import MODEL_ID, MODEL_REVISION, RELEASE_FORMAT, SPLITS
from .jsonio import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .luna_baseline import approved_luna_runtime_contract
from .tma_image import process_tma_image


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
    shutil.copyfile(dataset_manifest_path, root / files["dataset_manifest"])
    checkpoint_sha256 = sha256_json(
        {
            "adapter_config.json": sha256_file(root / files["adapter_config"]),
            "adapter_model.safetensors": sha256_file(root / files["adapter_weights"]),
        }
    )
    identity = {
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
    }
    write_json(
        root / files["test_gate"],
        {
            "schema_version": "1.0",
            "identity": identity,
            "identity_sha256": sha256_json(identity),
            "reference_sha256": dataset_manifest["files_sha256"]["test.jsonl"],
            "prediction_sha256": sha256_file(root / files["test_predictions"]),
            "luna_prediction_sha256": sha256_file(root / files["luna_test_predictions"]),
            "status": "completed",
            "metrics_sha256": sha256_file(root / files["test_metrics"]),
        },
    )
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


def create_synthetic_luna_evaluation(dataset: Path, output: Path) -> dict[str, Any]:
    """Create public synthetic current-TMA cache evidence for importer smoke tests."""
    validate_compiled_dataset(dataset)
    dataset_root = dataset.resolve()
    manifest = read_json(dataset_root / "manifest.json")
    if manifest.get("source_release_version") != "synthetic-v1":
        raise ValueError("Synthetic Luna evidence requires the public synthetic dataset")
    root = output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Synthetic TMA data output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    run_id = "public-synthetic-luna"
    evaluation = root / "evaluation" / run_id
    documents: list[dict[str, Any]] = []
    contract = approved_luna_runtime_contract()
    try:
        for index, row in enumerate(read_jsonl(dataset_root / "test.jsonl"), 1):
            document_id = f"synthetic-tma-{index:04d}"
            image_sha256 = sha256_file(dataset_root / row["image"])
            processed_bytes, processed_height, processed_width = process_tma_image(
                (dataset_root / row["image"]).read_bytes()
            )
            ocr = {
                "schema_version": "1.0",
                "provider": "public-synthetic",
                "image_width": 320,
                "image_height": 240,
                "spans": [
                    _span(1, "Grill", 0.1),
                    _span(2, "Steak $12", 0.3),
                    _span(3, "charred vegetables", 0.5),
                    _span(4, "Best served rare", 0.7),
                ],
            }
            inputs = {
                "contract": contract,
                "image_sha256": image_sha256,
                "ocr_sha256": _tma_digest(ocr),
            }
            cache_key = _tma_digest(inputs)
            call_path = f"baselines/tma-core/{cache_key}/call.json"
            execution = {
                "cache_key": cache_key,
                "cached": True,
                "state": "succeeded",
                "cost": {},
                "call_path": call_path,
                "provider_seconds": 0.01,
                "fallback_item_count": 0,
                "provider_error": None,
                "provider_finish_reason": "stop",
                "original_call_seconds": 0.01,
            }
            durable_item = {
                "section_id": "section-public-synthetic",
                "notes": [],
                "information_only": False,
                "source_text": "Steak $12",
                "locator_text": "Steak $12",
                "name": "Steak",
                "ocr_line_indices": [2],
                "translation": None,
                "description": "charred vegetables",
                "price": "$12",
                "category": None,
                "page_index": 0,
                "vision_index": 0,
                "confidence": 1.0,
                "is_ocr_fallback": False,
            }
            serialized_result = {
                "id": 0,
                "info": {
                    "section_id": "section-public-synthetic",
                    "text": "Steak",
                    "text_translation": "Steak",
                    "description": "charred vegetables",
                    "price": "$12",
                    "price_info": {
                        "raw": "$12",
                        "amount": 12.0,
                        "currency": "USD",
                        "source": "symbol",
                        "confidence": 0.9,
                    },
                    "category": None,
                    "confidence": 1.0,
                    "source_text": "Steak $12",
                    "locator_text": "Steak $12",
                    "page_index": 0,
                    "page_label": "Page 1",
                    "locations": [],
                    "img_src": [],
                    "ocr_line_indices": [2],
                },
            }
            tma_result = {
                "contract": contract,
                "results": [serialized_result],
                "items": [durable_item],
                "vision_item_count": 1,
                "fallback_item_count": 0,
                "provider_error": None,
                "provider_response_received": True,
                "provider_finish_reason": "stop",
                "provider_seconds": 0.01,
                "extraction_seconds": 0.02,
                "processed_image": {
                    "sha256": hashlib.sha256(processed_bytes).hexdigest(),
                    "width": processed_width,
                    "height": processed_height,
                },
            }
            write_json(
                evaluation / "references" / f"{document_id}.json",
                {
                    "document_id": document_id,
                    "image": {
                        "path": row["image"],
                        "sha256": image_sha256,
                        "original_name": "public-synthetic.ppm",
                    },
                    "ocr": ocr,
                    "annotation": {"public_synthetic_fixture": True},
                    "metadata": {"synthetic": True},
                    "source": {"kind": "public-synthetic"},
                    "validation_issues": [],
                },
            )
            write_json(
                evaluation / "predictions" / f"{document_id}.json",
                {
                    "prediction": {
                        "schema_version": "2.0",
                        "currency": None,
                        "sections": [],
                        "unsectioned_items": [],
                    },
                    "tma": tma_result,
                    "_evaluation": execution,
                },
            )
            write_json(
                root / call_path,
                {
                    "state": "succeeded",
                    "started_at": "2026-08-30T00:00:00Z",
                    "inputs": inputs,
                    "rate": {},
                    "cost": {},
                    "result_path": "tma.json",
                    "elapsed_seconds": 0.01,
                },
            )
            write_json(root / "baselines" / "tma-core" / cache_key / "tma.json", tma_result)
            write_json(
                root / "baselines" / "tma-core" / cache_key / "provider.raw.json",
                {
                    "id": f"chatcmpl-public-synthetic-{index}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-5.6-luna",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None,
                            "message": {
                                "role": "assistant",
                                "content": canonical_json(row["target"]),
                                "annotations": [],
                                "audio": None,
                                "refusal": None,
                                "tool_calls": None,
                                "function_call": None,
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
            documents.append(
                {
                    "document_id": document_id,
                    "schema_error": None,
                    "counts": {},
                    "metrics": {},
                    "matches": [],
                    "unmatched_gold": [],
                    "unmatched_predictions": [],
                    "latency_seconds": 0.01,
                    "execution": execution,
                }
            )
        baseline = {"contract": contract, "rate": {}, "unsupported_fields": []}
        metadata = {
            "run_id": run_id,
            "status": "complete",
            "model": "current-tma-core:gpt-5.6-luna",
            "created_at": "2026-08-30T00:00:00Z",
            "dataset_version": "synthetic-v1",
            "git_commit": "public-synthetic",
            "document_count": len(documents),
            "reference_kind": "public-synthetic",
            "metric_version": "public-synthetic-v1",
            "match_threshold": 0.5,
            "release": "public-synthetic",
            "release_caveats": [],
            "baseline": baseline,
        }
        write_json(
            evaluation / "report.json",
            {
                **metadata,
                "completed_document_count": len(documents),
                "metrics": {},
                "counts": {},
                "by_tag": {},
                "documents": documents,
                "worst_documents": [],
                "costs": {},
                "price_diagnostics": {},
            },
        )
        write_json(evaluation / "run.json", metadata)
        return {"output": str(root), "evaluation_run": run_id, "documents": len(documents)}
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _tma_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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
    image_relative = f"images/{document_id}.ppm"
    image_path = root / image_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    color = hashlib.sha256(document_id.encode()).digest()[:3]
    image_path.write_bytes(b"P6\n320 240\n255\n" + color * (320 * 240))
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
            "original_name": f"{document_id}.ppm",
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
