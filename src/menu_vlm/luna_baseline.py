import hashlib
import json
import math
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlsplit

from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from .compiler import validate_compiled_dataset
from .jsonio import (
    canonical_json,
    read_json,
    read_jsonl,
    safe_relative,
    sha256_file,
    verify_sha256_sidecar,
    write_json,
    write_jsonl,
    write_sha256_sidecar,
)
from .prompt import render_user_prompt
from .schemas import OCRDocument, validate_compact_output
from .tma_image import process_tma_image

_CURRENT_LUNA_MODEL = "current-tma-core:gpt-5.6-luna"
_PROVIDER_MODEL = "gpt-5.6-luna"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SOURCE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_CONTRACT_KEYS = {
    "adapter_version",
    "model",
    "endpoint",
    "reasoning_effort",
    "image_detail",
    "max_completion_tokens",
    "timeout_seconds",
    "source_sha256",
    "loaded_source_paths",
    "adapter_source_sha256",
    "source_inventory",
    "scope",
    "excluded_stages",
    "provider_retries",
    "runtime_versions",
}
_CONTRACT_FIXED = {
    "adapter_version": "tma-core-extraction-v2",
    "model": _PROVIDER_MODEL,
    "endpoint": "https://api.openai.com/v1",
    "reasoning_effort": "none",
    "image_detail": "low",
    "max_completion_tokens": 4096,
    "timeout_seconds": 75,
    "provider_retries": 0,
}
_SOURCE_INVENTORY = (
    "all Python source files under backend/src, a deliberate safe superset of local modules "
    "loaded by the replayed one-page path; installed packages and native extensions are "
    "represented only by the listed runtime versions"
)
_SCOPE = (
    "Current local one-page TMA preprocessing, vision response materialization, OCR fallback "
    "and result serialization using cached OCR"
)
_EXCLUDED_STAGES = [
    "fresh OCR",
    "translation",
    "enrichment",
    "API/auth/queue",
    "refine pass",
]
_RUNTIME_VERSION_KEYS = {"openai", "pydantic", "Pillow"}
_TMA_RESULT_KEYS = {
    "contract",
    "results",
    "items",
    "vision_item_count",
    "fallback_item_count",
    "provider_error",
    "provider_response_received",
    "provider_finish_reason",
    "provider_seconds",
    "extraction_seconds",
    "processed_image",
}
_PROVIDER_REQUIRED_KEYS = {"id", "object", "created", "model", "choices", "usage"}
_PROVIDER_ALLOWED_KEYS = _PROVIDER_REQUIRED_KEYS | {
    "moderation",
    "service_tier",
    "system_fingerprint",
}
_CHOICE_KEYS = {"finish_reason", "index", "logprobs", "message"}
_MESSAGE_KEYS = {
    "annotations",
    "audio",
    "content",
    "function_call",
    "refusal",
    "role",
    "tool_calls",
}
_USAGE_REQUIRED_KEYS = {"completion_tokens", "prompt_tokens", "total_tokens"}
_USAGE_ALLOWED_KEYS = _USAGE_REQUIRED_KEYS | {
    "completion_tokens_details",
    "prompt_tokens_details",
}
_COMPLETION_TOKEN_DETAIL_KEYS = {
    "accepted_prediction_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
}
_PROMPT_TOKEN_DETAIL_KEYS = {"audio_tokens", "cache_write_tokens", "cached_tokens"}
_SERVICE_TIERS = {"auto", "default", "flex", "scale", "priority", "fast"}
_REPORT_KEYS = {
    "run_id",
    "model",
    "created_at",
    "dataset_version",
    "git_commit",
    "document_count",
    "reference_kind",
    "metric_version",
    "match_threshold",
    "release",
    "release_caveats",
    "baseline",
    "status",
    "completed_document_count",
    "metrics",
    "counts",
    "by_tag",
    "documents",
    "worst_documents",
    "costs",
    "price_diagnostics",
}
_RUN_KEYS = _REPORT_KEYS - {
    "completed_document_count",
    "metrics",
    "counts",
    "by_tag",
    "documents",
    "worst_documents",
    "costs",
    "price_diagnostics",
}
_REPORT_BASELINE_KEYS = {"contract", "rate", "unsupported_fields"}
_REPORT_DOCUMENT_KEYS = {
    "document_id",
    "schema_error",
    "counts",
    "metrics",
    "matches",
    "unmatched_gold",
    "unmatched_predictions",
    "latency_seconds",
    "execution",
}
_EXECUTION_KEYS = {
    "cache_key",
    "cached",
    "state",
    "cost",
    "call_path",
    "provider_seconds",
    "fallback_item_count",
    "provider_error",
    "provider_finish_reason",
    "original_call_seconds",
}
_PREDICTION_KEYS = {"prediction", "tma", "_evaluation"}
_CALL_KEYS = {"state", "started_at", "inputs", "rate", "cost", "result_path", "elapsed_seconds"}
_RESULT_KEYS = {"id", "info"}
_ITEM_KEYS = {
    "section_id",
    "notes",
    "information_only",
    "source_text",
    "locator_text",
    "name",
    "ocr_line_indices",
    "translation",
    "description",
    "price",
    "category",
    "page_index",
    "vision_index",
    "confidence",
    "is_ocr_fallback",
}
_INFO_BASE_KEYS = {
    "text",
    "text_translation",
    "description",
    "price",
    "price_info",
    "category",
    "confidence",
    "source_text",
    "locator_text",
    "page_index",
    "page_label",
    "locations",
    "img_src",
    "ocr_line_indices",
}
_NOTE_KEYS = {
    "id",
    "page_index",
    "original_text",
    "ocr_line_indices",
    "locations",
    "translation",
    "translation_language",
    "translation_status",
}
_NOTE_LOCATION_KEYS = {
    "page_index",
    "page_label",
    "text",
    "bounding_box",
    "score",
    "source",
}
_BOUNDING_BOX_KEYS = {"x", "y", "w", "h"}
_APPROVED_CONTRACT_RESOURCE = "current_tma_contract_v1.json"
_APPROVED_CONTRACT_SHA256 = "952e255374b961c90949566bc5de8fa92dff2198e7ccfc476a7c6fe1c0818cd6"
LUNA_RESPONSE_PROVENANCE_NAME = "luna-response-provenance.json"
LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME = "luna-response-provenance.json.sha256"
CANONICAL_LUNA_BASELINE_SHA256 = (
    "320bb246b6844901daa4f32b030d621400a9b0fd9ba8b925eba8b90ea9fa1683"
)
APPROVED_LUNA_RESPONSE_PROVENANCE_SHA256 = (
    "5546ac0106fb4841e7d0f8ad889404380b6ff3af8c000b29c7d0d432388cb62e"
)
APPROVED_SYNTHETIC_RESPONSE_PROVENANCE_SHA256 = (
    "e8423f8534df56db666da46b02d56517c8af8450510925c8be02c254857d7c2f"
)
APPROVED_SYNTHETIC_DATASET_SHA256 = (
    "de593b7ce8617ecfd67227f71ce0349dc1d4314cdb518a06770e8ccb3000d215"
)
_PROVENANCE_KEYS = {
    "format",
    "schema_version",
    "evaluation_run",
    "dataset_sha256",
    "baseline_prediction_sha256",
    "documents",
}
_PROVENANCE_DOCUMENT_KEYS = {
    "document_id",
    "example_id",
    "image_sha256",
    "cache_key",
    "call_path",
    "provider_response_path",
    "provider_response_sha256",
}
_METRIC_KEYS = {
    "calorie_accuracy",
    "description_association",
    "dish_f1",
    "dish_hallucination_rate",
    "dish_name_accuracy",
    "dish_note_f1",
    "dish_note_precision",
    "dish_note_recall",
    "dish_precision",
    "dish_price_association",
    "dish_price_value_association",
    "dish_recall",
    "entity_hallucination_rate",
    "note_f1",
    "note_precision",
    "note_recall",
    "ocr_provenance_coverage",
    "ocr_provenance_validity",
    "price_exact_accuracy",
    "price_precision",
    "schema_validity",
    "section_assignment",
    "section_f1",
    "section_note_f1",
    "section_note_precision",
    "section_note_recall",
    "section_precision",
    "section_recall",
    "variant_association",
}
_COUNT_KEYS = {
    "calorie_correct",
    "calorie_targets",
    "description_correct",
    "description_false_positive",
    "description_targets",
    "dish_name_correct",
    "documents",
    "evidenced_fields",
    "gold_dish_notes",
    "gold_dishes",
    "gold_notes",
    "gold_prices",
    "gold_section_notes",
    "gold_sections",
    "matched_dish_notes",
    "matched_dishes",
    "matched_entities",
    "matched_notes",
    "matched_priced_dishes",
    "matched_prices",
    "matched_section_notes",
    "matched_sections",
    "matched_variants",
    "predicted_calories",
    "predicted_descriptions",
    "predicted_dish_notes",
    "predicted_dishes",
    "predicted_entities",
    "predicted_notes",
    "predicted_prices",
    "predicted_section_notes",
    "predicted_sections",
    "predicted_variants",
    "price_correct",
    "price_false_positive",
    "price_issue_different_price_values_or_count",
    "price_issue_missing_prices",
    "price_targets",
    "price_value_correct",
    "references",
    "schema_valid",
    "section_correct",
    "textual_fields",
    "valid_references",
    "variant_correct",
    "variant_targets",
}


@dataclass(frozen=True)
class _CompletedRun:
    contract: dict[str, Any]
    rate: dict[str, Any]
    executions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class _ReplayContext:
    document_id: str
    image_sha256: str
    compiled: dict[str, Any]
    reference: dict[str, Any]
    prediction: dict[str, Any]
    source_image: Path
    provider_response_sha256: str


def approved_luna_runtime_contract() -> dict[str, Any]:
    """Load the immutable public trust anchor for the approved current-TMA run."""
    payload = files("menu_vlm").joinpath(_APPROVED_CONTRACT_RESOURCE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != _APPROVED_CONTRACT_SHA256:
        raise RuntimeError("Approved current-TMA contract artifact checksum mismatch")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("Approved current-TMA contract artifact must be a JSON object")
    return value


def approved_response_provenance_sha256(dataset_manifest: dict[str, Any]) -> str:
    if dataset_manifest.get("source_release_version") == "synthetic-v1":
        if dataset_manifest.get("dataset_sha256") != APPROVED_SYNTHETIC_DATASET_SHA256:
            raise ValueError("Synthetic response provenance requires the pinned synthetic dataset")
        return APPROVED_SYNTHETIC_RESPONSE_PROVENANCE_SHA256
    return APPROVED_LUNA_RESPONSE_PROVENANCE_SHA256


def approve_luna_response_provenance(
    dataset: Path,
    tma_data_root: Path,
    evaluation_run: str,
    canonical_baseline: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze exact provider-response bytes after reproducing the canonical baseline."""
    validate_compiled_dataset(dataset)
    dataset_root = dataset.resolve()
    dataset_manifest = read_json(dataset_root / "manifest.json")
    if dataset_manifest.get("source_release_version") == "synthetic-v1":
        raise ValueError("Production response approval does not accept the synthetic dataset")
    baseline = canonical_baseline.absolute()
    baseline_sidecar = baseline.with_suffix(baseline.suffix + ".sha256")
    if baseline.name != "luna-test-predictions.jsonl":
        raise ValueError("Canonical Luna baseline must use its reserved fixed filename")
    _require_unaliased_file(baseline, "canonical Luna baseline")
    _require_unaliased_file(baseline_sidecar, "canonical Luna baseline sidecar")
    baseline_check = verify_sha256_sidecar(baseline, baseline_sidecar)
    if baseline_check["sha256"] != CANONICAL_LUNA_BASELINE_SHA256:
        raise ValueError("Canonical Luna baseline does not match the independent approval digest")
    destination, sidecar = _provenance_pair(output, must_exist=False)
    reserved = {baseline.resolve(), baseline_sidecar.resolve(), destination, sidecar}
    if len(reserved) != 4:
        raise ValueError("Response provenance output collides with a reserved evidence file")
    completed, contexts, documents = _prepare_replay(dataset_root, tma_data_root, evaluation_run)
    rows = _replay_rows(tma_data_root.resolve(), completed, contexts, documents)
    reproduced = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    if hashlib.sha256(reproduced).hexdigest() != CANONICAL_LUNA_BASELINE_SHA256:
        raise ValueError("Approved raw responses do not reproduce the canonical Luna baseline")
    if baseline.read_bytes() != reproduced:
        raise ValueError("Canonical Luna baseline bytes differ from the approved replay")
    manifest = _response_provenance_manifest(
        evaluation_run,
        str(dataset_manifest["dataset_sha256"]),
        CANONICAL_LUNA_BASELINE_SHA256,
        documents,
    )
    write_json(destination, manifest)
    digest = write_sha256_sidecar(destination, sidecar)
    return {
        "valid": True,
        "documents": len(documents),
        "manifest_sha256": digest,
        "output": str(destination),
        "sidecar": str(sidecar),
    }


def write_synthetic_response_provenance(
    dataset: Path, tma_data_root: Path, evaluation_run: str
) -> dict[str, Any]:
    """Create deterministic synthetic-only response provenance for public smoke tests."""
    validate_compiled_dataset(dataset)
    dataset_root = dataset.resolve()
    dataset_manifest = read_json(dataset_root / "manifest.json")
    if dataset_manifest.get("source_release_version") != "synthetic-v1":
        raise ValueError("Synthetic response provenance requires the explicit synthetic dataset")
    destination, sidecar = _provenance_pair(
        tma_data_root / LUNA_RESPONSE_PROVENANCE_NAME, must_exist=False
    )
    completed, contexts, documents = _prepare_replay(dataset_root, tma_data_root, evaluation_run)
    rows = _replay_rows(tma_data_root.resolve(), completed, contexts, documents)
    reproduced = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    baseline_sha256 = hashlib.sha256(reproduced).hexdigest()
    manifest = _response_provenance_manifest(
        evaluation_run,
        str(dataset_manifest["dataset_sha256"]),
        baseline_sha256,
        documents,
    )
    write_json(destination, manifest)
    digest = write_sha256_sidecar(destination, sidecar)
    return {"manifest": str(destination), "sidecar": str(sidecar), "sha256": digest}


def import_luna_baseline(
    dataset: Path,
    tma_data_root: Path,
    evaluation_run: str,
    response_provenance: Path,
    response_provenance_sidecar: Path,
    output: Path,
) -> dict[str, Any]:
    """Import one completed current-TMA Luna run into the prediction contract."""
    validate_compiled_dataset(dataset)
    dataset_root = dataset.resolve()
    dataset_manifest = read_json(dataset_root / "manifest.json")
    tma_root = tma_data_root.resolve()
    if not tma_root.is_dir():
        raise ValueError(f"TMA data root is not a directory: {tma_root}")
    if not _SAFE_ID.fullmatch(evaluation_run):
        raise ValueError("TMA evaluation run must be a safe identifier")
    destination = output.resolve()
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("Luna prediction output or checksum sidecar already exists")
    provenance_path, provenance_sidecar = _provenance_pair(
        response_provenance, response_provenance_sidecar, must_exist=True
    )
    if len({destination, sidecar, provenance_path, provenance_sidecar}) != 4:
        raise ValueError("Luna output collides with reserved response provenance evidence")
    provenance = _load_response_provenance(
        provenance_path, provenance_sidecar, dataset_manifest, evaluation_run
    )

    completed, contexts, actual_documents = _prepare_replay(
        dataset_root, tma_root, evaluation_run
    )
    expected_documents = provenance["documents"]
    if actual_documents != expected_documents:
        raise ValueError("TMA response provenance does not match the exact replay inventory")
    rows = _replay_rows(tma_root, completed, contexts, expected_documents)

    try:
        write_jsonl(destination, rows)
        digest = write_sha256_sidecar(destination, sidecar)
    except Exception:
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return {
        "valid": True,
        "examples": len(rows),
        "prediction_sha256": digest,
        "output": str(destination),
        "sidecar": str(sidecar),
    }


def _prepare_replay(
    dataset_root: Path, tma_data_root: Path, evaluation_run: str
) -> tuple[_CompletedRun, list[_ReplayContext], list[dict[str, Any]]]:
    tma_root = tma_data_root.resolve()
    compiled_by_sha = _compiled_test_by_sha(dataset_root)
    evaluation_root = _exact_relative(
        tma_root, f"evaluation/{evaluation_run}", label="evaluation"
    )
    if not evaluation_root.is_dir():
        raise ValueError(f"TMA evaluation run is not a directory: {evaluation_run}")
    dataset_manifest = read_json(dataset_root / "manifest.json")
    completed = _validate_completed_run(
        evaluation_root,
        evaluation_run,
        synthetic_dataset=dataset_manifest.get("source_release_version") == "synthetic-v1",
    )
    document_ids = set(completed.executions)
    references = _load_identity_files(
        _exact_relative(evaluation_root, "references", label="reference directory"),
        document_ids,
        "reference",
    )
    predictions = _load_identity_files(
        _exact_relative(evaluation_root, "predictions", label="prediction directory"),
        document_ids,
        "prediction",
    )
    reference_by_sha: dict[str, tuple[str, dict[str, Any]]] = {}
    for document_id, reference in references.items():
        image = reference.get("image")
        digest = image.get("sha256") if isinstance(image, dict) else None
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError(f"TMA reference has an invalid image identity: {document_id}")
        if digest in reference_by_sha:
            raise ValueError(f"TMA references contain a duplicate image identity: {digest}")
        reference_by_sha[digest] = (document_id, reference)
    if set(reference_by_sha) != set(compiled_by_sha):
        missing = sorted(set(compiled_by_sha) - set(reference_by_sha))
        extra = sorted(set(reference_by_sha) - set(compiled_by_sha))
        raise ValueError(
            "TMA reference images do not exactly match compiled test images; "
            f"missing={missing}, extra={extra}"
        )
    contexts: list[_ReplayContext] = []
    documents: list[dict[str, Any]] = []
    for image_sha256, compiled in sorted(
        compiled_by_sha.items(), key=lambda item: str(item[1]["example_id"])
    ):
        document_id, reference = reference_by_sha[image_sha256]
        execution = completed.executions[document_id]
        cache_key = execution.get("cache_key")
        call_path = execution.get("call_path")
        if not isinstance(cache_key, str) or not _DIGEST.fullmatch(cache_key):
            raise ValueError(f"TMA replay has an invalid cache identity: {document_id}")
        expected_call = PurePosixPath("baselines", "tma-core", cache_key, "call.json")
        if not isinstance(call_path, str) or PurePosixPath(call_path) != expected_call:
            raise ValueError(f"TMA replay has an unsafe or mismatched cache path: {document_id}")
        provider_response_path = str(expected_call.with_name("provider.raw.json"))
        raw_file = _exact_relative(tma_root, provider_response_path, label="provider response")
        if not raw_file.is_file():
            raise ValueError(f"TMA provider response is missing: {document_id}")
        raw_sha256 = sha256_file(raw_file)
        contexts.append(
            _ReplayContext(
                document_id=document_id,
                image_sha256=image_sha256,
                compiled=compiled,
                reference=reference,
                prediction=predictions[document_id],
                source_image=safe_relative(dataset_root, str(compiled["image"])),
                provider_response_sha256=raw_sha256,
            )
        )
        documents.append(
            {
                "document_id": document_id,
                "example_id": str(compiled["example_id"]),
                "image_sha256": image_sha256,
                "cache_key": cache_key,
                "call_path": call_path,
                "provider_response_path": provider_response_path,
                "provider_response_sha256": raw_sha256,
            }
        )
    return completed, contexts, documents


def _replay_rows(
    tma_root: Path,
    completed: _CompletedRun,
    contexts: list[_ReplayContext],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(contexts) != len(documents):
        raise ValueError("TMA response provenance has incomplete replay accounting")
    rows: list[dict[str, Any]] = []
    for context, provenance in zip(contexts, documents, strict=True):
        compact, raw_text = _read_provider_compact_output(
            tma_root,
            context.prediction,
            reference=context.reference,
            compiled=context.compiled,
            report_contract=completed.contract,
            report_rate=completed.rate,
            report_execution=completed.executions[context.document_id],
            document_id=context.document_id,
            image_sha256=context.image_sha256,
            source_image=context.source_image,
            expected_raw_sha256=str(provenance["provider_response_sha256"]),
        )
        rows.append(
            {
                "example_id": str(context.compiled["example_id"]),
                "prediction": compact,
                "raw_output": raw_text,
            }
        )
    return rows


def _response_provenance_manifest(
    evaluation_run: str,
    dataset_sha256: str,
    baseline_prediction_sha256: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "tma-luna-response-provenance-v1",
        "schema_version": "1.0",
        "evaluation_run": evaluation_run,
        "dataset_sha256": dataset_sha256,
        "baseline_prediction_sha256": baseline_prediction_sha256,
        "documents": documents,
    }


def _load_response_provenance(
    path: Path,
    sidecar: Path,
    dataset_manifest: dict[str, Any],
    evaluation_run: str,
) -> dict[str, Any]:
    verified = verify_sha256_sidecar(path, sidecar)
    expected_sha256 = approved_response_provenance_sha256(dataset_manifest)
    if verified["sha256"] != expected_sha256:
        raise ValueError("Luna response provenance does not match the approved trust root")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Luna response provenance is not valid JSON") from exc
    if not isinstance(value, dict) or payload != (canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError("Luna response provenance is not canonical exact JSON")
    _validate_response_provenance(value, dataset_manifest, evaluation_run)
    return value


def _validate_response_provenance(
    value: dict[str, Any], dataset_manifest: dict[str, Any], evaluation_run: str
) -> None:
    synthetic = dataset_manifest.get("source_release_version") == "synthetic-v1"
    expected_count = 1 if synthetic else 3
    if (
        set(value) != _PROVENANCE_KEYS
        or value.get("format") != "tma-luna-response-provenance-v1"
        or value.get("schema_version") != "1.0"
        or value.get("evaluation_run") != evaluation_run
        or value.get("dataset_sha256") != dataset_manifest.get("dataset_sha256")
        or not isinstance(value.get("baseline_prediction_sha256"), str)
        or not _DIGEST.fullmatch(value["baseline_prediction_sha256"])
    ):
        raise ValueError("Luna response provenance has an invalid identity schema")
    if not synthetic and value["baseline_prediction_sha256"] != CANONICAL_LUNA_BASELINE_SHA256:
        raise ValueError("Luna response provenance has an unapproved baseline identity")
    documents = value.get("documents")
    if not isinstance(documents, list) or len(documents) != expected_count:
        raise ValueError("Luna response provenance has incorrect document accounting")
    document_ids: set[str] = set()
    example_ids: set[str] = set()
    image_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, dict) or set(document) != _PROVENANCE_DOCUMENT_KEYS:
            raise ValueError("Luna response provenance document has an invalid schema")
        document_id = document.get("document_id")
        example_id = document.get("example_id")
        image_sha256 = document.get("image_sha256")
        cache_key = document.get("cache_key")
        call_path = document.get("call_path")
        provider_path = document.get("provider_response_path")
        raw_sha256 = document.get("provider_response_sha256")
        expected_call = PurePosixPath("baselines", "tma-core", str(cache_key), "call.json")
        if (
            not isinstance(document_id, str)
            or not _SAFE_ID.fullmatch(document_id)
            or not isinstance(example_id, str)
            or not _SAFE_ID.fullmatch(example_id)
            or not isinstance(image_sha256, str)
            or not _DIGEST.fullmatch(image_sha256)
            or not isinstance(cache_key, str)
            or not _DIGEST.fullmatch(cache_key)
            or not isinstance(call_path, str)
            or PurePosixPath(call_path) != expected_call
            or not isinstance(provider_path, str)
            or PurePosixPath(provider_path) != expected_call.with_name("provider.raw.json")
            or not isinstance(raw_sha256, str)
            or not _DIGEST.fullmatch(raw_sha256)
        ):
            raise ValueError("Luna response provenance document has invalid evidence identity")
        if (
            document_id in document_ids
            or example_id in example_ids
            or image_sha256 in image_ids
        ):
            raise ValueError("Luna response provenance contains duplicate document identity")
        document_ids.add(document_id)
        example_ids.add(example_id)
        image_ids.add(image_sha256)
    if [document["example_id"] for document in documents] != sorted(example_ids):
        raise ValueError("Luna response provenance documents are not canonically ordered")


def _provenance_pair(
    path: Path, sidecar: Path | None = None, *, must_exist: bool
) -> tuple[Path, Path]:
    manifest_input = path.absolute()
    checksum_input = (
        sidecar.absolute()
        if sidecar is not None
        else manifest_input.with_name(LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME)
    )
    if (
        manifest_input.name != LUNA_RESPONSE_PROVENANCE_NAME
        or checksum_input.name != LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME
        or ".." in manifest_input.parts
        or ".." in checksum_input.parts
    ):
        raise ValueError("Luna response provenance must use the reserved fixed filenames")
    if manifest_input.is_symlink() or checksum_input.is_symlink():
        raise ValueError("Luna response provenance paths must not use filesystem aliases")
    manifest = manifest_input.resolve()
    checksum = checksum_input.resolve()
    if checksum != manifest.with_name(LUNA_RESPONSE_PROVENANCE_SIDECAR_NAME):
        raise ValueError("Luna response provenance must use one fixed-name evidence pair")
    if must_exist:
        _require_unaliased_file(manifest, "response provenance manifest")
        _require_unaliased_file(checksum, "response provenance sidecar")
    elif manifest.exists() or checksum.exists():
        raise FileExistsError("Luna response provenance manifest or sidecar already exists")
    return manifest, checksum


def _require_unaliased_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f"TMA {label} must be an unaliased regular file")


def _compiled_test_by_sha(dataset_root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(dataset_root / "test.jsonl")
    if not rows:
        raise ValueError("Compiled test split is empty")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        image = safe_relative(dataset_root, str(row.get("image", "")))
        if not image.is_file():
            raise ValueError(f"Compiled test image is missing: {example_id}")
        digest = sha256_file(image)
        if digest in result:
            raise ValueError(f"Compiled test contains a duplicate image identity: {digest}")
        result[digest] = row
    return result


def _validate_completed_run(
    evaluation_root: Path, evaluation_run: str, *, synthetic_dataset: bool
) -> _CompletedRun:
    report = read_json(_exact_relative(evaluation_root, "report.json", label="report"))
    run = read_json(_exact_relative(evaluation_root, "run.json", label="run metadata"))
    if not isinstance(report, dict) or not isinstance(run, dict):
        raise ValueError("TMA evaluation metadata must be JSON objects")
    if set(report) != _REPORT_KEYS or set(run) != _RUN_KEYS:
        raise ValueError("TMA evaluation report/run has an unexpected key inventory")
    synthetic = report.get("dataset_version") == "synthetic-v1"
    if (
        not isinstance(report.get("created_at"), str)
        or not report["created_at"].strip()
        or not isinstance(report.get("dataset_version"), str)
        or not report["dataset_version"].strip()
        or not _valid_metrics(report.get("metrics"), synthetic=synthetic)
        or not _valid_counts(report.get("counts"), synthetic=synthetic, document=False)
        or synthetic != synthetic_dataset
    ):
        raise ValueError("TMA evaluation report has malformed audited metadata")
    if (
        report.get("run_id") != evaluation_run
        or report.get("status") != "complete"
        or run.get("run_id") != evaluation_run
        or run.get("status") != "complete"
    ):
        raise ValueError("TMA evaluation must be complete and match its run identity")
    if report.get("model") != _CURRENT_LUNA_MODEL or run.get("model") != _CURRENT_LUNA_MODEL:
        raise ValueError("TMA evaluation is not the required current Luna model")
    baseline = report.get("baseline")
    run_baseline = run.get("baseline")
    if (
        not isinstance(baseline, dict)
        or set(baseline) != _REPORT_BASELINE_KEYS
        or not isinstance(run_baseline, dict)
        or set(run_baseline) != _REPORT_BASELINE_KEYS
    ):
        raise ValueError("TMA evaluation has malformed baseline provenance")
    contract = baseline.get("contract") if isinstance(baseline, dict) else None
    run_contract = run_baseline.get("contract") if isinstance(run_baseline, dict) else None
    if not isinstance(contract, dict) or not isinstance(run_contract, dict):
        raise ValueError("TMA evaluation has mismatched Luna contract provenance")
    _validate_runtime_contract(contract)
    _validate_runtime_contract(run_contract)
    if baseline != run_baseline:
        raise ValueError("TMA evaluation has mismatched Luna contract provenance")
    rate = baseline.get("rate")
    if not isinstance(rate, dict):
        raise ValueError("TMA evaluation has malformed rate provenance")
    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("TMA evaluation report has no completed documents")
    executions: dict[str, dict[str, Any]] = {}
    for document in documents:
        if not isinstance(document, dict) or set(document) != _REPORT_DOCUMENT_KEYS:
            raise ValueError("TMA evaluation report document has an unexpected key inventory")
        document_id = document.get("document_id") if isinstance(document, dict) else None
        if not isinstance(document_id, str) or not _SAFE_ID.fullmatch(document_id):
            raise ValueError("TMA evaluation report has an invalid document identity")
        if document_id in executions:
            raise ValueError("TMA evaluation report has duplicate document identities")
        if (
            document.get("schema_error") is not None
            or not _valid_metrics(document.get("metrics"), synthetic=synthetic)
            or not _valid_counts(document.get("counts"), synthetic=synthetic, document=True)
        ):
            raise ValueError(f"TMA evaluation report document metrics are malformed: {document_id}")
        execution = document.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"TMA report execution is missing: {document_id}")
        if set(execution) != _EXECUTION_KEYS or not _valid_execution_fields(execution):
            raise ValueError(f"TMA report execution is malformed: {document_id}")
        if (
            execution.get("state") != "succeeded"
            or execution.get("provider_error") is not None
            or execution.get("provider_finish_reason") != "stop"
        ):
            raise ValueError(f"TMA report execution is not terminal-success: {document_id}")
        if not _is_nonnegative_number(document.get("latency_seconds")):
            raise ValueError(f"TMA report latency is malformed: {document_id}")
        executions[document_id] = execution
    expected = len(executions)
    if (
        not _is_nonnegative_int(report.get("document_count"))
        or not _is_nonnegative_int(report.get("completed_document_count"))
        or report.get("document_count") != expected
        or report.get("completed_document_count") != expected
    ):
        raise ValueError("TMA evaluation document accounting is incomplete")
    if not _is_nonnegative_int(run.get("document_count")) or run.get("document_count") != expected:
        raise ValueError("TMA evaluation run/report document accounting is mismatched")
    if any(report.get(key) != run.get(key) for key in _RUN_KEYS):
        raise ValueError("TMA evaluation report/run shared fields are mismatched")
    return _CompletedRun(contract=contract, rate=rate, executions=executions)


def _load_identity_files(
    directory: Path, expected: set[str], label: str
) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise ValueError(f"TMA evaluation {label} directory is missing")
    files = list(directory.glob("*.json"))
    names = {path.stem for path in files}
    if names != expected or len(files) != len(expected):
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise ValueError(
            f"TMA evaluation {label} identities do not exactly match the report; "
            f"missing={missing}, extra={extra}"
        )
    result = {}
    for document_id in sorted(expected):
        value = read_json(
            _exact_relative(directory, f"{document_id}.json", label=f"{label} file")
        )
        if not isinstance(value, dict):
            raise ValueError(f"TMA {label} must be a JSON object: {document_id}")
        if label == "reference" and value.get("document_id") != document_id:
            raise ValueError(f"TMA reference document identity mismatch: {document_id}")
        result[document_id] = value
    return result


def _read_provider_compact_output(
    tma_root: Path,
    prediction: dict[str, Any],
    *,
    reference: dict[str, Any],
    compiled: dict[str, Any],
    report_contract: dict[str, Any],
    report_rate: dict[str, Any],
    report_execution: dict[str, Any],
    document_id: str,
    image_sha256: str,
    source_image: Path,
    expected_raw_sha256: str,
) -> tuple[dict[str, Any], str]:
    if set(prediction) != _PREDICTION_KEYS:
        raise ValueError(f"TMA prediction has an unexpected key inventory: {document_id}")
    evaluation = prediction.get("_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("state") != "succeeded":
        raise ValueError(f"TMA prediction is not a successful cached call: {document_id}")
    if set(evaluation) != _EXECUTION_KEYS or not _valid_execution_fields(evaluation):
        raise ValueError(f"TMA prediction execution is malformed: {document_id}")
    if report_execution != evaluation:
        raise ValueError(f"TMA report/prediction execution identity mismatch: {document_id}")
    if evaluation.get("provider_error") is not None:
        raise ValueError(f"TMA prediction records a provider error: {document_id}")
    if evaluation.get("provider_finish_reason") != "stop":
        raise ValueError(f"TMA prediction has a non-terminal finish reason: {document_id}")
    ocr, ocr_line_count = _validate_reference_ocr(reference, compiled, document_id)
    cache_key = evaluation.get("cache_key")
    call_path = evaluation.get("call_path")
    if not isinstance(cache_key, str) or not _DIGEST.fullmatch(cache_key):
        raise ValueError(f"TMA prediction has an invalid cache identity: {document_id}")
    expected = PurePosixPath("baselines", "tma-core", cache_key, "call.json")
    if not isinstance(call_path, str) or PurePosixPath(call_path) != expected:
        raise ValueError(f"TMA prediction has an unsafe or mismatched cache path: {document_id}")
    call_file = _exact_relative(tma_root, call_path, label="call")
    if not call_file.is_file():
        raise ValueError(f"TMA call record is missing: {document_id}")
    call = read_json(call_file)
    if (
        not isinstance(call, dict)
        or set(call) != _CALL_KEYS
        or not _valid_call_fields(call)
    ):
        raise ValueError(f"TMA call record is malformed: {document_id}")
    inputs = call.get("inputs") if isinstance(call, dict) else None
    if not isinstance(inputs, dict) or set(inputs) != {
        "contract",
        "image_sha256",
        "ocr_sha256",
    }:
        raise ValueError(f"TMA call inputs are malformed: {document_id}")
    if call.get("state") != "succeeded" or call.get("result_path") != "tma.json":
        raise ValueError(f"TMA call is not a completed extraction: {document_id}")
    if (
        call.get("cost") != evaluation.get("cost")
        or call.get("rate") != report_rate
        or call.get("elapsed_seconds") != evaluation.get("original_call_seconds")
    ):
        raise ValueError(f"TMA call execution evidence is mismatched: {document_id}")
    if inputs.get("contract") != report_contract:
        raise ValueError(f"TMA call/report contract mismatch: {document_id}")
    if inputs.get("image_sha256") != image_sha256:
        raise ValueError(f"TMA call/image identity mismatch: {document_id}")
    if inputs.get("ocr_sha256") != _tma_digest(ocr):
        raise ValueError(f"TMA call/OCR identity mismatch: {document_id}")
    if _tma_digest(inputs) != cache_key:
        raise ValueError(f"TMA cache digest does not match exact call inputs: {document_id}")
    result_relative = expected.with_name("tma.json")
    result_file = _exact_relative(tma_root, str(result_relative), label="TMA result")
    if not result_file.is_file():
        raise ValueError(f"TMA materialized result is missing: {document_id}")
    result = read_json(result_file)
    _validate_materialized_result(
        result,
        report_contract,
        document_id,
        source_image=source_image,
        ocr_line_count=ocr_line_count,
    )
    if (
        result.get("fallback_item_count") != evaluation.get("fallback_item_count")
        or result.get("provider_seconds") != evaluation.get("provider_seconds")
    ):
        raise ValueError(f"TMA materialized execution evidence is mismatched: {document_id}")
    if prediction.get("tma") != result:
        raise ValueError(f"TMA prediction is not bound to its materialized result: {document_id}")
    raw_relative = expected.with_name("provider.raw.json")
    raw_file = _exact_relative(tma_root, str(raw_relative), label="provider response")
    if not raw_file.is_file():
        raise ValueError(f"TMA provider response is missing: {document_id}")
    if sha256_file(raw_file) != expected_raw_sha256:
        raise ValueError(f"TMA provider response bytes are not approved: {document_id}")
    raw = read_json(raw_file)
    if not isinstance(raw, dict):
        raise ValueError(f"TMA provider response must be a JSON object: {document_id}")
    if "error" in raw:
        raise ValueError(f"TMA provider response contains a top-level error: {document_id}")
    raw_keys = set(raw)
    if (
        not _PROVIDER_REQUIRED_KEYS.issubset(raw_keys)
        or not raw_keys.issubset(_PROVIDER_ALLOWED_KEYS)
        or raw.get("object") != "chat.completion"
        or not isinstance(raw.get("id"), str)
        or not raw["id"].strip()
        or not _is_nonnegative_int(raw.get("created"))
    ):
        raise ValueError(f"TMA provider response has an invalid completion envelope: {document_id}")
    model = raw.get("model")
    if model != _PROVIDER_MODEL:
        raise ValueError(f"TMA provider model identity mismatch: {document_id}")
    if not _valid_optional_provider_metadata(raw):
        raise ValueError(f"TMA provider response has malformed optional metadata: {document_id}")
    _validate_provider_usage(raw.get("usage"), document_id)
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(f"TMA provider response must have exactly one choice: {document_id}")
    choice = choices[0]
    if (
        not isinstance(choice, dict)
        or set(choice) != _CHOICE_KEYS
        or not _is_nonnegative_int(choice.get("index"))
        or choice.get("index") != 0
        or choice.get("logprobs") is not None
        or choice.get("finish_reason") != "stop"
    ):
        raise ValueError(f"TMA provider response did not finish with stop: {document_id}")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if (
        not isinstance(message, dict)
        or set(message) != _MESSAGE_KEYS
        or message.get("role") != "assistant"
        or not isinstance(content, str)
        or not _valid_annotations(message.get("annotations"))
        or message.get("audio") is not None
    ):
        raise ValueError(
            f"TMA provider response must have exactly one assistant text content: {document_id}"
        )
    if any(message.get(key) is not None for key in ("refusal", "tool_calls", "function_call")):
        raise ValueError(f"TMA assistant response contains a refusal or tool call: {document_id}")
    try:
        ChatCompletion.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise ValueError(
            f"TMA provider response violates the approved OpenAI schema: {document_id}"
        ) from exc
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"TMA assistant content is not compact JSON: {document_id}") from exc
    compact = validate_compact_output(payload, ocr_line_count=ocr_line_count)
    return compact.model_dump(by_alias=True), content


def _validate_reference_ocr(
    reference: dict[str, Any], compiled: dict[str, Any], document_id: str
) -> tuple[dict[str, Any], int]:
    raw_ocr = reference.get("ocr")
    if not isinstance(raw_ocr, dict):
        raise ValueError(f"TMA reference OCR is missing: {document_id}")
    ocr = OCRDocument.model_validate(raw_ocr, strict=True)
    count = compiled.get("ocr_line_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(ocr.spans):
        raise ValueError(f"TMA reference/compiled OCR count mismatch: {document_id}")
    messages = compiled.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Compiled conversation shape is invalid: {document_id}")
    user = messages[1]
    content = user.get("content") if isinstance(user, dict) else None
    if (
        not isinstance(user, dict)
        or user.get("role") != "user"
        or not isinstance(content, list)
        or len(content) != 2
        or not isinstance(content[0], dict)
        or not isinstance(content[1], dict)
        or content[0].get("type") != "text"
        or content[1] != {"type": "image"}
    ):
        raise ValueError(f"Compiled user prompt shape is invalid: {document_id}")
    expected = render_user_prompt(page_index=0, ocr_lines=[span.text for span in ocr.spans])
    if content[0].get("text") != expected:
        raise ValueError(f"TMA reference OCR does not match compiled prompt order: {document_id}")
    return raw_ocr, count


def _validate_runtime_contract(contract: dict[str, Any]) -> None:
    if set(contract) != _CONTRACT_KEYS:
        raise ValueError("TMA runtime contract has an unexpected key inventory")
    if any(contract.get(key) != value for key, value in _CONTRACT_FIXED.items()):
        raise ValueError("TMA runtime contract has unexpected critical settings")
    if any(
        not isinstance(contract.get(key), int) or isinstance(contract.get(key), bool)
        for key in ("max_completion_tokens", "timeout_seconds", "provider_retries")
    ):
        raise ValueError("TMA runtime contract has incorrectly typed critical settings")
    if contract.get("source_inventory") != _SOURCE_INVENTORY or contract.get("scope") != _SCOPE:
        raise ValueError("TMA runtime contract has unexpected source or scope provenance")
    if contract.get("excluded_stages") != _EXCLUDED_STAGES:
        raise ValueError("TMA runtime contract has unexpected excluded stages")
    source_sha256 = contract.get("source_sha256")
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise ValueError("TMA runtime contract has no source hash inventory")
    for path, digest in source_sha256.items():
        if (
            not _is_safe_source_path(path)
            or not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
        ):
            raise ValueError("TMA runtime contract has an invalid source hash inventory")
    loaded = contract.get("loaded_source_paths")
    if (
        not isinstance(loaded, list)
        or not loaded
        or any(not isinstance(path, str) for path in loaded)
        or len(loaded) != len(set(loaded))
        or not set(loaded).issubset(source_sha256)
    ):
        raise ValueError("TMA runtime contract has invalid loaded source provenance")
    adapter_digest = contract.get("adapter_source_sha256")
    if not isinstance(adapter_digest, str) or not _DIGEST.fullmatch(adapter_digest):
        raise ValueError("TMA runtime contract has an invalid adapter source hash")
    versions = contract.get("runtime_versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != _RUNTIME_VERSION_KEYS
        or any(not isinstance(value, str) or not value.strip() for value in versions.values())
    ):
        raise ValueError("TMA runtime contract has invalid runtime versions")
    if contract != approved_luna_runtime_contract():
        raise ValueError("TMA runtime contract does not match the approved provenance anchor")


def _validate_materialized_result(
    result: Any,
    report_contract: dict[str, Any],
    document_id: str,
    *,
    source_image: Path,
    ocr_line_count: int,
) -> None:
    if not isinstance(result, dict) or set(result) != _TMA_RESULT_KEYS:
        raise ValueError(f"TMA materialized result has an invalid key inventory: {document_id}")
    if result.get("contract") != report_contract:
        raise ValueError(f"TMA materialized result contract mismatch: {document_id}")
    results = result.get("results")
    items = result.get("items")
    if (
        not isinstance(results, list)
        or not isinstance(items, list)
        or any(not isinstance(value, dict) for value in results)
        or any(not isinstance(value, dict) for value in items)
    ):
        raise ValueError(f"TMA materialized result collections are malformed: {document_id}")
    vision_count = result.get("vision_item_count")
    fallback_count = result.get("fallback_item_count")
    if not _is_nonnegative_int(vision_count) or not _is_nonnegative_int(fallback_count):
        raise ValueError(f"TMA materialized result counts are malformed: {document_id}")
    total_count = cast(int, vision_count) + cast(int, fallback_count)
    if len(results) != len(items) or len(items) != total_count:
        raise ValueError(f"TMA materialized result accounting is inconsistent: {document_id}")
    for index, (serialized, item) in enumerate(zip(results, items, strict=True)):
        _validate_tma_item(item, ocr_line_count, document_id)
        _validate_tma_result_row(serialized, item, index, ocr_line_count, document_id)
    observed_fallback = sum(item["is_ocr_fallback"] is True for item in items)
    if observed_fallback != fallback_count or len(items) - observed_fallback != vision_count:
        raise ValueError(f"TMA materialized result item accounting is inconsistent: {document_id}")
    if (
        result.get("provider_error") is not None
        or result.get("provider_response_received") is not True
        or result.get("provider_finish_reason") != "stop"
    ):
        raise ValueError(
            f"TMA materialized result provider evidence is contradictory: {document_id}"
        )
    if not _is_nonnegative_number(result.get("provider_seconds")) or not _is_nonnegative_number(
        result.get("extraction_seconds")
    ):
        raise ValueError(f"TMA materialized result timings are malformed: {document_id}")
    processed = result.get("processed_image")
    if (
        not isinstance(processed, dict)
        or set(processed) != {"sha256", "width", "height"}
        or not isinstance(processed.get("sha256"), str)
        or not _DIGEST.fullmatch(processed["sha256"])
        or not _is_positive_int(processed.get("width"))
        or not _is_positive_int(processed.get("height"))
    ):
        raise ValueError(f"TMA materialized processed image evidence is malformed: {document_id}")
    processed_bytes, expected_height, expected_width = process_tma_image(source_image.read_bytes())
    expected_processed = {
        "sha256": hashlib.sha256(processed_bytes).hexdigest(),
        "width": expected_width,
        "height": expected_height,
    }
    if processed != expected_processed:
        raise ValueError(
            f"TMA materialized processed image does not match source processing: {document_id}"
        )


def _validate_tma_item(item: dict[str, Any], count: int, document_id: str) -> None:
    if set(item) != _ITEM_KEYS:
        raise ValueError(f"TMA materialized item has an invalid key inventory: {document_id}")
    nullable_strings = (
        "section_id",
        "locator_text",
        "translation",
        "description",
        "price",
        "category",
    )
    if any(
        item.get(key) is not None and not isinstance(item.get(key), str)
        for key in nullable_strings
    ):
        raise ValueError(f"TMA materialized item has malformed text fields: {document_id}")
    if (
        not isinstance(item.get("name"), str)
        or not item["name"].strip()
        or not isinstance(item.get("source_text"), str)
        or not isinstance(item.get("information_only"), bool)
        or not isinstance(item.get("is_ocr_fallback"), bool)
        or not _is_exact_zero_int(item.get("page_index"))
        or not _valid_confidence(item.get("confidence"))
        or not _valid_ocr_indices(item.get("ocr_line_indices"), count)
        or not _valid_notes(item.get("notes"), count)
    ):
        raise ValueError(f"TMA materialized item has malformed field types: {document_id}")
    vision_index = item.get("vision_index")
    if vision_index is not None and not _is_nonnegative_int(vision_index):
        raise ValueError(f"TMA materialized item has malformed vision identity: {document_id}")
    if item["is_ocr_fallback"] is (vision_index is not None):
        raise ValueError(
            f"TMA materialized item has contradictory fallback identity: {document_id}"
        )


def _validate_tma_result_row(
    row: dict[str, Any], item: dict[str, Any], index: int, count: int, document_id: str
) -> None:
    if (
        set(row) != _RESULT_KEYS
        or not _is_nonnegative_int(row.get("id"))
        or row.get("id") != index
        or not isinstance(row.get("info"), dict)
    ):
        raise ValueError(f"TMA materialized result row has an invalid inventory: {document_id}")
    info = row["info"]
    keys = set(info)
    if not _INFO_BASE_KEYS.issubset(keys) or not keys.issubset(
        _INFO_BASE_KEYS | {"section_id", "notes"}
    ):
        raise ValueError(f"TMA materialized result info has an invalid inventory: {document_id}")
    section_id = item["section_id"]
    notes = item["notes"]
    if (info.get("section_id") if "section_id" in info else None) != section_id:
        raise ValueError(f"TMA materialized result section binding is invalid: {document_id}")
    if (info.get("notes") if "notes" in info else []) != _serialize_tma_notes(notes):
        raise ValueError(f"TMA materialized result note binding is invalid: {document_id}")
    if not _is_exact_zero_int(info.get("page_index")):
        raise ValueError(f"TMA materialized result page identity is invalid: {document_id}")
    expected = {
        "text": item["name"],
        "text_translation": item["translation"] or item["name"],
        "description": item["description"],
        "price": item["price"],
        "category": item["category"],
        "confidence": item["confidence"],
        "source_text": item["source_text"],
        "locator_text": item["locator_text"] or item["source_text"] or item["name"],
        "page_index": item["page_index"],
        "page_label": "Page 1",
        "locations": [],
        "img_src": [],
        "ocr_line_indices": item["ocr_line_indices"],
    }
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError(f"TMA materialized result/item binding is invalid: {document_id}")
    if not _valid_price_info(info.get("price_info"), item["price"]):
        raise ValueError(f"TMA materialized result price info is malformed: {document_id}")
    if not _valid_ocr_indices(info.get("ocr_line_indices"), count):
        raise ValueError(f"TMA materialized result OCR identity is malformed: {document_id}")


def _valid_price_info(value: Any, raw_price: Any) -> bool:
    if raw_price is None:
        return value is None
    if not isinstance(value, dict):
        return False
    keys = set(value)
    if keys not in (
        {"raw", "source", "confidence"},
        {"raw", "amount", "currency", "source", "confidence"},
    ):
        return False
    if value.get("raw") != raw_price or not isinstance(value.get("source"), str):
        return False
    if not _valid_confidence(value.get("confidence")):
        return False
    if "amount" in value and not _is_nonnegative_number(value.get("amount")):
        return False
    return (
        "currency" not in value
        or value.get("currency") is None
        or isinstance(value["currency"], str)
    )


def _valid_notes(value: Any, count: int) -> bool:
    if not isinstance(value, list):
        return False
    for note in value:
        if not isinstance(note, dict) or set(note) != _NOTE_KEYS:
            return False
        if (
            not isinstance(note.get("id"), str)
            or not note["id"]
            or not _is_exact_zero_int(note.get("page_index"))
            or not isinstance(note.get("original_text"), str)
            or not note["original_text"]
            or not _valid_ocr_indices(note.get("ocr_line_indices"), count)
            or not _valid_note_locations(note.get("locations"))
            or (
                note.get("translation") is not None
                and not isinstance(note.get("translation"), str)
            )
            or (
                note.get("translation_language") is not None
                and not isinstance(note.get("translation_language"), str)
            )
            or note.get("translation_status") not in {"pending", "translated", "unavailable"}
        ):
            return False
    return True


def _valid_note_locations(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for location in value:
        if not isinstance(location, dict) or set(location) != _NOTE_LOCATION_KEYS:
            return False
        box = location.get("bounding_box")
        if (
            not _is_exact_zero_int(location.get("page_index"))
            or location.get("page_label") != "Page 1"
            or not isinstance(location.get("text"), str)
            or not location["text"]
            or location.get("source") != "ocr_reference"
            or not _valid_confidence(location.get("score"))
            or not isinstance(box, dict)
            or set(box) != _BOUNDING_BOX_KEYS
            or any(not _valid_unit_number(box.get(key)) for key in _BOUNDING_BOX_KEYS)
        ):
            return False
    return True


def _serialize_tma_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for note in notes:
        locations = []
        for location in note["locations"]:
            locations.append(
                {
                    key: value
                    for key, value in location.items()
                    if key != "bounding_box"
                }
                | {"boundingBox": location["bounding_box"]}
            )
        serialized.append(note | {"locations": locations})
    return serialized


def _valid_ocr_indices(value: Any, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(set(value))
        and all(_is_positive_int(index) and index <= count for index in value)
    )


def _valid_confidence(value: Any) -> bool:
    return _is_nonnegative_number(value) and value <= 1


def _valid_unit_number(value: Any) -> bool:
    return _is_nonnegative_number(value) and value <= 1


def _is_safe_source_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and len(path.parts) >= 2
        and path.parts[0] == "src"
        and path.suffix == ".py"
        and all(
            part not in {"", ".", ".."} and _SAFE_SOURCE_PART.fullmatch(part)
            for part in path.parts
        )
    )


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return _is_nonnegative_int(value) and value > 0


def _is_exact_zero_int(value: Any) -> bool:
    return _is_nonnegative_int(value) and value == 0


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _valid_metrics(value: Any, *, synthetic: bool) -> bool:
    if not isinstance(value, dict):
        return False
    if synthetic:
        return value == {}
    return set(value) == _METRIC_KEYS and all(
        item is None or (_is_nonnegative_number(item) and item <= 1)
        for item in value.values()
    )


def _valid_counts(value: Any, *, synthetic: bool, document: bool) -> bool:
    if not isinstance(value, dict):
        return False
    if synthetic:
        return value == {}
    keys = set(value)
    allowed = (_COUNT_KEYS, _COUNT_KEYS - {"price_issue_missing_prices"})
    if (document and keys not in allowed) or (not document and keys != _COUNT_KEYS):
        return False
    return all(_is_nonnegative_int(item) for item in value.values())


def _valid_optional_provider_metadata(raw: dict[str, Any]) -> bool:
    moderation = raw.get("moderation")
    service_tier = raw.get("service_tier")
    fingerprint = raw.get("system_fingerprint")
    return (
        moderation is None
        and (
            service_tier is None
            or (isinstance(service_tier, str) and service_tier in _SERVICE_TIERS)
        )
        and (fingerprint is None or isinstance(fingerprint, str))
    )


def _valid_execution_fields(value: dict[str, Any]) -> bool:
    return (
        isinstance(value.get("cached"), bool)
        and isinstance(value.get("cost"), dict)
        and _is_nonnegative_number(value.get("provider_seconds"))
        and _is_nonnegative_int(value.get("fallback_item_count"))
        and _is_nonnegative_number(value.get("original_call_seconds"))
    )


def _valid_call_fields(value: dict[str, Any]) -> bool:
    return (
        isinstance(value.get("started_at"), str)
        and bool(value["started_at"].strip())
        and isinstance(value.get("rate"), dict)
        and isinstance(value.get("cost"), dict)
        and _is_nonnegative_number(value.get("elapsed_seconds"))
    )


def _valid_annotations(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for annotation in value:
        if not isinstance(annotation, dict) or set(annotation) != {"type", "url_citation"}:
            return False
        citation = annotation.get("url_citation")
        if annotation.get("type") != "url_citation" or not isinstance(citation, dict):
            return False
        if set(citation) != {"start_index", "end_index", "title", "url"}:
            return False
        start = citation.get("start_index")
        end = citation.get("end_index")
        title = citation.get("title")
        url = citation.get("url")
        if (
            not _is_nonnegative_int(start)
            or not _is_nonnegative_int(end)
            or cast(int, end) < cast(int, start)
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(url, str)
        ):
            return False
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
    return True


def _validate_provider_usage(value: Any, document_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"TMA provider response has malformed token usage: {document_id}")
    keys = set(value)
    if not _USAGE_REQUIRED_KEYS.issubset(keys) or not keys.issubset(_USAGE_ALLOWED_KEYS):
        raise ValueError(f"TMA provider response has malformed token usage: {document_id}")
    if any(not _is_nonnegative_int(value.get(key)) for key in _USAGE_REQUIRED_KEYS):
        raise ValueError(f"TMA provider response has malformed token usage: {document_id}")
    if value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
        raise ValueError(f"TMA provider response has inconsistent token usage: {document_id}")
    _validate_token_details(
        value.get("completion_tokens_details"), _COMPLETION_TOKEN_DETAIL_KEYS, document_id
    )
    _validate_token_details(
        value.get("prompt_tokens_details"), _PROMPT_TOKEN_DETAIL_KEYS, document_id
    )


def _validate_token_details(value: Any, allowed: set[str], document_id: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not set(value).issubset(allowed):
        raise ValueError(f"TMA provider response has malformed token usage: {document_id}")
    if any(item is not None and not _is_nonnegative_int(item) for item in value.values()):
        raise ValueError(f"TMA provider response has malformed token usage: {document_id}")


def _tma_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _exact_relative(root: Path, relative: str, *, label: str) -> Path:
    target = safe_relative(root, relative)
    expected = root / relative
    if target != expected:
        raise ValueError(f"TMA {label} path must not use a filesystem alias")
    return target
