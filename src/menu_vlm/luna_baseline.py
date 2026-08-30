import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .compiler import validate_compiled_dataset
from .jsonio import (
    read_json,
    read_jsonl,
    safe_relative,
    sha256_file,
    write_jsonl,
    write_sha256_sidecar,
)
from .schemas import validate_compact_output

_CURRENT_LUNA_MODEL = "current-tma-core:gpt-5.6-luna"
_PROVIDER_MODEL = "gpt-5.6-luna"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def import_luna_baseline(
    dataset: Path,
    tma_data_root: Path,
    evaluation_run: str,
    output: Path,
) -> dict[str, Any]:
    """Import one completed current-TMA Luna run into the prediction contract."""
    validate_compiled_dataset(dataset)
    dataset_root = dataset.resolve()
    tma_root = tma_data_root.resolve()
    if not tma_root.is_dir():
        raise ValueError(f"TMA data root is not a directory: {tma_root}")
    if not _SAFE_ID.fullmatch(evaluation_run):
        raise ValueError("TMA evaluation run must be a safe identifier")
    destination = output.resolve()
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("Luna prediction output or checksum sidecar already exists")

    compiled_by_sha = _compiled_test_by_sha(dataset_root)
    evaluation_root = safe_relative(tma_root, f"evaluation/{evaluation_run}")
    expected_evaluation_root = tma_root / "evaluation" / evaluation_run
    if evaluation_root != expected_evaluation_root:
        raise ValueError("TMA evaluation path must not use a filesystem alias")
    if not evaluation_root.is_dir():
        raise ValueError(f"TMA evaluation run is not a directory: {evaluation_run}")
    document_ids = _validate_completed_run(evaluation_root, evaluation_run)
    references = _load_identity_files(
        safe_relative(evaluation_root, "references"), document_ids, "reference"
    )
    predictions = _load_identity_files(
        safe_relative(evaluation_root, "predictions"), document_ids, "prediction"
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

    rows: list[dict[str, Any]] = []
    for image_sha256, compiled in sorted(
        compiled_by_sha.items(), key=lambda item: str(item[1]["example_id"])
    ):
        document_id, _reference = reference_by_sha[image_sha256]
        compact, raw_text = _read_provider_compact_output(
            tma_root,
            predictions[document_id],
            document_id=document_id,
            image_sha256=image_sha256,
            ocr_line_count=int(compiled["ocr_line_count"]),
        )
        rows.append(
            {
                "example_id": str(compiled["example_id"]),
                "prediction": compact,
                "raw_output": raw_text,
            }
        )

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


def _validate_completed_run(evaluation_root: Path, evaluation_run: str) -> set[str]:
    report = read_json(safe_relative(evaluation_root, "report.json"))
    run = read_json(safe_relative(evaluation_root, "run.json"))
    if not isinstance(report, dict) or not isinstance(run, dict):
        raise ValueError("TMA evaluation metadata must be JSON objects")
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
    contract = baseline.get("contract") if isinstance(baseline, dict) else None
    if not isinstance(contract, dict) or contract.get("model") != _PROVIDER_MODEL:
        raise ValueError("TMA evaluation has mismatched Luna contract provenance")
    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("TMA evaluation report has no completed documents")
    document_ids = []
    for document in documents:
        document_id = document.get("document_id") if isinstance(document, dict) else None
        if not isinstance(document_id, str) or not _SAFE_ID.fullmatch(document_id):
            raise ValueError("TMA evaluation report has an invalid document identity")
        document_ids.append(document_id)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("TMA evaluation report has duplicate document identities")
    expected = len(document_ids)
    if (
        report.get("document_count") != expected
        or report.get("completed_document_count") != expected
    ):
        raise ValueError("TMA evaluation document accounting is incomplete")
    if run.get("document_count") != expected:
        raise ValueError("TMA evaluation run/report document accounting is mismatched")
    return set(document_ids)


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
        value = read_json(safe_relative(directory, f"{document_id}.json"))
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
    document_id: str,
    image_sha256: str,
    ocr_line_count: int,
) -> tuple[dict[str, Any], str]:
    evaluation = prediction.get("_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("state") != "succeeded":
        raise ValueError(f"TMA prediction is not a successful cached call: {document_id}")
    cache_key = evaluation.get("cache_key")
    call_path = evaluation.get("call_path")
    if not isinstance(cache_key, str) or not _DIGEST.fullmatch(cache_key):
        raise ValueError(f"TMA prediction has an invalid cache identity: {document_id}")
    expected = PurePosixPath("baselines", "tma-core", cache_key, "call.json")
    if not isinstance(call_path, str) or PurePosixPath(call_path) != expected:
        raise ValueError(f"TMA prediction has an unsafe or mismatched cache path: {document_id}")
    call_file = safe_relative(tma_root, call_path)
    if call_file != tma_root.joinpath(*expected.parts):
        raise ValueError(f"TMA call path must not use a filesystem alias: {document_id}")
    if not call_file.is_file():
        raise ValueError(f"TMA call record is missing: {document_id}")
    call = read_json(call_file)
    inputs = call.get("inputs") if isinstance(call, dict) else None
    if (
        not isinstance(inputs, dict)
        or call.get("state") != "succeeded"
        or inputs.get("image_sha256") != image_sha256
    ):
        raise ValueError(f"TMA call/image identity mismatch: {document_id}")
    raw_relative = expected.with_name("provider.raw.json")
    raw_file = safe_relative(tma_root, str(raw_relative))
    if raw_file != tma_root.joinpath(*raw_relative.parts):
        raise ValueError(f"TMA provider path must not use a filesystem alias: {document_id}")
    if not raw_file.is_file():
        raise ValueError(f"TMA provider response is missing: {document_id}")
    raw = read_json(raw_file)
    if not isinstance(raw, dict):
        raise ValueError(f"TMA provider response must be a JSON object: {document_id}")
    model = raw.get("model")
    if not isinstance(model, str) or not (
        model == _PROVIDER_MODEL or model.startswith(_PROVIDER_MODEL + "-")
    ):
        raise ValueError(f"TMA provider model identity mismatch: {document_id}")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(f"TMA provider response must have exactly one choice: {document_id}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if message is None or message.get("role") != "assistant" or not isinstance(content, str):
        raise ValueError(
            f"TMA provider response must have exactly one assistant text content: {document_id}"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"TMA assistant content is not compact JSON: {document_id}") from exc
    compact = validate_compact_output(payload, ocr_line_count=ocr_line_count)
    return compact.model_dump(by_alias=True), content
