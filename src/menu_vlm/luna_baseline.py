import hashlib
import json
import re
from dataclasses import dataclass
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
from .prompt import render_user_prompt
from .schemas import OCRDocument, validate_compact_output

_CURRENT_LUNA_MODEL = "current-tma-core:gpt-5.6-luna"
_PROVIDER_MODEL = "gpt-5.6-luna"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _CompletedRun:
    contract: dict[str, Any]
    executions: dict[str, dict[str, Any]]


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
    evaluation_root = _exact_relative(
        tma_root, f"evaluation/{evaluation_run}", label="evaluation"
    )
    if not evaluation_root.is_dir():
        raise ValueError(f"TMA evaluation run is not a directory: {evaluation_run}")
    completed = _validate_completed_run(evaluation_root, evaluation_run)
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

    rows: list[dict[str, Any]] = []
    for image_sha256, compiled in sorted(
        compiled_by_sha.items(), key=lambda item: str(item[1]["example_id"])
    ):
        document_id, reference = reference_by_sha[image_sha256]
        compact, raw_text = _read_provider_compact_output(
            tma_root,
            predictions[document_id],
            reference=reference,
            compiled=compiled,
            report_contract=completed.contract,
            report_execution=completed.executions[document_id],
            document_id=document_id,
            image_sha256=image_sha256,
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


def _validate_completed_run(evaluation_root: Path, evaluation_run: str) -> _CompletedRun:
    report = read_json(_exact_relative(evaluation_root, "report.json", label="report"))
    run = read_json(_exact_relative(evaluation_root, "run.json", label="run metadata"))
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
    executions: dict[str, dict[str, Any]] = {}
    for document in documents:
        document_id = document.get("document_id") if isinstance(document, dict) else None
        if not isinstance(document_id, str) or not _SAFE_ID.fullmatch(document_id):
            raise ValueError("TMA evaluation report has an invalid document identity")
        if document_id in executions:
            raise ValueError("TMA evaluation report has duplicate document identities")
        execution = document.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"TMA report execution is missing: {document_id}")
        if (
            execution.get("state") != "succeeded"
            or execution.get("provider_error") is not None
            or execution.get("provider_finish_reason") != "stop"
        ):
            raise ValueError(f"TMA report execution is not terminal-success: {document_id}")
        executions[document_id] = execution
    expected = len(executions)
    if (
        report.get("document_count") != expected
        or report.get("completed_document_count") != expected
    ):
        raise ValueError("TMA evaluation document accounting is incomplete")
    if run.get("document_count") != expected:
        raise ValueError("TMA evaluation run/report document accounting is mismatched")
    return _CompletedRun(contract=contract, executions=executions)


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
    report_execution: dict[str, Any],
    document_id: str,
    image_sha256: str,
) -> tuple[dict[str, Any], str]:
    evaluation = prediction.get("_evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("state") != "succeeded":
        raise ValueError(f"TMA prediction is not a successful cached call: {document_id}")
    evidence_keys = (
        "cache_key",
        "call_path",
        "state",
        "provider_error",
        "provider_finish_reason",
    )
    if any(report_execution.get(key) != evaluation.get(key) for key in evidence_keys):
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
    inputs = call.get("inputs") if isinstance(call, dict) else None
    if not isinstance(inputs, dict) or set(inputs) != {
        "contract",
        "image_sha256",
        "ocr_sha256",
    }:
        raise ValueError(f"TMA call inputs are malformed: {document_id}")
    if call.get("state") != "succeeded" or call.get("result_path") != "tma.json":
        raise ValueError(f"TMA call is not a completed extraction: {document_id}")
    if inputs.get("contract") != report_contract:
        raise ValueError(f"TMA call/report contract mismatch: {document_id}")
    if inputs.get("image_sha256") != image_sha256:
        raise ValueError(f"TMA call/image identity mismatch: {document_id}")
    if inputs.get("ocr_sha256") != _tma_digest(ocr):
        raise ValueError(f"TMA call/OCR identity mismatch: {document_id}")
    if _tma_digest(inputs) != cache_key:
        raise ValueError(f"TMA cache digest does not match exact call inputs: {document_id}")
    raw_relative = expected.with_name("provider.raw.json")
    raw_file = _exact_relative(tma_root, str(raw_relative), label="provider response")
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
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError(f"TMA provider response did not finish with stop: {document_id}")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(content, str)
    ):
        raise ValueError(
            f"TMA provider response must have exactly one assistant text content: {document_id}"
        )
    if any(message.get(key) is not None for key in ("refusal", "tool_calls", "function_call")):
        raise ValueError(f"TMA assistant response contains a refusal or tool call: {document_id}")
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
    ocr = OCRDocument.model_validate(raw_ocr)
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


def _tma_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _exact_relative(root: Path, relative: str, *, label: str) -> Path:
    target = safe_relative(root, relative)
    expected = root / relative
    if target != expected:
        raise ValueError(f"TMA {label} path must not use a filesystem alias")
    return target
