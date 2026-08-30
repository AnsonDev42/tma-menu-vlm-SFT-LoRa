import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifacts import create_artifact_bundle, package_directory, verify_archive
from .compiler import CompileOptions, compile_release, validate_compiled_dataset
from .constants import REPOSITORY_ROOT
from .evaluation import evaluate_files, evaluate_test_once, select_checkpoint
from .jsonio import canonical_json, verify_sha256_sidecar, write_json
from .luna_baseline import approve_luna_response_provenance, import_luna_baseline
from .prediction import predict_dataset
from .privacy import scan_public_repository
from .release import validate_release
from .synthetic import (
    create_reordered_predictions,
    create_synthetic_artifact_run,
    create_synthetic_luna_evaluation,
    create_synthetic_release,
)
from .training import preflight_config, run_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(canonical_json(result))
    if args.command == "scan-public" and not result["valid"]:
        return 1
    return 0


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "synthetic-release":
        return create_synthetic_release(args.output, include_splits=not args.unsplit)
    if args.command == "synthetic-predictions":
        return create_reordered_predictions(args.references, args.output)
    if args.command == "synthetic-artifact-run":
        return create_synthetic_artifact_run(
            args.output,
            dataset_manifest_path=args.dataset_manifest,
            response_provenance_path=args.response_provenance,
            luna_predictions_path=args.luna_predictions,
        )
    if args.command == "synthetic-luna-evaluation":
        return create_synthetic_luna_evaluation(args.dataset, args.output)
    if args.command == "import-luna-baseline":
        return import_luna_baseline(
            args.dataset,
            args.tma_data_root,
            args.evaluation_run,
            args.response_provenance,
            args.response_provenance_sidecar,
            args.output,
        )
    if args.command == "approve-luna-response-provenance":
        return approve_luna_response_provenance(
            args.dataset,
            args.tma_data_root,
            args.evaluation_run,
            args.canonical_baseline,
            args.output,
        )
    if args.command == "validate-release":
        release = validate_release(args.release, allow_unsplit=args.allow_unsplit)
        return {
            "valid": True,
            "format": release.manifest["format"],
            "records": len(release.records),
            "exclusions": len(release.exclusions),
            "manifest_sha256": release.manifest_sha256,
        }
    if args.command == "compile":
        return compile_release(
            CompileOptions(
                release=args.release,
                output=args.output,
                allow_unsplit=args.allow_unsplit,
                split_seed=args.seed,
            )
        )
    if args.command == "validate-dataset":
        return validate_compiled_dataset(args.dataset)
    if args.command == "preflight":
        return preflight_config(
            args.config,
            no_download=args.no_download,
            dataset_path=args.dataset,
        )
    if args.command == "train":
        return run_training(args.config, args.dataset, args.output)
    if args.command == "predict":
        return predict_dataset(
            args.config,
            args.dataset,
            args.split_file,
            args.adapter,
            args.output,
            max_new_tokens=args.max_new_tokens,
        )
    if args.command == "evaluate":
        metrics = evaluate_files(
            args.references,
            args.predictions,
            luna_predictions_path=args.luna_predictions,
        )
        write_json(args.output, metrics)
        return metrics
    if args.command == "select-checkpoint":
        selection = select_checkpoint(args.metrics)
        write_json(args.output, selection)
        return selection
    if args.command == "evaluate-test":
        return evaluate_test_once(
            args.references,
            args.predictions,
            args.output,
            dataset_sha256=args.dataset_sha256,
            dataset_manifest=args.dataset_manifest,
            checkpoint=args.checkpoint,
            gate_store=args.gate_store,
            luna_predictions=args.luna_predictions,
        )
    if args.command == "bundle":
        return create_artifact_bundle(args.run_root, args.spec, args.output)
    if args.command == "package":
        return package_directory(args.source, args.archive)
    if args.command == "verify-archive":
        return verify_archive(args.archive, args.output)
    if args.command == "verify-sidecar":
        return verify_sha256_sidecar(args.file, args.sidecar)
    if args.command == "scan-public":
        return scan_public_repository(args.repository)
    raise AssertionError(f"Unhandled command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="menu-vlm")
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("synthetic-release", help="create public synthetic input")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--unsplit", action="store_true")

    synthetic_predictions = commands.add_parser("synthetic-predictions")
    synthetic_predictions.add_argument("--references", type=Path, required=True)
    synthetic_predictions.add_argument("--output", type=Path, required=True)

    synthetic_artifacts = commands.add_parser("synthetic-artifact-run")
    synthetic_artifacts.add_argument("--dataset-manifest", type=Path, required=True)
    synthetic_artifacts.add_argument("--response-provenance", type=Path, required=True)
    synthetic_artifacts.add_argument("--luna-predictions", type=Path, required=True)
    synthetic_artifacts.add_argument("--output", type=Path, required=True)

    synthetic_luna = commands.add_parser("synthetic-luna-evaluation")
    synthetic_luna.add_argument("--dataset", type=Path, required=True)
    synthetic_luna.add_argument("--output", type=Path, required=True)

    luna = commands.add_parser("import-luna-baseline")
    luna.add_argument("--dataset", type=Path, required=True)
    luna.add_argument("--tma-data-root", type=Path, required=True)
    luna.add_argument("--evaluation-run", required=True)
    luna.add_argument("--response-provenance", type=Path, required=True)
    luna.add_argument("--response-provenance-sidecar", type=Path, required=True)
    luna.add_argument("--output", type=Path, required=True)

    approval = commands.add_parser("approve-luna-response-provenance")
    approval.add_argument("--dataset", type=Path, required=True)
    approval.add_argument("--tma-data-root", type=Path, required=True)
    approval.add_argument("--evaluation-run", required=True)
    approval.add_argument("--canonical-baseline", type=Path, required=True)
    approval.add_argument("--output", type=Path, required=True)

    validate_release_parser = commands.add_parser("validate-release")
    validate_release_parser.add_argument("--release", type=Path, required=True)
    validate_release_parser.add_argument("--allow-unsplit", action="store_true")

    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--release", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--allow-unsplit", action="store_true")
    compile_parser.add_argument("--seed", type=int, default=20260829)

    validate_dataset = commands.add_parser("validate-dataset")
    validate_dataset.add_argument("--dataset", type=Path, required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "qwen3-vl-4b-lora.json",
    )
    preflight.add_argument("--dataset", type=Path)
    preflight.add_argument("--no-download", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--dataset", type=Path, required=True)
    predict.add_argument("--split-file", required=True)
    predict.add_argument("--adapter", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--max-new-tokens", type=int, default=4096)

    evaluate = commands.add_parser("evaluate")
    _evaluation_args(evaluate, require_luna=False)

    selection = commands.add_parser("select-checkpoint")
    selection.add_argument("--metrics", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)

    test = commands.add_parser("evaluate-test")
    _evaluation_args(test, require_luna=True)
    test.add_argument("--dataset-sha256", required=True)
    test.add_argument("--dataset-manifest", type=Path, required=True)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--gate-store", type=Path, required=True)

    bundle = commands.add_parser("bundle")
    bundle.add_argument("--run-root", type=Path, required=True)
    bundle.add_argument("--spec", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)

    package = commands.add_parser("package")
    package.add_argument("--source", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)

    verify = commands.add_parser("verify-archive")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    verify_sidecar = commands.add_parser("verify-sidecar")
    verify_sidecar.add_argument("--file", type=Path, required=True)
    verify_sidecar.add_argument("--sidecar", type=Path, required=True)

    scan = commands.add_parser("scan-public")
    scan.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    return parser


def _evaluation_args(parser: argparse.ArgumentParser, *, require_luna: bool) -> None:
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--luna-predictions", type=Path, required=require_luna)
    parser.add_argument("--output", type=Path, required=True)


if __name__ == "__main__":
    raise SystemExit(main())
