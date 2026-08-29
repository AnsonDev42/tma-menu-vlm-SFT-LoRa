import json
import shutil
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DATASET_FORMAT, PROMPT_VERSION, SPLITS
from .jsonio import (
    canonical_json,
    read_json,
    read_jsonl,
    safe_relative,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .prompt import prompt_metadata, render_user_prompt, system_prompt
from .release import ValidatedRelease, validate_release
from .schemas import (
    CompactItem,
    CompactOutput,
    CompactSection,
    MenuItem,
    PrintedText,
    ReleaseRecord,
    SourceText,
    validate_compact_output,
)


@dataclass(frozen=True)
class CompileOptions:
    release: Path
    output: Path
    allow_unsplit: bool = False
    split_seed: int = 20260829
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)


def compile_release(options: CompileOptions) -> dict[str, Any]:
    release = validate_release(options.release, allow_unsplit=options.allow_unsplit)
    output = options.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        assignments, preserved, split_warnings = _assign_splits(release, options)
        lineage = {str(row["document_id"]): str(row["source_id"]) for row in release.lineage}
        examples: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
            "robustness_validation": [],
            "robustness_test": [],
        }
        failures: list[dict[str, str]] = []
        for record in sorted(release.records, key=lambda row: row.document_id):
            split = assignments[record.document_id]
            try:
                target = project_record(record)
            except ValueError as exc:
                failures.append({"document_id": record.document_id, "reason": str(exc)})
                continue
            image_relative = _copy_image(release.root, output, record)
            role = _role(record, split)
            example = _render_example(
                record=record,
                split=split,
                role=role,
                source_id=lineage.get(record.document_id, record.document_id),
                image_relative=image_relative,
                target=target,
            )
            examples[role].append(example)

        primary_splits = {
            split
            for split in SPLITS
            if any(row["kind"] == "source_reference" for row in examples[split])
        }
        if primary_splits != set(SPLITS):
            raise ValueError(
                "Projection failures produced degenerate primary splits; refusing compiled dataset"
            )
        files = {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "robustness_validation": "robustness_validation.jsonl",
            "robustness_test": "robustness_test.jsonl",
        }
        for name, relative in files.items():
            write_jsonl(output / relative, examples[name])
        write_json(output / "excluded.json", list(release.exclusions))
        write_json(output / "projection-failures.json", failures)
        write_json(
            output / "split-manifest.json",
            {
                "schema_version": "1.0",
                "source_release_splits_preserved": preserved,
                "seed": None if preserved else options.split_seed,
                "ratios": None if preserved else list(options.split_ratios),
                "assignments": dict(sorted(assignments.items())),
                "warnings": split_warnings,
            },
        )
        output_hashes = _hash_output_files(output)
        dataset_sha256 = sha256_json(output_hashes)
        counts = {name: len(rows) for name, rows in examples.items()}
        counts.update(excluded=len(release.exclusions), projection_failures=len(failures))
        manifest: dict[str, Any] = {
            "format": DATASET_FORMAT,
            "schema_version": "1.0",
            "source_release_format": release.manifest["format"],
            "source_release_version": release.manifest.get("version"),
            "source_release_manifest_sha256": release.manifest_sha256,
            "source_release_splits_preserved": preserved,
            "split_seed": None if preserved else options.split_seed,
            "split_assignments": dict(sorted(assignments.items())),
            "prompt": prompt_metadata(),
            "one_image_per_example": True,
            "counts": counts,
            "accounting": {
                "input_records": len(release.records),
                "compiled_records": sum(len(rows) for rows in examples.values()),
                "release_exclusions": len(release.exclusions),
            },
            "files": files,
            "files_sha256": output_hashes,
            "dataset_sha256": dataset_sha256,
        }
        write_json(output / "manifest.json", manifest)
        validate_compiled_dataset(output)
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def project_record(record: ReleaseRecord) -> dict[str, Any]:
    line_by_id = {span.id: index for index, span in enumerate(record.ocr.spans, 1)}
    sections: list[CompactSection] = []
    items: list[CompactItem] = []
    for section_index, section in enumerate(record.annotation.sections, 1):
        section_id = f"s{section_index}"
        sections.append(
            CompactSection(
                id=section_id,
                h=_printed_text(section.name, line_by_id, owner=f"section {section_index} heading"),
                notes=[
                    _printed_text(note, line_by_id, owner=f"section {section_index} note")
                    for note in section.notes
                ],
            )
        )
        items.extend(
            _compact_item(
                item, line_by_id, section_ref=section_id, owner=f"section {section_index}"
            )
            for item in section.items
        )
    items.extend(
        _compact_item(item, line_by_id, section_ref=None, owner="unsectioned")
        for item in record.annotation.unsectioned_items
    )
    compact = CompactOutput(s=sections, i=items).model_dump(by_alias=True)
    return validate_compact_output(compact, ocr_line_count=len(record.ocr.spans)).model_dump(
        by_alias=True
    )


def validate_compiled_dataset(path: Path) -> dict[str, Any]:
    root = path.resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("Unsupported compiled dataset format")
    hashes = manifest.get("files_sha256", {})
    actual_files = {
        str(file.relative_to(root))
        for file in root.rglob("*")
        if file.is_file() and file.name != "manifest.json"
    }
    if set(hashes) != actual_files:
        raise ValueError("Compiled manifest does not account for every generated file")
    for relative, expected in hashes.items():
        target = safe_relative(root, relative)
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Compiled dataset checksum drift: {relative}")
    if sha256_json(hashes) != manifest.get("dataset_sha256"):
        raise ValueError("Compiled dataset aggregate checksum drift")

    example_ids: set[str] = set()
    source_splits: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name, relative in manifest["files"].items():
        rows = read_jsonl(root / relative)
        counts[name] = len(rows)
        for row in rows:
            example_id = str(row.get("example_id", ""))
            if not example_id or example_id in example_ids:
                raise ValueError("Compiled examples have missing or duplicate IDs")
            example_ids.add(example_id)
            image = safe_relative(root, str(row.get("image", "")))
            if not image.is_file():
                raise ValueError(f"Compiled example image is missing: {example_id}")
            compact_output = validate_compact_output(
                row.get("target"), ocr_line_count=int(row.get("ocr_line_count", 0))
            )
            assistant = row["messages"][-1]["content"][0]["text"]
            if json.loads(assistant) != compact_output.model_dump(by_alias=True):
                raise ValueError(f"Assistant target diverges from canonical target: {example_id}")
            source_id = str(row["source_id"])
            split = str(row["split"])
            previous = source_splits.setdefault(source_id, split)
            if previous != split:
                raise ValueError("Compiled source lineage crosses splits")
            if name in {"validation", "test"} and row["kind"] != "source_reference":
                raise ValueError("Primary validation/test contains a derivative")
            if name.startswith("robustness_") and row["kind"] != "silver_augmented":
                raise ValueError("Robustness split contains a canonical source")
    for name, count in counts.items():
        if manifest["counts"].get(name) != count:
            raise ValueError(f"Compiled count drift: {name}")
    if manifest["accounting"]["input_records"] != (
        manifest["accounting"]["compiled_records"] + manifest["counts"]["projection_failures"]
    ):
        raise ValueError("Compiled record accounting is incomplete")
    return {
        "valid": True,
        "dataset_sha256": manifest["dataset_sha256"],
        "examples": len(example_ids),
        "counts": counts,
    }


def _printed_text(value: SourceText, line_by_id: dict[str, int], *, owner: str) -> PrintedText:
    lines = _line_indices(value.source_ids, line_by_id, owner=owner, require=False)
    return PrintedText(t=value.text, l=lines)


def _compact_item(
    item: MenuItem,
    line_by_id: dict[str, int],
    *,
    section_ref: str | None,
    owner: str,
) -> CompactItem:
    source_ids = list(item.name.source_ids)
    if item.description:
        source_ids.extend(item.description.source_ids)
    for price in item.prices:
        source_ids.extend(price.source_ids)
    if item.calories:
        source_ids.extend(item.calories.source_ids)
    for variant in item.variants:
        if variant.name:
            source_ids.extend(variant.name.source_ids)
        for price in variant.prices:
            source_ids.extend(price.source_ids)
    lines = _line_indices(source_ids, line_by_id, owner=f"{owner} item {item.name.text}")
    name_lines = _line_indices(
        item.name.source_ids, line_by_id, owner=f"{owner} item name {item.name.text}"
    )
    notes = [
        _printed_text(note, line_by_id, owner=f"{owner} item note {item.name.text}")
        for note in item.notes
    ]
    return CompactItem(
        l=lines,
        n=item.name.text,
        a=name_lines[0],
        c=1.0,
        s=section_ref,
        notes=notes,
    )


def _line_indices(
    source_ids: Iterable[str],
    line_by_id: dict[str, int],
    *,
    owner: str,
    require: bool = True,
) -> list[int]:
    identifiers = list(source_ids)
    unknown = sorted(set(identifiers) - set(line_by_id))
    if unknown:
        raise ValueError(f"{owner} cites unknown OCR source(s): {', '.join(unknown)}")
    lines = sorted({line_by_id[source_id] for source_id in identifiers})
    if require and not lines:
        raise ValueError(f"{owner} has no OCR evidence")
    return lines


def _render_example(
    *,
    record: ReleaseRecord,
    split: str,
    role: str,
    source_id: str,
    image_relative: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    ocr_lines = [span.text for span in record.ocr.spans]
    return {
        "schema_version": "1.0",
        "example_id": record.document_id,
        "source_id": source_id,
        "split": split,
        "kind": record.kind,
        "evaluation_role": role,
        "image": image_relative,
        "ocr_line_count": len(ocr_lines),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_metadata()["sha256"],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": render_user_prompt(page_index=0, ocr_lines=ocr_lines)},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": canonical_json(target)}],
            },
        ],
        "target": target,
    }


def _role(record: ReleaseRecord, split: str) -> str:
    if split == "train":
        return "train"
    if record.kind == "source_reference":
        return split
    return f"robustness_{split}"


def _copy_image(release_root: Path, output: Path, record: ReleaseRecord) -> str:
    source = safe_relative(release_root, record.image.path)
    suffix = source.suffix.lower() or ".img"
    relative = f"images/{record.document_id}{suffix}"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != record.image.sha256:
        raise ValueError(f"Copied image checksum drift: {record.document_id}")
    return relative


def _assign_splits(
    release: ValidatedRelease, options: CompileOptions
) -> tuple[dict[str, str], bool, list[str]]:
    if release.splits is not None:
        return (
            {str(key): str(value) for key, value in release.splits["documents"].items()},
            True,
            [],
        )
    if not options.allow_unsplit:
        raise ValueError("Unsplit input requires --allow-unsplit")
    sources = [record for record in release.records if record.kind == "source_reference"]
    source_groups = {
        record.document_id: record.metadata.restaurant_id.strip() or record.document_id
        for record in sources
    }
    group_assignments = deterministic_grouped_split(
        source_groups, seed=options.split_seed, ratios=options.split_ratios
    )
    assignments = {
        source_id: group_assignments[group] for source_id, group in source_groups.items()
    }
    for row in release.lineage:
        assignments[str(row["document_id"])] = assignments[str(row["source_id"])]
    fallback = [source_id for source_id, group in source_groups.items() if group == source_id]
    warnings = (
        [f"restaurant_id missing for {len(fallback)} source(s); fell back to canonical source ID"]
        if fallback
        else []
    )
    return assignments, False, warnings


def deterministic_grouped_split(
    documents: dict[str, str],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    import hashlib
    import math

    if len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("Three positive split ratios are required")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    groups: dict[str, list[str]] = defaultdict(list)
    for document_id, group in documents.items():
        groups[group].append(document_id)
    if len(groups) < 3:
        raise ValueError("At least three source groups are required for non-degenerate splits")
    ordered = sorted(
        groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    )
    counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}
    # Seed every requested split before deficit balancing. Small datasets would
    # otherwise allocate multiple early groups to train and can leave test empty.
    for group, split in zip(ordered[:3], SPLITS, strict=True):
        assignments[group] = split
        counts[split] += len(groups[group])
    for group in ordered[3:]:
        split = max(
            SPLITS,
            key=lambda name: (
                ratios[SPLITS.index(name)] * len(documents) - counts[name],
                -SPLITS.index(name),
            ),
        )
        assignments[group] = split
        counts[split] += len(groups[group])
    if any(counts[name] == 0 for name in SPLITS):
        raise ValueError("Grouped split allocation produced a degenerate split")
    return assignments


def _hash_output_files(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
