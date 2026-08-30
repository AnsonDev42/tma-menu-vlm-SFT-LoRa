import hashlib
import json
import re
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_sha256_sidecar(path: Path, sidecar: Path) -> str:
    digest = sha256_file(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def verify_sha256_sidecar(path: Path, sidecar: Path) -> dict[str, Any]:
    source = path.resolve()
    checksum = sidecar.resolve()
    if not source.is_file() or not checksum.is_file():
        raise ValueError("File and SHA-256 sidecar must both exist")
    expected_line = checksum.read_text(encoding="utf-8")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\\r\n]+)\n", expected_line)
    if match is None or match.group(2) != source.name:
        raise ValueError("SHA-256 sidecar format or file identity is invalid")
    actual = sha256_file(source)
    if actual != match.group(1):
        raise ValueError("File checksum does not match SHA-256 sidecar")
    return {"valid": True, "sha256": actual, "file": source.name}


def safe_relative(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"Path must be non-empty and relative: {relative!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes release root: {relative}")
    return target
