import re
import subprocess
from pathlib import Path
from typing import Any

_FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".j2k",
    ".jpeg",
    ".jp2",
    ".jpg",
    ".jxl",
    ".jsonl",
    ".parquet",
    ".pdf",
    ".png",
    ".pth",
    ".pt",
    ".safetensors",
    ".svg",
    ".tar",
    ".tgz",
    ".tif",
    ".tiff",
    ".webp",
    ".zip",
}
_FORBIDDEN_PARTS = {
    ".local",
    "artifacts",
    "checkpoints",
    "data",
    "datasets",
    "outputs",
    "predictions",
    "runs",
    "wandb",
}
_SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "Runpod key assignment": re.compile(r"RUNPOD_API_KEY\s*=\s*(?!<|your_|\$\{|\.{3})[^\s#]{12,}"),
    "bearer credential": re.compile(r"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{16,}"),
    "signed URL": re.compile(r"[?&](?:sig|signature|token|x-amz-signature)=[^\s&]+", re.I),
}


def scan_public_repository(root: Path) -> dict[str, Any]:
    repository = root.resolve()
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [Path(line) for line in result.stdout.splitlines() if line]
    findings: list[dict[str, str]] = []
    for relative in paths:
        if relative.name == ".env" or relative.suffix.lower() in _FORBIDDEN_SUFFIXES:
            findings.append(
                {"path": str(relative), "reason": "forbidden private/artifact file type"}
            )
        if any(part in _FORBIDDEN_PARTS for part in relative.parts):
            findings.append(
                {"path": str(relative), "reason": "forbidden private/artifact directory"}
            )
        path = repository / relative
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"path": str(relative), "reason": "unexpected binary tracked file"})
            continue
        for label, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": str(relative), "reason": label})
    return {"valid": not findings, "files_scanned": len(paths), "findings": findings}
