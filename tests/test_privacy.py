import subprocess
from pathlib import Path

import pytest

from menu_vlm.privacy import scan_public_repository


@pytest.mark.parametrize("suffix", [".svg", ".pdf", ".heic", ".heif"])
def test_public_scan_rejects_tracked_menu_document_formats(tmp_path: Path, suffix: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    candidate = tmp_path / f"public-looking-menu{suffix}"
    candidate.write_bytes(b"synthetic fixture")
    subprocess.run(["git", "add", candidate.name], cwd=tmp_path, check=True)

    report = scan_public_repository(tmp_path)

    assert report["valid"] is False
    assert report["findings"] == [
        {"path": candidate.name, "reason": "forbidden private/artifact file type"}
    ]
