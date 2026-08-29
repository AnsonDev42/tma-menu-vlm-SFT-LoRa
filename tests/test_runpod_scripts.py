import json
import os
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNPOD = ROOT / "scripts" / "runpod"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, env=env)


@pytest.mark.parametrize(
    "remote_root",
    [
        "/workspace",
        "/workspace/",
        "/workspace//",
        "/workspace/.",
        "/workspace/job/..",
        "/workspace/job/",
    ],
)
def test_shared_remote_root_guard_rejects_workspace_aliases(remote_root: str) -> None:
    result = _run(
        "bash",
        "-c",
        'source "$1"; validate_remote_root "$2"',
        "test",
        str(RUNPOD / "lib.sh"),
        remote_root,
    )
    assert result.returncode == 2
    assert "canonical directory below /workspace" in result.stderr


def test_cleanup_validates_delete_flag_before_any_remote_action(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        '#!/usr/bin/env bash\necho "$*" >> "$FAKE_INVOCATION_LOG"\nexit 99\n',
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "cleanup.sh"),
        "pod123",
        "/workspace/safe-run",
        "--surprise",
        env=env,
    )

    assert result.returncode == 2
    assert not invocation_log.exists()


@pytest.mark.parametrize(
    "ssh_info",
    [
        {"ip": "host;touch-pwned", "port": 22, "ssh_key": {"path": "/tmp/key"}},
        {"ip": "127.0.0.1", "port": 70000, "ssh_key": {"path": "/tmp/key"}},
        {"ip": "127.0.0.1", "port": 22, "ssh_key": {"path": "relative-key"}},
    ],
)
def test_shared_ssh_parser_rejects_unsafe_runpodctl_output(
    tmp_path: Path, ssh_info: dict[str, object]
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "runpodctl",
        f"#!/usr/bin/env bash\nprintf '%s\\n' '{json.dumps(ssh_info)}'\n",
    )
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = _run(
        "bash",
        "-c",
        'source "$1"; load_ssh_info pod123',
        "test",
        str(RUNPOD / "lib.sh"),
        env=env,
    )

    assert result.returncode == 2
    assert "unsafe SSH" in result.stderr


def test_cost_quote_uses_catalog_price_storage_and_fails_closed(tmp_path: Path) -> None:
    catalog = tmp_path / "gpu.json"
    user = tmp_path / "user.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "gpuId": "NVIDIA A40",
                    "securePricePerHr": 0.44,
                    "communityPricePerHr": None,
                    "dataCenterAvailability": [{"dataCenterId": "EU-RO-1", "stockStatus": "high"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    user.write_text(json.dumps({"clientBalance": 20.0}), encoding="utf-8")
    command = [
        "python3",
        str(RUNPOD / "cost_guard.py"),
        "quote",
        "--gpu-catalog",
        str(catalog),
        "--user",
        str(user),
        "--gpu-id",
        "NVIDIA A40",
        "--cloud-type",
        "SECURE",
        "--hours",
        "2",
        "--container-gb",
        "20",
        "--volume-gb",
        "100",
        "--data-center-id",
        "EU-RO-1",
        "--cap-usd",
        "15",
    ]

    quote = _run(*command)

    assert quote.returncode == 0, quote.stderr
    payload = json.loads(quote.stdout)
    assert Decimal(payload["compute_cost_usd"]) == Decimal("0.88")
    assert Decimal(payload["storage_cost_usd"]) > 0
    assert Decimal(payload["total_cost_usd"]) == (
        Decimal(payload["compute_cost_usd"]) + Decimal(payload["storage_cost_usd"])
    )
    over_cap_command = list(command)
    over_cap_command[over_cap_command.index("--hours") + 1] = "40"
    over_cap = _run(*over_cap_command)
    assert over_cap.returncode == 2
    assert "$15" in over_cap.stderr
    subsecond_command = list(command)
    subsecond_command[subsecond_command.index("--hours") + 1] = "0.0001"
    subsecond = _run(*subsecond_command)
    assert subsecond.returncode == 2
    assert "at least one second" in subsecond.stderr

    quote_path = tmp_path / "quote.json"
    quote_path.write_text(quote.stdout, encoding="utf-8")
    receipt_path = tmp_path / "pod123.json"
    receipt = _run(
        "python3",
        str(RUNPOD / "cost_guard.py"),
        "receipt",
        "--quote",
        str(quote_path),
        "--pod-id",
        "pod123",
        "--output",
        str(receipt_path),
    )
    assert receipt.returncode == 0, receipt.stderr
    remaining = _run(
        "python3",
        str(RUNPOD / "cost_guard.py"),
        "remaining",
        "--receipt",
        str(receipt_path),
        "--pod-id",
        "pod123",
    )
    assert json.loads(remaining.stdout)["remaining_seconds"] > 0
    wrong_pod = _run(
        "python3",
        str(RUNPOD / "cost_guard.py"),
        "remaining",
        "--receipt",
        str(receipt_path),
        "--pod-id",
        "pod999",
    )
    assert wrong_pod.returncode == 2
    assert "does not belong" in wrong_pod.stderr


def test_launch_uses_supported_cli_price_quote_and_deletion_watchdog(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "version ") echo 'runpodctl 2.12.0-51ca7f0' ;;
  "user ") echo '{"clientBalance":20.0}' ;;
  "gpu list") echo '[{"gpuId":"NVIDIA A40","securePricePerHr":0.44,
"communityPricePerHr":null,"dataCenterAvailability":[{"dataCenterId":"EU-RO-1",
"stockStatus":"high"}]}]' ;;
  "pod create") echo '{"id":"pod123","name":"synthetic"}' ;;
  "pod delete") echo '{"id":"pod123","deleted":true}' ;;
  *) exit 97 ;;
esac
""",
    )
    guard_dir = tmp_path / "guards"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUNPOD_API_KEY": "synthetic-test-key",
        "RUNPOD_GUARD_DIR": str(guard_dir),
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "launch-pod.sh"),
        "test-run",
        "template123",
        "NVIDIA A40",
        "0.0003",
        "20",
        "100",
        "EU-RO-1",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    receipt = Path(payload["cost_guard"]["receipt"])
    assert receipt.is_file()
    create = next(
        line for line in invocation_log.read_text().splitlines() if line.startswith("pod create")
    )
    assert "--terminate-after" not in create
    assert "--container-disk-in-gb 20" in create
    assert "--volume-in-gb 100" in create
    assert "--cloud-type SECURE" in create
    deadline = time.monotonic() + 3
    while "pod delete pod123" not in invocation_log.read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "pod delete pod123" in invocation_log.read_text()


def test_launch_deletes_created_pod_if_watchdog_setup_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "version ") echo 'runpodctl 2.12.0-51ca7f0' ;;
  "user ") echo '{"clientBalance":20.0}' ;;
  "gpu list") echo '[{"gpuId":"NVIDIA A40","securePricePerHr":0.44,"communityPricePerHr":null}]' ;;
  "pod create") echo '{"id":"pod123"}' ;;
  "pod delete") echo '{"id":"pod123","deleted":true}' ;;
  *) exit 97 ;;
esac
""",
    )
    guard_dir = tmp_path / "guards"
    guard_dir.mkdir()
    (guard_dir / "pod123.json").write_text("occupied\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUNPOD_API_KEY": "synthetic-test-key",
        "RUNPOD_GUARD_DIR": str(guard_dir),
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "launch-pod.sh"),
        "test-run",
        "template123",
        "NVIDIA A40",
        "1",
        "20",
        "100",
        env=env,
    )

    assert result.returncode != 0
    assert "pod delete pod123" in invocation_log.read_text()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
