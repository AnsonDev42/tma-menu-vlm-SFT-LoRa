import hashlib
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
    state_dir = tmp_path / "state"
    state_dir.mkdir()
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
  "pod get")
    if [[ -f "$FAKE_STATE_DIR/deleted" ]]; then
      echo '{"error":"pod not found","code":"not_found"}' >&2
      exit 1
    fi
    echo '{"id":"pod123"}'
    ;;
  "pod delete")
    touch "$FAKE_STATE_DIR/deleted"
    echo '{"id":"pod123","deleted":true}'
    ;;
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
        "FAKE_STATE_DIR": str(state_dir),
        "RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS": "0",
        "RUNPOD_DELETE_MAX_BACKOFF_SECONDS": "0",
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
    watchdog_log = Path(payload["cost_guard"]["watchdog_log"])
    deadline = time.monotonic() + 4
    while (
        (
            not watchdog_log.exists()
            or "Confirmed pod pod123 is absent" not in watchdog_log.read_text()
        )
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    invocations = invocation_log.read_text().splitlines()
    assert invocations.count("pod delete pod123") == 1
    assert invocations[-1] == "pod get pod123"


def test_launch_deletes_created_pod_if_watchdog_setup_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "version ") echo 'runpodctl 2.12.0-51ca7f0' ;;
  "user ") echo '{"clientBalance":20.0}' ;;
  "gpu list") echo '[{"gpuId":"NVIDIA A40","securePricePerHr":0.44,"communityPricePerHr":null}]' ;;
  "pod create") echo '{"id":"pod123"}' ;;
  "pod get")
    if [[ -f "$FAKE_STATE_DIR/deleted" ]]; then
      echo '{"error":"pod not found","code":"not_found"}' >&2
      exit 1
    fi
    echo '{"id":"pod123"}'
    ;;
  "pod delete")
    count_file="$FAKE_STATE_DIR/delete-count"
    count=0
    [[ ! -f "$count_file" ]] || count="$(<"$count_file")"
    count=$((count + 1))
    printf '%s\n' "$count" > "$count_file"
    if [[ "$count" -eq 1 ]]; then
      echo '{"error":"temporary outage","code":"network_error"}' >&2
      exit 1
    fi
    touch "$FAKE_STATE_DIR/deleted"
    echo '{"id":"pod123","deleted":true}'
    ;;
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
        "FAKE_STATE_DIR": str(state_dir),
        "RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS": "0",
        "RUNPOD_DELETE_MAX_BACKOFF_SECONDS": "0",
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
    assert state_dir.joinpath("deleted").is_file()
    invocations = invocation_log.read_text().splitlines()
    assert invocations.count("pod delete pod123") == 2
    assert invocations[-1] == "pod get pod123"


def test_delete_retries_transient_failure_until_absence_is_confirmed(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "pod get")
    if [[ -f "$FAKE_STATE_DIR/deleted" ]]; then
      echo '{"error":"pod not found","code":"not_found"}' >&2
      exit 1
    fi
    echo '{"id":"pod123"}'
    ;;
  "pod delete")
    count_file="$FAKE_STATE_DIR/delete-count"
    count=0
    [[ ! -f "$count_file" ]] || count="$(<"$count_file")"
    count=$((count + 1))
    printf '%s\n' "$count" > "$count_file"
    if [[ "$count" -eq 1 ]]; then
      echo '{"error":"temporary outage","code":"network_error"}' >&2
      exit 1
    fi
    touch "$FAKE_STATE_DIR/deleted"
    echo '{"id":"pod123","deleted":true}'
    ;;
  *) exit 97 ;;
esac
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
        "FAKE_STATE_DIR": str(state_dir),
        "RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS": "0",
        "RUNPOD_DELETE_MAX_BACKOFF_SECONDS": "0",
    }

    result = _run(
        "bash",
        "-c",
        'source "$1"; delete_pod_until_absent pod123',
        "test",
        str(RUNPOD / "lib.sh"),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert invocation_log.read_text().splitlines() == [
        "pod get pod123",
        "pod delete pod123",
        "pod get pod123",
        "pod delete pod123",
        "pod get pod123",
    ]


def test_delete_retries_when_success_is_visible_before_absence(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "pod get")
    count=0
    [[ ! -f "$FAKE_STATE_DIR/delete-count" ]] || count="$(<"$FAKE_STATE_DIR/delete-count")"
    if [[ "$count" -ge 2 ]]; then
      echo '{"error":"pod not found","code":"not_found"}' >&2
      exit 1
    fi
    echo '{"id":"pod123"}'
    ;;
  "pod delete")
    count=0
    [[ ! -f "$FAKE_STATE_DIR/delete-count" ]] || count="$(<"$FAKE_STATE_DIR/delete-count")"
    printf '%s\n' "$((count + 1))" > "$FAKE_STATE_DIR/delete-count"
    echo '{"id":"pod123","deleted":true}'
    ;;
  *) exit 97 ;;
esac
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
        "FAKE_STATE_DIR": str(state_dir),
        "RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS": "0",
        "RUNPOD_DELETE_MAX_BACKOFF_SECONDS": "0",
    }

    result = _run(
        "bash",
        "-c",
        'source "$1"; delete_pod_until_absent pod123',
        "test",
        str(RUNPOD / "lib.sh"),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert invocation_log.read_text().splitlines() == [
        "pod get pod123",
        "pod delete pod123",
        "pod get pod123",
        "pod delete pod123",
        "pod get pod123",
    ]


def test_delete_does_not_mutate_an_already_absent_pod(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "pod get")
    echo '{"error":"pod not found","code":"not_found"}' >&2
    exit 1
    ;;
  "pod delete") exit 98 ;;
  *) exit 97 ;;
esac
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        "-c",
        'source "$1"; delete_pod_until_absent pod123',
        "test",
        str(RUNPOD / "lib.sh"),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert invocation_log.read_text().splitlines() == ["pod get pod123"]


def test_deletion_watchdog_uses_confirmed_retry_path(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    _write_executable(
        fake_bin / "runpodctl",
        """#!/usr/bin/env bash
echo "$*" >> "$FAKE_INVOCATION_LOG"
case "$1 $2" in
  "pod get")
    if [[ -f "$FAKE_STATE_DIR/deleted" ]]; then
      echo '{"error":"pod not found","code":"not_found"}' >&2
      exit 1
    fi
    echo '{"id":"pod123"}'
    ;;
  "pod delete")
    count_file="$FAKE_STATE_DIR/delete-count"
    count=0
    [[ ! -f "$count_file" ]] || count="$(<"$count_file")"
    count=$((count + 1))
    printf '%s\n' "$count" > "$count_file"
    if [[ "$count" -eq 1 ]]; then
      echo '{"error":"temporary outage","code":"network_error"}' >&2
      exit 1
    fi
    touch "$FAKE_STATE_DIR/deleted"
    echo '{"id":"pod123","deleted":true}'
    ;;
  *) exit 97 ;;
esac
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
        "FAKE_STATE_DIR": str(state_dir),
        "RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS": "0",
        "RUNPOD_DELETE_MAX_BACKOFF_SECONDS": "0",
    }

    result = _run(
        "bash",
        str(RUNPOD / "delete-after.sh"),
        "pod123",
        "1",
        str(receipt),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "confirmed pod pod123 is absent" in result.stderr.casefold()
    assert invocation_log.read_text().splitlines() == [
        "pod get pod123",
        "pod delete pod123",
        "pod get pod123",
        "pod delete pod123",
        "pod get pod123",
    ]


def test_transfer_requires_and_sends_checksummed_luna_baseline(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        "#!/usr/bin/env bash\necho '{\"ip\":\"127.0.0.1\",\"port\":22,"
        "\"ssh_key\":{\"path\":\"/tmp/synthetic-key\"}}'\n",
    )
    _write_executable(
        fake_bin / "ssh",
        '#!/usr/bin/env bash\necho "ssh $*" >> "$FAKE_INVOCATION_LOG"\n',
    )
    _write_executable(
        fake_bin / "rsync",
        '#!/usr/bin/env bash\necho "rsync $*" >> "$FAKE_INVOCATION_LOG"\n',
    )
    archive = tmp_path / "dataset.tar.gz"
    archive.write_bytes(b"synthetic archive")
    Path(str(archive) + ".sha256").write_text("synthetic\n", encoding="utf-8")
    baseline = tmp_path / "luna-test-predictions.jsonl"
    baseline.write_text('{"example_id":"synthetic"}\n', encoding="utf-8")
    digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    sidecar = tmp_path / "luna-test-predictions.jsonl.sha256"
    sidecar.write_text(f"{digest}  {baseline.name}\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "transfer-to-pod.sh"),
        "pod123",
        str(ROOT),
        str(archive),
        str(baseline),
        str(sidecar),
        "/workspace/synthetic-run",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    invocations = invocation_log.read_text(encoding="utf-8")
    assert str(baseline) in invocations
    assert str(sidecar) in invocations
    assert "/workspace/synthetic-run/incoming/" in invocations


def test_transfer_rejects_luna_checksum_drift_before_remote_action(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        '#!/usr/bin/env bash\necho "$*" >> "$FAKE_INVOCATION_LOG"\nexit 99\n',
    )
    archive = tmp_path / "dataset.tar.gz"
    archive.write_bytes(b"synthetic archive")
    Path(str(archive) + ".sha256").write_text("synthetic\n", encoding="utf-8")
    baseline = tmp_path / "luna-test-predictions.jsonl"
    baseline.write_text("drift\n", encoding="utf-8")
    sidecar = tmp_path / "luna-test-predictions.jsonl.sha256"
    sidecar.write_text(f"{'a' * 64}  {baseline.name}\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "transfer-to-pod.sh"),
        "pod123",
        str(ROOT),
        str(archive),
        str(baseline),
        str(sidecar),
        "/workspace/synthetic-run",
        env=env,
    )

    assert result.returncode != 0
    assert "checksum does not match" in result.stderr
    assert not invocation_log.exists()


def test_run_training_forwards_required_luna_files_to_remote_job(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        "#!/usr/bin/env bash\necho '{\"ip\":\"127.0.0.1\",\"port\":22,"
        "\"ssh_key\":{\"path\":\"/tmp/synthetic-key\"}}'\n",
    )
    _write_executable(
        fake_bin / "ssh",
        '#!/usr/bin/env bash\necho "$*" >> "$FAKE_INVOCATION_LOG"\necho LAUNCHED\n',
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"pod_id": "pod123", "delete_after_epoch": int(time.time()) + 3600}),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "run-training.sh"),
        "pod123",
        "/workspace/synthetic-run",
        "dataset.tar.gz",
        "luna-test-predictions.jsonl",
        "luna-test-predictions.jsonl.sha256",
        str(receipt),
        env=env,
    )

    assert result.returncode == 0, result.stderr
    invocation = invocation_log.read_text(encoding="utf-8")
    assert "on-pod-train.sh" in invocation
    assert "luna-test-predictions.jsonl" in invocation
    assert "luna-test-predictions.jsonl.sha256" in invocation


def test_on_pod_script_always_verifies_evaluates_and_bundles_luna_baseline() -> None:
    script = (RUNPOD / "on-pod-train.sh").read_text(encoding="utf-8")

    assert script.count("verify-sidecar") == 3
    assert 'luna_run="$run/luna-test-predictions.jsonl"' in script
    assert '--luna-predictions "$luna_run"' in script
    assert '"luna_test_predictions":"luna-test-predictions.jsonl"' in script
    assert "LUNA_TEST_PREDICTIONS" not in script


@pytest.mark.parametrize(
    ("archive", "baseline", "sidecar"),
    [
        (
            "dataset.tar.gz",
            "caller-controlled.jsonl",
            "caller-controlled.jsonl.sha256",
        ),
        (
            "luna-test-predictions.jsonl",
            "luna-test-predictions.jsonl",
            "luna-test-predictions.jsonl.sha256",
        ),
        (
            "luna-test-predictions.jsonl.sha256",
            "luna-test-predictions.jsonl",
            "luna-test-predictions.jsonl.sha256",
        ),
    ],
)
def test_training_rejects_caller_controlled_or_colliding_luna_names_before_remote_action(
    tmp_path: Path, archive: str, baseline: str, sidecar: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations"
    _write_executable(
        fake_bin / "runpodctl",
        '#!/usr/bin/env bash\necho "$*" >> "$FAKE_INVOCATION_LOG"\nexit 99\n',
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_INVOCATION_LOG": str(invocation_log),
    }

    result = _run(
        "bash",
        str(RUNPOD / "run-training.sh"),
        "pod123",
        "/workspace/synthetic-run",
        archive,
        baseline,
        sidecar,
        str(receipt),
        env=env,
    )

    assert result.returncode == 2
    assert "reserved" in result.stderr
    assert not invocation_log.exists()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
