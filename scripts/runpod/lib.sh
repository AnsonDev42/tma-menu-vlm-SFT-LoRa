#!/usr/bin/env bash

runpod_die() {
  echo "$*" >&2
  exit 2
}

validate_pod_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || \
    runpod_die "POD_ID contains unsafe characters"
}

validate_remote_root() {
  local value="$1"
  local canonical
  [[ "$value" =~ ^/workspace/[A-Za-z0-9._/-]+$ ]] || \
    runpod_die "REMOTE_ROOT must be one canonical directory below /workspace"
  canonical="$(python3 -c 'import posixpath,sys; print(posixpath.normpath(sys.argv[1]))' "$value")" || \
    runpod_die "REMOTE_ROOT canonicalization failed"
  [[ "$canonical" == "$value" && "$canonical" != "/workspace" ]] || \
    runpod_die "REMOTE_ROOT must be one canonical directory below /workspace"
}

validate_archive_name() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
    runpod_die "ARCHIVE_NAME must be a safe basename"
}

validate_training_transfer_names() {
  local archive_name="$1"
  local luna_name="$2"
  local luna_sidecar_name="$3"
  validate_archive_name "$archive_name"
  validate_archive_name "$luna_name"
  validate_archive_name "$luna_sidecar_name"
  [[ "$luna_name" == "luna-test-predictions.jsonl" && \
    "$luna_sidecar_name" == "luna-test-predictions.jsonl.sha256" ]] || \
    runpod_die "Luna baseline files must use the reserved training artifact names"
  [[ "$archive_name" != "$luna_name" && "$archive_name" != "$luna_sidecar_name" ]] || \
    runpod_die "Dataset archive name collides with a reserved Luna artifact name"
}

validate_delete_retry_config() {
  local initial="${RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS:-1}"
  local maximum="${RUNPOD_DELETE_MAX_BACKOFF_SECONDS:-30}"
  [[ "$initial" =~ ^[0-9]+$ && "$maximum" =~ ^[0-9]+$ ]] || \
    runpod_die "Runpod deletion retry delays must be whole seconds"
  ((initial <= maximum && maximum <= 60)) || \
    runpod_die "Runpod deletion retry backoff must be ordered and capped at 60 seconds"
}

runpodctl_error_code() {
  python3 -c '
import json
import sys

for line in reversed(sys.stdin.read().splitlines()):
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and isinstance(value.get("code"), str):
        print(value["code"])
        raise SystemExit(0)
raise SystemExit(1)
' <<< "$1"
}

pod_absence_status() {
  local pod_id="$1"
  local error code
  validate_pod_id "$pod_id"
  if error="$(runpodctl pod get "$pod_id" 2>&1 >/dev/null)"; then
    return 1
  fi
  code="$(runpodctl_error_code "$error" 2>/dev/null)" || code="unknown"
  if [[ "$code" == "not_found" ]]; then
    return 0
  fi
  echo "Pod $pod_id absence check failed (code=$code); will retry." >&2
  return 2
}

delete_pod_until_absent() {
  local pod_id="$1"
  local delay="${RUNPOD_DELETE_INITIAL_BACKOFF_SECONDS:-1}"
  local maximum="${RUNPOD_DELETE_MAX_BACKOFF_SECONDS:-30}"
  local status delete_error delete_code next_delay
  validate_pod_id "$pod_id"
  validate_delete_retry_config

  while true; do
    if pod_absence_status "$pod_id"; then
      echo "Confirmed pod $pod_id is absent." >&2
      return 0
    else
      status=$?
    fi

    if [[ "$status" -eq 1 ]]; then
      if delete_error="$(runpodctl pod delete "$pod_id" 2>&1 >/dev/null)"; then
        echo "Pod $pod_id deletion accepted; confirming absence." >&2
      else
        delete_code="$(runpodctl_error_code "$delete_error" 2>/dev/null)" || \
          delete_code="unknown"
        echo "Pod $pod_id deletion failed (code=$delete_code); will retry." >&2
      fi
    fi

    sleep "$delay"
    if ((delay < maximum)); then
      next_delay=$((delay == 0 ? 1 : delay * 2))
      ((next_delay > maximum)) && next_delay="$maximum"
      delay="$next_delay"
    fi
  done
}

load_ssh_info() {
  local pod_id="$1"
  local raw parsed
  validate_pod_id "$pod_id"
  raw="$(runpodctl ssh info "$pod_id")" || return
  parsed="$(python3 -c '
import ipaddress, json, pathlib, re, sys
d = json.load(sys.stdin)
host = str(d.get("ip", ""))
try:
    ipaddress.ip_address(host)
except ValueError:
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host):
        raise SystemExit("unsafe SSH host")
port = int(d.get("port", 0))
if not 1 <= port <= 65535:
    raise SystemExit("unsafe SSH port")
key = str(d.get("ssh_key", {}).get("path", ""))
if not pathlib.PurePath(key).is_absolute() or any(c in key for c in "\n\r\t"):
    raise SystemExit("unsafe SSH key path")
print(f"{host}\t{port}\t{key}")
' <<< "$raw")" || runpod_die "runpodctl returned unsafe SSH connection data"
  IFS=$'\t' read -r RUNPOD_SSH_IP RUNPOD_SSH_PORT RUNPOD_SSH_KEY <<< "$parsed"
  [[ -n "$RUNPOD_SSH_IP" && -n "$RUNPOD_SSH_PORT" && -n "$RUNPOD_SSH_KEY" ]] || \
    runpod_die "runpodctl returned incomplete SSH connection data"
}

rsync_ssh_command() {
  printf 'ssh -i %q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p %q' \
    "$RUNPOD_SSH_KEY" "$RUNPOD_SSH_PORT"
}
