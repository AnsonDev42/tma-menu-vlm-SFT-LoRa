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
