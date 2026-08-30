#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -ne 6 ]]; then
  runpod_die "usage: $0 POD_ID REMOTE_ROOT ARCHIVE_NAME LUNA_BASELINE_NAME LUNA_BASELINE_SHA256_NAME GUARD_RECEIPT"
fi
pod_id="$1"
remote_root="$2"
archive_name="$3"
luna_baseline_name="$4"
luna_baseline_sidecar_name="$5"
guard_receipt="$6"
validate_pod_id "$pod_id"
validate_remote_root "$remote_root"
validate_archive_name "$archive_name"
validate_archive_name "$luna_baseline_name"
validate_archive_name "$luna_baseline_sidecar_name"
[[ "$archive_name" != "$luna_baseline_name" && \
  "$archive_name" != "$luna_baseline_sidecar_name" && \
  "$luna_baseline_name" != "$luna_baseline_sidecar_name" ]] || \
  runpod_die "Dataset and Luna file names must be distinct"
[[ -f "$guard_receipt" ]] || runpod_die "GUARD_RECEIPT must exist"
remaining_json="$(python3 "$script_dir/cost_guard.py" remaining \
  --receipt "$guard_receipt" --pod-id "$pod_id")"
max_seconds="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["remaining_seconds"])' \
  <<< "$remaining_json")"
[[ "$max_seconds" =~ ^[1-9][0-9]*$ ]] || runpod_die "guard has no positive remaining duration"

load_ssh_info "$pod_id"
ssh_args=(-i "$RUNPOD_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  -p "$RUNPOD_SSH_PORT")
remote_command="cd '$remote_root/project' && setsid timeout '$max_seconds' bash scripts/runpod/on-pod-train.sh '$remote_root' '$archive_name' '$luna_baseline_name' '$luna_baseline_sidecar_name' > '$remote_root/train.log' 2>&1 </dev/null & echo LAUNCHED"
ssh "${ssh_args[@]}" root@"$RUNPOD_SSH_IP" "$remote_command"
echo "Monitor with: runpodctl pod logs $pod_id --follow" >&2
echo "Or SSH-tail: $remote_root/train.log" >&2
