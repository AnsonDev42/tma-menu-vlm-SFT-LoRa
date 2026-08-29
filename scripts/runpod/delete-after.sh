#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -ne 3 ]]; then
  runpod_die "usage: $0 POD_ID MAX_SECONDS GUARD_RECEIPT"
fi
pod_id="$1"
max_seconds="$2"
receipt="$3"
validate_pod_id "$pod_id"
[[ "$max_seconds" =~ ^[1-9][0-9]*$ ]] || runpod_die "MAX_SECONDS must be positive"
[[ -f "$receipt" ]] || runpod_die "GUARD_RECEIPT must exist"
validate_delete_retry_config
if [[ -n "${RUNPOD_DELETE_READY_FILE:-}" ]]; then
  printf '%s\n' "$$" > "$RUNPOD_DELETE_READY_FILE"
fi

sleep "$max_seconds"
delete_pod_until_absent "$pod_id"
echo "Deletion watchdog confirmed pod $pod_id is absent after $max_seconds seconds" >&2
