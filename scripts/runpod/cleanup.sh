#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  runpod_die "usage: $0 POD_ID REMOTE_ROOT [--delete-pod]"
fi
pod_id="$1"
remote_root="$2"
delete_pod="${3:-}"
validate_pod_id "$pod_id"
validate_remote_root "$remote_root"
if [[ -n "$delete_pod" && "$delete_pod" != "--delete-pod" ]]; then
  runpod_die "third argument must be --delete-pod"
fi

load_ssh_info "$pod_id"
ssh -i "$RUNPOD_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  -p "$RUNPOD_SSH_PORT" root@"$RUNPOD_SSH_IP" rm -rf -- "$remote_root"
echo "Removed remote private workspace: $remote_root (not recoverable)" >&2
if [[ "$delete_pod" == "--delete-pod" ]]; then
  runpodctl pod delete "$pod_id"
  echo "Deleted pod: $pod_id" >&2
fi
