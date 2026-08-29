#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -ne 4 ]]; then
  runpod_die "usage: $0 POD_ID LOCAL_PROJECT_DIR LOCAL_DATASET_ARCHIVE REMOTE_ROOT"
fi
pod_id="$1"
remote_root="$4"
validate_pod_id "$pod_id"
validate_remote_root "$remote_root"
[[ -d "$2" ]] || runpod_die "LOCAL_PROJECT_DIR must exist"
[[ -f "$3" ]] || runpod_die "LOCAL_DATASET_ARCHIVE must exist"
local_project="$(cd "$2" && pwd)"
local_archive="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
if [[ ! -f "$local_archive" || ! -f "$local_archive.sha256" ]]; then
  runpod_die "dataset archive and .sha256 sidecar must both exist"
fi

load_ssh_info "$pod_id"
ssh_args=(-i "$RUNPOD_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  -p "$RUNPOD_SSH_PORT")
rsync_ssh="$(rsync_ssh_command)"
ssh "${ssh_args[@]}" root@"$RUNPOD_SSH_IP" mkdir -p -- \
  "$remote_root/project" "$remote_root/incoming"
rsync -az --delete --exclude .git --exclude .venv --exclude .env \
  -e "$rsync_ssh" "$local_project/" root@"$RUNPOD_SSH_IP":"$remote_root/project/"
rsync -az -e "$rsync_ssh" "$local_archive" "$local_archive.sha256" \
  root@"$RUNPOD_SSH_IP":"$remote_root/incoming/"
echo "Transfer complete: $remote_root" >&2
