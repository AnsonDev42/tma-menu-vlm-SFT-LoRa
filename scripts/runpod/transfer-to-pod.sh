#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -ne 6 ]]; then
  runpod_die "usage: $0 POD_ID LOCAL_PROJECT_DIR LOCAL_DATASET_ARCHIVE LOCAL_LUNA_BASELINE LOCAL_LUNA_BASELINE_SHA256 REMOTE_ROOT"
fi
pod_id="$1"
remote_root="$6"
validate_pod_id "$pod_id"
validate_remote_root "$remote_root"
[[ -d "$2" ]] || runpod_die "LOCAL_PROJECT_DIR must exist"
[[ -f "$3" ]] || runpod_die "LOCAL_DATASET_ARCHIVE must exist"
local_project="$(cd "$2" && pwd)"
local_archive="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
[[ -f "$4" ]] || runpod_die "LOCAL_LUNA_BASELINE must exist"
[[ -f "$5" ]] || runpod_die "LOCAL_LUNA_BASELINE_SHA256 must exist"
local_luna="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
local_luna_sidecar="$(cd "$(dirname "$5")" && pwd)/$(basename "$5")"
archive_name="$(basename "$local_archive")"
luna_baseline_name="$(basename "$local_luna")"
luna_baseline_sidecar_name="$(basename "$local_luna_sidecar")"
validate_training_transfer_names \
  "$archive_name" "$luna_baseline_name" "$luna_baseline_sidecar_name"
if [[ ! -f "$local_archive" || ! -f "$local_archive.sha256" ]]; then
  runpod_die "dataset archive and .sha256 sidecar must both exist"
fi
uv run --project "$local_project" menu-vlm verify-sidecar \
  --file "$local_luna" --sidecar "$local_luna_sidecar" >/dev/null

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
rsync -az -e "$rsync_ssh" "$local_luna" "$local_luna_sidecar" \
  root@"$RUNPOD_SSH_IP":"$remote_root/incoming/"
echo "Transfer complete: $remote_root" >&2
