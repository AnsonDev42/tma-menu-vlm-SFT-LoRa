#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -ne 3 ]]; then
  runpod_die "usage: $0 POD_ID REMOTE_ROOT LOCAL_OUTPUT_DIR"
fi
pod_id="$1"
remote_root="$2"
validate_pod_id "$pod_id"
validate_remote_root "$remote_root"
[[ -n "$3" ]] || runpod_die "LOCAL_OUTPUT_DIR must be non-empty"
local_output="$(mkdir -p "$3" && cd "$3" && pwd)"
load_ssh_info "$pod_id"
rsync_ssh="$(rsync_ssh_command)"
rsync -az --no-owner --no-group -e "$rsync_ssh" \
  root@"$RUNPOD_SSH_IP":"$remote_root/artifact-bundle.tar.gz" "$local_output/"
rsync -az --no-owner --no-group -e "$rsync_ssh" \
  root@"$RUNPOD_SSH_IP":"$remote_root/artifact-bundle.tar.gz.sha256" "$local_output/"
archive="$local_output/artifact-bundle.tar.gz"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
uv run --project "$project_root" menu-vlm verify-archive --archive "$archive" \
  --output "$local_output/artifact-bundle"
