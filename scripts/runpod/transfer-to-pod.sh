#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 POD_ID LOCAL_PROJECT_DIR LOCAL_DATASET_ARCHIVE REMOTE_ROOT" >&2
  exit 2
fi
pod_id="$1"
local_project="$(cd "$2" && pwd)"
local_archive="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
remote_root="$4"
if [[ ! "$remote_root" =~ ^/workspace/[A-Za-z0-9._/-]+$ ]] || [[ "$remote_root" == *".."* ]]; then
  echo "REMOTE_ROOT must be a specific directory below /workspace" >&2
  exit 2
fi
if [[ ! -f "$local_archive" || ! -f "$local_archive.sha256" ]]; then
  echo "dataset archive and .sha256 sidecar must both exist" >&2
  exit 2
fi

eval "$(runpodctl ssh info "$pod_id" | python3 -c '
import json, shlex, sys
d=json.load(sys.stdin)
print("IP="+shlex.quote(d["ip"]))
print("PORT="+shlex.quote(str(d["port"])))
print("KEY="+shlex.quote(d["ssh_key"]["path"]))
')"
ssh_args=(-i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT")
rsync_ssh="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT"
ssh "${ssh_args[@]}" root@"$IP" mkdir -p -- "$remote_root/project" "$remote_root/incoming"
rsync -az --delete --exclude .git --exclude .venv --exclude .env \
  -e "$rsync_ssh" "$local_project/" root@"$IP":"$remote_root/project/"
rsync -az -e "$rsync_ssh" "$local_archive" "$local_archive.sha256" \
  root@"$IP":"$remote_root/incoming/"
echo "Transfer complete: $remote_root" >&2
