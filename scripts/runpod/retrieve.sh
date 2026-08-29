#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 POD_ID REMOTE_ROOT LOCAL_OUTPUT_DIR" >&2
  exit 2
fi
pod_id="$1"
remote_root="$2"
local_output="$(mkdir -p "$3" && cd "$3" && pwd)"
if [[ ! "$remote_root" =~ ^/workspace/[A-Za-z0-9._/-]+$ ]] || [[ "$remote_root" == *".."* ]]; then
  echo "REMOTE_ROOT must be below /workspace" >&2
  exit 2
fi
eval "$(runpodctl ssh info "$pod_id" | python3 -c '
import json, shlex, sys
d=json.load(sys.stdin)
print("IP="+shlex.quote(d["ip"]))
print("PORT="+shlex.quote(str(d["port"])))
print("KEY="+shlex.quote(d["ssh_key"]["path"]))
')"
rsync_ssh="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT"
rsync -az -e "$rsync_ssh" \
  root@"$IP":"$remote_root/artifact-bundle.tar.gz" "$local_output/"
rsync -az -e "$rsync_ssh" \
  root@"$IP":"$remote_root/artifact-bundle.tar.gz.sha256" "$local_output/"
archive="$local_output/artifact-bundle.tar.gz"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
uv run --project "$project_root" menu-vlm verify-archive --archive "$archive" \
  --output "$local_output/artifact-bundle"
