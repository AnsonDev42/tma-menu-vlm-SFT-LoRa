#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 POD_ID REMOTE_ROOT [--delete-pod]" >&2
  exit 2
fi
pod_id="$1"
remote_root="$2"
delete_pod="${3:-}"
if [[ ! "$remote_root" =~ ^/workspace/[A-Za-z0-9._/-]+$ ]] || [[ "$remote_root" == *".."* ]]; then
  echo "REMOTE_ROOT must be a specific directory below /workspace" >&2
  exit 2
fi
eval "$(runpodctl ssh info "$pod_id" | python3 -c '
import json, shlex, sys
d=json.load(sys.stdin)
print("IP="+shlex.quote(d["ip"]))
print("PORT="+shlex.quote(str(d["port"])))
print("KEY="+shlex.quote(d["ssh_key"]["path"]))
')"
ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT" \
  root@"$IP" rm -rf -- "$remote_root"
echo "Removed remote private workspace: $remote_root (not recoverable)" >&2
if [[ "$delete_pod" == "--delete-pod" ]]; then
  runpodctl pod delete "$pod_id"
  echo "Deleted pod: $pod_id" >&2
elif [[ -n "$delete_pod" ]]; then
  echo "third argument must be --delete-pod" >&2
  exit 2
fi
