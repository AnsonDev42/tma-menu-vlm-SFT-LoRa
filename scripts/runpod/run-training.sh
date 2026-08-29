#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 POD_ID REMOTE_ROOT ARCHIVE_NAME MAX_HOURS HOURLY_RATE" >&2
  exit 2
fi
pod_id="$1"
remote_root="$2"
archive_name="$3"
max_hours="$4"
hourly_rate="$5"
if [[ ! "$remote_root" =~ ^/workspace/[A-Za-z0-9._/-]+$ ]] || [[ "$remote_root" == *".."* ]] || \
  [[ ! "$archive_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "REMOTE_ROOT must be below /workspace and ARCHIVE_NAME must be a basename" >&2
  exit 2
fi
max_seconds="$(python3 - "$max_hours" "$hourly_rate" <<'PY'
from decimal import Decimal
import sys
h, r = Decimal(sys.argv[1]), Decimal(sys.argv[2])
cost = h * r
if h <= 0 or cost <= 0 or cost > Decimal("15"):
    raise SystemExit(f"refusing run: projected cost ${cost} must be in (0, $15]")
print(int(h * 3600))
PY
)"

eval "$(runpodctl ssh info "$pod_id" | python3 -c '
import json, shlex, sys
d=json.load(sys.stdin)
print("IP="+shlex.quote(d["ip"]))
print("PORT="+shlex.quote(str(d["port"])))
print("KEY="+shlex.quote(d["ssh_key"]["path"]))
')"
ssh_args=(-i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$PORT")
remote_command="cd '$remote_root/project' && setsid timeout '$max_seconds' bash scripts/runpod/on-pod-train.sh '$remote_root' '$archive_name' > '$remote_root/train.log' 2>&1 </dev/null & echo LAUNCHED"
ssh "${ssh_args[@]}" root@"$IP" "$remote_command"
echo "Monitor with: runpodctl pod logs $pod_id --follow" >&2
echo "Or SSH-tail: $remote_root/train.log" >&2
