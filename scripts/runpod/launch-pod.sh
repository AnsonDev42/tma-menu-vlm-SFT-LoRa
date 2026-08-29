#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 NAME TEMPLATE_ID GPU_ID MAX_HOURS HOURLY_RATE [DATA_CENTER_ID]" >&2
  exit 2
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set in the environment}"

name="$1"
template_id="$2"
gpu_id="$3"
max_hours="$4"
hourly_rate="$5"
data_center_id="${6:-}"

read -r terminate_after projected_cost < <(python3 - "$max_hours" "$hourly_rate" <<'PY'
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import sys

hours = Decimal(sys.argv[1])
rate = Decimal(sys.argv[2])
cost = hours * rate
if hours <= 0 or cost <= 0 or cost > Decimal("15"):
    raise SystemExit(f"refusing pod launch: projected cost ${cost} must be in (0, $15]")
deadline = datetime.now(UTC) + timedelta(seconds=float(hours * 3600))
print(deadline.isoformat(timespec="seconds").replace("+00:00", "Z"), cost)
PY
)

runpodctl version >&2
runpodctl user >/dev/null
runpodctl pod create --help >/dev/null
args=(pod create --name "$name" --template-id "$template_id" --gpu-id "$gpu_id"
  --ssh --terminate-after "$terminate_after" --wait)
if [[ -n "$data_center_id" ]]; then
  args+=(--data-center-ids "$data_center_id")
fi
echo "Launching projected \$$projected_cost pod; automatic deletion at $terminate_after" >&2
exec runpodctl "${args[@]}"
