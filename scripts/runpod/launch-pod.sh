#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -lt 6 || $# -gt 7 ]]; then
  runpod_die "usage: $0 NAME TEMPLATE_ID GPU_ID MAX_HOURS CONTAINER_GB VOLUME_GB [DATA_CENTER_ID]"
fi
name="$1"
template_id="$2"
gpu_id="$3"
max_hours="$4"
container_gb="$5"
volume_gb="$6"
data_center_id="${7:-}"

[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || runpod_die "NAME is unsafe"
[[ "$template_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || runpod_die "TEMPLATE_ID is unsafe"
[[ "$gpu_id" =~ ^[A-Za-z0-9][A-Za-z0-9\ ._\(\)-]{0,127}$ ]] || runpod_die "GPU_ID is unsafe"
if [[ -n "$data_center_id" && ! "$data_center_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
  runpod_die "DATA_CENTER_ID is unsafe"
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set in the environment}"
: "${RUNPOD_MAX_COST_USD:=15}"
command -v runpodctl >/dev/null || runpod_die "runpodctl is not installed"
validate_delete_retry_config

version="$(runpodctl version)"
python3 -c '
import re, sys
match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", sys.argv[1])
if match is None or tuple(map(int, match.groups())) < (2, 8, 0):
    raise SystemExit("runpodctl >= 2.8.0 is required for priced GPU catalog output")
' "$version"
echo "$version" >&2

scratch="$(mktemp -d)"
pod_id=""
guard_armed=false
watchdog_pid=""
cleanup_launch_failure() {
  status=$?
  trap - EXIT
  rm -rf -- "$scratch"
  if [[ $status -ne 0 && -n "$pod_id" && "$guard_armed" != true ]]; then
    if [[ -n "$watchdog_pid" ]] && kill -0 "$watchdog_pid" 2>/dev/null; then
      kill "$watchdog_pid" 2>/dev/null || true
      wait "$watchdog_pid" 2>/dev/null || true
    fi
    echo "Launch setup failed after creation; deleting pod $pod_id before returning." >&2
    delete_pod_until_absent "$pod_id"
  fi
  exit "$status"
}
trap cleanup_launch_failure EXIT
runpodctl user > "$scratch/user.json"
runpodctl gpu list --include-unavailable > "$scratch/gpu-catalog.json"
quote_args=(quote --gpu-catalog "$scratch/gpu-catalog.json" --user "$scratch/user.json"
  --gpu-id "$gpu_id" --cloud-type SECURE --hours "$max_hours"
  --container-gb "$container_gb" --volume-gb "$volume_gb" --cap-usd "$RUNPOD_MAX_COST_USD")
if [[ -n "$data_center_id" ]]; then
  quote_args+=(--data-center-id "$data_center_id")
fi
python3 "$script_dir/cost_guard.py" "${quote_args[@]}" > "$scratch/quote.json"

create_args=(pod create --name "$name" --template-id "$template_id" --gpu-id "$gpu_id"
  --gpu-count 1 --cloud-type SECURE --container-disk-in-gb "$container_gb"
  --volume-in-gb "$volume_gb" --ssh)
if [[ -n "$data_center_id" ]]; then
  create_args+=(--data-center-ids "$data_center_id")
fi
pod_json="$(runpodctl "${create_args[@]}")"
pod_id="$(python3 -c '
import json,sys
value=json.load(sys.stdin)
pod_id=value.get("id") if isinstance(value,dict) else None
if not isinstance(pod_id,str) or not pod_id: raise SystemExit("pod create response has no id")
print(pod_id)
' <<< "$pod_json")"
validate_pod_id "$pod_id"

guard_dir="${RUNPOD_GUARD_DIR:-${TMPDIR:-/tmp}/tma-menu-vlm-runpod-guards}"
mkdir -p "$guard_dir"
receipt="$guard_dir/$pod_id.json"
python3 "$script_dir/cost_guard.py" receipt --quote "$scratch/quote.json" \
  --pod-id "$pod_id" --output "$receipt" >/dev/null
max_seconds="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_seconds"])' "$scratch/quote.json")"
watchdog_log="$guard_dir/$pod_id.watchdog.log"
watchdog_ready="$scratch/watchdog.ready"
RUNPOD_DELETE_READY_FILE="$watchdog_ready" \
  nohup bash "$script_dir/delete-after.sh" "$pod_id" "$max_seconds" "$receipt" \
  > "$watchdog_log" 2>&1 </dev/null &
watchdog_pid=$!
for _ in {1..100}; do
  [[ ! -f "$watchdog_ready" ]] || break
  kill -0 "$watchdog_pid" 2>/dev/null || runpod_die "deletion watchdog exited before arming"
  sleep 0.05
done
[[ -f "$watchdog_ready" ]] || runpod_die "deletion watchdog did not arm within five seconds"
guard_armed=true

python3 -c '
import json,sys
pod=json.load(sys.stdin)
pod["cost_guard"]={"receipt":sys.argv[1],"watchdog_pid":int(sys.argv[2]),
"watchdog_log":sys.argv[3],
"limitation":"local watchdog requires this launcher host to remain running and awake"}
print(json.dumps(pod,sort_keys=True,separators=(",",":")))
' "$receipt" "$watchdog_pid" "$watchdog_log" <<< "$pod_json"
echo "Pod $pod_id created; local deletion watchdog PID $watchdog_pid is armed." >&2
echo "runpodctl has no server-side terminate-after flag; keep this host awake and retain $receipt." >&2
