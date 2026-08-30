#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

if [[ $# -ne 4 ]]; then
  runpod_die "usage: $0 REMOTE_ROOT ARCHIVE_NAME LUNA_BASELINE_NAME LUNA_BASELINE_SHA256_NAME"
fi
remote_root="$1"
archive_name="$2"
luna_baseline_name="$3"
luna_baseline_sidecar_name="$4"
validate_remote_root "$remote_root"
validate_archive_name "$archive_name"
validate_archive_name "$luna_baseline_name"
validate_archive_name "$luna_baseline_sidecar_name"
[[ "$archive_name" != "$luna_baseline_name" && \
  "$archive_name" != "$luna_baseline_sidecar_name" && \
  "$luna_baseline_name" != "$luna_baseline_sidecar_name" ]] || \
  runpod_die "Dataset and Luna file names must be distinct"
project="$remote_root/project"
archive="$remote_root/incoming/$archive_name"
dataset="$remote_root/private-dataset"
run="$remote_root/run"
config="$project/configs/qwen3-vl-4b-lora.json"
luna_incoming="$remote_root/incoming/$luna_baseline_name"
luna_sidecar_incoming="$remote_root/incoming/$luna_baseline_sidecar_name"

export HF_HOME="$remote_root/hf-cache"
python3 -m pip install --break-system-packages "uv==0.12.6"
cd "$project"
uv sync --extra train --frozen
uv run menu-vlm verify-sidecar --file "$luna_incoming" \
  --sidecar "$luna_sidecar_incoming"
uv run menu-vlm verify-archive --archive "$archive" --output "$dataset"
uv run menu-vlm validate-dataset --dataset "$dataset"
uv run menu-vlm preflight --config "$config" --dataset "$dataset"
mkdir -p "$run/evaluations" "$run/repro"
cp "$luna_incoming" "$run/$luna_baseline_name"
cp "$luna_sidecar_incoming" "$run/$luna_baseline_sidecar_name"
uv run menu-vlm verify-sidecar --file "$run/$luna_baseline_name" \
  --sidecar "$run/$luna_baseline_sidecar_name"
nvidia-smi -q > "$run/hardware.txt"
printf '%q ' "$0" "$@" > "$run/commands.txt"
printf '\n' >> "$run/commands.txt"

set +e
uv run menu-vlm train --config "$config" --dataset "$dataset" --output "$run/training"
train_status=$?
set -e
if [[ $train_status -ne 0 ]]; then
  if grep -Eqi 'out of memory|cuda oom' "$remote_root/train.log"; then
    mv "$run/training" "$run/bf16-failed"
    python3 - "$config" "$run/repro/qlora-config.json" <<'PY'
import json, sys
config=json.load(open(sys.argv[1], encoding="utf-8"))
config["mode"]="qlora"
json.dump(config, open(sys.argv[2], "w", encoding="utf-8"), sort_keys=True, separators=(",", ":"))
PY
    config="$run/repro/qlora-config.json"
    uv run menu-vlm train --config "$config" --dataset "$dataset" --output "$run/training"
  else
    exit "$train_status"
  fi
fi

checkpoint_metrics="$run/evaluations/checkpoints.jsonl"
: > "$checkpoint_metrics"
for checkpoint in "$run"/training/checkpoints/checkpoint-*; do
  [[ -d "$checkpoint" ]] || continue
  name="$(basename "$checkpoint")"
  predictions="$run/evaluations/$name.validation.predictions.jsonl"
  metrics="$run/evaluations/$name.validation.metrics.json"
  uv run menu-vlm predict --config "$config" --dataset "$dataset" \
    --split-file validation.jsonl --adapter "$checkpoint" --output "$predictions"
  uv run menu-vlm evaluate --references "$dataset/validation.jsonl" \
    --predictions "$predictions" --output "$metrics"
  python3 - "$checkpoint" "$metrics" >> "$checkpoint_metrics" <<'PY'
import json, pathlib, sys
checkpoint=pathlib.Path(sys.argv[1]); metrics=json.load(open(sys.argv[2], encoding="utf-8"))
state=json.load(open(checkpoint / "trainer_state.json", encoding="utf-8"))
losses=[row["eval_loss"] for row in state.get("log_history", []) if "eval_loss" in row]
if not losses: raise SystemExit("checkpoint has no validation loss")
print(json.dumps({"checkpoint": str(checkpoint), "structural_f1": metrics["structural_f1"],
 "item_f1": metrics["item_f1"], "eval_loss": losses[-1]}, sort_keys=True, separators=(",", ":")))
PY
done
[[ -s "$checkpoint_metrics" ]] || { echo "no checkpoints were produced" >&2; exit 1; }
uv run menu-vlm select-checkpoint --metrics "$checkpoint_metrics" \
  --output "$run/checkpoint-selection.json"
selected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$run/checkpoint-selection.json")"
cp -a "$selected" "$run/selected-adapter"

uv run menu-vlm predict --config "$config" --dataset "$dataset" \
  --split-file validation.jsonl --adapter "$run/selected-adapter" \
  --output "$run/evaluations/selected.validation.predictions.jsonl"
uv run menu-vlm evaluate --references "$dataset/validation.jsonl" \
  --predictions "$run/evaluations/selected.validation.predictions.jsonl" \
  --output "$run/evaluations/selected.validation.metrics.json"
uv run menu-vlm predict --config "$config" --dataset "$dataset" \
  --split-file robustness_validation.jsonl --adapter "$run/selected-adapter" \
  --output "$run/evaluations/robustness-validation.predictions.jsonl"
uv run menu-vlm evaluate --references "$dataset/robustness_validation.jsonl" \
  --predictions "$run/evaluations/robustness-validation.predictions.jsonl" \
  --output "$run/evaluations/robustness-validation.metrics.json"
uv run menu-vlm predict --config "$config" --dataset "$dataset" --split-file test.jsonl \
  --adapter "$run/selected-adapter" --output "$run/test.predictions.jsonl"
dataset_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_sha256"])' "$dataset/manifest.json")"
gate_store="$remote_root/frozen-test-gates"
test_args=(evaluate-test --references "$dataset/test.jsonl" --predictions "$run/test.predictions.jsonl"
  --output "$run/test.metrics.json" --dataset-sha256 "$dataset_sha"
  --checkpoint "$run/selected-adapter" --gate-store "$gate_store"
  --luna-predictions "$run/$luna_baseline_name")
uv run menu-vlm "${test_args[@]}"
test_gate="$(find "$gate_store" -maxdepth 1 -type f -name '*.json' -print)"
[[ -f "$test_gate" && "$(find "$gate_store" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 1 ]] || \
  { echo "expected exactly one frozen test gate" >&2; exit 1; }
cp "$test_gate" "$run/test-gate.json"
uv run menu-vlm predict --config "$config" --dataset "$dataset" \
  --split-file robustness_test.jsonl --adapter "$run/selected-adapter" \
  --output "$run/evaluations/robustness-test.predictions.jsonl"
uv run menu-vlm evaluate --references "$dataset/robustness_test.jsonl" \
  --predictions "$run/evaluations/robustness-test.predictions.jsonl" \
  --output "$run/evaluations/robustness-test.metrics.json"

cp "$project/uv.lock" "$run/repro/uv.lock"
cp "$config" "$run/repro/training-config.json"
cp "$dataset/manifest.json" "$run/repro/dataset-manifest.json"
cp "$run/training/training-report.json" "$run/repro/processor-provenance.json"
cp "$remote_root/train.log" "$run/training.log"
python3 - "$run" "$luna_baseline_name" <<'PY'
import json, pathlib, sys
run=pathlib.Path(sys.argv[1])
luna_baseline_name=sys.argv[2]
spec={"schema_version":"1.0","model_id":"Qwen/Qwen3-VL-4B-Instruct",
"model_revision":"ebb281ec70b05090aa6165b016eac8ec08e71b17",
"dataset_sha256":json.load(open(run/"repro/dataset-manifest.json"))["dataset_sha256"],
"seed":20260829,"hardware":{"path":"hardware.txt"},"commands":["commands.txt"],"files":{
"adapter_config":"selected-adapter/adapter_config.json",
"adapter_weights":"selected-adapter/adapter_model.safetensors",
"processor_provenance":"repro/processor-provenance.json","dependency_lock":"repro/uv.lock",
"training_config":"repro/training-config.json","dataset_manifest":"repro/dataset-manifest.json",
"training_logs":"training.log","checkpoint_selection":"checkpoint-selection.json",
"checkpoint_candidates":"evaluations/checkpoints.jsonl",
"selected_validation_predictions":"evaluations/selected.validation.predictions.jsonl",
"selected_validation_metrics":"evaluations/selected.validation.metrics.json",
"robustness_validation_predictions":"evaluations/robustness-validation.predictions.jsonl",
"robustness_validation_metrics":"evaluations/robustness-validation.metrics.json",
"luna_test_predictions":luna_baseline_name,
"test_predictions":"test.predictions.jsonl","test_metrics":"test.metrics.json",
"robustness_test_predictions":"evaluations/robustness-test.predictions.jsonl",
"robustness_test_metrics":"evaluations/robustness-test.metrics.json",
"test_gate":"test-gate.json",
"hardware":"hardware.txt","commands":"commands.txt"}}
json.dump(spec, open(run/"artifact-spec.json","w"), sort_keys=True, separators=(",", ":"))
PY
uv run menu-vlm bundle --run-root "$run" --spec "$run/artifact-spec.json" \
  --output "$remote_root/artifact-bundle"
uv run menu-vlm package --source "$remote_root/artifact-bundle" \
  --archive "$remote_root/artifact-bundle.tar.gz"
echo TRAIN_EVAL_BUNDLE_DONE
