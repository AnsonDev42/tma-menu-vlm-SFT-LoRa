#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REMOTE_ROOT ARCHIVE_NAME" >&2
  exit 2
fi
remote_root="$1"
archive_name="$2"
if [[ ! "$remote_root" =~ ^/workspace/[A-Za-z0-9._/-]+$ ]] || [[ "$remote_root" == *".."* ]] || \
  [[ ! "$archive_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "unsafe remote root or archive name" >&2
  exit 2
fi
project="$remote_root/project"
archive="$remote_root/incoming/$archive_name"
dataset="$remote_root/private-dataset"
run="$remote_root/run"
config="$project/configs/qwen3-vl-4b-lora.json"

export HF_HOME="$remote_root/hf-cache"
python3 -m pip install --break-system-packages "uv==0.12.6"
cd "$project"
uv sync --extra train --frozen
uv run menu-vlm verify-archive --archive "$archive" --output "$dataset"
uv run menu-vlm validate-dataset --dataset "$dataset"
uv run menu-vlm preflight --config "$config" --dataset "$dataset"
mkdir -p "$run/evaluations" "$run/repro"
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
  --split-file robustness_validation.jsonl --adapter "$run/selected-adapter" \
  --output "$run/evaluations/robustness-validation.predictions.jsonl"
uv run menu-vlm evaluate --references "$dataset/robustness_validation.jsonl" \
  --predictions "$run/evaluations/robustness-validation.predictions.jsonl" \
  --output "$run/evaluations/robustness-validation.metrics.json"
uv run menu-vlm predict --config "$config" --dataset "$dataset" --split-file test.jsonl \
  --adapter "$run/selected-adapter" --output "$run/test.predictions.jsonl"
dataset_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_sha256"])' "$dataset/manifest.json")"
test_args=(evaluate-test --references "$dataset/test.jsonl" --predictions "$run/test.predictions.jsonl"
  --output "$run/test.metrics.json" --dataset-sha256 "$dataset_sha" --checkpoint "$selected")
if [[ -n "${LUNA_TEST_PREDICTIONS:-}" ]]; then
  test_args+=(--luna-predictions "$LUNA_TEST_PREDICTIONS")
fi
uv run menu-vlm "${test_args[@]}"

cp "$project/uv.lock" "$run/repro/uv.lock"
cp "$config" "$run/repro/training-config.json"
cp "$dataset/manifest.json" "$run/repro/dataset-manifest.json"
cp "$run/training/training-report.json" "$run/repro/processor-provenance.json"
cp "$remote_root/train.log" "$run/training.log"
python3 - "$run" <<'PY'
import json, pathlib, sys
run=pathlib.Path(sys.argv[1])
spec={"schema_version":"1.0","model_id":"Qwen/Qwen3-VL-4B-Instruct",
"model_revision":"ebb281ec70b05090aa6165b016eac8ec08e71b17",
"dataset_sha256":json.load(open(run/"repro/dataset-manifest.json"))["dataset_sha256"],
"seed":20260829,"hardware":{"path":"hardware.txt"},"commands":["commands.txt"],"files":{
"adapter_config":"selected-adapter/adapter_config.json",
"adapter_weights":"selected-adapter/adapter_model.safetensors",
"processor_provenance":"repro/processor-provenance.json","dependency_lock":"repro/uv.lock",
"training_config":"repro/training-config.json","dataset_manifest":"repro/dataset-manifest.json",
"training_logs":"training.log","checkpoint_selection":"checkpoint-selection.json",
"predictions":"test.predictions.jsonl","metrics":"test.metrics.json",
"hardware":"hardware.txt","commands":"commands.txt"}}
json.dump(spec, open(run/"artifact-spec.json","w"), sort_keys=True, separators=(",", ":"))
PY
uv run menu-vlm bundle --run-root "$run" --spec "$run/artifact-spec.json" \
  --output "$remote_root/artifact-bundle"
uv run menu-vlm package --source "$remote_root/artifact-bundle" \
  --archive "$remote_root/artifact-bundle.tar.gz"
echo TRAIN_EVAL_BUNDLE_DONE
