#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf -- "$scratch"' EXIT
cd "$repository_root"

uv run menu-vlm synthetic-release --output "$scratch/release"
uv run menu-vlm validate-release --release "$scratch/release"
uv run menu-vlm compile --release "$scratch/release" --output "$scratch/compiled-a"
uv run menu-vlm compile --release "$scratch/release" --output "$scratch/compiled-b"
cmp "$scratch/compiled-a/manifest.json" "$scratch/compiled-b/manifest.json"
uv run menu-vlm validate-dataset --dataset "$scratch/compiled-a"
uv run menu-vlm preflight --config configs/qwen3-vl-4b-lora.json \
  --dataset "$scratch/compiled-a" --no-download

uv run menu-vlm synthetic-luna-evaluation --dataset "$scratch/compiled-a" \
  --output "$scratch/tma-data"
uv run menu-vlm import-luna-baseline --dataset "$scratch/compiled-a" \
  --tma-data-root "$scratch/tma-data" --evaluation-run public-synthetic-luna \
  --response-provenance "$scratch/tma-data/luna-response-provenance.json" \
  --response-provenance-sidecar "$scratch/tma-data/luna-response-provenance.json.sha256" \
  --output "$scratch/luna-test-predictions.jsonl"
uv run menu-vlm verify-sidecar --file "$scratch/luna-test-predictions.jsonl" \
  --sidecar "$scratch/luna-test-predictions.jsonl.sha256"

uv run menu-vlm synthetic-predictions --references "$scratch/compiled-a/test.jsonl" \
  --output "$scratch/reordered-predictions.jsonl"
uv run menu-vlm evaluate --references "$scratch/compiled-a/test.jsonl" \
  --predictions "$scratch/reordered-predictions.jsonl" \
  --luna-predictions "$scratch/luna-test-predictions.jsonl" \
  --output "$scratch/reordered-metrics.json"
python3 - "$scratch/reordered-metrics.json" <<'PY'
import json, sys
metrics=json.load(open(sys.argv[1], encoding="utf-8"))
assert metrics["schema_valid_rate"] == metrics["structural_f1"] == 1.0, metrics
assert metrics["luna_baseline"]["schema_valid_rate"] == 1.0, metrics
PY

uv run menu-vlm synthetic-artifact-run --dataset-manifest "$scratch/compiled-a/manifest.json" \
  --response-provenance "$scratch/tma-data/luna-response-provenance.json" \
  --luna-predictions "$scratch/luna-test-predictions.jsonl" \
  --output "$scratch/synthetic-run"
uv run menu-vlm bundle --run-root "$scratch/synthetic-run" \
  --spec "$scratch/synthetic-run/artifact-spec.json" --output "$scratch/bundle"
uv run menu-vlm package --source "$scratch/bundle" --archive "$scratch/bundle.tar.gz"
uv run menu-vlm verify-archive --archive "$scratch/bundle.tar.gz" --output "$scratch/restored"
cmp "$scratch/bundle/artifact-manifest.json" "$scratch/restored/artifact-manifest.json"

uv run pytest
uv run ruff check .
uv run mypy src
uv run menu-vlm scan-public --repository "$repository_root"
echo "SYNTHETIC_SMOKE_DONE scratch=$scratch"
