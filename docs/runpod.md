# Private Runpod training workflow

This is a batch job, not a service: it exposes no port, runs detached, persists
logs/artifacts under an explicit `/workspace/...` root, and is complete only when
the bundle verifies locally. The flow follows Runpod's documented fine-tune pod
shape: [create with a termination guard, detach long work, monitor logs, retrieve,
then delete the pod](https://github.com/runpod/runpodctl).

## Cost and stop policy

The hard combined compute/storage policy for this experiment is `$15`, retaining
`$5` of the available credit for recovery. `launch-pod.sh` reads the selected
GPU's current on-demand price directly from `runpodctl gpu list
--include-unavailable`, requires explicit container/volume sizes, adds Runpod's
published running storage rates, checks the live account balance, and rejects a
combined quote above `$15`. Caller-provided hourly prices are not accepted.

Current `runpodctl` v2.12.0 has no `pod create --terminate-after` flag. The launch
script therefore creates the pod with supported flags and immediately arms a
detached local deletion watchdog. It writes a cost/expiry receipt; detached
training derives its GNU `timeout` from the receipt's remaining duration. The
watchdog is real but host-local: the Mac that launched it must remain running and
awake. It does not survive host power loss, reboot, forced process termination,
or the host sleeping through the deadline; after an ordinary delayed wake it can
only resume when the operating system schedules it. Explicit retrieval and
cleanup plus a final `runpodctl pod list` remain mandatory.

All scripted pod deletion is centralized on current `runpodctl` v2.12.0
`pod get`/`pod delete` commands. It retries CLI/control-plane failures with
exponential backoff capped at 30 seconds, repeats deletion when an accepted
delete remains visible, and finishes only after `pod get` returns typed
`not_found`. Launch setup failure after creation runs that confirmed deletion
synchronously and does not return while the pod may still be present. Permanent
authentication/configuration failures therefore intentionally leave cleanup
running until an operator restores access; they are not downgraded to success.

Start with one 48 GB A40 BF16 baseline, maximum three epochs. One BF16 CUDA OOM
may fall back to the pinned 4-bit QLoRA mode. A second OOM, non-finite loss,
checksum/schema/split failure, empty checkpoint set, or command failure stops the
workflow. Do not start a sweep. Pricing/availability are live state: inspect
`runpodctl gpu list --include-unavailable` immediately before launch.

## 1. Prerequisites and private package

```bash
runpodctl update
runpodctl version
export RUNPOD_API_KEY=...       # environment only; never a script/config value
runpodctl user                  # auth check before the first billable action
runpodctl ssh list-keys

uv run menu-vlm package --source /absolute/private/compiled \
  --archive /absolute/private/compiled.tar.gz

uv run menu-vlm approve-luna-response-provenance \
  --dataset /absolute/private/compiled \
  --tma-data-root /absolute/private/tma-menu-parser-data \
  --evaluation-run COMPLETED_CURRENT_LUNA_RUN \
  --canonical-baseline /absolute/private/luna-test-predictions.jsonl \
  --output /absolute/private/luna-response-provenance.json

uv run menu-vlm import-luna-baseline \
  --dataset /absolute/private/compiled \
  --tma-data-root /absolute/private/tma-menu-parser-data \
  --evaluation-run COMPLETED_CURRENT_LUNA_RUN \
  --response-provenance /absolute/private/luna-response-provenance.json \
  --response-provenance-sidecar /absolute/private/luna-response-provenance.json.sha256 \
  --output /absolute/private/luna-test-predictions.jsonl
```

The package command writes `compiled.tar.gz.sha256`; the importer writes
`luna-test-predictions.jsonl.sha256`. Keep those files and the approved fixed-name
provenance manifest/sidecar private. The importer
requires a complete `current-tma-core:gpt-5.6-luna` run, matches its evaluation
references to compiled test images by SHA-256, resolves provider responses only
through validated call paths under the explicit TMA data root, and rejects any
missing, duplicate, extra, malformed, or mismatched identity. It verifies every
raw-response digest against the checksum-verified, aggregate-pinned provenance
manifest before parsing; production has no fallback.

## 2. Launch with a bounded local deletion guard

Pick a maximum duration and explicit disk sizes whose live combined quote is at
most `$15`. Supply an official/current PyTorch template ID:

```bash
scripts/runpod/launch-pod.sh tma-qwen3-vl TEMPLATE_ID "NVIDIA A40" 8 20 100 DC_ID
```

The script requires `runpodctl >= 2.8.0`, checks auth/catalog/balance, creates no
public application port, enables SSH, prints the pod JSON plus `cost_guard`
receipt/watchdog fields, and returns once scheduled. Capture the pod ID and
receipt path, keep the launcher host running and awake, and independently confirm
the pod is gone in the Runpod console after the deadline. Retry the transfer
command if SSH is not ready yet.

## 3. Transfer code and private data

```bash
scripts/runpod/transfer-to-pod.sh POD_ID "$PWD" \
  /absolute/private/compiled.tar.gz \
  /absolute/private/luna-test-predictions.jsonl \
  /absolute/private/luna-test-predictions.jsonl.sha256 \
  /absolute/private/luna-response-provenance.json \
  /absolute/private/luna-response-provenance.json.sha256 \
  /workspace/tma-menu-vlm
```

This rsyncs public code without `.git`, `.venv`, or `.env`, and transfers the
private archive, Luna prediction JSONL, approved response provenance, and their
sidecars directly over SSH. It validates both Luna evidence checksums before any
remote action. No image, record, or Luna
prediction enters Git or a public artifact store. The Luna inputs must use the
reserved names shown above; the dataset archive may not reuse any of them.

## 4. Run detached and monitor

```bash
scripts/runpod/run-training.sh POD_ID /workspace/tma-menu-vlm \
  compiled.tar.gz luna-test-predictions.jsonl \
  luna-test-predictions.jsonl.sha256 luna-response-provenance.json \
  luna-response-provenance.json.sha256 /absolute/local/path/to/POD_ID.json

runpodctl pod logs POD_ID --follow
# Or use `runpodctl ssh info POD_ID`, then tail:
# /workspace/tma-menu-vlm/train.log
```

The remote script verifies checksums before extraction, installs the frozen
training environment, runs training, evaluates every validation checkpoint,
selects by structural/item/loss policy, reruns canonical selected validation,
evaluates validation/test robustness, consumes primary test once through its
fixed identity gate with the checksum-verified saved Luna baseline, and creates
a checksummed adapter bundle containing every prediction/metric stream plus the
exact Luna test predictions and approved raw-response provenance. The pod copies
both to their fixed run paths and rechecks both sidecars; it rechecks the baseline
again immediately before
the frozen test evaluation. Success ends with
`TRAIN_EVAL_BUNDLE_DONE`; a merely running pod is not proof.
Bundle creation independently revalidates the completed test gate and requires
its Luna SHA-256 to match the copied fixed-path baseline, closing any mutation
window between frozen evaluation and artifact packaging. The gate also freezes the
exact compiled manifest file SHA-256. Bundle creation binds the copied fixed-name
provenance role to the public approved aggregate anchor and requires its baseline
digest to equal the copied Luna prediction bytes.

## 5. Retrieve and verify before deletion

```bash
scripts/runpod/retrieve.sh POD_ID /workspace/tma-menu-vlm \
  /absolute/private/local-results
```

The script rsyncs archive and sidecar and runs the safe local verifier. Inspect
`artifact-bundle/artifact-manifest.json`, metrics, selection, and logs before
cleanup.

## 6. Remove private remote state and billing

After retrieval verifies:

```bash
scripts/runpod/cleanup.sh POD_ID /workspace/tma-menu-vlm --delete-pod
runpodctl pod list
```

Cleanup permanently removes the explicit remote private root, requests pod
deletion, and continues checking until `runpodctl pod get` confirms `not_found`.
If a separately created network volume was used, delete it only after the
returned adapter is verified; network-volume storage bills independently.
