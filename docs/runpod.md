# Private Runpod training workflow

This is a batch job, not a service: it exposes no port, runs detached, persists
logs/artifacts under an explicit `/workspace/...` root, and is complete only when
the bundle verifies locally. The flow follows Runpod's documented fine-tune pod
shape: [create with a termination guard, detach long work, monitor logs, retrieve,
then delete the pod](https://github.com/runpod/runpodctl).

## Cost and stop policy

The hard combined compute/storage policy for this experiment is `$15`, retaining
`$5` of the available credit for recovery. `launch-pod.sh` and `run-training.sh`
both require an observed hourly price and maximum hours, reject projected cost
above `$15`, and apply both Runpod `--terminate-after` and remote `timeout` guards.

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
```

The package command writes `compiled.tar.gz.sha256`. Keep both files.

## 2. Launch with automatic deletion

Read the selected GPU's live hourly price and pick a maximum duration whose
product is at most `$15`. Supply an official/current PyTorch template ID:

```bash
scripts/runpod/launch-pod.sh tma-qwen3-vl TEMPLATE_ID "NVIDIA A40" 8 0.44 DC_ID
```

The script checks live CLI help/auth, creates no public port, enables SSH, waits
for an SSH banner, and prints the pod JSON. Capture the returned pod ID.

## 3. Transfer code and private data

```bash
scripts/runpod/transfer-to-pod.sh POD_ID "$PWD" \
  /absolute/private/compiled.tar.gz /workspace/tma-menu-vlm
```

This rsyncs public code without `.git`, `.venv`, or `.env`, and transfers the
private archive plus sidecar directly over SSH. No image or record enters Git or
a public artifact store.

## 4. Run detached and monitor

```bash
scripts/runpod/run-training.sh POD_ID /workspace/tma-menu-vlm \
  compiled.tar.gz 8 0.44

runpodctl pod logs POD_ID --follow
# Or use `runpodctl ssh info POD_ID`, then tail:
# /workspace/tma-menu-vlm/train.log
```

The remote script verifies checksums before extraction, installs the frozen
training environment, runs training, evaluates every validation checkpoint,
selects by structural/item/loss policy, evaluates robustness, consumes test once,
and creates a checksummed adapter bundle. Success ends with
`TRAIN_EVAL_BUNDLE_DONE`; a merely running pod is not proof.

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

Cleanup permanently removes the explicit remote private root and deletes the
pod. If a separately created network volume was used, delete it only after the
returned adapter is verified; network-volume storage bills independently.
