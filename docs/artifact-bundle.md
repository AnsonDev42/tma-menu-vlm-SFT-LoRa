# Adapter artifact bundle

The return bundle is adapter-only and must contain exactly these roles:

| Role | Evidence |
| --- | --- |
| `adapter_config`, `adapter_weights` | Selected PEFT adapter |
| `processor_provenance` | Native processor/template snapshot or hashed provenance |
| `dependency_lock`, `training_config` | Exact software and run configuration |
| `dataset_manifest` | Private-free release/split hashes and prompt provenance |
| `training_logs` | Loss, stop, and failure history |
| `checkpoint_candidates`, `checkpoint_selection` | Every candidate and structural F1 -> item F1 -> loss rationale |
| `selected_validation_predictions`, `selected_validation_metrics` | Canonical selected-checkpoint validation evidence |
| `robustness_validation_predictions`, `robustness_validation_metrics` | Held-out validation derivative evidence |
| `luna_test_predictions` | Imported current-TMA Luna predictions from the fixed `luna-test-predictions.jsonl` run path used for the frozen comparison |
| `luna_response_provenance` | Approved fixed-name private manifest binding each Luna prediction to exact provider raw-response bytes |
| `test_predictions`, `test_metrics`, `test_gate` | Frozen primary test evidence and durable identity gate |
| `robustness_test_predictions`, `robustness_test_metrics` | Held-out test derivative evidence |
| `hardware`, `commands` | GPU/runtime facts and replay commands |

The artifact spec also requires model ID/revision, dataset SHA-256, the exact
dataset-manifest file SHA-256, seed, hardware, and commands. `menu-vlm bundle`
copies only explicit files, hashes each role, and
writes `artifact-manifest.json`. Before publishing that manifest, bundling requires
an exact completed test-gate schema, verifies its typed identity and digest fields,
and binds all frozen evidence to the copied bundle bytes. The gate's prediction,
metrics, and Luna prediction SHA-256 values must equal the copied
`test_predictions`, `test_metrics`, and `luna_test_predictions` roles. The gate's
checkpoint identity is recomputed from the copied `adapter_config` and
`adapter_weights` with the evaluation-time directory identity algorithm. The copied
dataset manifest's dataset SHA-256 must equal both the artifact spec and gate. Its
copied file SHA-256 must equal the frozen gate identity exactly, so changing split
assignments, prompt/source provenance, or count allocation while preserving the
generated-files aggregate still fails. Its
exact schema and split/source accounting are revalidated, its aggregate is
recomputed from `files_sha256`, and `files_sha256["test.jsonl"]` must equal the
gate's reference SHA-256. The copied `luna_response_provenance` role must use its
fixed name and hash to the approved production aggregate (or the separately pinned,
explicit synthetic-only aggregate). Paths, symlinks, reserved-name collisions, and
duplicate source aliases are rejected. Its `baseline_prediction_sha256` must equal
the copied `luna_test_predictions` role as well as the gate's Luna digest, preventing
a changed baseline plus newly forged gate from bypassing unchanged provenance.
Mutating or
replacing any of these roles after test-gate consumption therefore fails bundle
creation and removes the partial output.
`menu-vlm package` creates a deterministic tarball,
an internal file manifest, and an external SHA-256 sidecar. `verify-archive`
rejects checksum drift, path traversal, links, unsupported entries, and missing or
extra files before exposing the returned bundle.

Adapters and private evaluation output are not public release assets and are
ignored by Git. Do not merge the base model into the bundle.
