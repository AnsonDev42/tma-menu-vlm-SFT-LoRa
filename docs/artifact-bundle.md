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
| `luna_test_predictions` | Imported current-TMA Luna predictions used for the frozen comparison |
| `test_predictions`, `test_metrics`, `test_gate` | Frozen primary test evidence and durable identity gate |
| `robustness_test_predictions`, `robustness_test_metrics` | Held-out test derivative evidence |
| `hardware`, `commands` | GPU/runtime facts and replay commands |

The artifact spec also requires model ID/revision, dataset SHA-256, seed, hardware,
and commands. `menu-vlm bundle` copies only explicit files, hashes each role, and
writes `artifact-manifest.json`. `menu-vlm package` creates a deterministic tarball,
an internal file manifest, and an external SHA-256 sidecar. `verify-archive`
rejects checksum drift, path traversal, links, unsupported entries, and missing or
extra files before exposing the returned bundle.

Adapters and private evaluation output are not public release assets and are
ignored by Git. Do not merge the base model into the bundle.
