import copy
import json
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

from .compiler import validate_compiled_dataset
from .jsonio import read_json, read_jsonl, write_jsonl
from .training import preflight_config, validate_module_targets

InferenceMode = Literal["bf16", "int8"]


class _PhaseTimer:
    def __init__(self, now: Callable[[], float]) -> None:
        self._now = now
        self.started = now()
        self.first_token_at: float | None = None
        self.vision_seconds = 0.0
        self._vision_started: float | None = None

    def logits(self, _input_ids: Any, scores: Any) -> Any:
        if self.first_token_at is None:
            self.first_token_at = self._now()
        return scores

    def vision_start(self, _module: Any, _inputs: Any) -> None:
        self._vision_started = self._now()

    def vision_end(self, _module: Any, _inputs: Any, _output: Any) -> None:
        if self._vision_started is not None:
            self.vision_seconds += self._now() - self._vision_started
            self._vision_started = None


def predict_dataset(
    config_path: Path,
    dataset_path: Path,
    split_file: str,
    adapter_path: Path,
    output_path: Path,
    *,
    max_new_tokens: int = 4096,
    batch_size: int = 1,
    inference_mode: InferenceMode = "bf16",
    wandb_project: str | None = None,
) -> dict[str, Any]:
    """Generate batched, CUDA-only JSON predictions for one compiled split."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if inference_mode not in ("bf16", "int8"):
        raise ValueError("inference_mode must be bf16 or int8")
    validate_compiled_dataset(dataset_path)
    config = read_json(config_path)
    preflight_config(config_path, no_download=False, dataset_path=dataset_path)
    root = dataset_path.resolve()
    rows = read_jsonl(root / split_file)
    if not rows:
        raise ValueError(f"Prediction split is empty: {split_file}")
    if output_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")

    import torch  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    _require_cuda(torch)
    quantization = BitsAndBytesConfig(load_in_8bit=True) if inference_mode == "int8" else None
    processor = AutoProcessor.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    processor.tokenizer.padding_side = "left"
    base = AutoModelForMultimodalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
        quantization_config=quantization,
        trust_remote_code=False,
    )
    _validate_cuda_model(base)
    validate_module_targets(config, (name for name, _ in base.named_modules()))
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    _validate_cuda_model(model)
    model.eval()

    run = _start_wandb(
        wandb_project,
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        inference_mode=inference_mode,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    predictions: list[dict[str, Any]] = []
    try:
        for batch in _batches(rows, batch_size):
            predictions.extend(
                _predict_batch(
                    batch,
                    root=root,
                    processor=processor,
                    model=model,
                    torch=torch,
                    max_new_tokens=max_new_tokens,
                    inference_mode=inference_mode,
                )
            )
        if len(predictions) != len(rows) or [row["example_id"] for row in predictions] != [
            row["example_id"] for row in rows
        ]:
            raise RuntimeError("Batched prediction did not preserve exact example accounting")
        write_jsonl(output_path, predictions)
        if run is not None:
            run.log(_aggregate_telemetry(predictions))
    finally:
        if run is not None:
            run.finish()
    return {
        "examples": len(predictions),
        "output": str(output_path),
        "batch_size": batch_size,
        "inference_mode": inference_mode,
    }


def _predict_batch(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    processor: Any,
    model: Any,
    torch: Any,
    max_new_tokens: int,
    inference_mode: InferenceMode,
) -> list[dict[str, Any]]:
    conversations = [_prediction_messages(row, root) for row in rows]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    preprocess_started = time.monotonic()
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")
    torch.cuda.synchronize()
    preprocessing_seconds = time.monotonic() - preprocess_started

    timer = _PhaseTimer(lambda: _cuda_time(torch))
    vision = _vision_module(model)
    pre_handle = vision.register_forward_pre_hook(timer.vision_start)
    post_handle = vision.register_forward_hook(timer.vision_end)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                logits_processor=[timer.logits],
            )
        finished_at = _cuda_time(torch)
    finally:
        pre_handle.remove()
        post_handle.remove()
    if timer.first_token_at is None:
        raise RuntimeError("Generation returned without producing a token")

    input_width = int(inputs["input_ids"].shape[-1])
    generated_ids = generated[:, input_width:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    if len(decoded) != len(rows):
        raise RuntimeError("Generated batch size does not match input batch size")
    generated_token_counts = _generated_token_counts(generated_ids, processor)
    ttft_seconds = timer.first_token_at - timer.started
    prefill_seconds = max(0.0, ttft_seconds - timer.vision_seconds)
    decode_seconds = max(0.0, finished_at - timer.first_token_at)
    peak_memory = int(torch.cuda.max_memory_allocated())
    result = []
    for row, raw, generated_tokens in zip(rows, decoded, generated_token_counts, strict=True):
        text = raw.strip()
        try:
            prediction: Any = json.loads(text)
        except json.JSONDecodeError:
            prediction = text
        result.append(
            {
                "example_id": row["example_id"],
                "prediction": prediction,
                "raw_output": text,
                "inference_mode": inference_mode,
                "batch_size": len(rows),
                "preprocessing_seconds": preprocessing_seconds,
                "vision_encoder_seconds": timer.vision_seconds,
                "prefill_seconds": prefill_seconds,
                "ttft_seconds": ttft_seconds,
                "decode_seconds": decode_seconds,
                "generated_tokens": generated_tokens,
                "decode_tokens_per_second": (
                    generated_tokens / decode_seconds if decode_seconds else None
                ),
                "latency_seconds": finished_at - timer.started + preprocessing_seconds,
                "output_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / decode_seconds if decode_seconds else None,
                "peak_memory_bytes": peak_memory,
            }
        )
    return result


def _prediction_messages(row: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    image_path = (root / row["image"]).resolve()
    if not image_path.is_relative_to(root) or not image_path.is_file():
        raise ValueError(f"Prediction image is missing: {row['example_id']}")
    messages: list[dict[str, Any]] = copy.deepcopy(row["messages"][:-1])
    for message in messages:
        for content in message["content"]:
            if content.get("type") == "image":
                content["image"] = str(image_path)
    return messages


def _require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BF16/INT8 prediction requires CUDA; CPU and macOS are unsupported")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Prediction requires a CUDA device with BF16 support")


def _validate_cuda_model(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError("Loaded model did not expose an auditable CUDA device map")
    unsafe = {str(device) for device in device_map.values() if not _is_cuda_device(device)}
    if unsafe:
        raise RuntimeError(f"CPU/disk model offload is forbidden: {sorted(unsafe)}")
    implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if implementation != "flash_attention_2":
        raise RuntimeError("Model did not activate FlashAttention 2")


def _is_cuda_device(device: Any) -> bool:
    return isinstance(device, int) or str(device).startswith("cuda")


def _vision_module(model: Any) -> Any:
    candidates = (
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
    )
    for candidate in candidates:
        vision = getattr(candidate, "visual", None)
        if vision is not None:
            return vision
    raise RuntimeError("Unable to instrument the vision encoder")


def _generated_token_counts(generated_ids: Any, processor: Any) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ignored = {
        value
        for value in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        )
        if isinstance(value, int)
    }
    return [sum(int(token) not in ignored for token in row) for row in generated_ids.tolist()]


def _cuda_time(torch: Any) -> float:
    torch.cuda.synchronize()
    return time.monotonic()


def _batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def _start_wandb(project: str | None, **config: str | int) -> Any | None:
    if project is None:
        return None
    if not project.strip() or "/" in project or "\\" in project:
        raise ValueError("wandb_project must be a public-safe project name, not a path")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requires the telemetry dependency extra") from exc
    settings = wandb.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        x_disable_machine_info=True,
        x_disable_meta=True,
        x_disable_stats=True,
        x_save_requirements=False,
    )
    return wandb.init(project=project, config=config, save_code=False, settings=settings)


def _aggregate_telemetry(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    keys = (
        "preprocessing_seconds",
        "vision_encoder_seconds",
        "prefill_seconds",
        "ttft_seconds",
        "decode_seconds",
        "generated_tokens",
        "decode_tokens_per_second",
        "peak_memory_bytes",
    )
    result: dict[str, int | float] = {"examples": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), int | float)]
        if values:
            result[f"{key}_mean"] = statistics.fmean(values)
    return result
