import copy
import json
import time
from pathlib import Path
from typing import Any

from .compiler import validate_compiled_dataset
from .jsonio import read_json, read_jsonl, write_jsonl
from .training import preflight_config, validate_module_targets


def predict_dataset(
    config_path: Path,
    dataset_path: Path,
    split_file: str,
    adapter_path: Path,
    output_path: Path,
    *,
    max_new_tokens: int = 4096,
) -> dict[str, Any]:
    """Generate deterministic JSON predictions for one compiled split on Runpod."""
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

    quantization = None
    if config["mode"] == "qlora":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    processor = AutoProcessor.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=False
    )
    base = AutoModelForMultimodalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=False,
    )
    validate_module_targets(config, (name for name, _ in base.named_modules()))
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    predictions = []
    for row in rows:
        image_path = (root / row["image"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Prediction image is missing: {row['example_id']}")
        messages = copy.deepcopy(row["messages"][:-1])
        for message in messages:
            for content in message["content"]:
                if content.get("type") == "image":
                    content["image"] = str(image_path)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        elapsed = time.monotonic() - started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = int(generated.shape[-1]) - prompt_tokens
        decoded = processor.decode(generated[0][prompt_tokens:], skip_special_tokens=True).strip()
        try:
            prediction: Any = json.loads(decoded)
        except json.JSONDecodeError:
            prediction = decoded
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        predictions.append(
            {
                "example_id": row["example_id"],
                "prediction": prediction,
                "raw_output": decoded,
                "latency_seconds": elapsed,
                "output_tokens": output_tokens,
                "tokens_per_second": output_tokens / elapsed if elapsed else None,
                "peak_memory_bytes": peak_memory,
            }
        )
    write_jsonl(output_path, predictions)
    return {"examples": len(predictions), "output": str(output_path)}
