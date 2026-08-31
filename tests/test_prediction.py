import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menu_vlm import prediction


class _Cuda:
    def is_available(self) -> bool:
        return True

    def is_bf16_supported(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def max_memory_allocated(self) -> int:
        return 123


class _Tensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key: Any) -> "_Tensor":
        if isinstance(key, tuple):
            return _Tensor([row[key[1]] for row in self.rows])
        raise AssertionError(key)

    def tolist(self) -> list[list[int]]:
        return self.rows


class _Inputs(dict[str, Any]):
    def to(self, device: str) -> "_Inputs":
        assert device == "cuda:0"
        return self


class _Hook:
    def remove(self) -> None:
        pass


class _Vision:
    def register_forward_pre_hook(self, hook: Any) -> _Hook:
        self.pre = hook
        return _Hook()

    def register_forward_hook(self, hook: Any) -> _Hook:
        self.post = hook
        return _Hook()


class _Model:
    def __init__(self) -> None:
        self.hf_device_map = {"": "cuda:0"}
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        self.model = SimpleNamespace(visual=_Vision())

    def named_modules(self) -> list[tuple[str, object]]:
        return []

    def eval(self) -> None:
        pass

    def generate(self, **kwargs: Any) -> _Tensor:
        self.model.visual.pre(self.model.visual, ())
        self.model.visual.post(self.model.visual, (), None)
        kwargs["logits_processor"][0](None, None)
        batch = kwargs["input_ids"].shape[0]
        return _Tensor([[10, 11, 20 + index, 99] for index in range(batch)])


class _Processor:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99, padding_side="right")

    def apply_chat_template(self, conversations: list[Any], **kwargs: Any) -> _Inputs:
        assert len(conversations) == 2
        assert kwargs["padding"] is True
        return _Inputs(input_ids=_Tensor([[10, 11], [10, 11]]))

    def batch_decode(self, generated: _Tensor, **kwargs: Any) -> list[str]:
        return ['{"s":[],"i":[]}', '{"s":[],"i":[]}']


def test_predict_batches_two_examples_and_emits_phase_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (dataset / name).write_bytes(b"image")
    rows = [_row("a", "a.jpg"), _row("b", "b.jpg")]
    written: list[dict[str, Any]] = []
    load: dict[str, Any] = {}
    model = _Model()
    torch = SimpleNamespace(cuda=_Cuda(), bfloat16="bf16", inference_mode=lambda: _Context())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: _Processor()),
            AutoModelForMultimodalLM=SimpleNamespace(
                from_pretrained=lambda *a, **kwargs: load.update(kwargs) or model
            ),
            BitsAndBytesConfig=lambda **kwargs: load.update(quantization=kwargs) or kwargs,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: model)),
    )
    monkeypatch.setattr(prediction, "validate_compiled_dataset", lambda path: {})
    monkeypatch.setattr(prediction, "preflight_config", lambda *a, **k: {})
    monkeypatch.setattr(prediction, "validate_module_targets", lambda *a, **k: {})
    monkeypatch.setattr(prediction, "read_json", lambda path: _config())
    monkeypatch.setattr(prediction, "read_jsonl", lambda path: rows)
    monkeypatch.setattr(prediction, "write_jsonl", lambda path, values: written.extend(values))

    report = prediction.predict_dataset(
        tmp_path / "config.json",
        dataset,
        "test.jsonl",
        tmp_path / "adapter",
        tmp_path / "predictions.jsonl",
        batch_size=2,
        inference_mode="int8",
    )

    assert report["examples"] == 2
    assert load["quantization"] == {"load_in_8bit": True}
    assert load["device_map"] == {"": "cuda:0"}
    assert load["attn_implementation"] == "flash_attention_2"
    assert [row["example_id"] for row in written] == ["a", "b"]
    for row in written:
        assert row["batch_size"] == 2
        assert row["generated_tokens"] == 1
        assert row["decode_tokens"] == 0
        assert row["decode_tokens_per_second"] == 0.0
        assert row["batch_decode_tokens"] == 0
        assert row["batch_decode_tokens_per_second"] == 0.0
        assert row["tokens_per_second"] == 0.0
        assert set(row) >= {
            "preprocessing_seconds",
            "vision_encoder_seconds",
            "prefill_seconds",
            "ttft_seconds",
            "decode_seconds",
            "decode_tokens_per_second",
        }


def test_prediction_fails_closed_without_cuda() -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        prediction._require_cuda(torch)


def test_model_validation_rejects_offload_and_missing_flash_attention() -> None:
    with pytest.raises(RuntimeError, match="offload"):
        prediction._validate_cuda_model(
            SimpleNamespace(
                hf_device_map={"visual": "cuda:0", "language": "cpu"},
                config=SimpleNamespace(_attn_implementation="flash_attention_2"),
            )
        )
    with pytest.raises(RuntimeError, match="FlashAttention 2"):
        prediction._validate_cuda_model(
            SimpleNamespace(
                hf_device_map={"": "cuda:0"}, config=SimpleNamespace(_attn_implementation="sdpa")
            )
        )


def test_wandb_receives_only_public_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    run = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
            init=lambda **kwargs: captured.update(kwargs) or run,
        ),
    )
    result = prediction._start_wandb(
        "public-benchmark",
        model_id="public-model",
        model_revision="abc",
        inference_mode="int8",
        batch_size=2,
        max_new_tokens=100,
    )
    assert result is run
    assert captured["project"] == "public-benchmark"
    assert captured["save_code"] is False
    assert captured["settings"].x_disable_meta is True
    assert captured["settings"].x_disable_stats is True
    assert set(captured["config"]) == {
        "model_id",
        "model_revision",
        "inference_mode",
        "batch_size",
        "max_new_tokens",
    }


def test_aggregate_telemetry_cannot_include_payloads_or_paths() -> None:
    logged = prediction._aggregate_telemetry(
        [
            {
                "example_id": "private-id",
                "raw_output": "private output",
                "adapter_path": "/private/adapter",
                "inference_mode": "int8",
                "batch_index": 0,
                "preprocessing_seconds": 1.0,
                "generated_tokens": 2,
                "decode_tokens": 1,
                "decode_tokens_per_second": 4.0,
                "batch_decode_tokens": 3,
                "batch_decode_tokens_per_second": 12.0,
            },
            {
                "example_id": "another-private-id",
                "inference_mode": "int8",
                "batch_index": 0,
                "preprocessing_seconds": 999.0,
                "generated_tokens": 3,
                "decode_tokens": 2,
                "decode_tokens_per_second": 8.0,
                "batch_decode_tokens": 3,
                "batch_decode_tokens_per_second": 12.0,
            },
            {
                "example_id": "last-private-id",
                "inference_mode": "int8",
                "batch_index": 1,
                "preprocessing_seconds": 3.0,
                "generated_tokens": 1,
                "decode_tokens": 0,
                "decode_tokens_per_second": 0.0,
                "batch_decode_tokens": 0,
                "batch_decode_tokens_per_second": 0.0,
            },
        ]
    )
    assert logged == {
        "examples": 3,
        "batches": 2,
        "preprocessing_seconds_mean": 2.0,
        "generated_tokens_mean": 2.0,
        "decode_tokens_mean": 1.0,
        "decode_tokens_per_second_mean": 4.0,
        "batch_decode_tokens_mean": 1.5,
        "batch_decode_tokens_per_second_mean": 6.0,
    }


def test_decode_throughput_excludes_ttft_token_and_handles_zero_duration() -> None:
    assert prediction._decode_token_counts([1, 2, 5]) == [0, 1, 4]
    assert prediction._throughput(0, 0.0) == 0.0
    assert prediction._throughput(0, 1.0) == 0.0
    assert prediction._throughput(3, 0.0) is None
    assert prediction._throughput(3, 0.5) == 6.0


class _Context:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


def _row(example_id: str, image: str) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "image": image,
        "messages": [
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
    }


def _config() -> dict[str, str]:
    return {"model_id": "public-model", "model_revision": "abc", "mode": "bf16"}
