#!/usr/bin/env python3
"""Fail-closed Runpod cost quote and local deletion-guard receipt helpers."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

MONTH_HOURS = Decimal("730")
RUNNING_CONTAINER_USD_PER_GB_MONTH = Decimal("0.10")
RUNNING_VOLUME_USD_PER_GB_MONTH = Decimal("0.10")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    quote = commands.add_parser("quote")
    quote.add_argument("--gpu-catalog", type=Path, required=True)
    quote.add_argument("--user", type=Path, required=True)
    quote.add_argument("--gpu-id", required=True)
    quote.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), required=True)
    quote.add_argument("--hours", required=True)
    quote.add_argument("--container-gb", required=True)
    quote.add_argument("--volume-gb", required=True)
    quote.add_argument("--data-center-id")
    quote.add_argument("--cap-usd", required=True)

    receipt = commands.add_parser("receipt")
    receipt.add_argument("--quote", type=Path, required=True)
    receipt.add_argument("--pod-id", required=True)
    receipt.add_argument("--output", type=Path, required=True)

    remaining = commands.add_parser("remaining")
    remaining.add_argument("--receipt", type=Path, required=True)
    remaining.add_argument("--pod-id", required=True)

    args = parser.parse_args()
    try:
        if args.command == "quote":
            result = build_quote(args)
        elif args.command == "receipt":
            result = create_receipt(args.quote, args.pod_id, args.output)
        else:
            result = remaining_seconds(args.receipt, args.pod_id)
    except (FileExistsError, ValueError, json.JSONDecodeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def build_quote(args: argparse.Namespace) -> dict[str, Any]:
    catalog = _read_json(args.gpu_catalog)
    user = _read_json(args.user)
    if not isinstance(catalog, list) or not isinstance(user, dict):
        raise ValueError("Runpod catalog/user responses have unexpected JSON shapes")
    matches = [row for row in catalog if isinstance(row, dict) and row.get("gpuId") == args.gpu_id]
    if len(matches) != 1:
        raise ValueError("selected GPU must have exactly one current catalog entry")
    gpu = matches[0]
    price_key = "securePricePerHr" if args.cloud_type == "SECURE" else "communityPricePerHr"
    price = _positive_decimal(gpu.get(price_key), f"catalog {price_key}")
    hours = _positive_decimal(args.hours, "hours")
    seconds = int((hours * 3600).to_integral_value(rounding=ROUND_FLOOR))
    if seconds < 1:
        raise ValueError("training duration must be at least one second")
    container_gb = _nonnegative_integer(args.container_gb, "container-gb")
    volume_gb = _nonnegative_integer(args.volume_gb, "volume-gb")
    cap = _positive_decimal(args.cap_usd, "cap-usd")
    if cap > Decimal("15"):
        raise ValueError("the approved hard cost cap may not exceed $15")
    if args.data_center_id:
        placements = gpu.get("dataCenterAvailability")
        if not isinstance(placements, list):
            raise ValueError("catalog lacks per-data-center availability")
        placement = next(
            (
                row
                for row in placements
                if isinstance(row, dict) and row.get("dataCenterId") == args.data_center_id
            ),
            None,
        )
        if placement is None or str(placement.get("stockStatus", "none")).casefold() in {
            "none",
            "unavailable",
        }:
            raise ValueError("selected GPU is unavailable in the requested data center")
    compute_cost = hours * price
    storage_hourly = (
        Decimal(container_gb) * RUNNING_CONTAINER_USD_PER_GB_MONTH
        + Decimal(volume_gb) * RUNNING_VOLUME_USD_PER_GB_MONTH
    ) / MONTH_HOURS
    storage_cost = hours * storage_hourly
    total = compute_cost + storage_cost
    if total <= 0 or total > cap:
        raise ValueError(
            f"refusing launch: projected combined cost {_usd(total)} exceeds {_usd(cap)}"
        )
    balance = _positive_decimal(user.get("clientBalance"), "Runpod clientBalance")
    if total > balance:
        raise ValueError(
            f"refusing launch: projected combined cost {_usd(total)} "
            f"exceeds current balance {_usd(balance)}"
        )
    return {
        "schema_version": "1.0",
        "pricing_source": "runpodctl gpu list --include-unavailable",
        "storage_pricing_source": "Runpod pod pricing (running container/volume $0.10/GB/month)",
        "gpu_id": args.gpu_id,
        "cloud_type": args.cloud_type,
        "data_center_id": args.data_center_id,
        "hours": _decimal_text(hours),
        "max_seconds": seconds,
        "compute_price_per_hour_usd": _decimal_text(price),
        "container_gb": container_gb,
        "volume_gb": volume_gb,
        "compute_cost_usd": _decimal_text(compute_cost),
        "storage_cost_usd": _decimal_text(storage_cost),
        "total_cost_usd": _decimal_text(total),
        "cap_usd": _decimal_text(cap),
        "balance_usd": _decimal_text(balance),
    }


def create_receipt(quote_path: Path, pod_id: str, output: Path) -> dict[str, Any]:
    _validate_pod_id(pod_id)
    quote = _read_json(quote_path)
    if not isinstance(quote, dict):
        raise ValueError("cost quote has an invalid shape")
    cap = _positive_decimal(quote.get("cap_usd"), "quoted cap-usd")
    if cap > Decimal("15"):
        raise ValueError("cost quote exceeds the approved $15 maximum")
    seconds = int(quote.get("max_seconds", 0))
    if seconds < 1:
        raise ValueError("cost quote has no positive bounded duration")
    now = int(time.time())
    payload = {
        "schema_version": "1.0",
        "pod_id": pod_id,
        "created_epoch": now,
        "delete_after_epoch": now + seconds,
        "quote": quote,
        "watchdog_host_pid": os.getppid(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return payload


def remaining_seconds(receipt_path: Path, pod_id: str) -> dict[str, Any]:
    _validate_pod_id(pod_id)
    receipt = _read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("pod_id") != pod_id:
        raise ValueError("guard receipt does not belong to this pod")
    remaining = int(receipt.get("delete_after_epoch", 0)) - int(time.time())
    if remaining < 1:
        raise ValueError("pod deletion guard has expired")
    return {"pod_id": pod_id, "remaining_seconds": remaining}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    try:
        result = int(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be a nonnegative integer") from exc
    if str(result) != str(value) or result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _validate_pod_id(value: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None:
        raise ValueError("unsafe pod id")


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _usd(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.0001'))}"


if __name__ == "__main__":
    raise SystemExit(main())
