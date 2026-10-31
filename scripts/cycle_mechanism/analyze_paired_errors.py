"""Paired moving-block bootstrap for fixed-checkpoint diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _load_conditions(path: Path, baseline_name: str, treatment_name: str) -> dict:
    conditions = {baseline_name: {}, treatment_name: {}}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["Condition"]
            if name not in conditions:
                continue
            window = int(row["Window_Index"])
            conditions[name][window] = {
                metric: float(row[metric])
                for metric in ("Text_MSE", "Text_MAE", "Time_MSE", "Time_MAE")
            }
    baseline = conditions[baseline_name]
    treatment = conditions[treatment_name]
    if not baseline or baseline.keys() != treatment.keys():
        raise ValueError("Baseline and treatment must contain matching per-window origins.")
    indices = sorted(baseline)
    if indices != list(range(len(indices))):
        raise ValueError("Window_Index must be contiguous and start at zero.")
    return {
        "indices": indices,
        "baseline": [baseline[index] for index in indices],
        "treatment": [treatment[index] for index in indices],
    }


def _block_bootstrap(
    baseline: list[float],
    treatment: list[float],
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict:
    count = len(baseline)
    if block_length <= 0 or block_length > count:
        raise ValueError(f"Invalid block length {block_length} for {count} windows.")
    base_prefix = [0.0]
    treat_prefix = [0.0]
    for base_value, treat_value in zip(baseline, treatment):
        base_prefix.append(base_prefix[-1] + base_value)
        treat_prefix.append(treat_prefix[-1] + treat_value)

    rng = random.Random(seed)
    delta_samples = []
    relative_samples = []
    blocks_per_replicate = math.ceil(count / block_length)
    for _ in range(repetitions):
        remaining = count
        base_sum = 0.0
        treat_sum = 0.0
        for _ in range(blocks_per_replicate):
            if remaining <= 0:
                break
            start = rng.randint(0, count - block_length)
            length = min(block_length, remaining)
            end = start + length
            base_sum += base_prefix[end] - base_prefix[start]
            treat_sum += treat_prefix[end] - treat_prefix[start]
            remaining -= length
        base_mean = base_sum / count
        treat_mean = treat_sum / count
        delta_samples.append(treat_mean - base_mean)
        relative_samples.append((treat_mean / base_mean - 1) * 100 if base_mean else float("nan"))

    delta_samples.sort()
    relative_samples.sort()
    point_base = sum(baseline) / count
    point_treat = sum(treatment) / count
    return {
        "block_length_origins": block_length,
        "repetitions": repetitions,
        "mean_difference_treatment_minus_baseline": point_treat - point_base,
        "relative_change_pct": (point_treat / point_base - 1) * 100 if point_base else None,
        "difference_95pct_ci": [
            _percentile(delta_samples, 0.025),
            _percentile(delta_samples, 0.975),
        ],
        "relative_change_95pct_ci": [
            _percentile(relative_samples, 0.025),
            _percentile(relative_samples, 0.975),
        ],
    }


def summarize(
    input_path: Path,
    baseline_name: str,
    treatment_name: str,
    block_lengths: list[int],
    repetitions: int,
    seed: int,
    nonoverlap_stride: int,
) -> dict:
    loaded = _load_conditions(input_path, baseline_name, treatment_name)
    baseline_rows = loaded["baseline"]
    treatment_rows = loaded["treatment"]
    count = len(baseline_rows)
    metrics = ("Text_MSE", "Text_MAE", "Time_MSE", "Time_MAE")
    output = {
        "input_file": str(input_path),
        "baseline_condition": baseline_name,
        "treatment_condition": treatment_name,
        "windows": count,
        "bootstrap": {
            "method": "paired non-circular moving-block bootstrap over consecutive window origins",
            "seed": seed,
            "interval": "percentile 95% CI",
            "block_lengths_in_row_origins": block_lengths,
            "sampling_interval": "not converted to clock time; TXT timestamps are unavailable",
        },
        "fixed_checkpoint_metrics": {},
        "moving_block_bootstrap": {},
        "nonoverlap_origins": {},
        "interpretation_limit": "Intervals describe the fixed-checkpoint evaluation period only; they do not include training-seed uncertainty or establish a retraining gain.",
    }
    for metric in metrics:
        baseline = [row[metric] for row in baseline_rows]
        treatment = [row[metric] for row in treatment_rows]
        base_mean = sum(baseline) / count
        treat_mean = sum(treatment) / count
        output["fixed_checkpoint_metrics"][metric] = {
            "baseline_mean": base_mean,
            "treatment_mean": treat_mean,
            "difference_treatment_minus_baseline": treat_mean - base_mean,
            "relative_change_pct": (treat_mean / base_mean - 1) * 100 if base_mean else None,
        }
        output["moving_block_bootstrap"][metric] = [
            _block_bootstrap(baseline, treatment, block_length, repetitions, seed + i)
            for i, block_length in enumerate(block_lengths)
        ]
    stride_indices = list(range(0, count, nonoverlap_stride))
    output["nonoverlap_origins"] = {
        "stride_origins": nonoverlap_stride,
        "windows": len(stride_indices),
        "metrics": {
            metric: {
                "baseline_mean": sum(baseline_rows[i][metric] for i in stride_indices) / len(stride_indices),
                "treatment_mean": sum(treatment_rows[i][metric] for i in stride_indices) / len(stride_indices),
                "difference_treatment_minus_baseline": sum(
                    treatment_rows[i][metric] - baseline_rows[i][metric] for i in stride_indices
                ) / len(stride_indices),
            }
            for metric in metrics
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--baseline", default="Clean Before")
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--block-lengths", default="288,1008",
                        help="consecutive window-origin counts; not interpreted as clock days")
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--nonoverlap-stride", type=int, default=96)
    args = parser.parse_args()
    output_path = args.output or args.input.with_name("paired_block_bootstrap.json")
    result = summarize(
        args.input,
        args.baseline,
        args.treatment,
        [int(value) for value in args.block_lengths.split(",") if value.strip()],
        args.repetitions,
        args.seed,
        args.nonoverlap_stride,
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved paired bootstrap summary: {output_path}")


if __name__ == "__main__":
    main()
