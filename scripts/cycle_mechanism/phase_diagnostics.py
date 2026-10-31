"""Validation-only phase scan and reproducible inference diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot(root: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True,
                text=True, encoding="utf-8",
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def _unpack_cycle(batch_cycle, device):
    if isinstance(batch_cycle, (tuple, list)):
        cycle_index, future_cycle_index = batch_cycle
    else:
        cycle_index, future_cycle_index = batch_cycle, None
    return cycle_index.to(device), (
        future_cycle_index.to(device) if future_cycle_index is not None else None
    )


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_checkpoint(model, checkpoint_path: Path, device) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    incompatible = model.load_state_dict(_state_dict(checkpoint), strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    critical = (
        "in_layer.temporal_query",
        "in_layer.word_embedding_base",
        "cycle_queue",
        "cycle_scale",
    )
    missing_critical = [key for key in critical if key in missing]
    if missing_critical:
        raise RuntimeError(
            "Checkpoint is missing required cycle/WTE weights: "
            + ", ".join(missing_critical)
        )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "missing_critical_keys": missing_critical,
    }


def _make_conditions(args, cycle_len: int) -> list[dict]:
    if args.evaluation_split == "test":
        if args.confirm_shift is None:
            raise ValueError(
                "Test confirmation requires --confirm_shift selected from validation."
            )
        shift = int(args.confirm_shift)
        return [
            {"name": "Clean Before", "mode": "clean", "offset": None, "seed": args.seed},
            {
                "name": f"TQ Confirm ({shift:+d} slots)",
                "mode": "tq_shift",
                "offset": shift,
                "seed": args.seed,
            },
            {"name": "Clean After", "mode": "clean", "offset": None, "seed": args.seed},
        ]

    requested = [int(value.strip()) for value in args.phase_offsets.split(",") if value.strip()]
    # Keep only one representative for offsets that are identical modulo the cycle.
    unique = {}
    for offset in requested:
        unique.setdefault(offset % cycle_len, offset)
    conditions = [
        {"name": "Clean Before", "mode": "clean", "offset": None, "seed": args.seed}
    ]
    for modulo, requested_offset in unique.items():
        signed_offset = modulo if modulo <= cycle_len // 2 else modulo - cycle_len
        conditions.append({
            "name": f"TQ Shift ({signed_offset:+d} slots)",
            "mode": "tq_shift",
            "offset": signed_offset,
            "requested_offsets": [
                value for value in requested if value % cycle_len == modulo
            ],
            "seed": args.seed,
        })
    conditions.append(
        {"name": "TQ Zero", "mode": "tq_zero", "offset": None, "seed": args.seed}
    )
    for random_seed in (2025, 2026, 2027):
        conditions.append({
            "name": f"TQ Independent Random Index (seed {random_seed})",
            "mode": "tq_random_index",
            "offset": None,
            "seed": random_seed,
        })
    conditions.append({
        "name": "Clean After",
        "mode": "clean",
        "offset": None,
        "seed": args.seed,
    })
    return conditions


def _evaluate_condition(
    model,
    loader,
    dataset,
    args,
    device,
    condition: dict,
    original_tq: torch.Tensor,
    original_queue: torch.Tensor,
):
    mode = condition["mode"]
    offset = condition["offset"] or 0
    cycle_len = original_tq.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(condition["seed"]))
    model.in_layer.temporal_query.data.copy_(original_tq)
    model.cycle_queue.data.copy_(original_queue)

    records = []
    cursor = 0
    pred_len = int(args.pred_len)
    horizon_splits = np.array_split(np.arange(pred_len), min(4, pred_len))

    try:
        if mode == "tq_zero":
            model.in_layer.temporal_query.data.zero_()

        with torch.no_grad():
            for batch_x, batch_y, _, _, batch_cycle in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cycle_index, future_cycle_index = _unpack_cycle(batch_cycle, device)
                original_cycle_index = cycle_index

                if mode == "tq_shift":
                    cycle_index = (cycle_index + offset) % cycle_len
                elif mode == "tq_random_index":
                    cycle_index = torch.randint(
                        0, cycle_len, cycle_index.shape, device=device, generator=generator
                    )

                output = model(
                    batch_x,
                    cycle_index=cycle_index,
                    future_cycle_index=future_cycle_index,
                )
                pred_text = output["outputs_text"][:, -pred_len:, :]
                pred_time = output["outputs_time"][:, -pred_len:, :]
                truth = batch_y[:, -pred_len:, :]
                diff_text = pred_text - truth
                diff_time = pred_time - truth
                text_mse = diff_text.square().mean(dim=(1, 2)).cpu().tolist()
                text_mae = diff_text.abs().mean(dim=(1, 2)).cpu().tolist()
                time_mse = diff_time.square().mean(dim=(1, 2)).cpu().tolist()
                time_mae = diff_time.abs().mean(dim=(1, 2)).cpu().tolist()
                text_horizon = diff_text.square().mean(dim=2)
                time_horizon = diff_time.square().mean(dim=2)
                text_horizon_mae = diff_text.abs().mean(dim=2)
                time_horizon_mae = diff_time.abs().mean(dim=2)

                for local_index in range(batch_x.shape[0]):
                    window_index = cursor + local_index
                    start_index = int(dataset.split_start) + window_index
                    input_end_index = start_index + int(args.seq_len) - 1
                    forecast_start_index = input_end_index + 1
                    row = {
                        "Condition": condition["name"],
                        "Condition_Mode": mode,
                        "Condition_Seed": int(condition["seed"]),
                        "Requested_Offset": offset if mode.endswith("shift") else np.nan,
                        "Applied_TQ_Offset": offset if mode in ("tq_shift", "both_shift") else np.nan,
                        "Applied_Queue_Offset": offset if mode in ("queue_shift", "both_shift") else np.nan,
                        "Window_Index": window_index,
                        "Input_Start_Row": start_index,
                        "Input_End_Row": input_end_index,
                        "Forecast_Start_Row": forecast_start_index,
                        "Input_End_Cycle_Slot": int(original_cycle_index[local_index].item()),
                        "Text_MSE": text_mse[local_index],
                        "Text_MAE": text_mae[local_index],
                        "Time_MSE": time_mse[local_index],
                        "Time_MAE": time_mae[local_index],
                    }
                    for group_index, steps in enumerate(horizon_splits, start=1):
                        if len(steps) == 0:
                            continue
                        start_step, end_step = int(steps[0]) + 1, int(steps[-1]) + 1
                        suffix = f"H{start_step}_{end_step}"
                        row[f"Text_MSE_{suffix}"] = float(
                            text_horizon[local_index, int(steps[0]):int(steps[-1]) + 1].mean().item()
                        )
                        row[f"Time_MSE_{suffix}"] = float(
                            time_horizon[local_index, int(steps[0]):int(steps[-1]) + 1].mean().item()
                        )
                        row[f"Text_MAE_{suffix}"] = float(
                            text_horizon_mae[local_index, int(steps[0]):int(steps[-1]) + 1].mean().item()
                        )
                        row[f"Time_MAE_{suffix}"] = float(
                            time_horizon_mae[local_index, int(steps[0]):int(steps[-1]) + 1].mean().item()
                        )
                    records.append(row)
                cursor += batch_x.shape[0]
    finally:
        model.in_layer.temporal_query.data.copy_(original_tq)
        model.cycle_queue.data.copy_(original_queue)

    return records


def capture_attention_by_slot(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Capture attention returned by the actual first-layer forward pass."""
    cycle_len = int(model.in_layer.temporal_query.shape[0])
    prototype_count = int(model.in_layer.word_embedding_base.shape[0])
    attention_sum = np.zeros((cycle_len, prototype_count), dtype=np.float64)
    sample_count = np.zeros(cycle_len, dtype=np.int64)
    captured = {}

    def capture(_module, _inputs, output):
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError("MultiheadAttention did not return attention weights.")
        if weights.ndim == 4:
            weights = weights.mean(dim=1)
        if weights.ndim != 3:
            raise RuntimeError(f"Unexpected attention tensor shape: {tuple(weights.shape)}")
        captured["weights"] = weights.detach().mean(dim=1).cpu()

    attention_module = model.in_layer.layers[0]["cross_attention"]
    hook = attention_module.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for batch_x, _, _, _, batch_cycle in loader:
                batch_x = batch_x.float().to(device)
                cycle_index, future_cycle_index = _unpack_cycle(batch_cycle, device)
                captured.clear()
                model(
                    batch_x,
                    cycle_index=cycle_index,
                    future_cycle_index=future_cycle_index,
                )
                if "weights" not in captured:
                    raise RuntimeError("No attention weights were captured from the model forward.")
                weights = captured["weights"].numpy()
                slots = cycle_index.detach().cpu().numpy().astype(np.int64)
                for slot in np.unique(slots):
                    mask = slots == slot
                    attention_sum[slot] += weights[mask].sum(axis=0)
                    sample_count[slot] += int(mask.sum())
    finally:
        hook.remove()

    means = np.divide(
        attention_sum,
        sample_count[:, None],
        out=np.zeros_like(attention_sum),
        where=sample_count[:, None] > 0,
    )
    return means, sample_count


def _relative_file(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def run_phase_diagnostics(args, model_factory, data_provider) -> Path:
    """Run P0 on validation data, with one optional preselected test confirmation."""
    root = Path.cwd().resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.evaluation_split == "test" and args.confirm_shift is None:
        raise ValueError("Pass a single --confirm_shift selected on validation before test evaluation.")
    if args.evaluation_split == "val" and args.confirm_shift is not None:
        raise ValueError("--confirm_shift is only for the one-time test confirmation run.")

    device = torch.device(args.device)
    dataset, loader = data_provider(args, flag=args.evaluation_split)
    model = model_factory(args, device).to(device)
    checkpoint_path = _relative_file(root, args.checkpoint)
    load_report = _load_checkpoint(model, checkpoint_path, device)
    model.eval()
    original_tq = model.in_layer.temporal_query.detach().clone()
    original_queue = model.cycle_queue.detach().clone()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_s{args.seed}"
    output_dir = _relative_file(root, args.phase_output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    conditions = _make_conditions(args, int(original_tq.shape[0]))
    all_records = []
    for condition in conditions:
        print(f"[P0] {condition['name']} on {args.evaluation_split}: {len(dataset)} windows")
        all_records.extend(
            _evaluate_condition(
                model, loader, dataset, args, device, condition, original_tq, original_queue
            )
        )

    per_window = pd.DataFrame(all_records)
    per_window.to_csv(output_dir / "per_window_errors.csv", index=False, encoding="utf-8-sig")
    summary = per_window.groupby(
        ["Condition", "Condition_Mode", "Condition_Seed", "Applied_TQ_Offset", "Applied_Queue_Offset"],
        dropna=False,
    ).agg(
        Windows=("Window_Index", "count"),
        Text_MSE=("Text_MSE", "mean"),
        Text_MAE=("Text_MAE", "mean"),
        Time_MSE=("Time_MSE", "mean"),
        Time_MAE=("Time_MAE", "mean"),
    ).reset_index()
    baseline = summary.loc[summary["Condition"] == "Clean Before"].iloc[0]
    summary["Text_MSE_Delta_pct"] = (summary["Text_MSE"] / baseline["Text_MSE"] - 1) * 100
    summary["Time_MSE_Delta_pct"] = (summary["Time_MSE"] / baseline["Time_MSE"] - 1) * 100
    summary.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    horizon_columns = [
        column for column in per_window.columns
        if column.startswith(("Text_MSE_H", "Text_MAE_H", "Time_MSE_H", "Time_MAE_H"))
    ]
    horizon_summary = per_window.groupby(
        ["Condition", "Condition_Mode", "Condition_Seed"], dropna=False
    )[horizon_columns].mean().reset_index()
    horizon_summary.to_csv(output_dir / "horizon_summary.csv", index=False, encoding="utf-8-sig")

    phase_summary = per_window.groupby(
        ["Condition", "Condition_Seed", "Applied_TQ_Offset", "Input_End_Cycle_Slot"],
        dropna=False,
    ).agg(
        Windows=("Window_Index", "count"),
        Text_MSE=("Text_MSE", "mean"),
        Time_MSE=("Time_MSE", "mean"),
        Text_MAE=("Text_MAE", "mean"),
        Time_MAE=("Time_MAE", "mean"),
    ).reset_index()
    phase_summary.to_csv(output_dir / "phase_slot_summary.csv", index=False, encoding="utf-8-sig")

    attention, attention_counts = capture_attention_by_slot(model, loader, device)
    attention_rows = [
        {
            "Cycle_Slot": slot,
            "Sample_Count": int(attention_counts[slot]),
            "Prototype_Index": prototype,
            "Mean_Attention": float(attention[slot, prototype]),
        }
        for slot in range(attention.shape[0])
        for prototype in range(attention.shape[1])
    ]
    pd.DataFrame(attention_rows).to_csv(
        output_dir / "attention_by_cycle_slot.csv", index=False, encoding="utf-8-sig"
    )

    clean_after = summary.loc[summary["Condition"] == "Clean After"].iloc[0]
    clean_restoration = {
        metric: float(abs(clean_after[metric] - baseline[metric]))
        for metric in ("Text_MSE", "Text_MAE", "Time_MSE", "Time_MAE")
    }
    raw_data_path = Path(args.data_path)
    if raw_data_path.is_absolute():
        data_file = raw_data_path
    else:
        data_root = _relative_file(root, args.root_path)
        data_file = (data_root / raw_data_path).resolve()
    wte_file = _relative_file(root, args.word_embedding_path)
    tracked_code = [
        root / "validate_cycle_mechanism.py",
        root / "scripts/cycle_mechanism/phase_diagnostics.py",
        root / "models/CALF.py",
        root / "data_provider/data_loader.py",
        root / "data_provider/data_factory.py",
    ]
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "P0 fixed-checkpoint diagnostics; not evidence of training-time gain",
        "split": args.evaluation_split,
        "args": vars(args),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "device": str(device),
        },
        "git": _git_snapshot(root),
        "inputs": {
            "checkpoint": str(checkpoint_path.relative_to(root)) if checkpoint_path.is_relative_to(root) else str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "data_file": str(data_file.relative_to(root)) if data_file.is_relative_to(root) else str(data_file),
            "data_file_sha256": _sha256(data_file),
            "word_embedding_file": str(wte_file.relative_to(root)) if wte_file.is_relative_to(root) else str(wte_file),
            "word_embedding_sha256": _sha256(wte_file),
            "data_rows_total": getattr(dataset, "raw_data_rows", None),
            "split_start_row": getattr(dataset, "split_start", None),
            "split_end_row": getattr(dataset, "split_end", None),
            "windows": len(dataset),
            "normalization": {
                "dataset_scaler": "StandardScaler fitted on Solar training split",
                "scaler_mean": dataset.scaler.mean_.tolist(),
                "scaler_scale": dataset.scaler.scale_.tolist(),
                "model_use_norm": int(args.use_norm),
            },
            "cycle_slot_semantics": "Row index modulo cycle; source TXT has no timestamps, so slots are not labeled as clock time.",
        },
        "checkpoint_load": load_report,
        "conditions": conditions,
        "clean_restoration_absolute_difference": clean_restoration,
        "code_sha256": {
            str(p.relative_to(root)): _sha256(p) for p in tracked_code if p.is_file()
        },
        "outputs": [
            "per_window_errors.csv",
            "summary_metrics.csv",
            "horizon_summary.csv",
            "phase_slot_summary.csv",
            "attention_by_cycle_slot.csv",
        ],
        "interpretation_limit": "These are post-training inference interventions on one checkpoint. They do not establish that TQ or the word embedding improves the final time branch after retraining.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[P0] Results and manifest saved under {output_dir}")
    return output_dir
