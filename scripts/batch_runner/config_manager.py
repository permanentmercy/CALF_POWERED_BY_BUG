# -*- coding: utf-8 -*-
"""
Config Manager for Batch Runner
Handles configuration loading/saving, hyperparameter validation, and loop expansion (grid/zip).
"""
import os
import json
import itertools
import hashlib
from typing import Dict, Any, List, Tuple


def parse_range_spec(spec: str) -> List[Any]:
    """
    解析 '起点:步长:终点' 或 '起点:终点' 循环语法。
    示例:
      '0.1:0.1:0.5' -> [0.1, 0.2, 0.3, 0.4, 0.5]
      '2025:1:2027' -> [2025, 2026, 2027]
      '2025:2027'   -> [2025, 2026, 2027]
      '24:24:144'   -> [24, 48, 72, 96, 120, 144]
      '0.0001:0.0001:0.0004' -> [0.0001, 0.0002, 0.0003, 0.0004]
    """
    spec = spec.replace("：", ":").strip()
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    if len(parts) == 2:
        start_s, stop_s = parts[0], parts[1]
        step_s = "1"
    elif len(parts) == 3:
        start_s, step_s, stop_s = parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"无效的范围格式: '{spec}', 请使用 '起点:步长:终点' (例如 0.1:0.1:0.5 或 2025:1:2027)")

    is_int = ('.' not in start_s and '.' not in step_s and '.' not in stop_s)
    if is_int:
        start = int(start_s)
        step = int(step_s)
        stop = int(stop_s)
        if step == 0:
            raise ValueError("步长不能为0")
        vals = []
        cur = start
        if step > 0:
            while cur <= stop:
                vals.append(cur)
                cur += step
        else:
            while cur >= stop:
                vals.append(cur)
                cur += step
        return vals
    else:
        start = float(start_s)
        step = float(step_s)
        stop = float(stop_s)
        if step == 0:
            raise ValueError("步长不能为0")
        dec = max(
            len(start_s.split('.')[1]) if '.' in start_s else 0,
            len(step_s.split('.')[1]) if '.' in step_s else 0,
            len(stop_s.split('.')[1]) if '.' in stop_s else 0
        )
        dec = max(dec, 4)
        vals = []
        cur = start
        eps = abs(step) * 1e-6
        if step > 0:
            while cur <= stop + eps:
                vals.append(round(cur, dec))
                cur += step
        else:
            while cur >= stop - eps:
                vals.append(round(cur, dec))
                cur += step
        return vals


def parse_loop_values(raw: Any) -> List[Any]:
    """
    将用户输入解析为循环取值列表。
    支持逗号分隔枚举与 '起点:步长:终点' 范围写法。
    示例:
      '0.1:0.1:0.5' -> [0.1, 0.2, 0.3, 0.4, 0.5]
      '2025, 2026, 2027' -> [2025, 2026, 2027]
      '96, 192:96:384' -> [96, 192, 288, 384]
    """
    if isinstance(raw, list):
        res = []
        for item in raw:
            if isinstance(item, str) and (":" in item or "：" in item):
                res.extend(parse_range_spec(item))
            else:
                res.append(item)
        return res

    if not isinstance(raw, str):
        return [raw]

    raw_str = raw.strip()
    if not raw_str:
        return []

    tokens = [t.strip() for t in raw_str.split(",") if t.strip()]
    results = []
    for token in tokens:
        clean_tok = token.replace("：", ":")
        if ":" in clean_tok:
            results.extend(parse_range_spec(clean_tok))
        else:
            if clean_tok.isdigit():
                results.append(int(clean_tok))
            else:
                try:
                    results.append(float(clean_tok))
                except ValueError:
                    if clean_tok.lower() == "true":
                        results.append(True)
                    elif clean_tok.lower() == "false":
                        results.append(False)
                    else:
                        results.append(clean_tok)
    return results


class ConfigManager:
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            self.batch_name = "batch_run"
            self.checkpoints_root = "./checkpoints/batch_default"
            self.fixed_params = self._get_default_fixed_params()
            self.loop_params = {"random_seed": [2025, 2026, 2027]}
            self.mode = "grid"  # 'grid' (Cartesian product) or 'paired' (Zip)
        else:
            self.batch_name = config_dict.get("batch_name", "batch_run")
            self.checkpoints_root = config_dict.get("checkpoints_root", "./checkpoints/batch_default")
            self.fixed_params = config_dict.get("fixed_params", self._get_default_fixed_params())
            self.loop_params = config_dict.get("loop_params", {})
            self.mode = config_dict.get("mode", "grid")

    @staticmethod
    def _get_default_fixed_params() -> Dict[str, Any]:
        """Default baseline parameters for CALF Solar forecasting aligned with Solar_test.sh."""
        return {
            "task_name": "long_term_forecast",
            "is_training": 1,
            "root_path": "./datasets/Solar/",
            "data_path": "solar_AL.txt",
            "model_id": "CALF_96_96_p100",
            "model": "CALF",
            "data": "Solar",
            "seq_len": 96,
            "label_len": 0,
            "pred_len": 96,
            "percent": 100,
            "batch_size": 16,
            "accumulation_steps": 4,
            "learning_rate": 0.0004,
            "train_epochs": 100,
            "patience": 5,
            "d_model": 768,
            "n_heads": 4,
            "e_layers": 1,
            "d_layers": 1,
            "d_ff": 3072,
            "dropout": 0.1,
            "enc_in": 137,
            "c_out": 137,
            "gpt_layers": 1,
            "itr": 1,
            "cos": 1,
            "tmax": 10,
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "num_workers": 0,
            "task_w": 0.736,
            "feature_w": 0.05,
            "output_w": 0.214,
            "use_swa": 0,
            "tq_mode": "add",
            "use_tq_gate": 0,
            "layer_offset": 0,
            "test_branch": "time",
            "mlp_res_w": 0.0,
            "out_mlp_layers": 3,
            "cycle_prior_before_denorm": 0,
            "tq_dropout": 0.1,
            "tq_lr_factor": 5.0,
            "eval_test_every_epoch": True,
            "use_norm": 1,
            "use_direct_trend": 0,
            "tq_inject_target": "q",
            "gpt2_path": "./models/gpt2",
            "task_loss": "smooth_l1",
            "feature_loss": "smooth_l1",
            "output_loss": "smooth_l1",
            "cycle": 144,
            "data_split": "7:1:2",
            "bestmodel": True,
            "auto_resume": 1,
            "dec_in": 137,
            "checkpoint_path": "0"
        }

    def load_from_file(self, file_path: str):
        """Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.batch_name = data.get("batch_name", self.batch_name)
        self.checkpoints_root = data.get("checkpoints_root", self.checkpoints_root)
        self.fixed_params = data.get("fixed_params", self.fixed_params)
        self.loop_params = {k: parse_loop_values(v) for k, v in data.get("loop_params", {}).items()}
        self.mode = data.get("mode", "grid")

    def save_to_file(self, file_path: str):
        """Save current configuration to a JSON file."""
        folder = os.path.dirname(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        data = {
            "batch_name": self.batch_name,
            "checkpoints_root": self.checkpoints_root,
            "mode": self.mode,
            "loop_params": self.loop_params,
            "fixed_params": self.fixed_params
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def set_checkpoints_root(self, path: str):
        """Set master checkpoint root folder."""
        # Normalize to forward slashes for cross-platform compatibility
        normalized = path.replace("\\", "/").rstrip("/")
        self.checkpoints_root = normalized

    def set_fixed_param(self, key: str, value: Any):
        """Set a single fixed parameter value."""
        self.fixed_params[key] = value

    def set_loop_param(self, key: str, values: Any):
        """Set a parameter loop sweep, supporting range syntax (start:step:stop)."""
        self.loop_params[key] = parse_loop_values(values)

    def remove_param(self, key: str):
        """Remove a parameter from fixed or loop params."""
        self.fixed_params.pop(key, None)
        self.loop_params.pop(key, None)

    def generate_task_combos(self) -> List[Dict[str, Any]]:
        """
        Expands loop parameters into Cartesian product combinations (or paired zip).
        Merges with fixed parameters and attaches unified checkpoints directory.
        Returns a list of task specifications.
        """
        loop_keys = list(self.loop_params.keys())
        loop_val_lists = [self.loop_params[k] for k in loop_keys]

        combos = []
        if not loop_keys:
            # Single task without loop
            combos = [{}]
        elif self.mode == "paired":
            # Zip aligned
            min_len = min(len(v) for v in loop_val_lists)
            for i in range(min_len):
                combo = {loop_keys[j]: loop_val_lists[j][i] for j in range(len(loop_keys))}
                combos.append(combo)
        else:
            # Cartesian product
            for prod in itertools.product(*loop_val_lists):
                combo = {loop_keys[j]: prod[j] for j in range(len(loop_keys))}
                combos.append(combo)

        tasks = []
        task_counter = 1
        for loop_dict in combos:
            merged_params = dict(self.fixed_params)
            merged_params.update(loop_dict)

            # 1. 动态 output_w 计算与跳过约束: output_w = 1.0 - task_w - feature_w
            task_w = float(merged_params.get("task_w", 0.736))
            feature_w = float(merged_params.get("feature_w", 0.05))
            if "output_w" not in loop_dict or str(merged_params.get("output_w", "")).lower() in ["auto", "-1"]:
                calc_ow = round(1.0 - task_w - feature_w, 6)
                if calc_ow < 0:
                    # 跳过无效组合 (output_w < 0)
                    continue
                merged_params["output_w"] = calc_ow

            # 显式 output_w <= 0 同样视为无效并跳过
            if float(merged_params.get("output_w", 0)) <= 0:
                continue

            # 2. 动态 model_id 联动: {model}_{seq_len}_{pred_len}_p{percent}
            model = merged_params.get("model", "CALF")
            seq_len = merged_params.get("seq_len", 96)
            pred_len = merged_params.get("pred_len", 96)
            percent = merged_params.get("percent", 100)
            merged_params["model_id"] = f"{model}_{seq_len}_{pred_len}_p{percent:g}"

            # 4. 动态 d_ff 联动: d_ff = d_model * 4 (若 d_ff 未显式指定循环)
            if "d_ff" not in loop_dict:
                d_m = int(merged_params.get("d_model", 768))
                merged_params["d_ff"] = d_m * 4

            # Ensure master checkpoints root is strictly routed
            merged_params["checkpoints"] = self.checkpoints_root

            # Compute canonical signature hash
            canonical_repr = json.dumps(merged_params, sort_keys=True)
            task_hash = hashlib.md5(canonical_repr.encode("utf-8")).hexdigest()[:8]

            task_id = f"task_{task_counter:03d}"
            tasks.append({
                "task_idx": task_counter,
                "task_id": task_id,
                "task_hash": task_hash,
                "loop_signature": loop_dict,
                "params": merged_params
            })
            task_counter += 1

        return tasks

    @staticmethod
    def params_to_cmd_args(params: Dict[str, Any]) -> List[str]:
        """Convert a parameter dictionary into command-line arguments for run.py."""
        args = []
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            elif isinstance(v, (list, tuple)):
                args.append(f"--{k}")
                for item in v:
                    args.append(str(item))
            else:
                args.append(f"--{k}")
                args.append(str(v))
        return args
