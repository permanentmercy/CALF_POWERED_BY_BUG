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
        """Default baseline parameters for CALF Solar forecasting."""
        return {
            "task_name": "long_term_forecast",
            "is_training": 1,
            "root_path": "./dataset/Solar/",
            "data_path": "solar_AL.txt",
            "model_id": "CALF_96_96_p100",
            "model": "CALF",
            "data": "Solar",
            "seq_len": 96,
            "label_len": 0,
            "pred_len": 96,
            "d_model": 768,
            "n_heads": 4,
            "e_layers": 1,
            "d_layers": 1,
            "d_ff": 3072,
            "dropout": 0.1,
            "lora_dropout": 0.1,
            "r": 8,
            "train_epochs": 100,
            "batch_size": 16,
            "patience": 3,
            "learning_rate": 0.0004,
            "gpt_layers": 1,
            "feature_w": 0.05,
            "output_w": 0.214,
            "task_w": 1.0,
            "cycle": 24,
            "use_direct_trend": 0,
            "out_mlp_layers": 2,
            "data_split": "7:1:2",
            "percent": 100,
            "bestmodel": True
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
        self.loop_params = data.get("loop_params", self.loop_params)
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

    def set_loop_param(self, key: str, values: List[Any]):
        """Set a parameter loop sweep."""
        if not isinstance(values, list):
            values = [values]
        self.loop_params[key] = values

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
        for idx, loop_dict in enumerate(combos, 1):
            merged_params = dict(self.fixed_params)
            merged_params.update(loop_dict)

            # Ensure master checkpoints root is strictly routed
            merged_params["checkpoints"] = self.checkpoints_root

            # Compute canonical signature hash
            canonical_repr = json.dumps(merged_params, sort_keys=True)
            task_hash = hashlib.md5(canonical_repr.encode("utf-8")).hexdigest()[:8]

            task_id = f"task_{idx:03d}"
            tasks.append({
                "task_idx": idx,
                "task_id": task_id,
                "task_hash": task_hash,
                "loop_signature": loop_dict,
                "params": merged_params
            })

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
