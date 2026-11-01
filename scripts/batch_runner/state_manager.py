# -*- coding: utf-8 -*-
"""
State Manager for Batch Runner (v2.3)
Maintains batch_state.json for breakpoint resumption, progress tracking, and metric aggregation.
Features multi-tier task matching (hash -> loop signature -> setting -> physical disk probe)
and dynamic path synchronization to eliminate progress loss across restarts.
"""
import os
import re
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional


def compute_setting_string(p: Dict[str, Any], ii: int = 0) -> str:
    """
    Computes canonical experiment setting name exactly conforming to run.py.
    Used for disk checkpoint discovery and directory mapping.
    """
    task_name = p.get("task_name", "long_term_forecast")
    model_id = p.get("model_id", "CALF_96_96_p100")
    model = p.get("model", "CALF")
    data = p.get("data", "Solar")
    features = p.get("features", "M")
    seq_len = p.get("seq_len", 96)
    label_len = p.get("label_len", 0)
    pred_len = p.get("pred_len", 96)
    d_model = p.get("d_model", 768)
    n_heads = p.get("n_heads", 4)
    e_layers = p.get("e_layers", 1)
    d_layers = p.get("d_layers", 1)
    d_ff = p.get("d_ff", 3072)
    factor = p.get("factor", 1)
    embed = p.get("embed", "timeF")
    distil = p.get("distil", True)
    des = p.get("des", "test")
    gpt_layers = p.get("gpt_layers", 1)
    learning_rate = p.get("learning_rate", 0.0004)
    random_seed = p.get("random_seed", 2025)
    feature_w = p.get("feature_w", 0.05)
    output_w = p.get("output_w", 0.214)
    train_epochs = p.get("train_epochs", 100)
    batch_size = p.get("batch_size", 16)
    dropout = p.get("dropout", 0.1)
    lora_dropout = p.get("lora_dropout", 0.1)
    r = p.get("r", 8)

    setting = (
        f"{task_name}_{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_"
        f"dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}_eb{embed}_dt{distil}_"
        f"{des}_gpt{gpt_layers}_rl{learning_rate}_{random_seed}_{ii}_feature{feature_w}_output{output_w}_"
        f"train_epochs{train_epochs}_bs{batch_size}_dr{dropout}_ld{lora_dropout}_r{r}"
    )

    percent = float(p.get("percent", 100))
    _ckpt = str(p.get("checkpoint_path", "0")).strip()
    _ckpt_valid = _ckpt not in ["0", "None", ""]
    if percent < 100:
        p_tag = f"{percent:g}"
        if _ckpt_valid:
            setting += f"_fewshot{p_tag}pct"
        else:
            setting += f"_scratch{p_tag}pct"

    extra_tags = []
    if p.get("exp_tag", ""):
        extra_tags.append(str(p["exp_tag"]).strip())
    if float(p.get("task_w", 1.0)) != 1.0:
        extra_tags.append(f"tw{float(p['task_w']):g}")
    if int(p.get("cycle", 24)) != 24:
        extra_tags.append(f"cyc{p['cycle']}")
    if int(p.get("use_direct_trend", 0)) != 0:
        extra_tags.append(f"trend{p['use_direct_trend']}")
    if int(p.get("out_mlp_layers", 2)) != 2:
        extra_tags.append(f"omlp{p['out_mlp_layers']}")
    if str(p.get("tq_mode", "mul")) != "mul":
        extra_tags.append(f"tqm_{p['tq_mode']}")
    if str(p.get("data_split", "7:1:2")) != "7:1:2":
        extra_tags.append(f"split{str(p['data_split']).replace(':', '_')}")
    if str(p.get("lradj", "type1")) != "type1":
        extra_tags.append(f"lradj_{p['lradj']}")
    if int(p.get("use_tq", 1)) != 1:
        extra_tags.append("notq")

    if extra_tags:
        setting += "_" + "_".join(extra_tags)

    return setting


def probe_disk_checkpoint(
    checkpoints_root: str,
    task_id: str,
    loop_sig: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Scans physical disk under checkpoints_root, logs, and results directory
    to detect if this task has already been completed or partially trained.
    Returns dictionary with detection status and metrics.
    Strictly requires the setting directory or log to belong to this exact setting.
    """
    expected_setting = compute_setting_string(params)
    ckpt_dir = os.path.join(checkpoints_root, expected_setting).replace("\\", "/")
    has_dir = os.path.isdir(ckpt_dir)
    has_ckpt = os.path.exists(os.path.join(ckpt_dir, "checkpoint.pth"))

    mse_val: Optional[float] = None
    mae_val: Optional[float] = None
    is_completed = False

    # 1. Inspect logs: look in checkpoints_root/logs
    logs_dir = os.path.join(checkpoints_root, "logs").replace("\\", "/")
    cand_logs = []
    primary_log = os.path.join(logs_dir, f"{task_id}.log").replace("\\", "/")
    if os.path.exists(primary_log):
        cand_logs.append(primary_log)

    if os.path.isdir(logs_dir):
        for fname in os.listdir(logs_dir):
            if fname.endswith(".log") and fname != f"{task_id}.log":
                cand_logs.append(os.path.join(logs_dir, fname).replace("\\", "/"))

    for lpath in cand_logs:
        try:
            with open(lpath, "r", encoding="utf-8", errors="ignore") as lf:
                content = lf.read()
                # Must match this exact setting string in the log
                if expected_setting in content:
                    # Look for test results: mse:0.1699, mae:0.2253
                    m = re.findall(r"mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)", content)
                    if m:
                        mse_val = float(m[-1][0])
                        mae_val = float(m[-1][1])
                        is_completed = True
                        break
        except Exception:
            pass

    # 2. Inspect results/<setting>/metrics.json
    if not is_completed and (has_dir or has_ckpt):
        metrics_json = os.path.join("./results", expected_setting, "metrics.json")
        if os.path.exists(metrics_json):
            try:
                with open(metrics_json, "r", encoding="utf-8") as mj:
                    mdata = json.load(mj)
                    if mdata.get("mse") is not None and mdata.get("mae") is not None:
                        mse_val = float(mdata["mse"])
                        mae_val = float(mdata["mae"])
                        is_completed = True
            except Exception:
                pass

    # 3. Check setting folder's own metrics.json if present
    if not is_completed and has_dir:
        ckpt_metrics_json = os.path.join(ckpt_dir, "metrics.json")
        if os.path.exists(ckpt_metrics_json):
            try:
                with open(ckpt_metrics_json, "r", encoding="utf-8") as cmj:
                    mdata = json.load(cmj)
                    if mdata.get("mse") is not None and mdata.get("mae") is not None:
                        mse_val = float(mdata["mse"])
                        mae_val = float(mdata["mae"])
                        is_completed = True
            except Exception:
                pass

    return {
        "setting": expected_setting,
        "checkpoint_dir": ckpt_dir if (has_dir or has_ckpt) else "",
        "has_ckpt": has_ckpt,
        "is_completed": is_completed,
        "metrics": {"mse": mse_val, "mae": mae_val} if is_completed else None
    }


class StateManager:
    STATE_FILENAME = "batch_state.json"

    def __init__(self, checkpoints_root: str, batch_name: str = "batch_run"):
        self._checkpoints_root = ""
        self.state_file_path = ""
        self.checkpoints_root = checkpoints_root
        self.batch_name = batch_name
        self.state: Dict[str, Any] = {}

    @property
    def checkpoints_root(self) -> str:
        return self._checkpoints_root

    @checkpoints_root.setter
    def checkpoints_root(self, value: str):
        """Dynamic setter keeping state_file_path strictly synchronized."""
        norm_path = value.replace("\\", "/").rstrip("/") if value else "./checkpoints/batch_default"
        self._checkpoints_root = norm_path
        self.state_file_path = os.path.join(self._checkpoints_root, self.STATE_FILENAME).replace("\\", "/")

    def set_checkpoints_root(self, checkpoints_root: str, batch_name: Optional[str] = None):
        """Explicit method to safely switch checkpoints root and batch name."""
        self.checkpoints_root = checkpoints_root
        if batch_name:
            self.batch_name = batch_name

    def init_or_resume(self, tasks: List[Dict[str, Any]], reset_interrupted: bool = True) -> Dict[str, Any]:
        """
        Initializes new batch state or loads existing state to resume.
        Employs 4-tier matching (exact hash -> loop signature -> setting -> physical disk probe)
        to guarantee zero loss of completed progress.
        """
        if not os.path.exists(self.checkpoints_root):
            try:
                os.makedirs(self.checkpoints_root, exist_ok=True)
            except Exception:
                pass

        existing_state = {}
        # 1. Primary read: state_file_path in current checkpoints_root
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    existing_state = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read existing state file ({e}), creating backup.")
                backup_path = f"{self.state_file_path}.bak_{int(time.time())}"
                try:
                    os.rename(self.state_file_path, backup_path)
                except OSError:
                    pass
        else:
            # Fallback read: if batch_default has a state whose checkpoints_root matches ours
            default_state_path = "./checkpoints/batch_default/batch_state.json"
            if os.path.exists(default_state_path):
                try:
                    with open(default_state_path, "r", encoding="utf-8") as df:
                        cand = json.load(df)
                        cand_root = cand.get("checkpoints_root", "").replace("\\", "/").rstrip("/")
                        if cand_root == self.checkpoints_root:
                            existing_state = cand
                except Exception:
                    pass

        existing_tasks = existing_state.get("tasks", {})
        used_old_tids = set()
        merged_tasks = {}

        for t in tasks:
            tid = t["task_id"]
            thash = t["task_hash"]
            loop_sig = t.get("loop_signature", {})
            expected_setting = compute_setting_string(t["params"])

            matched_rec = None

            # Tier 1: Match by exact task_hash
            if tid in existing_tasks and existing_tasks[tid].get("task_hash") == thash and tid not in used_old_tids:
                matched_rec = existing_tasks[tid]
                used_old_tids.add(tid)
            else:
                for old_tid, old_rec in existing_tasks.items():
                    if old_tid not in used_old_tids and old_rec.get("task_hash") == thash:
                        matched_rec = old_rec
                        used_old_tids.add(old_tid)
                        break

            # Tier 2: Match by identical loop_signature
            if not matched_rec and loop_sig:
                for old_tid, old_rec in existing_tasks.items():
                    if old_tid not in used_old_tids:
                        old_sig = old_rec.get("loop_signature")
                        if old_sig and old_sig == loop_sig:
                            matched_rec = old_rec
                            used_old_tids.add(old_tid)
                            break

            # Tier 3: Match by setting string
            if not matched_rec:
                for old_tid, old_rec in existing_tasks.items():
                    if old_tid not in used_old_tids and old_rec.get("setting") == expected_setting:
                        matched_rec = old_rec
                        used_old_tids.add(old_tid)
                        break

            if matched_rec:
                status = matched_rec.get("status", "PENDING")
                # If was interrupted in the middle of running, reset to PENDING so it resumes cleanly
                if status == "RUNNING" and reset_interrupted:
                    status = "PENDING"
                    matched_rec["error"] = "Interrupted by user or shutdown; queued for resume"

                # Verify completed status against physical disk
                if status == "COMPLETED":
                    ckpt_dir = matched_rec.get("checkpoint_dir") or os.path.join(self.checkpoints_root, expected_setting).replace("\\", "/")
                    has_metrics = matched_rec.get("metrics") and matched_rec["metrics"].get("mse") is not None
                    has_real_ckpt = os.path.exists(os.path.join(ckpt_dir, "checkpoint.pth"))
                    has_setting_dir = os.path.isdir(ckpt_dir)
                    log_p = os.path.join(self.checkpoints_root, "logs", f"{tid}.log")
                    has_log = os.path.exists(log_p)

                    # If neither the checkpoint nor setting dir nor task log exists in this batch folder, reset to PENDING
                    if not has_real_ckpt and not has_setting_dir and not has_log:
                        status = "PENDING"
                        matched_rec["metrics"] = None
                        matched_rec["error"] = None
                    elif not has_metrics and not has_real_ckpt:
                        status = "PENDING"

                # If status still PENDING, perform Tier 4 disk probe to see if it actually finished
                if status == "PENDING":
                    disk_res = probe_disk_checkpoint(self.checkpoints_root, tid, loop_sig, t["params"])
                    if disk_res["is_completed"]:
                        status = "COMPLETED"
                        matched_rec["metrics"] = disk_res["metrics"]
                        matched_rec["setting"] = disk_res["setting"]
                        matched_rec["checkpoint_dir"] = disk_res["checkpoint_dir"]
                        matched_rec["error"] = None
                        print(f"[StateManager] Auto-recovered {tid} ({loop_sig}) as COMPLETED from disk (MSE: {disk_res['metrics']['mse']}, MAE: {disk_res['metrics']['mae']})")

                matched_rec["task_id"] = tid
                matched_rec["task_idx"] = t["task_idx"]
                matched_rec["task_hash"] = thash
                matched_rec["status"] = status
                matched_rec["params"] = t["params"]
                matched_rec["loop_signature"] = loop_sig
                if not matched_rec.get("setting"):
                    matched_rec["setting"] = expected_setting
                merged_tasks[tid] = matched_rec
            else:
                # Brand new task: Check Tier 4 disk probe before marking PENDING
                disk_res = probe_disk_checkpoint(self.checkpoints_root, tid, loop_sig, t["params"])
                init_status = "COMPLETED" if disk_res["is_completed"] else "PENDING"
                if disk_res["is_completed"]:
                    print(f"[StateManager] Auto-discovered completed task {tid} ({loop_sig}) on disk (MSE: {disk_res['metrics']['mse']}, MAE: {disk_res['metrics']['mae']})")

                merged_tasks[tid] = {
                    "task_idx": t["task_idx"],
                    "task_id": tid,
                    "task_hash": thash,
                    "status": init_status,
                    "loop_signature": loop_sig,
                    "setting": disk_res["setting"],
                    "checkpoint_dir": disk_res["checkpoint_dir"],
                    "metrics": disk_res["metrics"],
                    "start_time": None,
                    "end_time": None,
                    "duration_sec": None,
                    "error": None,
                    "params": t["params"]
                }

        self.state = {
            "batch_name": self.batch_name,
            "checkpoints_root": self.checkpoints_root,
            "created_at": existing_state.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(merged_tasks),
            "tasks": merged_tasks
        }
        self.save_state()
        return self.state

    def save_state(self):
        """Persist state dictionary atomically to batch_state.json in current checkpoints_root."""
        self.state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state["checkpoints_root"] = self.checkpoints_root
        self.state["batch_name"] = self.batch_name
        try:
            if not os.path.exists(self.checkpoints_root):
                os.makedirs(self.checkpoints_root, exist_ok=True)
            temp_path = f"{self.state_file_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.state_file_path):
                os.remove(self.state_file_path)
            os.rename(temp_path, self.state_file_path)
        except Exception as e:
            print(f"[Warning] Failed to atomically save state to {self.state_file_path}: {e}")

    def mark_running(self, task_id: str, setting: str = "", checkpoint_dir: str = ""):
        """Mark task as currently running."""
        if task_id in self.state.get("tasks", {}):
            task = self.state["tasks"][task_id]
            task["status"] = "RUNNING"
            task["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task["end_time"] = None
            task["duration_sec"] = None
            task["error"] = None
            if setting:
                task["setting"] = setting
            if checkpoint_dir:
                task["checkpoint_dir"] = checkpoint_dir
            self.save_state()

    def mark_completed(self, task_id: str, metrics: Dict[str, Any], duration_sec: float, setting: str = "", checkpoint_dir: str = ""):
        """Mark task as successfully completed."""
        if task_id in self.state.get("tasks", {}):
            task = self.state["tasks"][task_id]
            task["status"] = "COMPLETED"
            task["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task["duration_sec"] = round(duration_sec, 2)
            task["metrics"] = metrics
            task["error"] = None
            if setting:
                task["setting"] = setting
            if checkpoint_dir:
                task["checkpoint_dir"] = checkpoint_dir
            self.save_state()

    def mark_failed(self, task_id: str, error_msg: str, duration_sec: Optional[float] = None):
        """Mark task as failed."""
        if task_id in self.state.get("tasks", {}):
            task = self.state["tasks"][task_id]
            task["status"] = "FAILED"
            task["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if duration_sec:
                task["duration_sec"] = round(duration_sec, 2)
            task["error"] = str(error_msg)
            self.save_state()

    def reset_task(self, task_id: str):
        """Reset a single task back to PENDING."""
        if task_id in self.state.get("tasks", {}):
            task = self.state["tasks"][task_id]
            task["status"] = "PENDING"
            task["metrics"] = None
            task["error"] = None
            task["start_time"] = None
            task["end_time"] = None
            task["duration_sec"] = None
            self.save_state()

    def reset_all_failed_or_interrupted(self):
        """Reset all FAILED or RUNNING tasks back to PENDING."""
        for task in self.state.get("tasks", {}).values():
            if task["status"] in ["FAILED", "RUNNING"]:
                task["status"] = "PENDING"
                task["error"] = None
        self.save_state()

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Return list of tasks that have status PENDING."""
        tasks = []
        for task in self.state.get("tasks", {}).values():
            if task["status"] == "PENDING":
                tasks.append(task)
        # Sort by task_idx
        tasks.sort(key=lambda x: x.get("task_idx", 0))
        return tasks

    def get_summary(self) -> Dict[str, Any]:
        """Compute progress statistics."""
        tasks = list(self.state.get("tasks", {}).values())
        total = len(tasks)
        completed = sum(1 for t in tasks if t["status"] == "COMPLETED")
        failed = sum(1 for t in tasks if t["status"] == "FAILED")
        running = sum(1 for t in tasks if t["status"] == "RUNNING")
        pending = sum(1 for t in tasks if t["status"] == "PENDING")

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_pct": (completed / total * 100) if total > 0 else 0.0
        }

    def export_summary_csv(self, output_path: str):
        """Export completed results to a clean CSV."""
        tasks = sorted(list(self.state.get("tasks", {}).values()), key=lambda x: x.get("task_idx", 0))
        lines = []
        headers = ["task_id", "status", "mse", "mae", "duration_sec"]
        sig_keys = set()
        for t in tasks:
            sig_keys.update(t.get("loop_signature", {}).keys())
        sig_keys = sorted(list(sig_keys))
        full_headers = ["task_id", "status"] + sig_keys + ["mse", "mae", "duration_sec", "setting", "checkpoint_dir"]
        lines.append(",".join(full_headers))

        for t in tasks:
            row = [t.get("task_id", ""), t.get("status", "")]
            sig = t.get("loop_signature", {})
            for k in sig_keys:
                row.append(str(sig.get(k, "")))
            metrics = t.get("metrics") or {}
            row.append(f"{metrics.get('mse', ''):.6f}" if metrics.get("mse") is not None else "")
            row.append(f"{metrics.get('mae', ''):.6f}" if metrics.get("mae") is not None else "")
            row.append(str(t.get("duration_sec", "")))
            row.append(f'"{t.get("setting", "")}"')
            row.append(f'"{t.get("checkpoint_dir", "")}"')
            lines.append(",".join(row))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
