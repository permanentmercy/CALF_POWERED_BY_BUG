# -*- coding: utf-8 -*-
"""
State Manager for Batch Runner
Maintains batch_state.json for breakpoint resumption, progress tracking, and metric aggregation.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional


class StateManager:
    STATE_FILENAME = "batch_state.json"

    def __init__(self, checkpoints_root: str, batch_name: str = "batch_run"):
        self.checkpoints_root = checkpoints_root.replace("\\", "/").rstrip("/")
        self.batch_name = batch_name
        self.state_file_path = os.path.join(self.checkpoints_root, self.STATE_FILENAME).replace("\\", "/")
        self.state: Dict[str, Any] = {}

    def init_or_resume(self, tasks: List[Dict[str, Any]], reset_interrupted: bool = True) -> Dict[str, Any]:
        """
        Initializes new batch state or loads existing state to resume.
        """
        if not os.path.exists(self.checkpoints_root):
            os.makedirs(self.checkpoints_root, exist_ok=True)

        existing_state = {}
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

        existing_tasks = existing_state.get("tasks", {})
        merged_tasks = {}

        for t in tasks:
            tid = t["task_id"]
            thash = t["task_hash"]

            # Match by task_hash first, then task_id
            matched_rec = None
            if tid in existing_tasks and existing_tasks[tid].get("task_hash") == thash:
                matched_rec = existing_tasks[tid]
            else:
                for old_tid, old_rec in existing_tasks.items():
                    if old_rec.get("task_hash") == thash:
                        matched_rec = old_rec
                        break

            if matched_rec:
                status = matched_rec.get("status", "PENDING")
                # If was interrupted in the middle of running, reset to PENDING so it resumes cleanly
                if status == "RUNNING" and reset_interrupted:
                    status = "PENDING"
                    matched_rec["error"] = "Interrupted by user or shutdown; queued for resume"

                # Verify completed status
                if status == "COMPLETED":
                    ckpt_dir = matched_rec.get("checkpoint_dir")
                    # Check if at least metrics exist or checkpoint exists
                    has_metrics = matched_rec.get("metrics") and matched_rec["metrics"].get("mse") is not None
                    if not has_metrics and ckpt_dir and not os.path.exists(os.path.join(ckpt_dir, "checkpoint.pth")):
                        status = "PENDING"

                matched_rec["status"] = status
                matched_rec["params"] = t["params"]
                matched_rec["loop_signature"] = t["loop_signature"]
                merged_tasks[tid] = matched_rec
            else:
                merged_tasks[tid] = {
                    "task_idx": t["task_idx"],
                    "task_id": tid,
                    "task_hash": thash,
                    "status": "PENDING",
                    "loop_signature": t["loop_signature"],
                    "setting": "",
                    "checkpoint_dir": "",
                    "metrics": None,
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
        """Persist state dictionary to batch_state.json."""
        self.state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            temp_path = f"{self.state_file_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            if os.path.exists(self.state_file_path):
                os.remove(self.state_file_path)
            os.rename(temp_path, self.state_file_path)
        except Exception as e:
            print(f"[Warning] Failed to atomically save state: {e}")

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
        # Find all loop signature keys
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
