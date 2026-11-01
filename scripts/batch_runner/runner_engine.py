# -*- coding: utf-8 -*-
"""
Runner Engine for Batch Runner
Executes individual experiment runs via subprocess, streams stdout/stderr, extracts metrics,
writes per-task isolated logs, and handles graceful interruptions.
"""
import os
import sys
import re
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from .config_manager import ConfigManager
from .state_manager import StateManager


class RunnerEngine:
    def __init__(self, python_exe: Optional[str] = None):
        self.python_exe = self._resolve_python_exe(python_exe)
        self.current_process: Optional[subprocess.Popen] = None
        self.interrupted: bool = False

    @staticmethod
    def _resolve_python_exe(candidate: Optional[str]) -> str:
        """Find valid conda calf python or current interpreter."""
        candidates = [
            candidate,
            r"D:\software\conda\envs\calf\python.exe",
            sys.executable
        ]
        for p in candidates:
            if p and os.path.exists(p):
                return p.replace("\\", "/")
        return sys.executable.replace("\\", "/")

    def run_task(
        self,
        task: Dict[str, Any],
        state_mgr: StateManager,
        on_log_line: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Executes a single experiment task.
        Returns True if completed successfully, False otherwise.
        """
        task_id = task["task_id"]
        params = task["params"]
        checkpoints_root = state_mgr.checkpoints_root

        # Ensure logs directory exists
        logs_dir = os.path.join(checkpoints_root, "logs").replace("\\", "/")
        os.makedirs(logs_dir, exist_ok=True)

        cmd_args = ConfigManager.params_to_cmd_args(params)
        full_cmd = [self.python_exe, "run.py"] + cmd_args

        # Mark state as running
        state_mgr.mark_running(task_id)

        start_time = time.time()
        setting_captured = ""
        checkpoint_dir = ""
        mse_captured = None
        mae_captured = None

        log_file_path = os.path.join(logs_dir, f"{task_id}.log").replace("\\", "/")

        # Check if an intermediate epoch checkpoint exists to preserve previous logs
        from .state_manager import compute_setting_string
        expected_setting = compute_setting_string(params)
        latest_ckpt = os.path.join(checkpoints_root, expected_setting, "checkpoint_latest.pth").replace("\\", "/")
        is_resuming = os.path.exists(latest_ckpt) and os.path.exists(log_file_path)

        log_mode = "a" if is_resuming else "w"
        log_f = open(log_file_path, log_mode, encoding="utf-8")
        if is_resuming:
            log_f.write(f"\n{'='*70}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === RESUMING {task_id} FROM LAST COMPLETED EPOCH ===\n{'='*70}\n")
            log_f.flush()

        print(f"\n{'='*70}")
        print(f"[*] Starting {task_id} | Loop: {task.get('loop_signature', {})} | Hash: {task.get('task_hash')} | Resuming: {is_resuming}")
        print(f"[*] Command: {' '.join(full_cmd[:8])} ... [total {len(full_cmd)} args]")
        print(f"[*] Log file: {log_file_path} (mode={log_mode})")
        print(f"{'='*70}")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.current_process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1
            )

            # Stream output
            for line in self.current_process.stdout:
                line_str = line.rstrip()
                log_f.write(line_str + "\n")
                log_f.flush()

                if on_log_line:
                    on_log_line(line_str)
                else:
                    # Echo key progress lines to console
                    if any(kw in line_str for kw in [
                        "start training :", "testing :", "Epoch: ", "Best model stored",
                        "Validation loss decreased", "mse:", "mae:", "cost time:",
                        "[Epoch-Resume]", "[Resume Detection]", "[Resume Success]"
                    ]):
                        print(f"[{task_id}] {line_str}")

                # Capture setting
                if "start training :" in line_str:
                    match = re.search(r"start training\s*:\s*([^>]+)", line_str)
                    if match:
                        setting_captured = match.group(1).strip()
                        checkpoint_dir = os.path.join(checkpoints_root, setting_captured).replace("\\", "/")
                        state_mgr.mark_running(task_id, setting=setting_captured, checkpoint_dir=checkpoint_dir)
                elif "testing :" in line_str and not setting_captured:
                    match = re.search(r"testing\s*:\s*([^<]+)", line_str)
                    if match:
                        setting_captured = match.group(1).strip()
                        checkpoint_dir = os.path.join(checkpoints_root, setting_captured).replace("\\", "/")
                        state_mgr.mark_running(task_id, setting=setting_captured, checkpoint_dir=checkpoint_dir)

                # Capture MSE and MAE
                if "mse:" in line_str and "mae:" in line_str:
                    match = re.search(r"mse:\s*([0-9.]+),\s*mae:\s*([0-9.]+)", line_str)
                    if match:
                        mse_captured = float(match.group(1))
                        mae_captured = float(match.group(2))

            self.current_process.wait()
            return_code = self.current_process.returncode
            duration = time.time() - start_time
            self.current_process = None

            # Fallback: check results/<setting>/metrics.json if not captured from stdout
            if (mse_captured is None or mae_captured is None) and setting_captured:
                metrics_json_path = os.path.join("./results", setting_captured, "metrics.json")
                if os.path.exists(metrics_json_path):
                    try:
                        import json
                        with open(metrics_json_path, "r", encoding="utf-8") as f:
                            m_data = json.load(f)
                        mse_captured = m_data.get("mse")
                        mae_captured = m_data.get("mae")
                    except Exception:
                        pass

            if return_code == 0:
                metrics = {
                    "mse": mse_captured,
                    "mae": mae_captured
                }
                state_mgr.mark_completed(
                    task_id=task_id,
                    metrics=metrics,
                    duration_sec=duration,
                    setting=setting_captured,
                    checkpoint_dir=checkpoint_dir
                )
                print(f"[+] Task {task_id} COMPLETED in {duration:.1f}s | MSE: {mse_captured} | MAE: {mae_captured}")
                return True
            else:
                err_msg = f"Process exited with non-zero code: {return_code}"
                state_mgr.mark_failed(task_id, err_msg, duration_sec=duration)
                print(f"[-] Task {task_id} FAILED ({err_msg}) in {duration:.1f}s")
                return False

        except KeyboardInterrupt:
            self.interrupted = True
            print(f"\n[!] KeyboardInterrupt received. Terminating process for {task_id}...")
            if self.current_process:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
            duration = time.time() - start_time
            state_mgr.mark_failed(task_id, "Interrupted by user (Ctrl+C)", duration_sec=duration)
            raise

        except Exception as e:
            duration = time.time() - start_time
            state_mgr.mark_failed(task_id, str(e), duration_sec=duration)
            print(f"[-] Task {task_id} encountered exception: {e}")
            return False

        finally:
            log_f.close()

    def run_all_pending(
        self,
        tasks: List[Dict[str, Any]],
        state_mgr: StateManager
    ):
        """
        Sequentially runs all pending tasks in the batch.
        Automatically resumes from the last breakpoint.
        """
        self.interrupted = False
        summary = state_mgr.get_summary()
        pending_tasks = state_mgr.get_pending_tasks()

        print(f"\n{'='*70}")
        print(f"=== BATCH EXECUTION START: {state_mgr.batch_name} ===")
        print(f"Checkpoints Root: {state_mgr.checkpoints_root}")
        print(f"Total: {summary['total']} | Completed: {summary['completed']} | Pending: {len(pending_tasks)}")
        print(f"{'='*70}\n")

        if not pending_tasks:
            print("[OK] All tasks in this batch are already completed!")
            return

        for task in pending_tasks:
            if self.interrupted:
                break
            try:
                self.run_task(task, state_mgr)
            except KeyboardInterrupt:
                print("\n[!] Batch execution paused by user. Progress saved to batch_state.json.")
                break

        final_summary = state_mgr.get_summary()
        print(f"\n{'='*70}")
        print(f"=== BATCH EXECUTION FINISHED ===")
        print(f"Completed: {final_summary['completed']}/{final_summary['total']} ({final_summary['progress_pct']:.1f}%)")
        print(f"State saved at: {state_mgr.state_file_path}")
        print(f"{'='*70}\n")
