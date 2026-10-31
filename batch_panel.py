# -*- coding: utf-8 -*-
"""
CALF Batch Experiment Panel & Checkpoint Manager
Unified entry point for batch hyperparameter sweep, breakpoint continuation,
and centralized checkpoint management.

Usage:
  Interactive TUI:
    python batch_panel.py

  Headless CLI:
    python batch_panel.py --config configs/batch_templates/solar_al_seeds.json --run
    python batch_panel.py --config configs/my_run.json --checkpoints ./checkpoints/my_dir/ --resume
    python batch_panel.py --checkpoints ./checkpoints/my_dir/ --export results.csv
"""
import os
import sys
import argparse

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

from scripts.batch_runner.panel_ui import BatchPanelUI
from scripts.batch_runner.config_manager import ConfigManager
from scripts.batch_runner.state_manager import StateManager
from scripts.batch_runner.runner_engine import RunnerEngine
from scripts.batch_runner.gui_panel import launch_gui


def main():
    parser = argparse.ArgumentParser(
        description="CALF Batch Experiment Control Panel & Breakpoint Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to JSON configuration file (e.g. configs/batch_templates/solar_al_seeds.json)")
    parser.add_argument("--checkpoints", "-ckpt", type=str, default=None,
                        help="Target master checkpoint directory to store all models for this batch")
    parser.add_argument("--run", "-r", action="store_true", default=False,
                        help="Run or resume batch execution in headless/CLI mode")
    parser.add_argument("--reset-failed", action="store_true", default=False,
                        help="Reset failed or interrupted tasks before running")
    parser.add_argument("--export", type=str, default=None,
                        help="Export current batch results to specified CSV path and exit")
    parser.add_argument("--cli", "--tui", action="store_true", default=False,
                        help="Launch terminal CLI/TUI menu instead of graphical desktop GUI")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Force launch graphical desktop GUI (default mode)")

    args = parser.parse_args()

    # Headless CLI mode when --run or --export is specified
    if args.run or args.export:
        print("\n=== CALF Batch Runner (Headless CLI Mode) ===")
        cfg_mgr = ConfigManager()
        if args.config and os.path.exists(args.config):
            print(f"[*] Loading config from: {args.config}")
            cfg_mgr.load_from_file(args.config)
        else:
            print("[*] Using default baseline configuration")

        if args.checkpoints:
            print(f"[*] Overriding checkpoints root to: {args.checkpoints}")
            cfg_mgr.set_checkpoints_root(args.checkpoints)

        state_mgr = StateManager(cfg_mgr.checkpoints_root, cfg_mgr.batch_name)
        tasks = cfg_mgr.generate_task_combos()
        state_mgr.init_or_resume(tasks, reset_interrupted=True)

        if args.reset_failed:
            print("[*] Resetting failed/interrupted tasks...")
            state_mgr.reset_all_failed_or_interrupted()

        if args.run:
            engine = RunnerEngine()
            engine.run_all_pending(tasks, state_mgr)

        if args.export:
            print(f"[*] Exporting results to: {args.export}")
            state_mgr.export_summary_csv(args.export)
            print("[OK] Export completed.")
        return

    # Terminal CLI/TUI mode if explicitly requested
    if args.cli:
        panel = BatchPanelUI(config_path=args.config, checkpoints_root=args.checkpoints)
        panel.menu_main()
        return

    # Graphical Desktop GUI mode (Default)
    try:
        launch_gui(initial_config=args.config, initial_checkpoints=args.checkpoints)
    except Exception as e:
        print(f"[Warning] Failed to launch GUI ({e}). Falling back to terminal TUI menu.")
        panel = BatchPanelUI(config_path=args.config, checkpoints_root=args.checkpoints)
        panel.menu_main()


if __name__ == "__main__":
    main()
