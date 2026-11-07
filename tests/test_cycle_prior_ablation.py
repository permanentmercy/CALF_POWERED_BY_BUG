# -*- coding: utf-8 -*-
"""
Automated unit tests for cycle_queue ablation mechanism (--use_cycle_prior).
Verifies:
1. Model forward pass output difference between use_cycle_prior=1 and use_cycle_prior=0.
2. Gradient freeze (requires_grad=False) when use_cycle_prior=0.
3. Setting string generation isolation (_nocycq).
4. Batch template loading and task combinations in ConfigManager.
"""
import os
import sys
import torch
import torch.nn as nn

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.batch_runner.config_manager import ConfigManager
from scripts.batch_runner.state_manager import compute_setting_string


class DummyConfig:
    def __init__(self, use_cycle_prior=1):
        self.task_name = "long_term_forecast"
        self.seq_len = 96
        self.pred_len = 96
        self.d_model = 768
        self.n_heads = 4
        self.e_layers = 1
        self.d_layers = 1
        self.d_ff = 3072
        self.dropout = 0.1
        self.enc_in = 137
        self.c_out = 137
        self.cycle = 144
        self.use_cycle_prior = use_cycle_prior
        self.cycle_prior_before_denorm = 1
        self.use_norm = 1
        self.use_direct_trend = 0
        self.gpt_layers = 1
        self.gpt2_path = "./models/gpt2"
        self.word_embedding_path = "wte_pca_500.pt"
        self.r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.out_mlp_layers = 3
        self.tq_mode = "add"
        self.tq_inject_target = "q"
        self.use_tq = 1
        self.use_tq_gate = 0
        self.tq_dropout = 0.1
        self.mlp_res_w = 0.0


def test_setting_string():
    print("[Test 1/4] Verifying setting string isolation...")
    p_base = {
        "model_id": "CALF_96_96_p100", "model": "CALF", "data": "Solar",
        "seq_len": 96, "label_len": 0, "pred_len": 96,
        "d_model": 768, "n_heads": 4, "e_layers": 1, "d_layers": 1, "d_ff": 3072,
        "factor": 1, "embed": "timeF", "distil": True, "des": "test",
        "gpt_layers": 1, "learning_rate": 0.0004, "random_seed": 2025,
        "feature_w": 0.05, "output_w": 0.214, "train_epochs": 100,
        "batch_size": 16, "dropout": 0.1, "lora_dropout": 0.1, "r": 8,
        "use_cycle_prior": 1
    }
    s_base = compute_setting_string(p_base)
    assert "_nocycq" not in s_base, f"Baseline setting should NOT contain _nocycq: {s_base}"

    p_ablate = dict(p_base)
    p_ablate["use_cycle_prior"] = 0
    s_ablate = compute_setting_string(p_ablate)
    assert "_nocycq" in s_ablate, f"Ablation setting MUST contain _nocycq: {s_ablate}"
    assert s_ablate == s_base + "_nocycq", "Setting string mismatch for ablation tag!"
    print(f"  [PASS] Baseline setting: ...{s_base[-30:]}")
    print(f"  [PASS] Ablation setting: ...{s_ablate[-35:]}")


def test_batch_template():
    print("[Test 2/4] Verifying cycle_queue_ablation.json template and task combos...")
    template_path = os.path.join(project_root, "configs", "batch_templates", "cycle_queue_ablation.json")
    assert os.path.exists(template_path), f"Template not found: {template_path}"

    cm = ConfigManager()
    cm.load_from_file(template_path)
    tasks = cm.generate_task_combos()
    assert len(tasks) == 6, f"Expected 6 tasks (2 switches x 3 seeds), got {len(tasks)}"

    priors = [t["params"]["use_cycle_prior"] for t in tasks]
    assert priors.count(1) == 3, f"Expected 3 tasks with use_cycle_prior=1, got {priors.count(1)}"
    assert priors.count(0) == 3, f"Expected 3 tasks with use_cycle_prior=0, got {priors.count(0)}"

    for t in tasks:
        args = ConfigManager.params_to_cmd_args(t["params"])
        assert "--use_cycle_prior" in args, f"--use_cycle_prior missing from args: {args}"
        idx = args.index("--use_cycle_prior")
        val = int(args[idx + 1])
        assert val in [0, 1], f"Invalid arg value: {val}"
    print(f"  [PASS] Successfully expanded 6 tasks with valid --use_cycle_prior args.")


def test_model_requires_grad():
    print("[Test 3/4] Verifying model parameter gradient freeze...")
    from models.CALF import Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg_on = DummyConfig(use_cycle_prior=1)
    model_on = Model(cfg_on, device).to(device)
    assert model_on.cycle_queue.requires_grad is True, "cycle_queue should require grad when use_cycle_prior=1"
    assert model_on.cycle_scale.requires_grad is True, "cycle_scale should require grad when use_cycle_prior=1"

    cfg_off = DummyConfig(use_cycle_prior=0)
    model_off = Model(cfg_off, device).to(device)
    assert model_off.cycle_queue.requires_grad is False, "cycle_queue MUST NOT require grad when use_cycle_prior=0"
    assert model_off.cycle_scale.requires_grad is False, "cycle_scale MUST NOT require grad when use_cycle_prior=0"
    print("  [PASS] Gradient requirements verified (On=True, Off=False).")


def test_model_forward_ablation():
    print("[Test 4/4] Verifying forward output divergence between On and Off...")
    from models.CALF import Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(2025)
    cfg_on = DummyConfig(use_cycle_prior=1)
    model_on = Model(cfg_on, device).to(device)
    model_on.eval()

    cfg_off = DummyConfig(use_cycle_prior=0)
    model_off = Model(cfg_off, device).to(device)
    # Copy exact weights from model_on to model_off
    model_off.load_state_dict(model_on.state_dict(), strict=True)
    model_off.eval()

    # Give cycle_queue noticeable non-zero values to prove presence vs absence
    with torch.no_grad():
        nn.init.normal_(model_on.cycle_queue, mean=1.0, std=0.5)
        model_off.cycle_queue.copy_(model_on.cycle_queue)

    B, L, M = 2, 96, 137
    x = torch.randn(B, L, M, device=device)
    cycle_index = torch.zeros(B, dtype=torch.long, device=device)
    future_cycle_index = torch.zeros(B, 96, dtype=torch.long, device=device)

    with torch.no_grad():
        out_on = model_on(x, cycle_index=cycle_index, future_cycle_index=future_cycle_index)
        out_off = model_off(x, cycle_index=cycle_index, future_cycle_index=future_cycle_index)

    diff_time = torch.abs(out_on["outputs_time"] - out_off["outputs_time"]).mean().item()
    diff_text = torch.abs(out_on["outputs_text"] - out_off["outputs_text"]).mean().item()

    assert diff_time > 1e-4, f"outputs_time difference too small ({diff_time})! Cycle queue ablation did not take effect!"
    assert diff_text > 1e-4, f"outputs_text difference too small ({diff_text})! Cycle queue ablation did not take effect!"
    print(f"  [PASS] Forward output mean difference: time={diff_time:.6f}, text={diff_text:.6f} (Ablation successfully bypassing residual!)")


if __name__ == "__main__":
    print("=" * 70)
    print(">>> Running cycle_queue Ablation Verification Suite")
    print("=" * 70)
    test_setting_string()
    test_batch_template()
    test_model_requires_grad()
    test_model_forward_ablation()
    print("=" * 70)
    print(">>> ALL 4 TEST SUITES PASSED PERFECTLY!")
    print("=" * 70)
