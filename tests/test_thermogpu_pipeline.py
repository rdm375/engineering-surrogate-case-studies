from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "case-studies/thermogpu/scripts/run_study.py"


def load_module():
    spec = importlib.util.spec_from_file_location("thermogpu_run_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_direct_stem():
    m = load_module()
    assert m.canonical_direct_stem([1, 10, 1000]) == "scaling-b1-10-1000"


def test_outputs_ready_requires_all_nonempty(tmp_path):
    m = load_module()
    a, b = tmp_path / "a", tmp_path / "b"
    stage = m.Stage("x", "x", ["true"], (a, b))
    assert not m.outputs_ready(stage)
    a.write_text("a")
    b.write_text("")
    assert not m.outputs_ready(stage)
    b.write_text("b")
    assert m.outputs_ready(stage)


def test_calibration_stage_freezes_screen_confirm_protocol(tmp_path):
    m = load_module()

    class Args:
        config = ROOT / "case-studies/thermogpu/configs/study-v1.toml"
        thermogpu_root = Path("/nvme/Sync/ThermoGPU")
        tdar_root = Path("/nvme/Sync/tdar")
        cpwa_relu_root = Path("/nvme/Sync/cpwa-relu")
        cpwa_python = Path("/nvme/Sync/cpwa-relu/.venv/bin/python")
        dtype = "float32"
        repeats = 7
        warmup = 2
        calibration_seed = 20261004
        benchmark_seed = 20261005

    stages = {stage.name: stage for stage in m.build_stages(Args())}
    command = stages["calibrate"].command

    assert command[command.index("--screen-repeats") + 1] == "2"
    assert command[command.index("--confirm-top") + 1] == "2"
    assert command[command.index("--near-tie-percent") + 1] == "10"
    assert command[command.index("--repeats") + 1] == "7"
    assert command[command.index("--warmup") + 1] == "2"


def test_pipeline_includes_reusable_analysis_stage():
    m = load_module()

    class Args:
        config = ROOT / "case-studies/thermogpu/configs/study-v1.toml"
        thermogpu_root = Path("/nvme/Sync/ThermoGPU")
        tdar_root = Path("/nvme/Sync/tdar")
        cpwa_relu_root = Path("/nvme/Sync/cpwa-relu")
        cpwa_python = Path("/nvme/Sync/cpwa-relu/.venv/bin/python")
        dtype = "float32"
        repeats = 7
        warmup = 2
        calibration_seed = 20261004
        benchmark_seed = 20261005

    stages = {stage.name: stage for stage in m.build_stages(Args())}
    assert "master-pareto" in stages
    assert "analysis" in stages
    assert any(str(x).endswith("study-summary.json") for x in stages["analysis"].outputs)
