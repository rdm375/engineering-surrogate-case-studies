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


def load_analysis_module():
    script = ROOT / "case-studies/thermogpu/scripts/analyze_study.py"
    spec = importlib.util.spec_from_file_location("thermogpu_analyze_study", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scaling_analysis_recovers_power_law():
    m = load_analysis_module()

    rows = [
        {
            "budget": b,
            "simplices": 2 * b,
            "min_nodes": b**2,
            "total_nodes": 3 * b**2,
            "linf": b**-1.5,
            "steady_state_ns_per_eval": 5 * b,
        }
        for b in (16, 32, 64, 128)
    ]

    _, representation, accuracy, incremental = m.scaling_analysis(rows)

    assert abs(representation["simplices_vs_budget_exponent"] - 1.0) < 1e-12
    assert abs(representation["min_nodes_vs_budget_exponent"] - 2.0) < 1e-12
    assert abs(representation["total_nodes_vs_budget_exponent"] - 2.0) < 1e-12
    assert abs(accuracy["linf_vs_budget_exponent"] + 1.5) < 1e-12
    assert len(incremental) == 3


def test_feasibility_distinguishes_resource_failure_and_deferred(tmp_path, monkeypatch):
    m = load_analysis_module()
    monkeypatch.setattr(m, "ROOT", tmp_path)

    failure = tmp_path / "oom.failed.json"
    failure.write_text(
        """{
          "backend": "gpu",
          "device": "cuda:0",
          "all_failures_resource_related": true,
          "policy_failures": {
            "compact-min-live": "JaxRuntimeError: RESOURCE_EXHAUSTED: out of memory"
          }
        }"""
    )

    rows = [
        {
            "budget": "128",
            "batch_size": "1000000",
            "status": "failed",
            "evidence": "oom.failed.json",
        },
        {
            "budget": "256",
            "batch_size": "10000",
            "status": "pending",
            "evidence": "",
        },
    ]

    result = m.feasibility(rows, {(256, 10000)})

    assert result[0]["status"] == "resource_failure"
    assert result[0]["failure_kind"] == "gpu_memory"
    assert result[0]["hardware"] == "gpu:cuda:0"

    assert result[1]["raw_status"] == "pending"
    assert result[1]["status"] == "deferred"


def test_conclusion_analysis_accepts_deferred_but_not_pending():
    m = load_analysis_module()

    cross = [
        {
            "batch_size": 1,
            "preferred_exact_method": "CPU scalar",
        },
        {
            "batch_size": 1000,
            "preferred_exact_method": "GPU direct resident",
        },
    ]

    costs = [
        {"global_pareto": False},
        {"global_pareto": False},
    ]

    dom = [
        {
            "best_exact_dominates_surrogate": True,
            "surrogate_dominates_best_exact": False,
        },
        {
            "best_exact_dominates_surrogate": True,
            "surrogate_dominates_best_exact": False,
        },
    ]

    representation = {"total_nodes_vs_budget_exponent": 2.0}
    accuracy = {"linf_vs_budget_exponent": -1.5}

    feasible = [
        {
            "budget": 16,
            "batch_size": 1,
            "status": "complete",
            "failure_kind": "",
            "hardware": "",
            "evidence": "",
        },
        {
            "budget": 256,
            "batch_size": 10000,
            "status": "deferred",
            "failure_kind": "",
            "hardware": "",
            "evidence": "",
        },
    ]

    result = m.conclusion_analysis(
        cross, costs, dom, [], feasible, representation, accuracy
    )

    assert result["surrogate_computational_advantage_observed"] is False
    assert result["all_measured_surrogate_points_dominated_same_batch"] is True
    assert (
        result["measurement_sufficiency"]
        ["sufficient_for_current_v1_conclusion"]
        is True
    )
    assert (
        result["measurement_sufficiency"]
        ["additional_measurements_required_for_current_conclusion"]
        is False
    )

    feasible.append(
        {
            "budget": 256,
            "batch_size": 100000,
            "status": "pending",
            "failure_kind": "",
            "hardware": "",
            "evidence": "",
        }
    )

    result = m.conclusion_analysis(
        cross, costs, dom, [], feasible, representation, accuracy
    )

    assert (
        result["measurement_sufficiency"]
        ["sufficient_for_current_v1_conclusion"]
        is False
    )
    assert (
        result["measurement_sufficiency"]
        ["additional_measurements_required_for_current_conclusion"]
        is True
    )


def test_pipeline_classifies_evidence_producers():
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

    expected = {
        "environment",
        "direct",
        "surrogates",
        "native-dags",
        "calibrate",
        "surrogate-benchmark",
    }

    assert {
        name for name, stage in stages.items() if stage.evidence_producer
    } == expected

    assert not stages["direct-process"].evidence_producer
    assert not stages["master-pareto"].evidence_producer
    assert not stages["analysis"].evidence_producer
    assert not stages["report"].evidence_producer


def test_analysis_stage_declares_scaling_output():
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

    assert any(
        str(path).endswith("scaling-analysis.csv")
        for path in stages["analysis"].outputs
    )


def test_reproduction_excludes_legacy_performance_aggregate():
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

    assert stages["performance-process"].reproduce is False
    assert stages["performance-plot"].reproduce is False

    assert stages["master-pareto"].reproduce is True
    assert stages["analysis"].reproduce is True
    assert stages["report"].reproduce is True


def test_calibration_reproduction_uses_raw_success_failure_and_deferrals(
    tmp_path, monkeypatch
):
    m = load_module()

    raw = tmp_path / "case-studies/thermogpu/raw-results"
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "RAW", raw)

    batches = (1, 10, 100, 1000, 10000, 100000, 1000000)
    budgets = (16, 32, 64, 128, 256)
    deferred = set(m.V1_DEFERRED_CALIBRATIONS)

    outputs = []
    for budget in budgets:
        for batch in batches:
            nominal = (
                raw
                / "surrogate-performance"
                / f"budget-{budget}"
                / "calibration"
                / f"float32-b{batch}.json"
            )
            outputs.append(nominal)

            if (budget, batch) in deferred:
                continue

            nominal.parent.mkdir(parents=True, exist_ok=True)

            if (budget, batch) == (128, 1000000):
                nominal.with_suffix(".failed.json").write_text(
                    '{"all_failures_resource_related": true}\n'
                )
            else:
                nominal.write_text('{"status": "complete"}\n')

    stage = m.Stage(
        "calibrate",
        "calibrate",
        ["true"],
        tuple(outputs),
        evidence_producer=True,
    )

    ready, missing, evidence, declarations = m.retained_evidence_ready(stage)

    assert ready
    assert missing == []
    assert len(evidence) == 32
    assert sum(path.name.endswith(".failed.json") for path in evidence) == 1
    assert not any(
        path.name in {
            "float32-b10000.json",
            "float32-b100000.json",
            "float32-b1000000.json",
        }
        and path.parents[1].name == "budget-256"
        for path in evidence
    )
    assert declarations == [
        "budget=256,batch=10000:deferred",
        "budget=256,batch=100000:deferred",
        "budget=256,batch=1000000:deferred",
    ]


def test_calibration_reproduction_rejects_unexplained_missing_cell(
    tmp_path, monkeypatch
):
    m = load_module()

    raw = tmp_path / "case-studies/thermogpu/raw-results"
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "RAW", raw)

    outputs = []
    for budget in (16, 32, 64, 128, 256):
        for batch in (1, 10, 100, 1000, 10000, 100000, 1000000):
            nominal = (
                raw
                / "surrogate-performance"
                / f"budget-{budget}"
                / "calibration"
                / f"float32-b{batch}.json"
            )
            outputs.append(nominal)

            if (budget, batch) in set(m.V1_DEFERRED_CALIBRATIONS):
                continue

            # Deliberately leave B64 x 1000 unexplained.
            if (budget, batch) == (64, 1000):
                continue

            nominal.parent.mkdir(parents=True, exist_ok=True)
            if (budget, batch) == (128, 1000000):
                nominal.with_suffix(".failed.json").write_text(
                    '{"all_failures_resource_related": true}\n'
                )
            else:
                nominal.write_text('{"status": "complete"}\n')

    stage = m.Stage(
        "calibrate",
        "calibrate",
        ["true"],
        tuple(outputs),
        evidence_producer=True,
    )

    ready, missing, evidence, declarations = m.retained_evidence_ready(stage)

    assert not ready
    assert missing == ["budget=64,batch=1000"]
    assert len(evidence) == 31
    assert len(declarations) == 3


def test_legacy_surrogate_benchmark_is_not_required_reproduction_evidence():
    m = load_module()

    stage = m.Stage(
        "surrogate-benchmark",
        "legacy benchmark aggregate",
        ["true"],
        (
            m.RAW / "surrogate-performance/benchmark-float32.csv",
            m.RAW / "surrogate-performance/benchmark-float32.json",
        ),
        evidence_producer=True,
    )

    ready, missing, evidence, declarations = m.retained_evidence_ready(stage)

    assert ready
    assert missing == []
    assert evidence == []
    assert declarations == [
        "legacy aggregate not required; authoritative performance evidence "
        "is retained by calibrate"
    ]
