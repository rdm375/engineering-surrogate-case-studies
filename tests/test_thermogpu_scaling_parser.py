from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = (
    ROOT
    / "case-studies"
    / "thermogpu"
    / "scripts"
    / "collect_direct.py"
)

spec = spec_from_file_location("collect_direct", COLLECTOR)
module = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

parse_scaling_output = module.parse_scaling_output


SAMPLE = """\
ThermoGPU M7 component-count scaling; samples=9 target_ms=200 max_threads=8
components,states,backend,threads,calls_per_sample,median_seconds,ns_per_state,states_per_second,mad_percent,speedup_vs_scalar,speedup_vs_best_cpu
1,1000,scalar,1,627,0.00029348235087719299,293.48235087719297,3407359.921341395,2.1511576160249266,1,1
1,1000,openmp,4,1277,0.00018066140877055599,180.66140877055597,5535216.4405516302,9.7564383832297779,1.6244883335871807,1
1,1000,cuda_resident,0,7870,2.1067817916137231e-05,21.067817916137233,47465760.525395185,0.31929680326184495,13.930362985166846,8.5752311648837392
1,1000,cuda_e2e,0,663,0.00029891888235294116,298.91888235294113,3345389.1976595661,0.53033047162282854,0.98181268632829588,0.60438272533497717
"""


def test_parse_scaling_output():
    preamble, rows = parse_scaling_output(SAMPLE)

    assert preamble.startswith("ThermoGPU M7 component-count scaling")
    assert len(rows) == 4

    scalar = rows[0]
    assert scalar["components"] == 1
    assert scalar["states"] == 1000
    assert scalar["backend"] == "scalar"
    assert scalar["threads"] == 1
    assert scalar["ns_per_state"] == 293.48235087719297

    resident = rows[2]
    assert resident["backend"] == "cuda_resident"
    assert resident["threads"] == 0
    assert resident["ns_per_state"] == 21.067817916137233
    assert resident["states_per_second"] == 47465760.525395185


def test_parser_preserves_all_execution_modes():
    _, rows = parse_scaling_output(SAMPLE)

    assert {r["backend"] for r in rows} == {
        "scalar",
        "openmp",
        "cuda_resident",
        "cuda_e2e",
    }


def test_parser_rejects_missing_csv_header():
    bad = """\
ThermoGPU benchmark
this,is,not,the,expected,schema
1,2,3,4,5
"""

    try:
        parse_scaling_output(bad)
    except ValueError as exc:
        assert "CSV header not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")
