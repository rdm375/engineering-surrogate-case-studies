import pytest
from engineering_case_studies.collection import parse_performance

def test_parse_ns_eval():
    ns, rate=parse_performance("median=270805.088 ns/eval",1)
    assert ns == pytest.approx(270805.088)
    assert rate == pytest.approx(1e9/270805.088)

def test_parse_json_line():
    ns, _=parse_performance('{"ns_per_eval": 12.5}',100)
    assert ns == 12.5

def test_unparseable_fails():
    with pytest.raises(ValueError): parse_performance("hello",1)
