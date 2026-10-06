from engineering_case_studies.pareto import nondominated


def test_nondominated_tradeoff_curve():
    assert nondominated([(1.0, 1.0), (0.5, 2.0), (0.25, 4.0)]) == [True, True, True]


def test_dominated_point():
    assert nondominated([(1.0, 2.0), (0.5, 1.0), (0.25, 4.0)]) == [False, True, True]


def test_equal_point_does_not_dominate():
    assert nondominated([(1.0, 1.0), (1.0, 1.0)]) == [True, True]
