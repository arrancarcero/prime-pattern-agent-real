from ppa_real.truth import truth_tuple


def test_truth_tuple_self_test_to_1e7():
    assert all(v.ok is not False for v in truth_tuple.self_test(7))


def test_truth_tuple_is_frozen():
    assert isinstance(truth_tuple.TRUTH, tuple)
    assert truth_tuple.verify("pi", 10 ** 8, 5761455).ok
    assert truth_tuple.verify("pi", 10 ** 8, 5761456).ok is False


