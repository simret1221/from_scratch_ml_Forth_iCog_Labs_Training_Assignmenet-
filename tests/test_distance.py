from core.distance import euclidean_distance


def test_same_point():
    result = euclidean_distance([2, 3], [2, 3])

    assert result == 0


def test_simple_distance():
    result = euclidean_distance([0, 0], [3, 4])

    assert result == 5


def test_dimension_mismatch():
    try:
        euclidean_distance([1, 2], [1, 2, 3])
        assert False
    except ValueError:
        assert True


print("Distance tests passed.")