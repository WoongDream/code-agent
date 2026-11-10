import pytest
from stats import mean, median


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


def test_median_single():
    assert median([7]) == 7


def test_median_two():
    assert median([1, 3]) == 2


def test_median_even_unsorted():
    assert median([10, 2, 8, 4]) == 6


def test_median_odd_unsorted():
    assert median([5, 1, 9, 3, 7]) == 5


def test_median_negative():
    assert median([-5, -1, -3, -2]) == -2.5


def test_median_float():
    assert median([0.5, 1.5, 2.5, 3.5]) == pytest.approx(2.0)


def test_mean_unchanged():
    assert mean([1, 2, 3]) == 2
