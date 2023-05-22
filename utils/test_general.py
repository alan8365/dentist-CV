import numpy as np

from general import intersection


def test_intersection():
    p1 = (0, 0)
    p2 = (10, 10)

    p3 = (10, 0)
    p4 = (0, 10)
    assert intersection(p1, p2, p3, p4) == (5.0, 5.0)


def test_same_line_intersection():
    p1 = (0, 0)
    p2 = (1, 1)

    p3 = (2, 2)
    p4 = (3, 3)
    assert intersection(p1, p2, p3, p4) == (None, None)


def test_parallel_intersection():
    p1 = (0, 0)
    p2 = (1, 1)

    p3 = (1, 0)
    p4 = (2, 1)
    assert intersection(p1, p2, p3, p4) == (np.Inf, np.Inf)
