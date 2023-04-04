from general import intersection


def test_intersection():
    p1 = (0, 0)
    p2 = (10, 10)

    p3 = (10, 0)
    p4 = (0, 10)
    assert intersection(p1, p2, p3, p4) == (5.0, 5.0)
