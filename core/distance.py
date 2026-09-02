import math


def euclidean_distance(point_a, point_b):
    """
    Calculate Euclidean distance between two points.

    d = sqrt(sum((a_i - b_i)^2))
    """

    if len(point_a) != len(point_b):
        raise ValueError(
            "Points must have the same number of dimensions."
        )

    squared_sum = 0

    for i in range(len(point_a)):
        difference = point_a[i] - point_b[i]
        squared_sum += difference * difference

    return math.sqrt(squared_sum)