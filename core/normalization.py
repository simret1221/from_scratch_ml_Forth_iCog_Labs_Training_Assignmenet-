def min_max_normalize(data):
    """
    Normalize each feature to the range [0, 1].
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    dimensions = len(data[0])

    for point in data:
        if len(point) != dimensions:
            raise ValueError("All points must have same dimensions.")

    mins = []
    maxs = []

    for d in range(dimensions):
        values = []

        for point in data:
            values.append(point[d])

        mins.append(min(values))
        maxs.append(max(values))

    normalized = []

    for point in data:
        new_point = []

        for d in range(dimensions):

            if maxs[d] == mins[d]:
                new_point.append(0.0)
            else:
                value = (point[d] - mins[d]) / (maxs[d] - mins[d])
                new_point.append(value)

        normalized.append(new_point)

    return normalized