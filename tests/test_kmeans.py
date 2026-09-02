from kmeans.kmeans import KMeans
from data.kmeans_data import KMEANS_DATA


def test_kmeans():

    model = KMeans(k=2)

    model.fit(KMEANS_DATA)

    assert len(model.centroids) == 2
    assert len(model.labels) == len(KMEANS_DATA)

    for label in model.labels:
        assert label in [0, 1]

    print("K-Means tests passed.")
    print("Final centroids:", model.centroids)
    print("Cluster assignments:", model.labels)


def test_empty_data():

    model = KMeans(k=2)

    try:
        model.fit([])
        assert False
    except ValueError:
        assert True

    print("Empty data test passed.")


def test_invalid_k():

    try:
        KMeans(k=0)
        assert False
    except ValueError:
        assert True

    print("Invalid K test passed.")


def test_k_larger_than_dataset():

    model = KMeans(k=10)

    try:
        model.fit(KMEANS_DATA)
        assert False
    except ValueError:
        assert True

    print("K larger than dataset test passed.")


def test_empty_cluster_handling():

    # All points are identical.
    # This creates a situation where multiple centroids
    # can receive no points.

    data = [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ]

    model = KMeans(k=3)
    model.fit(data)

    assert len(model.centroids) == 3

    print("Empty cluster handling test passed.")


test_kmeans()
test_empty_data()
test_invalid_k()
test_k_larger_than_dataset()
test_empty_cluster_handling()