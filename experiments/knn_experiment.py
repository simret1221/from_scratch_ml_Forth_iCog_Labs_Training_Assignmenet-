from knn.knn import KNN
from knn.weighted_knn import WeightedKNN
from data.knn_data import TRAINING_DATA

from experiments.experiment_utils import (
    print_experiment_header,
    accuracy
)


TEST_DATA = [
    ([1.8, 1.7], "A"),
    ([2.2, 1.8], "A"),
    ([7.8, 7.8], "B"),
    ([7.2, 7.4], "B"),
]


def run_k_selection():

    print_experiment_header(
        "KNN K-SELECTION EXPERIMENT"
    )

    actual = []

    for _, label in TEST_DATA:
        actual.append(label)

    for k in [1, 3, 5]:

        model = KNN(k=k)
        model.fit(TRAINING_DATA)

        predictions = []

        for point, _ in TEST_DATA:
            predictions.append(
                model.predict(point)
            )

        score = accuracy(
            predictions,
            actual
        )

        print(
            "K =", k,
            "| Predictions =", predictions,
            "| Accuracy =", round(score * 100, 2), "%"
        )


def compare_weighted():

    print_experiment_header(
        "STANDARD VS WEIGHTED KNN"
    )

    standard = KNN(k=3)
    weighted = WeightedKNN(k=3)

    standard.fit(TRAINING_DATA)
    weighted.fit(TRAINING_DATA)

    for point, actual in TEST_DATA:

        p1 = standard.predict(point)
        p2 = weighted.predict(point)

        print(
            "Query:", point,
            "| Actual:", actual,
            "| Standard:", p1,
            "| Weighted:", p2
        )


if __name__ == "__main__":
    run_k_selection()
    compare_weighted()

