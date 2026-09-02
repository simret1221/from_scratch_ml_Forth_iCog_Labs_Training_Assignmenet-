from core.distance import euclidean_distance
from core.sorting import quick_sort


class WeightedKNN:

    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("K must be greater than zero.")

        self.k = k
        self.training_data = []

    def fit(self, training_data):
        """
        Store labeled training data.

        Expected format:
        [
            ([x1, x2], "A"),
            ([x1, x2], "B"),
            ...
        ]
        """

        if not training_data:
            raise ValueError("Training data cannot be empty.")

        self.training_data = training_data

    def predict(self, query_point):

        if not self.training_data:
            raise ValueError(
                "Model has not been trained. Call fit() first."
            )

        if self.k > len(self.training_data):
            raise ValueError(
                "K cannot be greater than the number of training points."
            )

        distances = []

        # Calculate distance to every training point
        for index, (point, label) in enumerate(self.training_data):

            distance = euclidean_distance(
                query_point,
                point
            )

            distances.append(
                (distance, label, index)
            )

        # Use the existing custom sorting algorithm
        sorted_neighbors = quick_sort(distances)

        # Select K nearest neighbors
        nearest_neighbors = sorted_neighbors[:self.k]

        # Weighted voting
        votes = {}

        for distance, label, _ in nearest_neighbors:

            # Prevent division by zero when query point
            # exactly matches a training point.
            if distance == 0:
                return label

            weight = 1 / distance

            if label not in votes:
                votes[label] = 0

            votes[label] += weight

        # Find class with highest total weight
        best_label = None
        best_weight = -1

        for label in votes:

            if votes[label] > best_weight:
                best_weight = votes[label]
                best_label = label

        return best_label