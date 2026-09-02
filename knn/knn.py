from core.distance import euclidean_distance
from core.sorting import quick_sort


class KNN:

    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("K must be greater than zero.")

        self.k = k
        self.training_data = []

    def fit(self, training_data):
        """
        Store training data.

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

        # Custom sorting
        sorted_neighbors = quick_sort(distances)

        # Select K nearest neighbors
        nearest_neighbors = sorted_neighbors[:self.k]

        # Majority voting
        votes = {}

        for _, label, _ in nearest_neighbors:

            if label not in votes:
                votes[label] = 0

            votes[label] += 1

        # Find majority class
        best_label = None
        best_votes = -1

        for label in votes:

            if votes[label] > best_votes:
                best_votes = votes[label]
                best_label = label

        return best_label