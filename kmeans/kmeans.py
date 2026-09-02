class KMeans:

    def __init__(self, k=2, max_iterations=100):
        if k <= 0:
            raise ValueError("K must be greater than zero.")

        if max_iterations <= 0:
            raise ValueError(
                "Maximum iterations must be greater than zero."
            )

        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.labels = []

    def fit(self, data):
        """
        Train K-Means on unlabeled data.

        Expected format:

        [
            [x1, x2],
            [x1, x2],
            ...
        ]
        """

        if not data:
            raise ValueError("Data cannot be empty.")

        if self.k > len(data):
            raise ValueError(
                "K cannot be greater than the number of data points."
            )

        # Check that all points have the same dimensions
        dimensions = len(data[0])

        for point in data:
            if len(point) != dimensions:
                raise ValueError(
                    "All data points must have the same dimensions."
                )

        # Initialize centroids using the first K data points
        self.centroids = []

        for i in range(self.k):
            self.centroids.append(data[i][:])

        for iteration in range(self.max_iterations):

            # Step 1: Assign every point to nearest centroid
            new_labels = []

            for point in data:

                best_cluster = 0
                best_distance = self._distance(
                    point,
                    self.centroids[0]
                )

                for cluster_index in range(1, self.k):

                    distance = self._distance(
                        point,
                        self.centroids[cluster_index]
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_cluster = cluster_index

                new_labels.append(best_cluster)

            # Step 2: Recalculate centroids
            new_centroids = []

            for cluster_index in range(self.k):

                cluster_points = []

                for i in range(len(data)):

                    if new_labels[i] == cluster_index:
                        cluster_points.append(data[i])

                # Keep old centroid if cluster is empty
                if not cluster_points:
                    new_centroids.append(
                        self.centroids[cluster_index][:]
                    )
                    continue

                mean_point = []

                for dimension in range(dimensions):

                    total = 0

                    for point in cluster_points:
                        total += point[dimension]

                    mean = total / len(cluster_points)
                    mean_point.append(mean)

                new_centroids.append(mean_point)

            # Step 3: Check convergence
            if self._centroids_equal(
                self.centroids,
                new_centroids
            ):
                self.centroids = new_centroids
                self.labels = new_labels
                break

            self.centroids = new_centroids
            self.labels = new_labels

        return self

    def _distance(self, point_a, point_b):
        """
        Calculate Euclidean distance.
        """

        squared_sum = 0

        for i in range(len(point_a)):
            difference = point_a[i] - point_b[i]
            squared_sum += difference * difference

        return squared_sum ** 0.5

    def _centroids_equal(
        self,
        centroids_a,
        centroids_b,
        tolerance=0.000001
    ):
        """
        Check whether centroids have converged.
        """

        for i in range(len(centroids_a)):

            for j in range(len(centroids_a[i])):

                difference = abs(
                    centroids_a[i][j]
                    - centroids_b[i][j]
                )

                if difference > tolerance:
                    return False

        return True