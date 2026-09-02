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
        self.convergence_history = []

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

        # Reset convergence history for a new training run
        self.convergence_history = []

        # Initialize centroids using K-Means++
        self.initialize_kmeans_plus_plus(data)

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

            # Step 3: Track centroid movement
            movement = 0

            for i in range(self.k):

                for j in range(dimensions):

                    difference = (
                        self.centroids[i][j]
                        - new_centroids[i][j]
                    )

                    movement += difference * difference

            self.convergence_history.append(
                movement ** 0.5
            )

            # Step 4: Check convergence
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

    def initialize_kmeans_plus_plus(self, data):

        self.centroids = []

        # First centroid
        self.centroids.append(data[0][:])

        while len(self.centroids) < self.k:

            best_point = None
            best_distance = -1

            for point in data:

                min_distance = None

                for centroid in self.centroids:

                    distance = self._distance(
                        point,
                        centroid
                    )

                    if min_distance is None or distance < min_distance:
                        min_distance = distance

                if min_distance > best_distance:

                    best_distance = min_distance
                    best_point = point

            self.centroids.append(best_point[:])

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

    def inertia(self, data):

        if not self.centroids or not self.labels:
            raise ValueError("Model must be fitted first.")

        total = 0

        for i in range(len(data)):

            point = data[i]
            centroid = self.centroids[self.labels[i]]

            total += self._distance(
                point,
                centroid
            ) ** 2

        return total

