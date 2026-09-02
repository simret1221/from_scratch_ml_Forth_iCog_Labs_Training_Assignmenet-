from kmeans.kmeans import KMeans
from knn.knn import KNN

from core.normalization import min_max_normalize
from core.clustering_metrics import silhouette_score

from application.customer_data import CUSTOMER_DATA


class CustomerIntelligence:

    def __init__(self, k=3):
        self.k = k
        self.kmeans = None
        self.knn = None
        self.normalized_data = []
        self.cluster_labels = []
        self.cluster_profiles = {}

        # Keep training normalization parameters so new customers
        # are transformed using the same scale as the training data.
        self.mins = []
        self.maxs = []

    def _fit_normalization(self):
        dimensions = len(CUSTOMER_DATA[0])
        self.mins = []
        self.maxs = []

        for d in range(dimensions):
            values = [point[d] for point in CUSTOMER_DATA]
            self.mins.append(min(values))
            self.maxs.append(max(values))

        self.normalized_data = []

        for point in CUSTOMER_DATA:
            self.normalized_data.append(
                self._normalize_point(point)
            )

    def _normalize_point(self, point):
        normalized = []

        for d in range(len(point)):
            if self.maxs[d] == self.mins[d]:
                normalized.append(0.0)
            else:
                value = (point[d] - self.mins[d]) / (
                    self.maxs[d] - self.mins[d]
                )
                normalized.append(value)

        return normalized

    def segment_customers(self):
        self._fit_normalization()

        self.kmeans = KMeans(k=self.k)
        self.kmeans.fit(self.normalized_data)

        self.cluster_labels = self.kmeans.labels

        self._build_cluster_profiles()

        silhouette = silhouette_score(
            self.normalized_data,
            self.cluster_labels
        )

        return {
            "clusters": self.cluster_labels,
            "centroids": self.kmeans.centroids,
            "inertia": self.kmeans.inertia(
                self.normalized_data
            ),
            "silhouette": silhouette
        }

    def _build_cluster_profiles(self):
        profiles = {}

        for cluster in range(self.k):
            indices = []

            for i, label in enumerate(self.cluster_labels):
                if label == cluster:
                    indices.append(i)

            if not indices:
                continue

            totals = [0, 0, 0, 0]

            for index in indices:
                for feature in range(4):
                    totals[feature] += CUSTOMER_DATA[index][feature]

            averages = []

            for total in totals:
                averages.append(total / len(indices))

            profiles[cluster] = {
                "size": len(indices),
                "averages": averages
            }

        self.cluster_profiles = profiles

    def train_knn(self):
        if not self.cluster_labels:
            raise ValueError("Run segment_customers() first.")

        labeled_data = []

        for point, label in zip(
            self.normalized_data,
            self.cluster_labels
        ):
            labeled_data.append((point, label))

        self.knn = KNN(k=5)
        self.knn.fit(labeled_data)

    def classify_customer(self, customer):
        if self.knn is None:
            raise ValueError("KNN has not been trained.")

        normalized_customer = self._normalize_point(customer)

        result = self.knn.predict_with_confidence(
            normalized_customer
        )

        cluster = result["prediction"]
        profile = self.cluster_profiles[cluster]

        recommendation = self._recommend(profile)

        return {
            "cluster": cluster,
            "confidence": result["confidence"],
            "tie": result["tie"],
            "votes": result["votes"],
            "recommendation": recommendation
        }

    def _recommend(self, profile):
        averages = profile["averages"]

        spending = averages[2]
        frequency = averages[3]

        if spending >= 70 and frequency >= 7:
            return "VIP rewards and loyalty campaign"

        if spending < 40:
            return "Targeted promotions and personalized offers"

        if frequency >= 5:
            return (
                "Engagement campaign and "
                "repeat-purchase incentives"
            )

        return "General promotional campaign"
