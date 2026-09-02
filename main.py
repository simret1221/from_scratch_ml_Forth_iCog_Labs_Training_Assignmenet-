from knn.knn import KNN
from kmeans.kmeans import KMeans

from data.knn_data import TRAINING_DATA
from data.kmeans_data import KMEANS_DATA


def run_knn_demo():

    print("=" * 40)
    print("KNN DEMONSTRATION")
    print("=" * 40)

    model = KNN(k=3)
    model.fit(TRAINING_DATA)

    query = [1.8, 1.7]
    prediction = model.predict(query)

    print("Query:", query)
    print("Prediction:", prediction)


def run_kmeans_demo():

    print()
    print("=" * 40)
    print("K-MEANS DEMONSTRATION")
    print("=" * 40)

    model = KMeans(k=2)
    model.fit(KMEANS_DATA)

    print("Final centroids:", model.centroids)
    print("Cluster assignments:", model.labels)


def main():

    run_knn_demo()
    run_kmeans_demo()


if __name__ == "__main__":
    main()