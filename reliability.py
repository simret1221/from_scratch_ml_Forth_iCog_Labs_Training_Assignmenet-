from knn.knn import KNN
from kmeans.kmeans import KMeans
from core.clustering_metrics import silhouette_score
from data.knn_data import TRAINING_DATA
from data.kmeans_data import KMEANS_DATA


def analyze_kmeans_result(k, inertia, silhouette):

    if silhouette >= 0.7:
        quality = "Strong clustering"

    elif silhouette >= 0.5:
        quality = "Reasonable clustering"

    elif silhouette >= 0.25:
        quality = "Weak clustering"

    else:
        quality = "Poor clustering"

    print(
        "K =", k,
        "| Inertia =", round(inertia, 3),
        "| Silhouette =", round(silhouette, 3),
        "| Quality =", quality
    )


def run_reliability_lab():

    print()
    print("=" * 55)
    print("        ALGORITHM RELIABILITY LAB")
    print("=" * 55)

    # ----------------------------------------
    # 1. KNN STABILITY
    # ----------------------------------------

    print("\n[1] KNN STABILITY")

    query = [1.8, 1.7]

    for k in [1, 3, 5]:

        model = KNN(k=k)
        model.fit(TRAINING_DATA)

        result = model.predict_with_confidence(query)

        print(
            "K =", k,
            "| Prediction =", result["prediction"],
            "| Confidence =", result["confidence"],
            "| Tie =", result["tie"]
        )

    # ----------------------------------------
    # 2. K-MEANS QUALITY
    # ----------------------------------------

    print("\n[2] K-MEANS QUALITY")

    for k in [2, 3, 4]:

        model = KMeans(k=k)
        model.fit(KMEANS_DATA)

        score = silhouette_score(
            KMEANS_DATA,
            model.labels
        )

        analyze_kmeans_result(
            k,
            model.inertia(KMEANS_DATA),
            score
        )

    # ----------------------------------------
    # 3. RELIABILITY INTERPRETATION
    # ----------------------------------------

    print("\n[3] RELIABILITY INTERPRETATION")

    print(
        "• Lower K may make KNN sensitive to individual points."
    )

    print(
        "• Higher K may smooth local decision boundaries."
    )

    print(
        "• High K-Means silhouette indicates strong cluster separation."
    )

    print(
        "• Inertia decreases as K increases, so use the elbow idea."
    )

    print(
        "• Different initialization can affect K-Means results."
    )

    print("\nReliability Lab complete.")


if __name__ == "__main__":
    run_reliability_lab()

