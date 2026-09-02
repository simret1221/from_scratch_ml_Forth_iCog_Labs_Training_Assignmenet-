from kmeans.kmeans import KMeans
from core.clustering_metrics import silhouette_score
from data.kmeans_data import KMEANS_DATA
from experiments.experiment_utils import print_experiment_header


def run_kmeans_experiment():

    print_experiment_header(
        "K-MEANS QUALITY EXPERIMENT"
    )

    print(
        f"{'K':<5}"
        f"{'Inertia':<15}"
        f"{'Silhouette':<15}"
        f"{'Iterations':<15}"
    )

    print("-" * 50)

    for k in [2, 3, 4]:

        model = KMeans(k=k)
        model.fit(KMEANS_DATA)

        inertia = model.inertia(KMEANS_DATA)

        silhouette = silhouette_score(
            KMEANS_DATA,
            model.labels
        )

        iterations = len(
            model.convergence_history
        )

        print(
            f"{k:<5}"
            f"{inertia:<15.3f}"
            f"{silhouette:<15.3f}"
            f"{iterations:<15}"
        )


if __name__ == "__main__":
    run_kmeans_experiment()