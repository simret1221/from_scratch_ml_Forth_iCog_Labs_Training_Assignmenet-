from core.distance import euclidean_distance


def silhouette_score(data, labels):

    if not data:
        raise ValueError("Data cannot be empty.")

    scores = []

    for i in range(len(data)):

        same_cluster = []
        other_clusters = {}

        for j in range(len(data)):

            if i == j:
                continue

            if labels[i] == labels[j]:
                same_cluster.append(j)
            else:
                cluster = labels[j]

                if cluster not in other_clusters:
                    other_clusters[cluster] = []

                other_clusters[cluster].append(j)

        # a(i)
        if same_cluster:
            total = 0

            for j in same_cluster:
                total += euclidean_distance(data[i], data[j])

            a = total / len(same_cluster)
        else:
            a = 0

        # b(i)
        b = None

        for cluster in other_clusters:

            total = 0

            for j in other_clusters[cluster]:
                total += euclidean_distance(data[i], data[j])

            average = total / len(other_clusters[cluster])

            if b is None or average < b:
                b = average

        if b is None:
            score = 0
        else:
            denominator = max(a, b)

            if denominator == 0:
                score = 0
            else:
                score = (b - a) / denominator

        scores.append(score)

    total = 0

    for score in scores:
        total += score

    return total / len(scores)