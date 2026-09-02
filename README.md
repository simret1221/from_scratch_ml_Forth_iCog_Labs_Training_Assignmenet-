# From-Scratch Machine Learning

## KNN & K-Means Without ML Libraries

A from-scratch implementation of two fundamental machine learning
algorithms:

- K-Nearest Neighbors (KNN)
- K-Means Clustering

The project is designed around the assignment requirement of
understanding and implementing the core mathematics and logic
without relying on machine-learning libraries.

---

## Project Goals

The main goals are:

1. Understand how KNN works internally.
2. Implement Euclidean distance manually.
3. Implement custom sorting instead of built-in sorting.
4. Implement KNN majority voting.
5. Understand how K-Means performs clustering.
6. Implement centroid initialization.
7. Implement nearest-centroid assignment.
8. Calculate new centroids using means.
9. Detect convergence.
10. Handle important edge cases.

---

## Project Structure

```text
from_scratch_ml/
│
├── main.py
├── README.md
│
├── core/
│   ├── distance.py
│   ├── sorting.py
│   └── __init__.py
│
├── data/
│   ├── knn_data.py
│   ├── kmeans_data.py
│   └── __init__.py
│
├── knn/
│   ├── knn.py
│   └── __init__.py
│
├── kmeans/
│   ├── kmeans.py
│   └── __init__.py
│
├── tests/
│   ├── test_distance.py
│   ├── test_sorting.py
│   ├── test_knn.py
│   ├── test_kmeans.py
│   └── __init__.py
│
└── results/
    └── baseline_results.txt