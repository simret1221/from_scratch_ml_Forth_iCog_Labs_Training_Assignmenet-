# From-Scratch Machine Learning

## KNN & K-Means Without ML Libraries

A from-scratch implementation of **K-Nearest Neighbors (KNN)** and **K-Means Clustering**, built by reconstructing the core machine-learning operations using basic Python data structures, loops, and mathematics.

The project follows an engineering progression:

```text
V1.0 BASELINE
      ↓
LIMITATIONS DISCOVERED
      ↓
ENGINEERING DECISIONS
      ↓
V2.0 IMPROVEMENTS
      ↓
EXPERIMENTS & EVIDENCE
      ↓
REAL-WORLD APPLICATION
```

The goal is not simply to make KNN and K-Means work, but to understand **what happens inside the algorithms, why each operation is required, and how the system can be made more reliable and useful in practice.**

---

# 1. INTRODUCTION

## Project Goal

The project explores two fundamental machine-learning paradigms:

### K-Nearest Neighbors — Classification

KNN is a **supervised learning** algorithm.

It receives labeled training data and classifies a new point according to its nearest neighbors.

The basic process is:

```text
New Query
    ↓
Calculate distance to every training point
    ↓
Rank the distances
    ↓
Select K nearest neighbors
    ↓
Count their labels
    ↓
Majority vote
    ↓
Prediction
```

The mathematical foundation is:

### Euclidean Distance

```math
d(q,x_i)=
\sqrt{\sum_{j=1}^{d}(q_j-x_{ij})^2}
```

### K Nearest Neighbors

```math
N_k(q)=
\text{the }k\text{ training points with the smallest distance}
```

### Majority Voting

```math
\hat{y}=
\arg\max_y
\sum_{i\in N_k(q)}
I(y_i=y)
```

In simple terms:

> **Distance determines who is close, K determines who gets to vote, and majority voting determines the prediction.**

---

## K-Means — Clustering

K-Means is an **unsupervised learning** algorithm.

Unlike KNN, the data does not need predefined labels.

Its goal is to divide data into `K` groups by repeatedly moving points toward their nearest centroid and updating the centroid positions.

The process is:

```text
Initial Centroids
       ↓
Assign each point to nearest centroid
       ↓
Calculate new centroid means
       ↓
Check convergence
       ↓
Repeat if necessary
       ↓
Final Clusters
```

### Objective Function

```math
J=
\sum_{i=1}^{n}
\|x_i-\mu_{c_i}\|^2
```

K-Means tries to minimize the total squared distance between each point and the centroid of its assigned cluster.

### Assignment

```math
c_i=
\arg\min_j
\|x_i-\mu_j\|^2
```

> For each point, choose the nearest centroid.

### Centroid Update

```math
\mu_j=
\frac{1}{|C_j|}
\sum_{x_i\in C_j}x_i
```

> Recalculate the centroid as the mean of all points assigned to that cluster.

---

# 2. V1.0 — BASELINE IMPLEMENTATION

The first version focuses on implementing the fundamental algorithms correctly and transparently.

The main principle was:

> **Remove black-box abstractions and rebuild the core operations ourselves.**

Instead of using:

```python
from sklearn.neighbors import KNeighborsClassifier
```

or:

```python
from sklearn.cluster import KMeans
```

the project implements the underlying logic directly.

The assignment requires native data structures, loops, mathematical operations, custom sorting, and manually implemented algorithmic logic.

---

## V1.0 Architecture

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
```

---

## V1.0 — KNN Implementation

The KNN implementation was decomposed into three major operations.

### 1. Distance Calculation

Implemented manually in:

```text
core/distance.py
```

The project calculates Euclidean distance rather than relying on a library distance function.

---

### 2. Custom Sorting

Implemented in:

```text
core/sorting.py
```

A custom `quick_sort()` implementation is used to rank the distance records.

KNN uses:

```python
sorted_neighbors = quick_sort(distances)
```

Then:

```python
nearest_neighbors = sorted_neighbors[:self.k]
```

The distance records contain:

```python
(distance, label, index)
```

where:

* `distance` → how close the point is
* `label` → its class
* `index` → its original position in the dataset

The assignment specifically requires custom sorting instead of built-in sorting.

---

### 3. Majority Voting

After selecting the nearest `K` points, their labels are counted.

The class with the highest number of votes becomes the prediction.

```text
Nearest Neighbors
       ↓
A
A
B
       ↓
A = 2 votes
B = 1 vote
       ↓
Prediction = A
```

---

## V1.0 — K-Means Implementation

The baseline K-Means implementation contains the fundamental iterative process.

### 1. Centroid Initialization

Initial centroid positions are selected from the data.

### 2. Assignment

Every point is compared with every centroid and assigned to the nearest one.

### 3. Update

For every cluster, the centroid is recalculated using the mean of its assigned points.

### 4. Convergence

The algorithm checks whether centroid movement has become sufficiently small.

The assignment describes this core workflow as centroid initialization, nearest-centroid assignment, mean-based centroid updates, and iteration until convergence.

---

# V1.0 — Edge Cases & Reliability

The baseline implementation also considers important failure conditions.

### KNN

Handled cases include:

* Invalid `K`
* Empty training data
* `K` larger than the training dataset
* Prediction before training
* Exact-match queries
* Voting ties

### K-Means

Handled cases include:

* Empty input data
* Invalid `K`
* Dimension inconsistencies
* Empty clusters
* Convergence detection

These cases are important because a machine-learning algorithm is not reliable simply because it works on one normal example.

The assignment explicitly expects discussion of edge cases such as voting ties and empty clusters.

---

# V1.0 — Testing

The implementation is tested through separate test modules:

```text
tests/
├── test_distance.py
├── test_sorting.py
├── test_knn.py
└── test_kmeans.py
```

The tests verify the fundamental operations independently before relying on them in the larger algorithms.

Example KNN validation:

```text
Weighted KNN prediction test passed.
Weighted KNN empty data test passed.
Weighted KNN invalid K test passed.
Weighted KNN exact-match test passed.
```

---

# 3. V1.0 → V2.0

## Why V2.0?

V1.0 successfully implements the algorithms, but implementation alone does not answer several engineering questions.

For example:

* Is KNN stable when `K` changes?
* How confident is a KNN prediction?
* Does feature scale affect distance calculations?
* How good are the K-Means clusters?
* Which value of `K` is appropriate?
* How sensitive is K-Means to initialization?
* Can the algorithms solve a realistic problem?

These limitations motivated the V2.0 improvements.

---

# V1.0 Limitations

| V1.0 Limitation                        | V2.0 Solution                |
| -------------------------------------- | ---------------------------- |
| Raw features may have different scales | Normalization                |
| Only basic KNN prediction              | Weighted KNN                 |
| No confidence information              | Confidence + tie analysis    |
| No systematic K comparison             | K-selection experiment       |
| No clustering quality measurement      | Inertia                      |
| No cluster separation measurement      | Silhouette score             |
| Limited initialization analysis        | Improved initialization      |
| No convergence evidence                | Convergence tracking         |
| No structured experiment framework     | Reliability Lab              |
| No practical application               | Customer Intelligence System |

---

# V2.0 — Engineering Improvements

## 1. Feature Normalization

Added:

```text
core/normalization.py
```

Min-Max normalization scales features into a common range:

```math
x'=
\frac{x-x_{min}}
{x_{max}-x_{min}}
```

This is important because distance-based algorithms can be dominated by features with larger numerical scales.

For the customer application, the normalization parameters are learned from the training data and reused when a new customer is classified.

---

# 2. Weighted KNN

V1 uses equal voting:

```text
Neighbor 1 → A
Neighbor 2 → A
Neighbor 3 → B
```

V2 introduces distance-weighted voting.

Closer neighbors receive greater influence.

A typical weight is:

```math
w_i=
\frac{1}{d_i+\epsilon}
```

Therefore:

```text
Closer point
     ↓
Higher weight
     ↓
Greater influence
```

This improves the model's ability to distinguish between very close and relatively distant neighbors.

---

# 3. KNN Confidence & Tie Analysis

V2 adds:

```python
predict_with_confidence()
```

The result includes:

```text
Prediction
Confidence
Votes
Tie
```

For example:

```text
K = 5
Votes = {0: 5}
Confidence = 1.0
Tie = False
```

This means:

> **All five selected neighbors voted for the same class.**

The confidence value represents **neighbor-vote agreement**, not a calibrated probability.

---

# 4. K Selection Experiment

V2 compares different values of `K`.

Example:

```text
K = 1
K = 3
K = 5
```

The purpose is not to claim that one value of K is universally optimal.

Instead, the experiment checks whether the prediction is stable as `K` changes.

---

# 5. Inertia

V2 adds clustering evaluation through inertia.

```math
J=
\sum_{i=1}^{n}
\|x_i-\mu_{c_i}\|^2
```

Inertia measures how close points are to the centroid of their assigned cluster.

```text
Lower inertia
      ↓
More compact clusters
```

However, inertia normally decreases as `K` increases, so it should not be used alone to select the number of clusters.

---

# 6. Silhouette Score

V2 adds silhouette analysis:

```math
s(i)=
\frac{b(i)-a(i)}
{\max(a(i),b(i))}
```

Silhouette evaluates both:

* How close a point is to its own cluster
* How far it is from the nearest other cluster

Therefore:

```text
High silhouette
      ↓
Good separation

Around 0
      ↓
Cluster overlap

Negative
      ↓
Possible poor assignment
```

This provides information that inertia alone cannot provide.

---

# 7. Improved K-Means Initialization

V2 improves centroid initialization using a deterministic farthest-point strategy inspired by the K-Means++ idea.

The purpose is to start centroids farther apart rather than relying only on arbitrary initial positions.

This helps provide better-separated starting representatives and makes the initialization strategy explicit.

---

# 8. Convergence Tracking

V2 tracks centroid movement during training.

Conceptually:

```text
Iteration 1
Centroids move significantly
        ↓
Iteration 2
Centroids move less
        ↓
Iteration 3
Centroids barely move
        ↓
Convergence
```

This provides evidence that the iterative optimization process is actually stabilizing.

---

# 9. Reliability Lab

A dedicated experiment framework was added:

```text
reliability.py
```

It evaluates:

### KNN

* Different values of K
* Prediction stability
* Vote confidence
* Ties

### K-Means

* Inertia
* Silhouette score
* Cluster quality
* Different K values

Example:

```text
K    Inertia    Silhouette
2    3.875      0.873
3    2.604      0.700
4    1.333      0.533
```

The experiment demonstrates that:

* Inertia decreases as K increases.
* Silhouette provides information about cluster separation.
* KNN predictions can be examined for stability rather than relying on one arbitrary K.

---

# V2.0 Structural Evolution

The project grew from a basic algorithm implementation into a more structured machine-learning system.

```text
V1.0

core/
├── distance.py
└── sorting.py

knn/
└── knn.py

kmeans/
└── kmeans.py

tests/
└── algorithm tests
```

↓

```text
V2.0

core/
├── distance.py
├── sorting.py
├── normalization.py
└── clustering_metrics.py

knn/
└── knn.py

kmeans/
└── kmeans.py

experiments/
└── reliability.py

application/
├── customer_data.py
├── customer_segmentation.py
└── app.py

tests/
└── extended tests
```

The structural change separates:

```text
Core Algorithms
      ↓
Supporting Mathematics
      ↓
Evaluation
      ↓
Experiments
      ↓
Application
```

This makes the project easier to test, extend, and explain.

---

# 4. CONCLUSION — FROM ALGORITHM TO APPLICATION

The final stage demonstrates why KNN and K-Means are useful beyond the classroom implementation.

The project applies both algorithms to a:

# Customer Intelligence & Segmentation System

The application uses customer features such as:

* Age
* Annual Income
* Spending Score
* Purchase Frequency

---

## K-Means in the Application

K-Means discovers customer groups without predefined labels.

The system identifies three customer segments.

### Cluster 0

```text
Younger customers
Lower average income
High spending
High purchase frequency
```

Interpretation:

> High-engagement / VIP-type customers.

### Cluster 1

```text
Older customers
Higher average income
Low spending
Low purchase frequency
```

Interpretation:

> High-income but low-engagement customers.

### Cluster 2

```text
Middle-range age
Middle-range income
Moderate spending
Moderate purchase frequency
```

Interpretation:

> Moderate-engagement customers.

---

# KNN in the Application

After K-Means discovers the customer groups, KNN is used to classify a **new customer** into an existing segment.

Example:

```text
Age:                 24
Annual Income:       $35,000
Spending Score:      82
Purchase Frequency:  8
```

The system predicts:

```text
Cluster: 0
KNN Vote Confidence: 100%
Tie: No
Votes: {0: 5}
```

The prediction means:

> **All five selected neighbors belong to Cluster 0.**

---

# Business Value

The machine-learning result is then converted into an actionable recommendation.

For example:

```text
Customer Segment
       ↓
Behavior Profile
       ↓
KNN Classification
       ↓
Recommended Action
```

For a high-spending, high-frequency customer:

```text
VIP rewards
+
Loyalty campaign
```

For low-spending customers:

```text
Targeted promotions
+
Personalized offers
```

For moderate-frequency customers:

```text
Engagement campaigns
+
Repeat-purchase incentives
```

This demonstrates the transition from:

> **Mathematical algorithm → software implementation → evaluation → practical decision support.**

---

# Live Application

The project includes an interactive Streamlit interface.

Run:

```bash
python -m streamlit run application/app.py
```

The application provides:

* Customer segmentation overview
* Cluster profiles
* Silhouette score
* Inertia
* New customer classification
* KNN confidence
* Voting information
* Business recommendations

---

# Final Architecture

```text
                    FROM-SCRATCH ML
                          │
             ┌────────────┴────────────┐
             │                         │
            KNN                     K-MEANS
             │                         │
       Classification              Clustering
             │                         │
       Distance + Sort          Assignment + Update
             │                         │
          Voting                  Centroids
             │                         │
             └────────────┬────────────┘
                          │
                     V2.0 LAYER
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
 Normalization       Evaluation         Experiments
       │                  │                  │
       │          Inertia + Silhouette      │
       │                  │            Reliability Lab
       └──────────────────┼──────────────────┘
                          │
                          ↓
              CUSTOMER INTELLIGENCE
                          │
                ┌─────────┴─────────┐
                │                   │
            Segmentation       New Customer
                │                   │
             K-Means                KNN
                │                   │
                └─────────┬─────────┘
                          ↓
                  BUSINESS ACTION
```

---

# Project Philosophy

This project is not only about reproducing KNN and K-Means.

It demonstrates an engineering workflow:

```text
UNDERSTAND
    ↓
IMPLEMENT
    ↓
TEST
    ↓
IDENTIFY LIMITATIONS
    ↓
DESIGN IMPROVEMENTS
    ↓
IMPLEMENT V2.0
    ↓
MEASURE
    ↓
INTERPRET
    ↓
APPLY TO A REAL PROBLEM
```

The central lesson is:

> **A machine-learning system is not complete when the algorithm runs. It becomes useful when its behavior can be evaluated, its limitations are understood, and its output can support a real decision.**

---

# Technologies

### Core Machine Learning

* Python
* Native Python lists
* Loops
* Functions
* Mathematical operations

### Application

* Streamlit

### Important Constraint

The **core ML algorithms** are implemented from scratch and do not rely on:

```text
scikit-learn
NumPy
Pandas
```

for the KNN and K-Means logic.

The application interface uses Streamlit separately from the core algorithm implementation.

---

# Key Learning Outcomes

Through this project, the following concepts were implemented and explored:

* Euclidean distance
* Custom sorting
* KNN neighbor selection
* Majority voting
* Weighted KNN
* Confidence and tie analysis
* Feature normalization
* K-Means centroid initialization
* Nearest-centroid assignment
* Centroid updating
* Convergence
* Inertia
* Silhouette score
* K selection
* Reliability experiments
* Edge-case handling
* Modular ML architecture
* Real-world customer segmentation
* ML-driven business recommendations

---

# Final Result

The project evolved from a **V1.0 educational implementation** into a **V2.0 evaluated and applied machine-learning system**.

```text
V1.0
Basic KNN + K-Means
        ↓
Correctness
        ↓
Limitations
        ↓
V2.0
Normalization
Weighted KNN
Confidence
Evaluation Metrics
Improved Initialization
Convergence Tracking
Reliability Experiments
        ↓
Customer Intelligence Application
        ↓
Practical Business Value
```

**From mathematical foundations to working algorithms, from working algorithms to measurable reliability, and from measurable reliability to a real-world application.**
