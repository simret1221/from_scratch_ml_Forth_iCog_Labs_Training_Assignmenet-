from knn.weighted_knn import WeightedKNN
from data.knn_data import TRAINING_DATA


def test_weighted_knn_prediction():

    model = WeightedKNN(k=3)
    model.fit(TRAINING_DATA)

    prediction = model.predict([1.8, 1.7])

    assert prediction == "A"

    print("Weighted KNN prediction test passed.")
    print("Query: [1.8, 1.7]")
    print("Prediction:", prediction)


def test_empty_training_data():

    model = WeightedKNN(k=3)

    try:
        model.fit([])
        assert False
    except ValueError:
        assert True

    print("Weighted KNN empty data test passed.")


def test_invalid_k():

    try:
        WeightedKNN(k=0)
        assert False
    except ValueError:
        assert True

    print("Weighted KNN invalid K test passed.")


def test_exact_match():

    model = WeightedKNN(k=3)
    model.fit(TRAINING_DATA)

    prediction = model.predict([1.0, 1.0])

    assert prediction == "A"

    print("Weighted KNN exact-match test passed.")


test_weighted_knn_prediction()
test_empty_training_data()
test_invalid_k()
test_exact_match()