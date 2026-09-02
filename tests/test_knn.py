from knn.knn import KNN
from data.knn_data import TRAINING_DATA


def test_knn_prediction():

    model = KNN(k=3)
    model.fit(TRAINING_DATA)

    prediction = model.predict([1.8, 1.7])

    assert prediction == "A"

    print("KNN prediction test passed.")
    print("Query: [1.8, 1.7]")
    print("Prediction:", prediction)


def test_empty_training_data():

    model = KNN(k=3)

    try:
        model.fit([])
        assert False
    except ValueError:
        assert True


def test_invalid_k():

    try:
        KNN(k=0)
        assert False
    except ValueError:
        assert True


def test_k_larger_than_dataset():

    model = KNN(k=10)
    model.fit(TRAINING_DATA)

    try:
        model.predict([1.8, 1.7])
        assert False
    except ValueError:
        assert True


test_knn_prediction()
test_empty_training_data()
test_invalid_k()
test_k_larger_than_dataset()