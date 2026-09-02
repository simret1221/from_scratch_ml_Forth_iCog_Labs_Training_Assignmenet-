def print_experiment_header(title):
    print()
    print("=" * 50)
    print(title)
    print("=" * 50)


def print_result(name, value):
    print(f"{name}: {value}")


def accuracy(predictions, actual):

    if not predictions:
        raise ValueError("Predictions cannot be empty.")

    if len(predictions) != len(actual):
        raise ValueError(
            "Predictions and actual labels must have same length."
        )

    correct = 0

    for prediction, target in zip(predictions, actual):

        if prediction == target:
            correct += 1

    return correct / len(actual)