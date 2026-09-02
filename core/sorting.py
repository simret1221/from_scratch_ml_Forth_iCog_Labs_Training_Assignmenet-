def quick_sort(items):
    """
    Sort items by their first element using Quick Sort.

    Each item is expected to have the form:

        (distance, label, index)
    """

    if len(items) <= 1:
        return items[:]

    pivot = items[len(items) // 2][0]

    left = []
    middle = []
    right = []

    for item in items:

        if item[0] < pivot:
            left.append(item)

        elif item[0] > pivot:
            right.append(item)

        else:
            middle.append(item)

    return (
        quick_sort(left)
        + middle
        + quick_sort(right)
    )