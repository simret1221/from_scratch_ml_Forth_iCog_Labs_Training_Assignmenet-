from core.sorting import quick_sort


items = [
    (5.0, "A", 0),
    (1.0, "B", 1),
    (3.0, "A", 2),
    (2.0, "B", 3)
]

result = quick_sort(items)

distances = [item[0] for item in result]

assert distances == [1.0, 2.0, 3.0, 5.0]

print("Sorting tests passed.")