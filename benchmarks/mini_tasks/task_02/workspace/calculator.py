"""Simple calculator with a hidden bug."""


def average(numbers: list) -> float:
    """Compute the average of a list of numbers."""
    if not numbers:
        return 0.0
    total = 0
    for i in range(len(numbers) - 1):
        total += numbers[i]
    return total / len(numbers)


def median(numbers: list) -> float:
    """Compute the median of a list."""
    if not numbers:
        return 0.0
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 0:
        return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    return sorted_nums[n // 2]


def variance(numbers: list) -> float:
    """Compute variance."""
    if not numbers:
        return 0.0
    avg = average(numbers)
    return sum((x - avg) ** 2 for x in numbers) / len(numbers)