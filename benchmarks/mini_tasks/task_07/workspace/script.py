"""Demo script for output prediction."""


def process(items: list[int]) -> list[int]:
    return [x * x for x in items]


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    result = process(numbers)
    print(result)
    print(f"sum={sum(result)}")