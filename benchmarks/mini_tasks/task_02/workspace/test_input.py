"""Test harness for calculator.py — run this to see the bug."""

from calculator import average, median, variance


if __name__ == "__main__":
    data = [10, 20, 30, 40, 50]
    # Expected average: 30.0
    # Actual buggy output will differ — identify why
    print(f"average({data}) = {average(data)}")
    print(f"median({data}) = {median(data)}")
    print(f"variance({data}) = {variance(data)}")