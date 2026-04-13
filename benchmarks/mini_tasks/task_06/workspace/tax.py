"""Tax calculation module."""


def calculate_tax(income: float) -> float:
    """Calculate tax using a progressive tax system.

    Tax brackets:
    - Income below 10,000: 10% tax
    - Income 10,000 to 50,000: 20% tax
    - Income above 50,000: 30% tax

    Args:
        income: Annual income in dollars.

    Returns:
        Tax amount owed.
    """
    if income < 0:
        raise ValueError("Income cannot be negative")
    if income < 10000:
        return income * 0.10
    elif income < 50000:
        return income * 0.20
    else:
        return income * 0.30


def net_income(gross: float) -> float:
    """Compute net income after tax."""
    return gross - calculate_tax(gross)