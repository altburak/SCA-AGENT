"""Process users.csv and filter eligible users."""
import csv


def load_users(path: str) -> list[dict]:
    """Load users from CSV file."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def is_eligible(user: dict) -> bool:
    """Check if a user is eligible.

    Eligibility rules:
    - User must be at least 30 years old.
    - User status must be 'active'.
    """
    age = int(user["age"])
    status = user["status"]
    return age >= 30 and status == "active"


def filter_eligible(users: list[dict]) -> list[dict]:
    """Return only eligible users."""
    return [u for u in users if is_eligible(u)]


if __name__ == "__main__":
    users = load_users("users.csv")
    eligible = filter_eligible(users)
    print(f"{len(eligible)} eligible users:")
    for u in eligible:
        print(f"  - {u['name']}")