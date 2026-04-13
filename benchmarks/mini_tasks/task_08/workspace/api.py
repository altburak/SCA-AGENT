"""User API implementation."""


_users: dict[int, dict] = {}


def get_user(user_id: int) -> dict:
    """Get a user by ID."""
    return _users[user_id]  # Raises KeyError if missing — contradicts docs


def add_user(user_id: int, name: str, email: str) -> None:
    """Add a new user."""
    if user_id in _users:
        raise ValueError(f"User {user_id} already exists")
    _users[user_id] = {"id": user_id, "name": name, "email": email}


def list_users() -> list[dict]:
    """Return all users in insertion order (by id)."""
    # Docs say "sorted by name" but this returns by id
    return list(_users.values())


def delete_user(user_id: int) -> bool:
    """Delete a user."""
    if user_id in _users:
        del _users[user_id]
        return True
    return False