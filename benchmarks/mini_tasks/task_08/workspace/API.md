# User API Documentation

## Functions

### `get_user(user_id: int) -> dict | None`

Retrieves a user by their ID. Returns a dictionary with user data.

- **Returns:** User dict, or `None` if the user is not found.
- **Raises:** Nothing. Safe to call with any ID.

### `add_user(user_id: int, name: str, email: str) -> None`

Adds a new user to the store. Raises `ValueError` if the user_id already exists.

### `list_users() -> list[dict]`

Returns all users as a list, **sorted alphabetically by name**.

### `delete_user(user_id: int) -> bool`

Removes a user by ID. Returns `True` if deleted, `False` if not found.