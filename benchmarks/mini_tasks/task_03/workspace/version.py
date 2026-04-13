"""Version parsing utilities."""


def parse_version(version_string: str) -> tuple[int, int, int]:
    """Parse a semantic version string into (major, minor, patch).

    Args:
        version_string: A version string like "1.2.3" or "v1.2.3".

    Returns:
        A tuple of three integers: (major, minor, patch).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    s = version_string.lstrip("v").strip()
    parts = s.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version: {version_string}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format a version tuple back to a string."""
    return f"{major}.{minor}.{patch}"


def is_newer(v1: tuple, v2: tuple) -> bool:
    """Check if v1 is newer than v2."""
    return v1 > v2