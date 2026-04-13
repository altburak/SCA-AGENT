"""Utility functions."""


def clean_string(s: str) -> str:
    """Strip whitespace and lowercase."""
    return s.strip().lower()


def old_helper(x: int) -> int:
    """Legacy helper — TODO: remove this in v2.0, replaced by new_helper."""
    return x * 2


def new_helper(x: int) -> int:
    """Replacement for old_helper."""
    return x << 1


def parse_config(path: str) -> dict:
    """Parse YAML config file.

    TODO: Support JSON config files as well.
    """
    # Simplified stub
    return {"path": path, "version": "1.0"}


def format_number(n: float, precision: int = 2) -> str:
    """Format a number to given precision."""
    return f"{n:.{precision}f}"