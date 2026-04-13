"""Main application module."""

from utils import parse_config, clean_string


def start(config_path: str) -> dict:
    """Initialize the application."""
    config = parse_config(config_path)
    return config


def process_data(items: list) -> list:
    """Process the given items.

    # TODO: Add input validation for empty lists
    """
    result = []
    for item in items:
        cleaned = clean_string(str(item))
        result.append(cleaned.upper())
    return result


def finalize(data: list) -> str:
    """Produce final output."""
    return "\n".join(data)


if __name__ == "__main__":
    cfg = start("config.yaml")
    processed = process_data(["hello", "world"])
    print(finalize(processed))