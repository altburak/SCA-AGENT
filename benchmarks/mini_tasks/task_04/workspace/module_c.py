"""Module C — imports D (creates cycle)."""
from module_d import utility


def process(source: str):
    utility(source)