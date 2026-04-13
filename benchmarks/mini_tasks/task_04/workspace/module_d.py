"""Module D — imports C (cycle!)."""
import module_c


def utility(source: str):
    print(f"utility called from {source}")
    # This creates a cycle:
    module_c.process(source + "_reentered")