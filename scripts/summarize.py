"""Aggregate per-run outputs into tables, uncertainty estimates and plots (spec 10, 11.4)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("P4")


if __name__ == "__main__":
    main()
