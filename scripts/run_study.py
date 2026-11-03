"""Run the frozen study matrix; refuses to run unless the config is frozen (spec 3, 12)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("P4")


if __name__ == "__main__":
    main()
