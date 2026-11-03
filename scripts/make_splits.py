"""Build disjoint, hashed calibration/search/selection/test splits by original image ID (spec 8)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("P0")


if __name__ == "__main__":
    main()
