"""Fast end-to-end smoke run on configs/smoke.yaml (spec 12.2)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("P2")


if __name__ == "__main__":
    main()
