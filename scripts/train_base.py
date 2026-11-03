"""Train and checkpoint one clean base model per seed (spec 8, Task A/B step 1)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("P0")


if __name__ == "__main__":
    main()
