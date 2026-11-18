"""Command-line entry point for the industrial-investment simulator.

Default: run a scripted 24-month episode.
--catalog: show the discrete action mapping.
Neither option runs a trained agent or graphical dashboard yet.
"""

import argparse

from environment.actions import ACTIONS, ACTION_SPACE
from scripts.preview_environment import main as preview_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial investment RL simulator")
    parser.add_argument(
        "--catalog", action="store_true", help="Show the 17 available investment actions"
    )
    args = parser.parse_args()
    if args.catalog:
        print(f"Action space: {ACTION_SPACE}")
        for action in ACTIONS:
            print(f"{action.action_id:2d} | {action.description}")
    else:
        preview_main()


if __name__ == "__main__":
    main()
