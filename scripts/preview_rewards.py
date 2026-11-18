
from dataclasses import replace

from environment.rewards import (
    RewardSnapshot,
    calculate_reward,
)


def main():
    print("=" * 65)
    print("INTELLIGENT INDUSTRIAL INVESTMENT DECISION SIMULATOR")
    print("PHASE 2 - ACTIVITY 5: REWARD FUNCTION")
    print("=" * 65)

    initial = RewardSnapshot(
        net_worth=100.0,
        employment_index=0.0,
        import_substitution_index=0.0,
        risk_exposure_index=0.0,
    )

    scenarios = [
        (
            "Hold capital - no changes",
            initial,
            True,
        ),
        (
            "Feasibility study costs 3 units",
            replace(initial, net_worth=97.0),
            True,
        ),
        (
            "Fund pilot - capital transferred to asset",
            replace(initial, risk_exposure_index=0.4),
            True,
        ),
        (
            "Profitable operating investment",
            replace(
                initial,
                net_worth=108.0,
                employment_index=0.4,
                import_substitution_index=0.2,
                risk_exposure_index=0.2,
            ),
            True,
        ),
        (
            "Invalid investment decision",
            initial,
            False,
        ),
    ]

    print("\nSIMULATED SCENARIOS\n")

    for description, after, valid in scenarios:
        result = calculate_reward(
            before=initial,
            after=after,
            action_valid=valid,
        )

        print(f"Scenario: {description}")
        print(f"  Financial:       {result.financial:+.4f}")
        print(f"  Employment:      {result.employment:+.4f}")
        print(
            f"  Import subst.:   "
            f"{result.import_substitution:+.4f}"
        )
        print(
            f"  Risk penalty:    "
            f"{result.risk_penalty:+.4f}"
        )
        print(
            f"  Invalid penalty: "
            f"{result.invalid_action_penalty:+.4f}"
        )
        print(f"  TOTAL REWARD:    {result.total:+.4f}")
        print("-" * 55)


if __name__ == "__main__":
    main()
