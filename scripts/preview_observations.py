
from environment.observations import (
    OBSERVATION_NAMES,
    OBSERVATION_SPACE,
    encode_observation,
    make_demo_snapshot,
)


def main():
    snapshot = make_demo_snapshot()
    observation = encode_observation(snapshot)

    print("=" * 65)
    print("INDUSTRIAL INVESTMENT SIMULATOR")
    print("PHASE 2 - ACTIVITY 4: OBSERVATION SPACE")
    print("=" * 65)

    print(f"\nObservation space: {OBSERVATION_SPACE}")
    print(f"Observation shape: {observation.shape}")
    print(f"Observation dtype: {observation.dtype}")
    print(
        f"Valid observation: "
        f"{OBSERVATION_SPACE.contains(observation)}"
    )

    print("\nAgent observations:")
    print("-" * 65)

    for index, (name, value) in enumerate(
        zip(OBSERVATION_NAMES, observation)
    ):
        print(f"{index:02d} | {name:47s} | {value:.3f}")

    print("-" * 65)
    print(f"Total features: {len(observation)}")


if __name__ == "__main__":
    main()
    