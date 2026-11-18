"""Demonstrate one deterministic, manually selected investment episode."""
from environment.custom_env import IndustrialInvestmentEnv


def main():
    env = IndustrialInvestmentEnv(render_mode="ansi")
    observation, info = env.reset(seed=42)

    print("INDUSTRIAL INVESTMENT RL — ENVIRONMENT PREVIEW")
    print(f"Observation shape: {observation.shape}")
    print(f"Valid observation: {env.observation_space.contains(observation)}")
    print(env.render())
    print("-" * 76)

    # Agro-processing: study -> pilot -> mature -> expand -> operate -> exit.
    # Other months deliberately hold capital.
    planned_actions = [1, 5, 0, 9, 0, 13] + [0] * 18
    episode_return = 0.0

    for action_id in planned_actions:
        observation, reward, terminated, truncated, info = env.step(action_id)
        episode_return += reward
        state = info["portfolio"]
        print(
            f"Month {state['month']:02d} | "
            f"Action: {info['action_description']:<41} | "
            f"Valid: {info['action_valid']} | "
            f"Reward: {reward:+.4f} | "
            f"Net worth: {state['net_worth']:.2f}"
        )
        if terminated or truncated:
            break

    print("-" * 76)
    print(f"Episode finished after month {env.month}")
    print(f"Total episode reward: {episode_return:+.4f}")
    print("Final public state:", env.public_state())
    env.close()


if __name__ == "__main__":
    main()
