"""Four genuine RL training loops using TorchRL data structures and PyTorch.

Algorithms: replay/target-network DQN, Monte-Carlo REINFORCE,
GAE-based A2C and clipped-surrogate PPO. TorchRL's TensorDict and
TensorDictReplayBuffer manage trajectories/replay. Loss formulas are
implemented explicitly using torch.nn and torch.distributions so every
term can be inspected in the summative report.

All train/eval runs use the same frozen 53-feature IndustrialInvestmentEnv.
The action mask is used consistently by all four algorithms.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from environment.custom_env import IndustrialInvestmentEnv
from environment.trade_env import TradeSignalEnv

OBS_DIM = 53
N_ACTIONS = 17


@dataclass
class Config:
    algorithm: str = "dqn"
    episodes: int = 20
    seed: int = 42
    run_id: str = "smoke"
    gamma: float = 0.98
    lr: float = 0.0005
    hidden: int = 64
    batch_size: int = 32
    replay_size: int = 5000
    warmup: int = 64
    target_update: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 800
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 24
    eval_episodes: int = 10
    env_mode: str = "synthetic"  # "historical" for all full training runs
    output_dir: str = "logs/training"
    models_dir: str = "models"


def network(output: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, output))


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def tensors(observation: np.ndarray, mask: np.ndarray):
    return (torch.as_tensor(observation, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool))


def select_action(net: nn.Module, obs: np.ndarray, mask: np.ndarray,
                  *, algorithm: str, epsilon: float = 0.0, rng: np.random.Generator,
                  greedy: bool = False):
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        raise ValueError("No legal actions available")
    if algorithm == "dqn" and not greedy and rng.random() < epsilon:
        return int(rng.choice(legal))
    o, m = tensors(obs, mask)
    with torch.no_grad():
        logits = net(o).masked_fill(~m, -1e9)
        if algorithm == "dqn" or greedy:
            return int(logits.argmax().item())
        return int(Categorical(logits=logits).sample().item())


def discounted_returns(rewards: torch.Tensor, gamma: float):
    out = torch.empty_like(rewards)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        carry = float(rewards[i]) + gamma * carry
        out[i] = carry
    return out


def compute_gae(rewards, values, gamma, lam):
    # Full episodes terminate after 24 decisions; terminal bootstrap is zero.
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros((), dtype=torch.float32)
    running = torch.zeros((), dtype=torch.float32)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value - values[i]
        running = delta + gamma * lam * running
        advantages[i] = running
        next_value = values[i]
    return advantages, advantages + values


def build_env(mode: str, *, holdout: bool = False):
    if mode == "synthetic":
        if holdout:
            raise ValueError("Synthetic mode has no historical holdout")
        return IndustrialInvestmentEnv()
    if mode == "historical":
        return TradeSignalEnv(split="holdout" if holdout else "historical")
    raise ValueError(f"Unknown environment mode: {mode}")


def train_dqn(cfg: Config):
    q = network(N_ACTIONS, cfg.hidden)
    target = network(N_ACTIONS, cfg.hidden)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    replay = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.replay_size), batch_size=cfg.batch_size)
    rng = np.random.default_rng(cfg.seed + 100_000)
    step = 0
    logs = []
    env = build_env(cfg.env_mode)
    try:
        for ep in range(cfg.episodes):
            obs, info = env.reset(seed=cfg.seed + ep)
            total, episode_losses, episode_steps = 0.0, [], 0
            for _ in range(24):
                frac = min(step / max(cfg.eps_decay_steps, 1), 1.0)
                epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
                mask = info["action_mask"]
                action = select_action(q, obs, mask, algorithm="dqn", epsilon=epsilon, rng=rng)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                item = TensorDict({
                    "observation": torch.as_tensor(obs).float().unsqueeze(0),
                    "action": torch.tensor([action], dtype=torch.int64),
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "next_observation": torch.as_tensor(next_obs).float().unsqueeze(0),
                    "next_mask": torch.as_tensor(next_info["action_mask"], dtype=torch.bool).unsqueeze(0),
                    "done": torch.tensor([done], dtype=torch.float32),
                }, batch_size=[1])
                replay.extend(item)
                total += reward
                obs, info = next_obs, next_info
                step += 1
                episode_steps += 1
                if step >= cfg.warmup and len(replay) >= cfg.batch_size:
                    batch = replay.sample()
                    q_selected = q(batch["observation"]).gather(1, batch["action"].unsqueeze(-1)).squeeze(-1)
                    with torch.no_grad():
                        next_q = target(batch["next_observation"]).masked_fill(~batch["next_mask"], -1e9)
                        y = batch["reward"] + cfg.gamma * (1.0-batch["done"]) * next_q.max(dim=1).values
                    loss = nn.functional.smooth_l1_loss(q_selected, y)
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
                    episode_losses.append(float(loss.detach()))
                if step % cfg.target_update == 0:
                    target.load_state_dict(q.state_dict())
                if done:
                    break
            logs.append({"episode": ep+1, "reward": total, "loss": float(np.mean(episode_losses)) if episode_losses else float("nan"),
                         "entropy": float("nan"), "epsilon": epsilon, "steps": episode_steps, "env_steps": step})
            if (ep+1) % max(1,cfg.episodes//5)==0:
                print(f"DQN episode {ep+1}/{cfg.episodes}: reward={total:+.4f} loss={logs[-1]['loss']:.4f} eps={epsilon:.3f}")
    finally:
        env.close()
    return q, logs


def train_policy_gradient(cfg: Config):
    policy = network(N_ACTIONS, cfg.hidden)
    critic = network(1, cfg.hidden) if cfg.algorithm in ("ppo", "a2c") else None
    params = list(policy.parameters()) + (list(critic.parameters()) if critic else [])
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed + 100_000)
    logs=[]
    env=build_env(cfg.env_mode)
    try:
        for ep in range(cfg.episodes):
            obs, info = env.reset(seed=cfg.seed+ep)
            store={k:[] for k in ("obs", "mask", "action", "reward", "old_log_prob", "old_value")}
            total=0.0
            for _ in range(24):
                old_o, old_m = tensors(obs, info["action_mask"])
                with torch.no_grad():
                    dist = Categorical(logits=policy(old_o).masked_fill(~old_m,-1e9))
                    action = int(dist.sample().item())
                    lp = float(dist.log_prob(torch.tensor(action)))
                    val = float(critic(old_o).squeeze()) if critic is not None else 0.0
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                for k,v in (("obs",old_o),("mask",old_m),("action",action),("reward",reward),("old_log_prob",lp),("old_value",val)):
                    store[k].append(v)
                total += reward
                obs, info = next_obs, next_info
                if terminated or truncated:
                    break
            # TorchRL's native trajectory container — fixed 24-step episodes.
            traj = TensorDict({
                "obs":torch.stack(store["obs"]), "mask":torch.stack(store["mask"]),
                "action":torch.tensor(store["action"],dtype=torch.int64),
                "reward":torch.tensor(store["reward"],dtype=torch.float32),
                "old_log_prob":torch.tensor(store["old_log_prob"],dtype=torch.float32),
                "old_value":torch.tensor(store["old_value"],dtype=torch.float32),
            }, batch_size=[len(store["action"])])
            rewards=traj["reward"]
            if cfg.algorithm == "reinforce":
                # True Monte Carlo REINFORCE: no learned critic, no bootstrapping.
                ret=discounted_returns(rewards,cfg.gamma)
                ret=(ret-ret.mean())/(ret.std(unbiased=False)+1e-8)
                dist=Categorical(logits=policy(traj["obs"]).masked_fill(~traj["mask"],-1e9))
                loss_actor=-(dist.log_prob(traj["action"])*ret.detach()).mean()
                entropy=dist.entropy().mean()
                loss=loss_actor-cfg.entropy_coeff*entropy
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
                loss_value=torch.tensor(0.0)
            else:
                advantage, targets=compute_gae(rewards,traj["old_value"],cfg.gamma,cfg.gae_lambda)
                advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
                traj.set("advantage",advantage.detach())
                traj.set("target_value",targets.detach())
                losses=[]; entropies=[]; values=[]
                # A2C: single full-rollout update; PPO: multiple clipped minibatch epochs.
                epochs=cfg.update_epochs if cfg.algorithm=="ppo" else 1
                size=len(traj)
                batch_size=min(cfg.minibatch_size,size) if cfg.algorithm=="ppo" else size
                for _epoch in range(epochs):
                    for ids in torch.randperm(size).split(batch_size):
                        b=traj[ids]
                        dist=Categorical(logits=policy(b["obs"]).masked_fill(~b["mask"],-1e9))
                        logp=dist.log_prob(b["action"])
                        if cfg.algorithm=="ppo":
                            ratio=torch.exp(logp-b["old_log_prob"])
                            unclipped=ratio*b["advantage"]
                            clipped=ratio.clamp(1-cfg.clip_epsilon,1+cfg.clip_epsilon)*b["advantage"]
                            actor_loss=-torch.minimum(unclipped,clipped).mean()
                        else:
                            actor_loss=-(logp*b["advantage"]).mean()
                        value_loss=nn.functional.mse_loss(critic(b["obs"]).squeeze(-1),b["target_value"])
                        entropy=dist.entropy().mean()
                        loss=actor_loss+cfg.value_coeff*value_loss-cfg.entropy_coeff*entropy
                        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
                        losses.append(float(loss.detach()));entropies.append(float(entropy.detach()));values.append(float(value_loss.detach()))
                entropy=torch.tensor(np.mean(entropies));loss=torch.tensor(np.mean(losses));loss_value=torch.tensor(np.mean(values))
            logs.append({"episode":ep+1,"reward":total,"loss":float(loss.detach()),"entropy":float(entropy.detach()),"value_loss":float(loss_value.detach()),"steps":len(traj)})
            if (ep+1)%max(1,cfg.episodes//5)==0:
                print(f"{cfg.algorithm.upper()} episode {ep+1}/{cfg.episodes}: reward={total:+.4f} loss={float(loss.detach()):.4f} entropy={float(entropy.detach()):.3f}")
    finally:
        env.close()
    return policy, logs


def evaluate(net:nn.Module, algorithm:str, episodes:int, seed:int, *, env_mode: str = "synthetic", holdout: bool = False, stochastic: bool = False):
    rng=np.random.default_rng(seed+100_000)
    results=[]
    env=build_env(env_mode, holdout=holdout)
    try:
        net.eval()
        for ep in range(episodes):
            obs,info=env.reset(seed=seed+ep)
            total=0.0
            for _ in range(24):
                action=select_action(net,obs,info["action_mask"],algorithm=algorithm,rng=rng,greedy=not stochastic)
                obs,r,terminated,truncated,info=env.step(action)
                total+=r
                if terminated or truncated:break
            results.append(total)
    finally:
        env.close()
    return float(np.mean(results)),float(np.std(results))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--algorithm",choices=["dqn","reinforce","ppo","a2c"],required=True)
    p.add_argument("--episodes",type=int,default=20)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--run-id",default="smoke")
    p.add_argument("--lr",type=float,default=0.0005)
    p.add_argument("--gamma",type=float,default=0.98)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--replay-size",type=int,default=5000)
    p.add_argument("--warmup",type=int,default=64)
    p.add_argument("--target-update",type=int,default=100)
    p.add_argument("--eps-start",type=float,default=1.0)
    p.add_argument("--eps-end",type=float,default=0.05)
    p.add_argument("--eps-decay-steps",type=int,default=800)
    p.add_argument("--entropy-coeff",type=float,default=0.01)
    p.add_argument("--gae-lambda",type=float,default=0.95)
    p.add_argument("--clip-epsilon",type=float,default=0.2)
    p.add_argument("--update-epochs",type=int,default=4)
    p.add_argument("--minibatch-size",type=int,default=24)
    p.add_argument("--eval-episodes",type=int,default=10)
    p.add_argument("--env-mode",choices=["synthetic","historical"],default="synthetic")
    p.add_argument("--value-coeff",type=float,default=0.5)
    args=p.parse_args()
    cfg=Config(**vars(args))
    if cfg.episodes < 1 or cfg.eval_episodes < 1 or cfg.batch_size < 1 or cfg.replay_size < cfg.batch_size:
        p.error("episodes/eval-episodes/batch-size must be positive and replay size >= batch size")
    if cfg.algorithm=="dqn" and (cfg.target_update < 1 or cfg.warmup < 1):
        p.error("target update and warmup must be positive")
    fix_seed(cfg.seed)
    model,logs=train_dqn(cfg) if cfg.algorithm=="dqn" else train_policy_gradient(cfg)
    # This is a validation on unseen simulator seeds, NOT the 2024 HS4 holdout.
    val_mean,val_std=evaluate(model,cfg.algorithm,cfg.eval_episodes,seed=50000,env_mode=cfg.env_mode)
    if cfg.algorithm == "dqn":
        val_sample_mean, val_sample_std = None, None
    else:
        val_sample_mean,val_sample_std=evaluate(model,cfg.algorithm,cfg.eval_episodes,seed=50000,env_mode=cfg.env_mode,stochastic=True)
    path=Path(cfg.output_dir)/cfg.algorithm/cfg.run_id
    path.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(x.keys() for x in logs)))
    with (path/"episodes.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(logs)
    model_dir=Path(cfg.models_dir)/cfg.algorithm
    model_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=model_dir/(cfg.run_id+".pt")
    torch.save({"algorithm":cfg.algorithm,"model_state":model.state_dict(),"config":asdict(cfg),"obs_dim":OBS_DIM,"n_actions":N_ACTIONS},checkpoint)
    summary={"run_id":cfg.run_id,"algorithm":cfg.algorithm,"train_episodes":cfg.episodes,
             "last_10_train_mean":float(np.mean([x["reward"] for x in logs[-10:]])),
             "validation_mean_reward":val_mean,"validation_std_reward":val_std,
             "validation_stochastic_mean_reward":val_sample_mean,
             "validation_stochastic_std_reward":val_sample_std,
             "evaluation_split":"historical indicators + unseen simulator seeds (not 2024)",
             "eval_seeds":"50000.."+str(50000+cfg.eval_episodes-1),"model":str(checkpoint),"config":asdict(cfg)}
    (path/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Saved:",checkpoint,path/"episodes.csv",path/"summary.json")
    print("Validation greedy reward:",f"{val_mean:+.4f} +/- {val_std:.4f}")
    if val_sample_mean is not None:
        print("Validation sampled reward:",f"{val_sample_mean:+.4f} +/- {val_sample_std:.4f}")
    print("Note: initial smoke runs do NOT count toward the required ten meaningful configurations per method.")


if __name__=="__main__":main()
