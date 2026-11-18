"""One-time 2024 evaluation after choosing configurations on 2016-2023 validation.

This script chooses the best config PER ALGORITHM using historical validation
results only; it does not tune or retrain on 2024.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from training.experiment import N_ACTIONS, evaluate, fix_seed, network
from evaluation.baselines import choose_action
from environment.trade_env import TradeSignalEnv


def evaluate_baseline(policy: str, episodes=100, seed=60000):
    vals=[]
    env=TradeSignalEnv(split='holdout')
    try:
        for i in range(episodes):
            obs,info=env.reset(seed=seed+i)
            rng=np.random.default_rng(seed+i+100000)
            score=0.0
            for _ in range(24):
                a=choose_action(policy,obs,info['action_mask'],rng)
                obs,r,terminated,truncated,info=env.step(a)
                score+=float(r)
                if terminated or truncated:break
            vals.append(score)
    finally:
        env.close()
    return float(np.mean(vals)),float(np.std(vals))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=100)
    args=p.parse_args()
    if args.episodes != 100:
        p.error('The frozen final protocol uses exactly 100 held-out episodes')
    root=Path('logs/training')
    rows=[]
    for algorithm in ('dqn','reinforce','ppo','a2c'):
        candidates=[]
        for result_path in (root/algorithm).glob('*/summary.json'):
            entry=json.loads(result_path.read_text(encoding='utf-8'))
            if entry.get('train_episodes') == 1000 and entry.get('config',{}).get('env_mode') == 'historical':
                candidates.append(entry)
        if len(candidates) != 10:
            raise ValueError(f'{algorithm}: expected 10 completed 1000-episode configs; found {len(candidates)}')
        chosen=max(candidates,key=lambda x:x['validation_mean_reward'])
        cfg=chosen['config']
        fix_seed(int(cfg['seed']))
        model=network(N_ACTIONS,int(cfg['hidden']))
        checkpoint=torch.load(chosen['model'],map_location='cpu',weights_only=True)
        if checkpoint['algorithm'] != algorithm or checkpoint['config']['env_mode'] != 'historical':
            raise ValueError('Wrong model checkpoint or environment settings')
        model.load_state_dict(checkpoint['model_state'])
        greedy_mean,greedy_std=evaluate(model,algorithm,100,seed=60000,env_mode='historical',holdout=True)
        stochastic_mean,stochastic_std=(None,None)
        if algorithm!='dqn':
            stochastic_mean,stochastic_std=evaluate(model,algorithm,100,seed=60000,env_mode='historical',holdout=True,stochastic=True)
        rows.append({'algorithm':algorithm,'selected_run_id':chosen['run_id'],
                     'historical_validation_mean':chosen['validation_mean_reward'],
                     'heldout_2024_greedy_mean':greedy_mean,'heldout_2024_greedy_std':greedy_std,
                     'heldout_2024_sampled_mean':stochastic_mean,'heldout_2024_sampled_std':stochastic_std,
                     'checkpoint':chosen['model']})
        print(f'{algorithm:10s} best={chosen["run_id"]:18s} historical-val={chosen["validation_mean_reward"]:+.4f} 2024-greedy={greedy_mean:+.4f}')
    for baseline in ('always_hold','random_valid','rule_based'):
        mean,std=evaluate_baseline(baseline)
        rows.append({'algorithm':baseline,'selected_run_id':'baseline','historical_validation_mean':'',
                     'heldout_2024_greedy_mean':mean,'heldout_2024_greedy_std':std,
                     'heldout_2024_sampled_mean':'','heldout_2024_sampled_std':'','checkpoint':''})
    result_path=root/'final_2024_evaluation.csv'
    with result_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('Saved',result_path)
    print('Interpretation: real HS4 signals initialize synthetic 24-month economics; these are NOT observed returns or actual 2024 firm outcomes.')

if __name__=='__main__':main()
