"""Fixed, resumable CPU sweep: 10 meaningful configurations × 4 algorithms.

Usage: uv run python -m training.run_sweep --dry-run
       uv run python -m training.run_sweep
       uv run python -m training.run_sweep --algorithm dqn

Do not tune on 2024. All runs have 1000 episodes, 100 validation episodes,
seed 42, and use the identical historical-data initialized environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

EPISODES = 1000
EVAL_EPISODES = 100
SEED = 42

# Each configuration varies at least one meaningful shared/method-specific control.
CONFIGS = {
    'dqn': [
        {'lr': 0.0005, 'gamma': .98, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0002, 'gamma': .98, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0010, 'gamma': .98, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .95, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .995, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .98, 'hidden': 128, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .98, 'hidden': 64, 'batch_size': 64, 'replay_size': 10000, 'target_update': 100, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .98, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 250, 'eps_decay_steps': 16000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .98, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 8000, 'warmup': 250},
        {'lr': 0.0005, 'gamma': .98, 'hidden': 64, 'batch_size': 32, 'replay_size': 5000, 'target_update': 100, 'eps_decay_steps': 20000, 'warmup': 250, 'eps_end': .10},
    ],
    'reinforce': [
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'entropy_coeff': .01},
        {'lr': .0001, 'gamma': .98, 'hidden': 64, 'entropy_coeff': .01},
        {'lr': .0010, 'gamma': .98, 'hidden': 64, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .95, 'hidden': 64, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .995, 'hidden': 64, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 128, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 32, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'entropy_coeff': 0.0},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'entropy_coeff': .05},
        {'lr': .0002, 'gamma': .95, 'hidden': 128, 'entropy_coeff': .03},
    ],
    'ppo': [
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0002, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0010, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .1, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .3, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 2, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 8, 'gae_lambda': .95, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 4, 'gae_lambda': .90, 'entropy_coeff': .01},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'clip_epsilon': .2, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': .05},
        {'lr': .0005, 'gamma': .98, 'hidden': 128, 'clip_epsilon': .2, 'update_epochs': 4, 'gae_lambda': .95, 'entropy_coeff': 0.0},
    ],
    'a2c': [
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0002, 'gamma': .98, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0010, 'gamma': .98, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0005, 'gamma': .95, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0005, 'gamma': .995, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'gae_lambda': .90, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'gae_lambda': 1.00, 'entropy_coeff': .01, 'value_coeff': .5},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .05, 'value_coeff': .5},
        {'lr': .0005, 'gamma': .98, 'hidden': 64, 'gae_lambda': .95, 'entropy_coeff': .01, 'value_coeff': 1.0},
        {'lr': .0002, 'gamma': .98, 'hidden': 128, 'gae_lambda': .95, 'entropy_coeff': 0.0, 'value_coeff': .25},
    ],
}


def result_rows() -> list[dict]:
    rows=[]
    for algorithm, params in CONFIGS.items():
        assert len(params) == 10, algorithm
        assert len({json.dumps(p, sort_keys=True) for p in params}) == 10, algorithm
        for n, options in enumerate(params, 1):
            rows.append({'algorithm': algorithm, 'run_id': f'{algorithm}_{n:02d}', **options})
    assert len(rows) == 40
    return rows


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm', choices=sorted(CONFIGS), help='Run just one algorithm (10 configurations)')
    p.add_argument('--dry-run', action='store_true', help='Print all planned configurations but do not train')
    args=p.parse_args()
    rows=result_rows()
    out=Path('logs/training')
    out.mkdir(parents=True, exist_ok=True)
    selected=[r for r in rows if args.algorithm is None or r['algorithm'] == args.algorithm]
    # Frozen *predeclared* protocol, not an adaptive hyperparameter search.
    with (out/'experiment_plan.json').open('w',encoding='utf-8') as f:
        json.dump({'episodes_per_run':EPISODES,'eval_episodes_per_run':EVAL_EPISODES,'train_seed':SEED,'validation_seed_range':[50000,50099],'env_mode':'historical','runs':rows},f,indent=2)
    for i, run in enumerate(selected, 1):
        target = out/run['algorithm']/run['run_id']/'summary.json'
        if target.is_file():
            completed=json.loads(target.read_text(encoding='utf-8'))
            if completed.get('train_episodes') == EPISODES and completed.get('config',{}).get('env_mode') == 'historical':
                print(f'[{i}/{len(selected)}] SKIP already completed: {run["run_id"]}',flush=True)
                continue
            raise RuntimeError(f'Incomplete or conflicting existing result; inspect first: {target}')
        cmd=[sys.executable,'-m','training.experiment','--algorithm',run['algorithm'],'--run-id',run['run_id'],
             '--episodes',str(EPISODES),'--eval-episodes',str(EVAL_EPISODES),'--seed',str(SEED),'--env-mode','historical']
        for k,v in run.items():
            if k in ('algorithm','run_id'):continue
            cmd.extend(['--'+k.replace('_','-'),str(v)])
        print(f'[{i}/{len(selected)}] {run["run_id"]}: {run}',flush=True)
        if not args.dry_run:
            subprocess.run(cmd,check=True)
    if not args.dry_run:
        summary=[]
        for run in rows:
            f=out/run['algorithm']/run['run_id']/'summary.json'
            if f.is_file():
                x=json.loads(f.read_text(encoding='utf-8'))
                if x.get('train_episodes') == EPISODES and x['config'].get('env_mode') == 'historical':
                    summary.append({'algorithm':run['algorithm'],'run_id':run['run_id'],
                                    'validation_mean_reward':x['validation_mean_reward'],
                                    'validation_std_reward':x['validation_std_reward'],
                                    'validation_stochastic_mean_reward':x.get('validation_stochastic_mean_reward'),
                                    'last_10_train_mean':x['last_10_train_mean'],
                                    **{k:v for k,v in run.items() if k not in ('algorithm','run_id')}})
        if summary:
            keys=list(dict.fromkeys(k for row in summary for k in row))
            with (out/'sweep_summary.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f, fieldnames=keys)
                writer.writeheader();writer.writerows(summary)
            print(f'Saved {out/"sweep_summary.csv"} with {len(summary)}/40 completed configurations')


if __name__ == '__main__':main()
