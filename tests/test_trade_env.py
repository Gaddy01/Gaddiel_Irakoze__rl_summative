"""Local integration tests for documented historical/2024 split."""
import csv
import pytest

from environment.actions import SECTORS
from environment.observations import OBSERVATION_NAMES
from environment.trade_env import TradeSignalEnv, read_signals


def write_fixture(tmp_path):
    for split, years in [('historical',range(2016,2024)),('holdout',[2024])]:
        path=tmp_path/('sector_signals_historical.csv' if split=='historical' else 'sector_signals_2024_holdout.csv')
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['sector','signal_year','demand_score','growth_signal'])
            writer.writeheader()
            for year in years:
                for sid in SECTORS:
                    writer.writerow({'sector':sid,'signal_year':year,'demand_score':round(.1+.05*(year-2016),4),'growth_signal':.55})


def test_trade_environment_initializes_all_53_features(tmp_path):
    write_fixture(tmp_path)
    env=TradeSignalEnv('historical',tmp_path)
    obs,info=env.reset(seed=42)
    assert obs.shape==(53,)
    assert env.observation_space.contains(obs)
    assert info['signal_year']==2018
    i=OBSERVATION_NAMES.index('agro_processing.demand_score')
    assert float(obs[i])==pytest.approx(.20)
    env.close()


def test_historical_never_uses_2024_and_holdout_only_uses_2024(tmp_path):
    write_fixture(tmp_path)
    train=TradeSignalEnv('historical',tmp_path)
    test=TradeSignalEnv('holdout',tmp_path)
    assert {train.reset(seed=s)[1]['signal_year'] for s in range(40)}==set(range(2016,2024))
    assert {test.reset(seed=s)[1]['signal_year'] for s in range(40)}=={2024}
    train.close();test.close()


def test_data_completeness_is_enforced(tmp_path):
    write_fixture(tmp_path)
    path=tmp_path/'sector_signals_historical.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows[:-1])
    with pytest.raises(ValueError,match='Incomplete sectors'):
        read_signals('historical',tmp_path)
