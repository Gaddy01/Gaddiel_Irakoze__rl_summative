"""Backend/API and checkpoint-replay verification for the Three.js control room."""
from http.server import ThreadingHTTPServer
import csv
import json
import threading
from urllib.request import Request, urlopen
from pathlib import Path

import numpy as np
import pytest

from visualization.session import SimulationController
from visualization.server import make_handler


class FakeEnv:
    """Test double, never used by the actual running application."""
    def __init__(self, split):
        self.split, self.month, self.signal_year = split, 0, (2024 if split=="holdout" else 2023)
        self.total = 100.0

    def reset(self, *, seed):
        self.month, self.total = 0, 100.0
        return np.zeros(53, np.float32), {"action_mask": np.ones(17,dtype=bool)}

    def public_state(self):
        return {
            "month":self.month,"cash":100.0,"net_worth":self.total,
            "employment_index":.1,"import_substitution_index":.2,"risk_exposure_index":.05,
            "sectors":{k:{"stage":"unstudied","invested_capital":0.0,"asset_value":0.0,
                         "progress":0.0,"demand_score":.2,"feasibility_estimate":.5,"uncertainty":.3}
                       for k in ("agro_processing","textiles","construction_materials","pharmaceuticals")}
        }

    def step(self, action):
        self.month += 1
        reward=.01 if action==0 else .02
        self.total += reward
        return (np.zeros(53,np.float32),reward,self.month==24,False,
                {"action_mask":np.ones(17,dtype=bool),"action_valid":True,
                 "reward_breakdown":{"total":reward}})

    def close(self): pass


def controller():
    return SimulationController(env_factory=FakeEnv,policy=lambda obs,mask:0)


def test_reset_and_public_state_have_no_hidden_feasibility():
    c=controller()
    try:
        state=c.snapshot()
        assert state["meta"]["year"]==2024 and state["meta"]["month"]==0
        assert len(state["action_catalog"])==17
        assert len(state["policies"]["ppo"]["portfolio"]["sectors"])==4
        assert state["policies"]["ppo"]["portfolio"]==state["policies"]["rule_based"]["portfolio"]
        assert "true_feasibility" not in json.dumps(state)
    finally:c.close()


def test_24_step_limit_and_reset():
    c=controller()
    try:
        for index in range(24):
            state=c.step()
            assert state["meta"]["month"]==index+1
            assert len(state["policies"]["ppo"]["history"])==index+2
        assert state["meta"]["done"] and len(state["events"])==48
        assert c.step()["meta"]["month"]==24
        restored=c.reset(seed=60007,split="historical")
        assert restored["meta"]["year"]==2023 and restored["meta"]["month"]==0
    finally:c.close()


@pytest.mark.parametrize("seed,split",[(-1,"holdout"),(True,"holdout"),(2_000_000_001,"holdout"),(60000,"future")])
def test_invalid_reset_rejected(seed,split):
    c=controller()
    try:
        with pytest.raises(ValueError):c.reset(seed=seed,split=split)
    finally:c.close()


def test_http_routes_and_api():
    c=controller()
    server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(c))
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    url="http://127.0.0.1:"+str(server.server_port)
    try:
        for path in ("/","/app.js","/world.js","/styles.css","/favicon.svg"):
            with urlopen(url+path,timeout=3) as response:
                assert response.status==200 and response.read()
        with urlopen(url+"/api/health",timeout=3) as response:
            assert json.load(response)["model"]=="ppo_03"
        step=Request(url+"/api/step",data=b"{}",headers={"Content-Type":"application/json"},method="POST")
        with urlopen(step,timeout=3) as response:
            assert json.load(response)["meta"]["month"]==1
        reset=Request(url+"/api/reset",data=b'{"seed":1,"split":"historical"}',headers={"Content-Type":"application/json"},method="POST")
        with urlopen(reset,timeout=3) as response:
            state=json.load(response)
            assert state["meta"]["month"]==0 and state["meta"]["split"]=="historical"
    finally:
        server.shutdown();server.server_close();c.close();worker.join(timeout=3)


def test_real_ppo_checkpoint_replays_saved_2024_seed60000():
    """Verify the actual TorchRL actor against the preexisting holdout evidence."""
    reference=Path("analysis_outputs/tables/holdout_2024_per_seed_replayed.csv")
    checkpoint=Path("models/ppo/ppo_03.pt")
    if not reference.is_file() or not checkpoint.is_file():
        pytest.skip("Real checkpoint and reference data not available")
    with reference.open(newline="",encoding="utf-8-sig") as file:
        expected=[row for row in csv.DictReader(file) if row["policy"]=="ppo" and int(row["seed"])==60000]
    assert len(expected)==1
    controller=SimulationController(checkpoint=checkpoint,seed=60000,split="holdout")
    try:
        for _ in range(24):controller.step()
        actual=controller.snapshot()["policies"]["ppo"]
        assert actual["total_reward"]==pytest.approx(float(expected[0]["reward"]),abs=1e-5)
        assert actual["portfolio"]["net_worth"]==pytest.approx(float(expected[0]["net_worth"]),abs=1e-3)
        assert controller.snapshot()["meta"]["done"]
    finally:controller.close()
