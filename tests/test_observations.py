
from dataclasses import replace

import numpy as np
import pytest

from environment.observations import (
    OBSERVATION_NAMES,
    OBSERVATION_SPACE,
    ProjectStage,
    encode_observation,
    make_demo_snapshot,
)


def test_observation_shape():
    observation = encode_observation(make_demo_snapshot())

    assert observation.shape == (53,)
    assert len(OBSERVATION_NAMES) == 53


def test_observation_dtype():
    observation = encode_observation(make_demo_snapshot())

    assert observation.dtype == np.float32


def test_observation_inside_space():
    observation = encode_observation(make_demo_snapshot())

    assert OBSERVATION_SPACE.contains(observation)


def test_initial_financial_state():
    observation = encode_observation(make_demo_snapshot())

    assert observation[0] == pytest.approx(0.0)
    assert observation[1] == pytest.approx(0.5)
    assert observation[2] == pytest.approx(0.0)


def test_initial_sector_stage():
    observation = encode_observation(make_demo_snapshot())

    stage_names = [
        "agro_processing.stage_unstudied",
        "agro_processing.stage_studied",
        "agro_processing.stage_pilot",
        "agro_processing.stage_operating",
        "agro_processing.stage_exited",
    ]

    values = [
        observation[OBSERVATION_NAMES.index(name)]
        for name in stage_names
    ]

    assert values == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_time_changes_observation():
    snapshot = make_demo_snapshot()

    later_snapshot = replace(snapshot, month=12)

    observation = encode_observation(later_snapshot)

    assert observation[0] == pytest.approx(0.5)


def test_study_information_changes_observation():
    snapshot = make_demo_snapshot()

    updated_sectors = dict(snapshot.sectors)

    updated_sectors["agro_processing"] = replace(
        updated_sectors["agro_processing"],
        feasibility_estimate=0.72,
        uncertainty=0.30,
        stage=ProjectStage.STUDIED,
    )

    updated_snapshot = replace(
        snapshot,
        sectors=updated_sectors,
    )

    before = encode_observation(snapshot)
    after = encode_observation(updated_snapshot)

    assert not np.array_equal(before, after)

    uncertainty_index = OBSERVATION_NAMES.index(
        "agro_processing.uncertainty"
    )

    assert after[uncertainty_index] == pytest.approx(0.30)


def test_invalid_score_rejected():
    snapshot = make_demo_snapshot()

    updated_sectors = dict(snapshot.sectors)

    updated_sectors["textiles"] = replace(
        updated_sectors["textiles"],
        demand_score=1.5,
    )

    invalid_snapshot = replace(
        snapshot,
        sectors=updated_sectors,
    )

    with pytest.raises(ValueError):
        encode_observation(invalid_snapshot)


def test_missing_sector_rejected():
    snapshot = make_demo_snapshot()

    incomplete_sectors = dict(snapshot.sectors)
    incomplete_sectors.pop("pharmaceuticals")

    invalid_snapshot = replace(
        snapshot,
        sectors=incomplete_sectors,
    )

    with pytest.raises(ValueError):
        encode_observation(invalid_snapshot)
