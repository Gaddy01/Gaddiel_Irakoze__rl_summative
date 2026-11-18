"""Pure offline validation of the data preprocessing rules (no network)."""
import pytest
from scripts.fetch_comtrade import SECTORS
from scripts.prepare_trade_data import compute_signals


def sample_rows():
    out = []
    for i, (sector, (code, _)) in enumerate(SECTORS.items()):
        for year in (2022, 2023, 2024):
            out.append(dict(sector=sector, hs4_code=code, flow_code="M", partner_code="0",
                            year=str(year), import_value_usd=str((i+1)*1000+(year-2022)*200)))
    return out


def test_splits_are_not_equal_and_codes_preserved():
    train, test=compute_signals(sample_rows())
    assert len(train)==len(test)==4
    assert all(x["signal_year"]==2023 for x in train)
    assert all(x["signal_year"]==2024 for x in test)
    assert all(0 <= x["demand_score"] <= 1 for x in train+test)


def test_missing_trade_rows_fail_closed():
    with pytest.raises(ValueError,match="Missing records"):
        compute_signals(sample_rows()[:-1])


def test_duplicate_trade_rows_fail_closed():
    rows=sample_rows()
    with pytest.raises(ValueError,match="Duplicate"):
        compute_signals(rows+[rows[0]])
