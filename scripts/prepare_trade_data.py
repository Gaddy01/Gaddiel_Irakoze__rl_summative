
"""Prepare historical Rwanda HS4 import indicators for the RL project.

Input: 2015-2024 annual records downloaded from UN Comtrade.
Training signals: 2016-2023 (2015 supplies the first growth baseline).
Held-out signals: 2024 (excluded when fitting the demand scaler).

This DOES NOT alter or feed data into the RL environment automatically.
"""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scripts.fetch_comtrade import SECTORS

SOURCE = "UN Comtrade public preview API (Rwanda imports from World)"
FIELDS = (
    "sector",
    "hs4_code",
    "signal_year",
    "import_value_usd",
    "import_growth_yoy",
    "demand_score",
    "growth_signal",
    "demand_outside_training_range",
    "source_is_reported",
    "source",
)


def compute_signals(
    rows: list[dict],
    train_year: int = 2023,
    test_year: int = 2024,
    start_year: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return historical and held-out observations.

    Normalization uses historical training years only.
    start_year=None allows compatibility with existing tests.
    """
    if test_year != train_year + 1:
        raise ValueError(
            "The holdout year must immediately follow the training end year"
        )

    if not rows:
        raise ValueError("Input CSV contains no trade records")

    imports: dict[str, dict[int, float]] = {
        sector: {} for sector in SECTORS
    }
    reporting: dict[str, dict[int, str]] = {
        sector: {} for sector in SECTORS
    }
    observed_years = []

    # Validate and index the original UN Comtrade records.
    for row in rows:
        sector = str(row.get("sector", ""))

        if sector not in SECTORS:
            raise ValueError(f"Unknown sector: {sector!r}")

        hs4 = SECTORS[sector][0]

        if str(row.get("hs4_code", "")) != hs4:
            raise ValueError(
                f"Incorrect HS4 code for {sector}: "
                f"{row.get('hs4_code')}"
            )

        if (
            str(row.get("partner_code", "")) != "0"
            or str(row.get("flow_code", "")) != "M"
        ):
            raise ValueError(
                f"Expected World-partner import record: {row}"
            )

        reporter = str(row.get("reporter_code", ""))

        if reporter and reporter != "646":
            raise ValueError(
                f"Expected Rwanda reporter 646, got {reporter}"
            )

        try:
            year = int(row["year"])
            value = float(row["import_value_usd"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid year or import value: {row}"
            ) from exc

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Import value must be finite and nonnegative: "
                f"{sector}/{year}"
            )

        if year in imports[sector]:
            raise ValueError(
                f"Duplicate sector-year: {sector}/{year}"
            )

        observed_years.append(year)
        imports[sector][year] = value
        reporting[sector][year] = str(
            row.get("is_reported", "unknown")
        )

    first_year = (
        min(observed_years)
        if start_year is None
        else start_year
    )

    if first_year >= train_year:
        raise ValueError(
            "At least one earlier baseline year is needed "
            "for year-on-year growth"
        )

    # Check completeness for all sectors and all years.
    required_years = range(first_year, test_year + 1)

    for sector in SECTORS:
        missing = [
            year
            for year in required_years
            if year not in imports[sector]
        ]

        if missing:
            raise ValueError(
                f"Missing records for {sector}: {missing}"
            )

    # Historical signals begin one year after the baseline.
    historical_years = range(
        first_year + 1,
        train_year + 1,
    )

    # Fit the demand normalization using ONLY historical years.
    # The held-out 2024 data must not influence these bounds.
    reference_logs = [
        math.log1p(imports[sector][year])
        for year in historical_years
        for sector in SECTORS
    ]

    lo = min(reference_logs)
    hi = max(reference_logs)

    def build_row(sector: str, year: int) -> dict:
        current = imports[sector][year]
        previous = imports[sector][year - 1]

        if previous == 0:
            raise ValueError(
                f"Cannot calculate YoY growth: "
                f"zero prior imports for {sector}/{year}"
            )

        # Annual growth in imported value.
        growth = (current - previous) / previous

        # Log-scale normalization of import market size.
        log_value = math.log1p(current)

        if hi == lo:
            raw_score = 0.5
        else:
            raw_score = (log_value - lo) / (hi - lo)

        demand_score = min(
            1.0,
            max(0.0, raw_score),
        )

        # Map growth to a bounded indicator.
        growth_score = min(
            1.0,
            max(0.0, 0.5 + 0.25 * growth),
        )

        return {
            "sector": sector,
            "hs4_code": SECTORS[sector][0],
            "signal_year": year,
            "import_value_usd": round(current, 2),
            "import_growth_yoy": round(growth, 6),
            "demand_score": round(demand_score, 6),
            "growth_signal": round(growth_score, 6),
            "demand_outside_training_range": bool(
                raw_score < 0 or raw_score > 1
            ),
            "source_is_reported": reporting[sector][year],
            "source": SOURCE,
        }

    historical = [
        build_row(sector, year)
        for year in historical_years
        for sector in SECTORS
    ]

    holdout = [
        build_row(sector, test_year)
        for sector in SECTORS
    ]

    return historical, holdout


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save processed indicators with a consistent schema."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare historical Rwanda HS4 import signals"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/rwanda_hs4_imports.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
    )

    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2023,
    )

    parser.add_argument(
        "--holdout-year",
        type=int,
        default=2024,
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"Input file not found: {args.input}. "
            "Run fetch_comtrade first"
        )

    # Read the downloaded CSV without modifying it.
    with args.input.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    # Generate historical and held-out indicators.
    historical, holdout = compute_signals(
        rows,
        train_year=args.train_end_year,
        test_year=args.holdout_year,
        start_year=args.start_year,
    )

    # Save all historical observations.
    write_csv(
        args.output_dir / "sector_signals_historical.csv",
        historical,
    )

    # Save 2024 separately for evaluation.
    write_csv(
        args.output_dir
        / f"sector_signals_{args.holdout_year}_holdout.csv",
        holdout,
    )

    # Document data provenance and exact transformations.
    reference_values = [
        math.log1p(float(row["import_value_usd"]))
        for row in historical
    ]

    provenance = {
        "source": (
            "UN Comtrade, United Nations Statistics Division "
            "— public preview API"
        ),
        "reporter": "Rwanda (646)",
        "partner": "World (0)",
        "flow": "Imports (M)",
        "classification": (
            "HS; HS4 codes as specified in raw source rows"
        ),
        "hs4_sector_mapping": {
            sector: details[0]
            for sector, details in SECTORS.items()
        },
        "raw_file": str(args.input.as_posix()),
        "raw_file_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "raw_row_count": len(rows),
        "baseline_year_for_growth": args.start_year,
        "historical_signal_years": [
            args.start_year + 1,
            args.train_end_year,
        ],
        "held_out_year": args.holdout_year,
        "historical_row_count": len(historical),
        "held_out_row_count": len(holdout),
        "normalization": {
            "demand_score": (
                "Clip((ln(1+value)-min_train_log)/"
                "(max_train_log-min_train_log), 0, 1)"
            ),
            "fitted_on": (
                "All four HS4 sector observations from "
                f"{args.start_year + 1} through "
                f"{args.train_end_year}; excludes held-out year"
            ),
            "minimum_log_imports": min(reference_values),
            "maximum_log_imports": max(reference_values),
            "constant_range_fallback": 0.5,
            "growth_signal": (
                "Clip(0.5 + 0.25*((value_t-value_t-1)"
                "/value_t-1), 0, 1)"
            ),
        },
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "limitations": [
            (
                "HS4 import value is a proxy for import "
                "market size, not domestic investment viability."
            ),
            (
                "World-partner totals do not measure "
                "supplier concentration."
            ),
            (
                "No observed employment or firm-level "
                "financial profitability is inferred."
            ),
            (
                "The processed data is not yet wired "
                "into the current synthetic RL environment."
            ),
            (
                "Check classification consistency, "
                "is_reported flags, and revisions "
                "before publication."
            ),
        ],
    }

    output_path = args.output_dir / "provenance.json"

    output_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {output_path}")

    print(
        f"Historical: {len(historical)} rows "
        f"({args.start_year + 1}–{args.train_end_year})"
    )

    print(
        f"Holdout: {len(holdout)} rows "
        f"({args.holdout_year})"
    )

    print(
        "NOTE: RL environment still uses synthetic default "
        "indicators; integration is a separate step."
    )


if __name__ == "__main__":
    main()
