"""Fetch *observed* Rwanda HS4 imports from the official UN Comtrade public preview API.

Fails closed on unavailable/missing rows: never substitutes invented economic values.
This is a data acquisition tool; training still defaults to frozen synthetic Env V2.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
SECTORS = {
    "agro_processing": ("2009", "Fruit/nut and vegetable juices"),
    "textiles": ("6109", "T-shirts, singlets and other vests, knitted or crocheted"),
    "construction_materials": ("2523", "Hydraulic cements"),
    "pharmaceuticals": ("3004", "Medicaments put up in measured doses/retail packs"),
}
FIELDNAMES = ("year", "reporter_code", "partner_code", "flow_code", "hs4_code", "sector", "description", "import_value_usd", "classification", "is_reported", "source_url")


def request_import(year: int, code: str, timeout: int = 35) -> dict:
    params = {
    "period": str(year),
    "reporterCode": "646",
    "partnerCode": "0",
    "flowCode": "M",
    "cmdCode": code,
    "maxRecords": "500",
    }
    # cmdCode selects one HS4 heading; leave aggregation at API defaults.
    url = API + "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "ML-Techniq-II-research/1.0", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as reply:
                payload = json.loads(reply.read().decode("utf-8"))
            break
        except (URLError, HTTPError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"UN Comtrade unavailable: {url} : {exc}") from exc
            time.sleep(2 * (attempt + 1))
    if payload.get("error"):
        raise ValueError(f"UN Comtrade API error: {payload['error']}")
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected response format: {url}")
    valid = [r for r in rows if (
        str(r.get("cmdCode")) == code
        and int(r.get("refYear", r.get("period", 0))) == year
        and str(r.get("reporterCode")) == "646"
        and str(r.get("partnerCode")) == "0"
        and str(r.get("flowCode")) == "M"
        and str(r.get("partner2Code") or "0") == "0"
    )]
    # A world aggregate must have exactly one total row, not a mix of mode/customs rows.
    exact = [r for r in valid if str(r.get("customsCode") or "C00") in ("C00", "0") and str(r.get("motCode") or "0") in ("0", "TOTAL")]
    if len(exact) != 1:
        raise ValueError(f"Expected one HS4 World import record for {code}/{year}; received {len(exact)}. Inspect API response: {url}")
    r = exact[0]
    v = r.get("primaryValue")
    if v is None or float(v) < 0:
        raise ValueError(f"Missing/invalid primaryValue for {code}/{year}: {url}")
    return {"year": year, "reporter_code": 646, "partner_code": 0, "flow_code": "M", "hs4_code": code,
            "import_value_usd": float(v), "classification": str(r.get("classificationCode", "HS")),
            "is_reported": r.get("isReported", ""), "source_url": url}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=Path("data/raw/rwanda_hs4_imports.csv"))
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()
    if args.start_year < 2000 or args.end_year > 2025 or args.start_year > args.end_year:
        parser.error("Check requested annual range (2000–2025) before downloading.")
    result = []
    for sector, (code, description) in SECTORS.items():
        for year in range(args.start_year, args.end_year + 1):
            row = request_import(year, code)
            row.update(sector=sector, description=description)
            result.append(row)
            print(f"{year} HS {code} {sector}: USD {row['import_value_usd']:,.0f}")
            time.sleep(args.pause)
    if len(result) != len(SECTORS) * (args.end_year - args.start_year + 1):
        raise RuntimeError("Incomplete data. No partial file written.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader(); writer.writerows(result)
    print("Saved", args.output)
    print("Attribution: UN Comtrade, United Nations Statistics Division. Public preview API; verify reporting coverage.")


if __name__ == "__main__":
    main()
