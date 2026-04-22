from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


SERIES_CONFIG = {
    "CPI_YOY": {
        "series_id": "CPIAUCSL",
        "description": "Consumer Price Index for All Urban Consumers: All Items",
    },
    "UNRATE": {
        "series_id": "UNRATE",
        "description": "Unemployment Rate",
    },
    "FEDFUNDS": {
        "series_id": "FEDFUNDS",
        "description": "Effective Federal Funds Rate",
    },
    "GS10": {
        "series_id": "GS10",
        "description": "10-Year Treasury Constant Maturity Rate",
    },
}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "benchmarks" / "data"
OUT_PATH = OUT_DIR / "fred_macro_context.json"


@dataclass
class FredObservation:
    date: str
    value: Optional[float]


def fetch_fred_series(series_id: str, api_key: str) -> List[FredObservation]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    out: List[FredObservation] = []
    for row in payload.get("observations", []):
        raw_val = row.get("value")
        val: Optional[float]
        if raw_val in (None, ".", ""):
            val = None
        else:
            try:
                val = float(raw_val)
            except ValueError:
                val = None
        out.append(FredObservation(date=row["date"], value=val))
    return out


def latest_valid(obs: List[FredObservation]) -> FredObservation:
    for row in reversed(obs):
        if row.value is not None:
            return row
    raise ValueError("No valid observations found.")


def nearest_on_or_before(obs: List[FredObservation], target_date: str) -> FredObservation:
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    candidates = []
    for row in obs:
        row_date = datetime.strptime(row.date, "%Y-%m-%d").date()
        if row_date <= target and row.value is not None:
            candidates.append(row)
    if not candidates:
        raise ValueError(f"No valid observation on or before {target_date}.")
    return candidates[-1]


def add_months_approx(date_str: str, months_back: int) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    month = dt.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    day = min(dt.day, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def compute_cpi_yoy(cpi_obs: List[FredObservation]) -> Dict[str, FredObservation]:
    value_by_date = {row.date: row.value for row in cpi_obs if row.value is not None}
    dates = [row.date for row in cpi_obs]

    out: Dict[str, FredObservation] = {}
    for i, date_str in enumerate(dates):
        current = value_by_date.get(date_str)
        if current is None or i < 12:
            continue
        prev_date = dates[i - 12]
        prev = value_by_date.get(prev_date)
        if prev is None or prev == 0:
            continue
        yoy = ((current / prev) - 1.0) * 100.0
        out[date_str] = FredObservation(date=date_str, value=round(yoy, 3))
    return out


def fmt_snapshot(snapshot_date: str, values: Dict[str, float]) -> str:
    return (
        f"Date: {snapshot_date}\n"
        f"CPI_YOY: {values['CPI_YOY']:.2f}\n"
        f"UNRATE: {values['UNRATE']:.2f}\n"
        f"FEDFUNDS: {values['FEDFUNDS']:.2f}\n"
        f"GS10: {values['GS10']:.2f}"
    )


def fmt_comparison(date_a: str, vals_a: Dict[str, float], date_b: str, vals_b: Dict[str, float]) -> str:
    return (
        f"Date A: {date_a}\n"
        f"CPI_YOY: {vals_a['CPI_YOY']:.2f}\n"
        f"UNRATE: {vals_a['UNRATE']:.2f}\n"
        f"FEDFUNDS: {vals_a['FEDFUNDS']:.2f}\n"
        f"GS10: {vals_a['GS10']:.2f}\n\n"
        f"Date B: {date_b}\n"
        f"CPI_YOY: {vals_b['CPI_YOY']:.2f}\n"
        f"UNRATE: {vals_b['UNRATE']:.2f}\n"
        f"FEDFUNDS: {vals_b['FEDFUNDS']:.2f}\n"
        f"GS10: {vals_b['GS10']:.2f}"
    )
    

def build_snapshot_values(
    snapshot_date: str,
    cpi_yoy_series: Dict[str, FredObservation],
    raw_series: Dict[str, List[FredObservation]],
) -> Dict[str, float]:
    values = {
        "CPI_YOY": nearest_on_or_before(list(cpi_yoy_series.values()), snapshot_date).value,
        "UNRATE": nearest_on_or_before(raw_series["UNRATE"], snapshot_date).value,
        "FEDFUNDS": nearest_on_or_before(raw_series["FEDFUNDS"], snapshot_date).value,
        "GS10": nearest_on_or_before(raw_series["GS10"], snapshot_date).value,
    }
    if any(v is None for v in values.values()):
        raise RuntimeError(f"Missing values in snapshot construction for {snapshot_date}.")
    return values  # type: ignore[return-value]


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set.")

    raw_series = {
        key: fetch_fred_series(cfg["series_id"], api_key)
        for key, cfg in SERIES_CONFIG.items()
    }

    cpi_yoy_series = compute_cpi_yoy(raw_series["CPI_YOY"])

    latest_date = latest_valid(raw_series["UNRATE"]).date
    prior_date_12m = add_months_approx(latest_date, 12)
    prior_date_24m = add_months_approx(latest_date, 24)
    prior_date_6m = add_months_approx(latest_date, 6)

    latest_values = build_snapshot_values(latest_date, cpi_yoy_series, raw_series)
    prior_values_12m = build_snapshot_values(prior_date_12m, cpi_yoy_series, raw_series)
    prior_values_24m = build_snapshot_values(prior_date_24m, cpi_yoy_series, raw_series)
    prior_values_6m = build_snapshot_values(prior_date_6m, cpi_yoy_series, raw_series)

    latest_text = fmt_snapshot(latest_date, latest_values)
    snapshot_alt_1_text = fmt_snapshot(prior_date_24m, prior_values_24m)

    comparison_12m_text = fmt_comparison(
        prior_date_12m, prior_values_12m, latest_date, latest_values
    )
    comparison_24m_text = fmt_comparison(
        prior_date_24m, prior_values_24m, latest_date, latest_values
    )
    comparison_6m_text = fmt_comparison(
        prior_date_6m, prior_values_6m, latest_date, latest_values
    )
    
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "latest_date": latest_date,
        "prior_date_6m": prior_date_6m,
        "prior_date_12m": prior_date_12m,
        "prior_date_24m": prior_date_24m,
        "series": SERIES_CONFIG,
        "contexts": {
            "latest_snapshot": latest_text,
            "snapshot_alt_1": snapshot_alt_1_text,
            "comparison_6m": comparison_6m_text,
            "comparison_12m": comparison_12m_text,
            "comparison_24m": comparison_24m_text,
        },
        "latest_values": latest_values,
        "prior_values_6m": prior_values_6m,
        "prior_values_12m": prior_values_12m,
        "prior_values_24m": prior_values_24m,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()