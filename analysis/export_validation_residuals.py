"""Export ordered time-rescaled gaps from the frozen ZN arrival models."""

import json
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import flow
from run import BASE, CFG, FILES, KINDS, OUT, REFINED, ROOT


COLUMNS = {
    BASE: "rescaled_gap_constant_rate",
    "time of day": "rescaled_gap_time_of_day",
    "time of day + news": "rescaled_gap_time_of_day_news",
    REFINED: "rescaled_gap_self_exciting",
}


def validation_files(start, stop):
    """Locate the monthly validation files using the naming convention in fetch.py."""
    edges = pd.date_range(start, stop, freq="MS").union(pd.DatetimeIndex([start, stop]))
    available = sorted({path for pattern in FILES for path in glob(pattern)})
    for first, following in zip(edges[:-1], edges[1:]):
        name = f"{first:%Y%m%d}_{following:%Y%m%d}.parquet"
        matches = [path for path in available if Path(path).name == name]
        if not matches:
            raise FileNotFoundError(f"Missing validation file {name}. See the README for the validation download command.")
        if len(matches) > 1:
            raise ValueError(f"Multiple files match {name}. Check 'files' in analysis/config.json.")
        yield first, following - pd.Timedelta(days=1), matches[0]


def main():
    start = pd.Timestamp(CFG["fit_end"]) + pd.Timedelta(days=1)
    stop = pd.Timestamp(CFG["validate_end"]) + pd.Timedelta(days=1)
    model = json.loads((OUT / "model.json").read_text())
    arrivals = {column: flow.Arrivals(**model["arrivals"][name]) for name, column in COLUMNS.items()}
    months = list(validation_files(start, stop))
    skip = flow.skipped_days([path for _, _, path in months])
    sessions = []
    for first, last, path in months:
        trades, _ = flow.load_trades([path], first.date(), last.date(), skip)
        orders = flow.to_orders(trades, model["merge_us"] * 1000)
        for day, group in orders.groupby("day", sort=True):
            kind = KINDS.get(day)
            times = group.t.to_numpy()
            rows = {
                "session_date_ny": day.isoformat(),
                "session_kind": kind or "none",
                "order_in_session": np.arange(2, len(group) + 1, dtype=np.int32),
            }
            for column, arrival in arrivals.items():
                rows[column] = arrival.day(times, kind)[1]
            sessions.append(pd.DataFrame(rows))

    residuals = pd.concat(sessions, ignore_index=True)
    target = ROOT / "data/validation_arrival_residuals.parquet"
    with TemporaryDirectory(dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        residuals.to_parquet(temporary, index=False)
        temporary.replace(target)
    print(f"Wrote {len(residuals):,} gaps across {residuals.session_date_ny.nunique()} sessions to {target}")


if __name__ == "__main__":
    main()
