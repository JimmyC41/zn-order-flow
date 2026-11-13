"""Download trades with quotes from Databento, one file per month. Months already on disk are skipped, so a re-run never pays twice."""
import sys
from pathlib import Path

import databento as db
import pandas as pd

dataset, symbol, start, end, out = sys.argv[1:6]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
client = db.Historical()
edges = pd.date_range(start, end, freq="MS").union(pd.DatetimeIndex([start, end])).sort_values()
for a, b in zip(edges[:-1], edges[1:]):
    target = out / f"{a:%Y%m%d}_{b:%Y%m%d}.parquet"
    if target.exists():
        continue
    part = out / (target.name + ".part")
    client.timeseries.get_range(
        dataset=dataset, symbols=[symbol], stype_in="continuous", schema="tbbo",
        start=f"{a:%Y-%m-%d}", end=f"{b:%Y-%m-%d}",
    ).to_parquet(part)
    part.rename(target)
    print(target.name, flush=True)
