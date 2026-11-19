"""Export diagnostic inputs from the existing frozen ZN project. Never fits or tests.

Read the four monthly validation files from the project data directory.
The export preserves unavailable horizons instead of the upstream global dropna.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from glob import glob
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd

REFINED = "time of day + news + self-exciting"


# Earlier verified exports lack per-file digests in their manifests. Keep the
# digests of those artifacts so existing results can still be checked.
LEGACY_SHARED_INPUTS = {
    "frozen_config.json": "12f45a8cc5f68dce63820182230c2207840f7ab7c52a9056f5726897ab9e01a8",
    "frozen_model.json": "0614dbf3c9269d556578673975e113f8dded3b6bf4e610239b7cda05d1b7e875",
    "news_releases.csv": "c23e1f20b3fcdfcbd805b0fb441d4f4245ab7df36fc080878f1c766cd3e8e311"
}
LEGACY_EXPORTS = {
    "1c7d22d400fb226abba22fc842be2f56dc967af6171915bd3f125df9b33fc986": {
        "validation_events.csv.gz": "f7d6a741545b19fb65f0f5d510bedd6a6a929eee6a73a822baa93f1816c32118",
        "validation_orders.csv.gz": "1dc936af6642fd95d6a2e14bde6795fcad5d5f1cc4dc08c6b21ca9edcc722610",
        "validation_residuals.csv.gz": "abd746b21b70b838c87a1991c9be0d7be087f85c2129de373412e48613c4988e"
    },
    "fe741af7e1c6365160a070a58fd8dafbb92215bdb50b904b08723374000a1305": {
        "validation_events.csv.gz": "8fbbb849f485e1ff548b75ba683b71648df964424ef4f2433a94cf1cdf6aa3f5",
        "validation_orders.csv.gz": "23f45b3e43320e85570fc59cb886e239b9f1e7b3a437f9ef139b737d91cf7f6f",
        "validation_residuals.csv.gz": "0a474d9461c2c8d819d2a677ada01af76bbd6be9bf302ab79356152ef3a156fd"
    }
}
EXPORT_FILES = (
    "frozen_config.json", "frozen_model.json", "news_releases.csv",
    "validation_events.csv.gz", "validation_orders.csv.gz", "validation_residuals.csv.gz",
)
SOURCE_DIGEST = "734d44f2d8cee13569433ffbccfc0e3a7b2d118d0af44790fdb8a3f9ee755d03"
RAW_FILES_DIGEST = "0c08d35ae057e804c23dec026ab46d17d2fe7d698711155578dee2068cf00b9e"
CALENDAR_DIGEST = "04e2fcefdf89472a886b2aa9ca710b2224be3366e0f7d4ecdfbe391da42143e0"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_text_digest(path):
    # Git may convert LF to CRLF on checkout.
    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_digest(document):
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_export_manifest(manifest):
    if "schema_version" not in manifest:
        if document_digest(manifest) not in LEGACY_EXPORTS:
            raise ValueError("Unrecognized legacy manifest. Regenerate the validation export.")
        return
    if manifest["schema_version"] != 2 or manifest.get("status") != "REAL_VALIDATION_EXPORT":
        raise ValueError("Expected a completed version 2 validation export")
    days = manifest.get("days", [])
    if (len(days) != 80 or not all(isinstance(d, str) for d in days)
            or hashlib.sha256("\n".join(days).encode()).hexdigest() != CALENDAR_DIGEST
            or manifest.get("n_orders") != 2_414_025 or manifest.get("n_residuals") != 2_413_945):
        raise ValueError("Unexpected validation calendar, order count or gap count")
    if document_digest(manifest.get("source_hashes")) != SOURCE_DIGEST:
        raise ValueError("The source model, configuration, calendar or order reconstruction differs from the frozen run")
    if document_digest(manifest.get("raw_sha256")) != RAW_FILES_DIGEST:
        raise ValueError("Validation inputs differ from the four reference Parquet files")
    hashes = manifest.get("output_sha256", {})
    if set(hashes) != set(EXPORT_FILES) or any(
            not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef")
            for value in hashes.values()):
        raise ValueError("Missing or invalid export file digests")


def verify_export_input(manifest, filename, digest):
    validate_export_manifest(manifest)
    hashes = manifest.get("output_sha256")
    if hashes is None:
        hashes = {**LEGACY_SHARED_INPUTS, **LEGACY_EXPORTS[document_digest(manifest)]}
    if filename not in hashes or hashes[filename] != digest:
        raise ValueError(f"{filename} does not match its validation export manifest")


@contextmanager
def new_output_directory(destination):
    """Make results visible only after every output has been written."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        yield stage
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        stage.rename(destination)


def day_frame(trades, orders, arrivals, kind, cfg):
    """Match upstream events exactly, then add VWAP and horizon provenance."""
    if len(trades) == 0 or len(orders) == 0:
        raise ValueError("Empty session")
    if trades.day.nunique() != 1 or orders.day.nunique() != 1 or trades.day.iloc[0] != orders.day.iloc[0]:
        raise ValueError("Pass one session at a time")
    if not trades.ts.is_monotonic_increasing:
        raise ValueError("Trades must be sorted by exchange timestamp")
    # Match upstream exactly. At large epoch-nanosecond values, pandas groupby
    # diff and Series.diff can round differently near the 100us boundary.
    # Preserve the frozen source's inferred orders; do not silently redefine them.
    gap = trades.groupby('day').ts.diff()
    new = gap.isna() | (gap > cfg['merge_us'] * 1000) | (trades.sgn != trades.sgn.shift())
    if not np.isfinite(trades['size']).all() or (trades['size'] <= 0).any():
        raise ValueError("Trade sizes must be finite and positive")
    grouped = trades.assign(notional=trades.px * trades['size']).groupby(new.cumsum())
    aggregates = grouped.agg(ts=('ts', 'first'), end=('ts', 'last'), volume=('size', 'sum'), notional=('notional', 'sum'))
    if len(aggregates) != len(orders) or not np.array_equal(aggregates.ts.to_numpy(), orders.ts.to_numpy()) or not np.allclose(aggregates.volume, orders['size'], rtol=0, atol=0):
        raise ValueError("Source aggregation differs from expected merge rule; inspect before exporting")
    if np.isinf(trades.px).any():
        raise ValueError("Infinite trade price")
    # A missing fill price makes its order VWAP unavailable. The first-fill
    # measurement can still be valid on the wider sample.
    complete_prices = grouped.px.count() == grouped.size()
    vwap = (aggregates.notional / aggregates.volume).where(complete_prices)
    t, ts = orders.t.to_numpy(), orders.ts.to_numpy(dtype=np.int64)
    bucket = np.minimum((t // cfg['bucket_seconds']).astype(int), len(arrivals.shape) - 1)
    base = arrivals.base(kind)[bucket]
    if np.any(base <= 0):
        raise ValueError("Nonpositive baseline intensity")
    sgn = orders.sgn.to_numpy()
    r = pd.DataFrame({'day': str(orders.day.iloc[0]), 'cluster_id': np.arange(len(orders)),
                      't': t, 'ts_ns': ts, 'cluster_end_ns': aggregates.end.to_numpy(dtype=np.int64),
                      'side': sgn, 'size': orders['size'].to_numpy(), 'kind': kind or 'none',
                      'earned': sgn * (orders.px.to_numpy() - orders.mid.to_numpy()) / cfg['tick'],
                      'earned_vwap': sgn * (vwap.to_numpy() - orders.mid.to_numpy()) / cfg['tick']})
    r['cluster_duration_ms'] = (r.cluster_end_ns - r.ts_ns) / 1e6
    for ms in cfg['reaction_ms']:
        r[f'seen_{ms}'] = arrivals.rate_seen(t, kind, ms / 1000) / base
    trade_ts, trade_mid = trades.ts.to_numpy(dtype=np.int64), trades.mid.to_numpy()
    for h in cfg['horizons_s']:
        target = ts + int(h * 1e9)
        pos = np.searchsorted(trade_ts, target, side='left')
        valid = pos < len(trade_ts)
        matched = pd.array([pd.NA] * len(r), dtype='Int64')
        matched[valid] = trade_ts[pos[valid]]
        future = np.full(len(r), np.nan)
        future[valid] = trade_mid[pos[valid]]
        lag = np.full(len(r), np.nan)
        lag[valid] = (trade_ts[pos[valid]] - target[valid]) / 1e6
        r[f'target_{h}_ns'] = target
        r[f'match_{h}_ns'] = matched
        r[f'lag_{h}_ms'] = lag
        r[f'back_{h}'] = sgn * (future - orders.mid.to_numpy()) / cfg['tick']
    r['news_window'] = ((r.kind == 'data release') & (r.t >= 600) & (r.t < 1800)) | ((r.kind == 'fed decision') & (r.t >= 20400))
    return r


def _run(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    cfg_path, model_path = repo/'analysis/config.json', repo/'output/model.json'
    cfg, model = json.loads(cfg_path.read_text()), json.loads(model_path.read_text())
    if cfg['fit_end'] != '2026-04-30' or cfg['validate_end'] != '2026-08-31':
        raise ValueError('This extension is restricted to the frozen May–August 2026 validation design')
    if model['merge_us'] != cfg['merge_us']:
        raise ValueError('Model and config merge windows differ')
    if cfg['reaction_ms'] != [1, 100, 1000] or cfg['horizons_s'] != [1, 5, 10, 30, 60]:
        raise ValueError('Unexpected reaction delays or horizons; review the export contract')
    news_path = repo/cfg['news_file']
    before = {p.relative_to(repo).as_posix(): source_text_digest(p) for p in [cfg_path, model_path, repo/'analysis/flow.py', news_path]}
    # Load the inspected data/model definitions, not run.py and not any fitting stage.
    spec = importlib.util.spec_from_file_location('zn_source_flow', repo/'analysis/flow.py')
    flow = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = flow
    spec.loader.exec_module(flow)
    flow.configure(cfg['session_open'], cfg['session_close'], cfg['bucket_seconds'])
    available = [Path(f) for pattern in cfg['files'] for f in glob(str(repo / pattern))]
    boundaries = pd.date_range('2026-05-01', '2026-09-01', freq='MS')
    files = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        name = f'{start:%Y%m%d}_{stop:%Y%m%d}.parquet'
        matches = [p for p in available if p.name == name]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one validation input named {name}, found {len(matches)}')
        files.append(str(matches[0]))
    if document_digest(before) != SOURCE_DIGEST:
        raise ValueError("Expected the frozen model, configuration, calendar and order reconstruction")
    raw_hashes = {Path(f).name: sha256(f) for f in files}
    if document_digest(raw_hashes) != RAW_FILES_DIGEST:
        raise ValueError("Validation inputs differ from the four reference Parquet files")
    skip = flow.skipped_days(files)
    news = pd.read_csv(repo/cfg['news_file']).dropna(subset=['date'])
    kinds = {pd.Timestamp(d).date(): ('fed decision' if r == 'FOMC' else 'data release') for d, r in zip(news.date, news.release)}
    m = model['arrivals'][REFINED]
    arrival = flow.Arrivals(**m)
    (output/'frozen_model.json').write_bytes(json.dumps(model, indent=2).encode())
    (output/'frozen_config.json').write_bytes(json.dumps(cfg, indent=2).encode())
    (output/'news_releases.csv').write_bytes(news_path.read_text(encoding='utf-8').encode())
    counts, days, n_orders, n_residuals = [], [], 0, 0
    with gzip.open(output/'validation_events.csv.gz', 'wt') as ef, gzip.open(output/'validation_orders.csv.gz', 'wt') as of, gzip.open(output/'validation_residuals.csv.gz', 'wt') as rf:
        for month in ['2026-05', '2026-06', '2026-07', '2026-08']:
            start = pd.Timestamp(month+'-01')
            end = start + pd.offsets.MonthEnd(1)
            trades, qa = flow.load_trades(files, start.date().isoformat(), end.date().isoformat(), skip)
            counts.append({'month': month, **qa})
            for day, tr in trades.groupby('day', sort=True):
                tr = tr.reset_index(drop=True)
                orders = flow.to_orders(tr, cfg['merge_us']*1000)
                frame = day_frame(tr, orders, arrival, kinds.get(day), cfg)
                frame.to_csv(ef, index=False, header=n_orders == 0)
                frame[['day', 't', 'ts_ns', 'cluster_end_ns', 'size', 'kind']].to_csv(of, index=False, header=n_orders == 0)
                _, z, _ = arrival.day(orders.t.to_numpy(), kinds.get(day))
                pd.DataFrame({'day': str(day), 'model': REFINED, 'residual_index': np.arange(len(z)), 'z': z}).to_csv(rf, index=False, header=n_residuals == 0)
                n_orders += len(frame)
                n_residuals += len(z)
                days.append(str(day))
                print(f'Exported {day}: {len(frame)} orders', flush=True)
    if n_orders == 0 or not days:
        raise ValueError('No validation events exported; directory is incomplete and must not be used')
    after = {name: source_text_digest(repo/name) for name in before}
    if before != after or raw_hashes != {Path(f).name: sha256(f) for f in files}:
        raise RuntimeError('Source files changed during export; do not use this export')
    manifest = {'schema_version': 2, 'raw_sha256': raw_hashes,
                'output_sha256': {name: sha256(output / name) for name in EXPORT_FILES}, 'status': 'REAL_VALIDATION_EXPORT', 'days': days, 'n_orders': n_orders, 'n_residuals': n_residuals,
                'source_hashes': before, 'source_unchanged': True, 'load_counts': counts,
                'reference_sample': {'days': 80, 'orders': 2414025},
                'matches_reference_counts': len(days) == 80 and n_orders == 2414025,
                'retained_before_horizon_dropna': True, 'residual_first_event': 'Excluded to match source np.diff(comp)',
                'warning': 'Merged events are offline proxies; TBBO future matches are trade-sampled pretrade quotes, not exact wall-clock quotes. No fit or test evaluation run.'}
    validate_export_manifest(manifest)
    (output/'export_manifest.json').write_text(json.dumps(manifest, indent=2))


def run(repo, output):
    # Source imports/JIT may otherwise create caches inside the inspected checkout.
    from numba import config as numba_config
    old_cache, old_bytecode = numba_config.CACHE_DIR, sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix='zn_validation_jit_') as cache:
        try:
            numba_config.CACHE_DIR, sys.dont_write_bytecode = cache, True
            repo, output = Path(repo).resolve(), Path(output).resolve()
            if output == repo or repo in output.parents:
                raise ValueError('Use an output directory outside the source repository')
            with new_output_directory(output) as stage:
                _run(repo, stage)
        finally:
            numba_config.CACHE_DIR, sys.dont_write_bytecode = old_cache, old_bytecode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.repo, args.out)
