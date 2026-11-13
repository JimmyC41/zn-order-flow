"""Order flow analysis: raw files to every table and figure.

    python analysis/run.py fit        fit months only: background, tuning, frozen models
    python analysis/run.py quoting    adds the quoting rule to a fitted model without refitting it
    python analysis/run.py validate   scores the frozen models on the validation months
    python analysis/run.py test       scores the frozen models on the locked test months, run once

To use other data, point the config at the files and move the three dates.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flow
from flow import Arrivals, Direction, Sizes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CFG = json.loads((Path(__file__).parent / "config.json").read_text())
FILES = [str(ROOT / p) for p in CFG["files"]]
TICK = CFG["tick"]
flow.configure(CFG["session_open"], CFG["session_close"], CFG["bucket_seconds"])
B, T, DUR = flow.B, flow.T, flow.DUR

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "lines.linewidth": 2, "figure.dpi": 160})
BASE, REFINED = "constant rate", "time of day + news + self-exciting"
SIZE_BINS = [(1, 1, "1"), (2, 9, "2 to 9"), (10, 99, "10 to 99"), (100, 999, "100 to 999"), (1000, 1e9, "1,000+")]


def save(name, obj):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(OUT / "tables" / f"{name}.csv")
        print(f"\n== {name}\n{obj.round(4).to_string()}", flush=True)
    else:
        (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def fig(name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"{name}.png")
    plt.close()


def clock(b):
    s = flow.OPEN_S + b * flow.BUCKET_S
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


# ---------------------------------------------------------------- data

def news_kinds():
    """Release day -> kind. Inflation and jobs reports come out at 8:30, Fed decisions at 14:00."""
    n = pd.read_csv(ROOT / CFG["news_file"]).dropna(subset=["date"])
    return {pd.Timestamp(d).date(): ("fed decision" if r == "FOMC" else "data release") for d, r in zip(n.date, n.release)}


KINDS = news_kinds()


def load(start, end):
    trades, counts = flow.load_trades(FILES, start, end, flow.skipped_days(FILES))
    orders = flow.to_orders(trades, CFG["merge_us"] * 1000)
    return trades, counts, add_features(orders)


def add_features(o):
    """Inputs to the direction model, all known before the order arrives."""
    o["buy"] = o.sgn == 1
    o["prev"] = o.groupby("day").sgn.shift(1).fillna(0)
    o["prev5"] = o.groupby("day").sgn.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    o["prev_x_size"] = o.prev * np.log(o.groupby("day")["size"].shift(1).fillna(1))
    o["queue"] = (o.bsz - o.asz) / (o.bsz + o.asz)
    return o


def split_days(days, share=0.6):
    k = int(round(len(days) * share))
    return days[:k], days[k:]


def bucket_counts(orders):
    return orders.groupby("day").t.apply(lambda t: pd.Series(np.bincount(flow.bucket(t.values), minlength=B))).unstack()


def count_scores(orders, arrivals):
    """Miss in orders per bucket, as a share of actual orders. With each day's total known, only the miss
    within the day is left. Carry-over is the correlation between one bucket's actual / expected and the
    next bucket's on the same day: how much the last 10 minutes say about the next 10."""
    counts = bucket_counts(orders)
    actual = counts.values
    expected = np.array([arrivals.expected_counts(KINDS.get(d)) for d in counts.index])
    err = lambda e: float(np.abs(actual - e).sum() / actual.sum())
    surprise = actual / expected
    return {"count_error": err(expected),
            "count_error_day_total_known": err(expected * (actual.sum(1) / expected.sum(1))[:, None]),
            "count_carry_over": float(np.corrcoef(surprise[:, :-1].ravel(), surprise[:, 1:].ravel())[0, 1])}


def direction_by_gap(model, orders):
    """Direction accuracy by the time since the previous order: how much of it holds outside a burst."""
    direction = build(model)[2]
    gap = orders.groupby("day").ts.diff() / 1e9
    band = pd.cut(gap, [0, 1e-3, 0.1, 1, np.inf], right=False, labels=["under 1 ms", "1 to 100 ms", "0.1 to 1 s", "over 1 s"])
    rows = {}
    for name, g in orders.groupby(band, observed=True):
        rows[name] = {"share of orders": len(g) / gap.notna().sum()}
        for k in list(DIRECTION_MODELS)[2:]:
            rows[name][f"{k} | accuracy"] = direction[k].score(g[DIRECTION_MODELS[k]].values, g.buy.values)["accuracy"]
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- models

def news_multipliers(orders):
    """Orders per bucket on each kind of release day, relative to days with no release."""
    counts = bucket_counts(orders)
    kind = pd.Series([KINDS.get(d, "none") for d in counts.index], index=counts.index)
    normal = counts[kind == "none"].mean()
    return {k: (counts[kind == k].mean() / normal).tolist() for k in ("data release", "fed decision") if (kind == k).any()}, \
        (normal / DUR).values


def fit_arrivals(orders, starts):
    """Baseline and refined arrival models. The decay speeds are fitted on every n-th day to keep it fast."""
    news, shape = news_multipliers(orders)
    const = np.full(B, len(orders) / (orders.day.nunique() * T))
    days = sorted(orders.day.unique())[::CFG["arrival_fit_day_step"]]
    se, objective = flow.fit_self_exciting(orders[orders.day.isin(days)], shape, news, KINDS, starts)
    models = {BASE: Arrivals(const), "time of day": Arrivals(shape),
              "time of day + news": Arrivals(shape, news=news), REFINED: se}
    return models, objective


SIZE_MODELS = {"lognormal": {}, "lognormal + common sizes": {"spikes": CFG["size_spikes"]},
               "lognormal + common sizes + separate tail": {"spikes": CFG["size_spikes"], "tail_share": CFG["size_tail_share"]}}
DIRECTION_MODELS = {"fixed chance": [], "previous order": ["prev"], "previous order and its size": ["prev", "prev_x_size"],
                    "previous order and its size + queue sizes": ["prev", "prev_x_size", "queue"]}
SIMULATED_DIRECTION = "previous order and its size"


def fit_models(orders):
    arrivals, objective = fit_arrivals(orders, CFG["decay_starts"])
    sizes = {k: Sizes.fit(orders["size"].values, **kw) for k, kw in SIZE_MODELS.items()}
    direction = {k: Direction.fit(orders[cols].values, orders.buy.values) for k, cols in DIRECTION_MODELS.items()}
    model = {"merge_us": CFG["merge_us"], "fit_days": int(orders.day.nunique()), "arrival_fit_objective": objective,
             "arrivals": {k: {"shape": m.shape.tolist(), "c": m.c, "alpha": m.alpha.tolist(), "beta": m.beta.tolist(),
                              "news": {a: b.tolist() for a, b in m.news.items()}} for k, m in arrivals.items()},
             "sizes": {k: s.params() for k, s in sizes.items()},
             "direction": {k: d.w.tolist() for k, d in direction.items()}}
    return model


def build(model):
    arrivals = {k: Arrivals(v["shape"], v["c"], v["alpha"], v["beta"], v["news"]) for k, v in model["arrivals"].items()}
    sizes = {k: Sizes(**v) for k, v in model["sizes"].items()}
    direction = {k: Direction(w) for k, w in model["direction"].items()}
    return arrivals, sizes, direction


def score_all(model, orders, label):
    arrivals, sizes, direction = build(model)
    a, resid = {}, {}
    for k, m in arrivals.items():
        a[k], resid[k] = m.score(orders, KINDS)
        a[k].update(count_scores(orders, m))
    a = pd.DataFrame(a).T
    s = pd.DataFrame({k: m.score(orders["size"].values) for k, m in sizes.items()}).T
    d = pd.DataFrame({k: m.score(orders[DIRECTION_MODELS[k]].values, orders.buy.values) for k, m in direction.items()}).T
    for name, t in (("arrivals", a), ("sizes", s), ("direction", d)):
        save(f"{label}_{name}", t)
    joint = pd.Series({"baseline": a.loglik_per_order.iloc[0] + s.loglik_per_order.iloc[0] - d.log_loss.iloc[0],
                       "refined": a.loglik_per_order.iloc[-1] + s.loglik_per_order.iloc[-1] - d.log_loss.iloc[-1]})
    save(f"{label}_main_score", joint.to_frame("joint log-likelihood per order"))
    return resid


# ---------------------------------------------------------------- fit stage

def background(trades, counts, orders):
    days = sorted(orders.day.unique())
    gap = trades.groupby("day").ts.diff()
    same = trades.sgn == trades.sgn.shift()
    edges = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e13]
    labels = ["under 1 µs", "1 to 10 µs", "10 to 100 µs", "0.1 to 1 ms", "1 to 10 ms", "10 to 100 ms", "0.1 to 1 s", "over 1 s"]
    g = pd.DataFrame({"same side": pd.cut(gap[same], edges, labels=labels).value_counts(sort=False),
                      "opposite side": pd.cut(gap[~same], edges, labels=labels).value_counts(sort=False)})
    g["same per opposite"] = g["same side"] / g["opposite side"]
    save("fit_gaps_by_side", g)

    ogap = orders.groupby("day").ts.diff().dropna() / 1e9
    per_day = orders.groupby("day").size()
    kind = pd.Series([KINDS.get(d, "none") for d in per_day.index], index=per_day.index)
    s = orders.sgn.values
    summary = {
        "days": len(days), "first_day": str(days[0]), "last_day": str(days[-1]),
        "trades_in_session": counts["session_trades"], "share_no_exchange_side": counts["no_exchange_side"] / counts["session_trades"],
        "trades_signed": len(trades), "orders": len(orders), "days_by_kind": kind.value_counts().to_dict(),
        "orders_per_minute": len(orders) / (len(days) * T / 60),
        "trade_share_size_1": float((trades["size"] == 1).mean()),
        "order_size_median": float(orders["size"].median()), "order_size_mean": float(orders["size"].mean()),
        "order_share_size_1": float((orders["size"] == 1).mean()),
        "volume_share_top_1pct_orders": float(orders["size"].nlargest(len(orders) // 100).sum() / orders["size"].sum()),
        "buy_volume_share_day_min": float(orders.groupby("day").apply(lambda g: g["size"][g.buy].sum() / g["size"].sum()).min()),
        "buy_volume_share_day_max": float(orders.groupby("day").apply(lambda g: g["size"][g.buy].sum() / g["size"].sum()).max()),
        "gap_median_s": float(ogap.median()), "gap_mean_s": float(ogap.mean()),
        "spread_one_tick_share": float((np.round((orders.ask - orders.bid) / TICK) == 1).mean()),
        "quote_size_median": float(((orders.bsz + orders.asz) / 2).median()),
        "direction_autocorrelation": {k: float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in (1, 100)},
    }
    save("background", summary)

    counts_b = bucket_counts(orders)
    plt.figure(figsize=(6.6, 3.0))
    for k, colour, lab in (("none", BLUE, "No release"), ("data release", ORANGE, "Inflation or jobs report at 8:30"),
                           ("fed decision", AQUA, "Fed decision at 14:00")):
        plt.plot(range(B), counts_b[kind == k].mean(), color=colour, label=f"{lab} ({(kind == k).sum()} days)")
    plt.xticks(range(0, B, 4), [clock(b) for b in range(0, B, 4)])
    plt.ylabel("Orders per 10 minutes")
    plt.xlabel("New York time")
    plt.legend(frameon=False, fontsize=8)
    plt.title("Figure 1. Orders by time of day, with and without a scheduled release")
    fig("fig1_orders_by_time")

    share = [float(((orders["size"] >= lo) & (orders["size"] <= hi)).mean()) for lo, hi, _ in SIZE_BINS]
    vol = [float(orders["size"][(orders["size"] >= lo) & (orders["size"] <= hi)].sum() / orders["size"].sum()) for lo, hi, _ in SIZE_BINS]
    x = np.arange(len(SIZE_BINS))
    plt.figure(figsize=(6.6, 3.0))
    b1 = plt.bar(x - 0.2, np.array(share) * 100, width=0.38, color=BLUE, label="Share of orders")
    b2 = plt.bar(x + 0.2, np.array(vol) * 100, width=0.38, color=ORANGE, label="Share of contracts traded")
    plt.bar_label(b1, [f"{v * 100:.0f}%" for v in share], fontsize=8, color=INK, padding=2)
    plt.bar_label(b2, [f"{v * 100:.0f}%" for v in vol], fontsize=8, color=INK, padding=2)
    plt.xticks(x, [b[2] for b in SIZE_BINS])
    plt.grid(axis="x", visible=False)
    plt.xlabel("Order size, contracts")
    plt.ylabel("%")
    plt.legend(frameon=False, fontsize=8)
    plt.title("Figure 2. Most orders are small, most volume comes from large orders")
    fig("fig2_order_sizes")

    plt.figure(figsize=(6.6, 3.0))
    plt.bar(range(len(g)), g["same per opposite"], width=0.72, color=BLUE)
    plt.yscale("log")
    plt.xticks(range(len(g)), labels, fontsize=7.5)
    plt.grid(axis="x", visible=False)
    plt.ylabel("Same-side gaps per opposite-side gap")
    plt.xlabel("Gap since the previous trade")
    plt.title("Figure 3. Trades under 0.1 ms apart are almost always on the same side")
    fig("fig3_gaps_by_side")
    return summary


def tune(trades):
    """Choices made on the fit months only: fitted on the earlier 60% of days, scored on the later 40%."""
    days = sorted(trades.day.unique())
    early, late = split_days(days)
    variants = 0
    grid = {}
    starts = {1: [[0.5, [0.5], [100]], [0.5, [0.4], [2000]]],
              2: [[0.4, [0.3, 0.3], [1000, 5]], [0.4, [0.3, 0.2], [5000, 20]]],
              3: [[0.3, [0.25, 0.2, 0.2], [3000, 60, 1]], [0.3, [0.3, 0.15, 0.1], [10000, 200, 5]]]}
    for us in (10, 100, 1000):
        o = add_features(flow.to_orders(trades, us * 1000))
        f, v = o[o.day.isin(early)], o[o.day.isin(late)]
        for k in (1, 2, 3):
            m = fit_arrivals(f, starts[k])[0]
            if k == 1:
                grid[(us, "no self-excitation")] = m["time of day + news"].score(v, KINDS)[0]["fit_error"]
                variants += 1
            grid[(us, f"{k} decay speeds")] = m[REFINED].score(v, KINDS)[0]["fit_error"]
            grid[(us, f"triggered share, {k}")] = float(m[REFINED].alpha.sum())
            variants += 1
        if us == CFG["merge_us"]:
            sz = {}
            for name, kw in [(k, kw) for k, kw in SIZE_MODELS.items()] + \
                            [(f"tail share {q}", {"spikes": CFG["size_spikes"], "tail_share": q}) for q in (0.005, 0.01, 0.05)]:
                sz[name] = Sizes.fit(f["size"].values, **kw).score(v["size"].values)
                variants += 1
            save("fit_tuning_sizes", pd.DataFrame(sz).T)
            tried = DIRECTION_MODELS | {"previous five orders": ["prev", "prev5"]}
            dr = {k: Direction.fit(f[cols].values, f.buy.values).score(v[cols].values, v.buy.values) for k, cols in tried.items()}
            variants += len(dr)
            save("fit_tuning_direction", pd.DataFrame(dr).T)
    g = pd.Series(grid).unstack()
    g.index.name = "merge window, microseconds"
    save("fit_tuning_arrivals", g)
    return variants


def stage_fit():
    trades, counts, orders = load(CFG["fit_start"], CFG["fit_end"])
    background(trades, counts, orders)
    variants = tune(trades)
    model = fit_models(orders)
    variants += len(model["arrivals"]) + len(model["sizes"]) + len(model["direction"])
    save("variants", {"model_variants": variants})
    evaluate(model, trades, orders, "fit")
    save("model", model)


# ---------------------------------------------------------------- evaluation, shared by every stage

def order_frame(model, trades, orders):
    """Per order: what a resting quote earns when hit and how far the price then moves against it, in ticks,
    with the model's view of how busy the market is. Earned = distance from the midpoint to the trade price.
    Given back = midpoint move in the order's direction after each horizon."""
    arrivals = build(model)[0][REFINED]
    rows = []
    for d, o in orders.groupby("day"):
        tr = trades[trades.day == d]
        ts, mid = tr.ts.values, tr.mid.values
        t, kind = o.t.values, KINDS.get(d)
        base = arrivals.base(kind)[flow.bucket(t)]
        r = pd.DataFrame({"day": d, "size": o["size"].values, "t": t, "kind": KINDS.get(d, "none"),
                          "burst": arrivals.day(t, kind)[2] / base,
                          "earned": o.sgn.values * (o.px.values - o.mid.values) / TICK})
        for ms in CFG["reaction_ms"]:
            r[f"seen_{ms}"] = arrivals.rate_seen(t, kind, ms / 1000) / base
        for h in CFG["horizons_s"]:
            j = np.searchsorted(ts, o.ts.values + int(h * 1e9))
            later = np.where(j < len(ts), mid[np.minimum(j, len(ts) - 1)], np.nan)
            r[f"back_{h}"] = o.sgn.values * (later - o.mid.values) / TICK
        rows.append(r)
    r = pd.concat(rows).dropna()
    r["news_window"] = ((r.kind == "data release") & (r.t >= 600) & (r.t < 1800)) | ((r.kind == "fed decision") & (r.t >= 20400))
    return r


def quoting_rule(model, r, label):
    """Pull quotes in news windows and when the model's rate, as seen with a reaction delay, is above a threshold.
    The threshold for each delay is the one that kept the most spread per day on the fit months."""
    value = r["size"] * (r.earned - r[f"back_{CFG['quote_horizon_s']}"])       # ticks x contracts kept if this order hits a quote
    if label == "fit":
        model["quoting"] = {}
        for ms in CFG["reaction_ms"]:
            grid = r[f"seen_{ms}"].quantile(np.arange(0.05, 1.0, 0.05)).values
            total = [value[~r.news_window & (r[f"seen_{ms}"] <= g)].sum() for g in grid]
            model["quoting"][str(ms)] = float(grid[int(np.argmax(total))])
    rows = {}
    for name, on in [("always quote", pd.Series(True, index=r.index))] + \
                    [(f"news windows and busy periods, {ms} ms reaction", ~r.news_window & (r[f"seen_{ms}"] <= model["quoting"][str(ms)]))
                     for ms in CFG["reaction_ms"]]:
        daily = (value[on].groupby(r.day[on]).sum() / r["size"][on].groupby(r.day[on]).sum()).reindex(r.day.unique())
        always = daily if name == "always quote" else always
        gain = daily - always                     # same days under both rules, so the day-to-day noise cancels
        rows[name] = {"share of orders quoted": on.mean(),
                      "kept per contract quoted": value[on].sum() / r["size"][on].sum(),
                      "kept per contract, standard error across days": daily.std() / np.sqrt(daily.notna().sum()),
                      "gain over always quote, per contract": gain.mean(),
                      "gain over always quote, standard error across days": gain.std() / np.sqrt(gain.notna().sum())}
    return pd.DataFrame(rows).T


def spreads(r):
    hs = CFG["horizons_s"]

    def table(groups):
        out = {}
        for name, g in groups:
            dm = g.groupby("day")[["earned"] + [f"back_{h}" for h in hs]].mean()
            row = {"orders": len(g), "earned": dm.earned.mean()}
            for h in hs:
                row[f"given back after {h}s"] = dm[f"back_{h}"].mean()
            kept = dm.earned - dm["back_10"]
            row["kept after 10s"] = kept.mean()
            row["kept after 10s, standard error"] = kept.std() / np.sqrt(len(kept))
            out[name] = row
        return pd.DataFrame(out).T

    q = r.burst.quantile([0.5, 0.9]).values
    groups = [("all orders", r), ("size: 1 contract", r[r["size"] == 1]), ("size: 100+", r[r["size"] >= 100]),
              ("activity: calm half", r[r.burst <= q[0]]), ("activity: burst, top 10%", r[r.burst > q[1]])]
    weighted = r.assign(**{c: r[c] * r["size"] for c in ["earned"] + [f"back_{h}" for h in hs]})
    w = weighted.groupby("day")[["earned"] + [f"back_{h}" for h in hs]].sum().div(r.groupby("day")["size"].sum(), axis=0)
    per_contract = {"orders": len(r), "earned": w.earned.mean(), **{f"given back after {h}s": w[f"back_{h}"].mean() for h in hs},
                    "kept after 10s": (w.earned - w.back_10).mean(),
                    "kept after 10s, standard error": (w.earned - w.back_10).std() / np.sqrt(len(w))}
    return pd.concat([pd.DataFrame({"all contracts (orders weighted by size)": per_contract}).T, table(groups)])


def market_maker_view(model, orders):
    """Burstiness and inventory pressure on real days against simulated days from each model."""
    arrivals, sizes, direction = build(model)
    normal = orders[~orders.day.isin(KINDS)]
    real = pd.DataFrame([flow.flow_stats(g.t.values, g.sgn.values, g["size"].values) for _, g in normal.groupby("day")])
    sims = {"baseline model": flow.simulate_flow(arrivals[BASE], sizes["lognormal"], direction["fixed chance"], CFG["sim_days"], CFG["seed"]),
            "refined model": flow.simulate_flow(arrivals[REFINED], sizes["lognormal + common sizes + separate tail"],
                                                direction[SIMULATED_DIRECTION], CFG["sim_days"], CFG["seed"])}
    out = {"actual, average day": real.mean(), "actual, 5th percentile day": real.quantile(.05), "actual, 95th percentile day": real.quantile(.95)}
    for k, s in sims.items():
        out[f"{k}, average day"] = s.mean()
        out[f"{k}, 5th percentile day"] = s.quantile(.05)
        out[f"{k}, 95th percentile day"] = s.quantile(.95)
    return pd.DataFrame(out)


def gap_figure(resid, name, title):
    plt.figure(figsize=(6.6, 3.0))
    x = np.logspace(-5, 1.2, 300)
    plt.plot(x, 1 - np.exp(-x), color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="A correct model")
    for key, colour, lab in ((BASE, ORANGE, "Baseline: constant rate"), (REFINED, BLUE, "Refined: time of day, news, self-exciting")):
        z = np.sort(resid[key])[::max(1, len(resid[key]) // 20000)]
        plt.plot(z, np.linspace(0, 1, len(z)), color=colour, label=lab)
    plt.xscale("log")
    plt.xlim(1e-5, 15)
    plt.xlabel("Gap between orders, rescaled by the model's predicted rate (1 = average)")
    plt.ylabel("Share of gaps shorter")
    plt.legend(frameon=False, fontsize=8, loc="upper left")
    plt.title(title)
    fig(name)


def evaluate(model, trades, orders, label):
    resid = score_all(model, orders, label)
    arrivals = build(model)[0]
    month = orders.day.map(lambda d: f"{d:%Y-%m}")
    kind = orders.day.map(lambda d: KINDS.get(d, "none"))
    rows = {}
    for name, g in [(f"month {k}", g) for k, g in orders.groupby(month)] + [(f"days: {k}", g) for k, g in orders.groupby(kind)]:
        rows[name] = {"days": g.day.nunique(), "orders per day": len(g) / g.day.nunique()}
        for k in (BASE, "time of day + news", REFINED):
            s = arrivals[k].score(g, KINDS)[0]
            rows[name][f"{k} | fit error"] = s["fit_error"]
            rows[name][f"{k} | count error"] = count_scores(g, arrivals[k])["count_error"]
    save(f"{label}_stability", pd.DataFrame(rows).T)
    save(f"{label}_direction_by_gap", direction_by_gap(model, orders))
    r = order_frame(model, trades, orders)
    save(f"{label}_spreads", spreads(r))
    save(f"{label}_quoting_rule", quoting_rule(model, r, label))
    save(f"{label}_market_maker_view", market_maker_view(model, orders))
    gap_figure(resid, f"fig4_gaps_{label}", f"Figure 4. Gaps between orders, actual against each model ({label} months)")

    actual = bucket_counts(orders)
    normal = actual[[d not in KINDS for d in actual.index]].mean()
    plt.figure(figsize=(6.6, 3.0))
    plt.bar(range(B), normal, width=0.72, color=BLUE, label="Actual, average day with no release")
    plt.plot(range(B), arrivals[REFINED].expected_counts(), color=ORANGE, marker="o", markersize=3, label="Refined model")
    plt.plot(range(B), arrivals[BASE].expected_counts(), color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="Baseline: constant rate")
    plt.xticks(range(0, B, 4), [clock(b) for b in range(0, B, 4)])
    plt.grid(axis="x", visible=False)
    plt.ylabel("Orders per 10 minutes")
    plt.legend(frameon=False, fontsize=8)
    plt.title(f"Figure 5. Predicted against actual orders by time of day ({label} months)")
    fig(f"fig5_predicted_vs_actual_{label}")


def stage_held_out(label, start, end):
    model = json.loads((OUT / "model.json").read_text())
    trades, _, orders = load(start, end)
    print(label, "days", orders.day.nunique(), min(orders.day), max(orders.day))
    evaluate(model, trades, orders, label)


if __name__ == "__main__":
    stage = sys.argv[1]
    next_day = lambda d: str((pd.Timestamp(d) + pd.Timedelta(days=1)).date())
    if stage == "fit":
        stage_fit()
    elif stage == "quoting":
        model = json.loads((OUT / "model.json").read_text())
        trades, _, orders = load(CFG["fit_start"], CFG["fit_end"])
        save("fit_quoting_rule", quoting_rule(model, order_frame(model, trades, orders), "fit"))
        save("model", model)
    elif stage == "validate":
        stage_held_out("validation", next_day(CFG["fit_end"]), CFG["validate_end"])
    elif stage == "test":
        stage_held_out("test", next_day(CFG["validate_end"]), CFG["test_end"])
