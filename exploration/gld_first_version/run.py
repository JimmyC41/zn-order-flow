"""Order flow analysis: raw files to every table and figure.

    python analysis/run.py fit        fit days only, freezes the models
    python analysis/run.py validate   scores the frozen models on the held-out days, run once
    python analysis/run.py sixmonth   checks the activity forecast on the 1-second file
    python analysis/run.py all

To use more data, add files to the config and move fit_end. Days after fit_end are held out.
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flow
from flow import B, DUR, T, Arrivals, Direction, Sizes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "gld"
CFG = json.loads((Path(__file__).parent / "config.json").read_text())
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HALF_HOURS = [f"{9 + (30 + 30 * b) // 60:02d}:{(30 + 30 * b) % 60:02d}" for b in range(B)]
SIZE_BINS = [(1, 1, "1"), (2, 9, "2 to 9"), (10, 99, "10 to 99"), (100, 100, "100"),
             (101, 999, "101 to 999"), (1000, 10**9, "1,000+")]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "lines.linewidth": 2, "figure.dpi": 160})


def paths(patterns):
    return [str(ROOT / p) for p in patterns]


def save(name, obj):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(OUT / "tables" / f"{name}.csv")
        print(f"\n== {name}\n{obj.round(4).to_string()}")
    else:
        (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def fig(name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"{name}.png")
    plt.close()


def split_days(days, share=0.6):
    k = max(1, int(round(len(days) * share)))
    return days[:k], days[k:]


def half_hour_counts(orders):
    return orders.groupby("day").t.apply(lambda t: pd.Series(np.bincount(flow.bucket(t.values), minlength=B))).unstack()


def count_error(orders, expected):
    actual = half_hour_counts(orders)
    return float((np.abs(actual - expected) / actual).mean().mean())


# ---------------------------------------------------------------- fit stage

def background(fills, counts, raw, orders):
    days = sorted(orders.day.unique())
    gaps = lambda o: (o.groupby("day").ts.diff().dropna() / 1e9)
    per_s = orders.groupby("day").t.apply(lambda t: pd.Series(np.bincount(t.astype(int), minlength=int(T))))
    lag1 = lambda o: float(np.corrcoef(o.sgn.values[:-1], o.sgn.values[1:])[0, 1])
    vol = fills.assign(v=fills["size"])
    summary = {
        "days": len(days), "first_day": str(days[0]), "last_day": str(days[-1]),
        "fills_regular_hours": counts["regular_hours"], "fills_opening_second_dropped": counts["opening_second"],
        "fills_unsigned_dropped": counts["unsigned"], "fills_continuous": counts["continuous"],
        "share_no_exchange_side": counts["no_exchange_side"] / (counts["continuous"] + counts["unsigned"]),
        "share_signed_by_tick_rule": counts["signed_by_tick_rule"] / (counts["continuous"] + counts["unsigned"]),
        "orders_unmerged": len(raw), "orders_merged": len(orders),
        "orders_per_day_min": int(orders.groupby("day").size().min()), "orders_per_day_max": int(orders.groupby("day").size().max()),
        "orders_per_minute": len(orders) / (len(days) * T / 60), "orders_per_hour": len(orders) / (len(days) * T / 3600),
        "fill_size_median": float(fills["size"].median()), "fill_size_mean": float(fills["size"].mean()),
        "fill_share_under_100": float((fills["size"] < 100).mean()),
        "fill_volume_share_under_100": float(fills["size"][fills["size"] < 100].sum() / fills["size"].sum()),
        "order_size_median": float(orders["size"].median()), "order_size_mean": float(orders["size"].mean()),
        "order_size_p99": float(orders["size"].quantile(.99)), "order_size_max": int(orders["size"].max()),
        "buy_volume_share_by_day": {str(d): float(g["size"][g.sgn == 1].sum() / g["size"].sum()) for d, g in vol.groupby("day")},
        "gap_median_s_unmerged": float(gaps(raw).median()), "gap_share_under_1ms_unmerged": float((gaps(raw) < 1e-3).mean()),
        "gap_median_s_merged": float(gaps(orders).median()), "gap_mean_s_merged": float(gaps(orders).mean()),
        "gap_cv_merged": float(gaps(orders).std() / gaps(orders).mean()),
        "count_dispersion_1s": float(per_s.var() / per_s.mean()),
        "direction_lag1_unmerged": lag1(raw), "direction_lag1_merged": lag1(orders),
    }
    minute = orders.assign(m=(orders.t // 60).astype(int), v=orders.sgn * orders["size"]).groupby(["day", "m"]).agg(v=("v", "sum"), tv=("size", "sum"))
    summary["imbalance_1min_sd"] = float((minute.v / minute.tv).std())
    save("background", summary)

    per_bucket = half_hour_counts(orders).mean()
    plt.figure(figsize=(6.4, 2.8))
    plt.bar(range(B), per_bucket, width=0.72, color=BLUE)
    plt.xticks(range(B), HALF_HOURS, fontsize=7.5)
    plt.grid(axis="x", visible=False)
    plt.ylabel("Orders per half hour")
    plt.title("Figure 1. Orders arrive mostly at the open and the close")
    fig("fig1_orders_by_half_hour")

    share = [float(((orders["size"] >= lo) & (orders["size"] <= hi)).mean()) for lo, hi, _ in SIZE_BINS]
    plt.figure(figsize=(6.4, 2.8))
    bars = plt.bar(range(len(SIZE_BINS)), np.array(share) * 100, width=0.72, color=BLUE)
    plt.bar_label(bars, [f"{s * 100:.1f}%" for s in share], fontsize=8, color=INK, padding=2)
    plt.xticks(range(len(SIZE_BINS)), [b[2] for b in SIZE_BINS])
    plt.grid(axis="x", visible=False)
    plt.xlabel("Order size, shares")
    plt.ylabel("Share of orders, %")
    plt.title("Figure 2. Four in five orders are under 100 shares")
    fig("fig2_order_sizes")
    return summary


def fit_models(raw, orders):
    """Fit baseline and refined parts on the fit days. Returns the frozen model."""
    n_days = orders.day.nunique()
    shape = flow.tod_rate(orders)
    const = lambda o: np.full(B, len(o) / (o.day.nunique() * T))
    model = {
        "merge_ms": CFG["merge_ms"],
        "arrivals": {
            "constant, unmerged orders": {"shape": const(raw).tolist()},
            "constant": {"shape": const(orders).tolist()},
            "time of day": {"shape": shape.tolist()},
        },
    }
    se = flow.fit_self_exciting(orders, shape, CFG["decay_speeds"])
    model["arrivals"]["time of day + self-exciting"] = {"shape": shape.tolist(), "c": se.c, "alpha": se.alpha.tolist(), "beta": se.beta.tolist()}
    base_sizes, ref_sizes = Sizes(orders["size"].values), Sizes(orders["size"].values, flow.SIZE_SPIKES)
    model["sizes"] = {"smooth curve": {"mu": base_sizes.mu, "sigma": base_sizes.sigma},
                      "smooth curve + round-number spikes": {"mu": ref_sizes.mu, "sigma": ref_sizes.sigma, "p_spike": ref_sizes.p_spike}}
    d = Direction(orders.sgn.values, orders.day.values)
    model["direction"] = {"coin flip": {"p_after_buy": .5, "p_after_sell": .5},
                          "depends on previous order": {"p_after_buy": d.p_after_buy, "p_after_sell": d.p_after_sell}}
    model["fit_days"] = n_days
    save("model", model)
    return model


def build(model):
    arr = {k: Arrivals(np.array(v["shape"]), v.get("c", 1.0), v.get("alpha", ()), v.get("beta", ())) for k, v in model["arrivals"].items()}
    sizes = {}
    for k, v in model["sizes"].items():
        s = Sizes.__new__(Sizes)
        s.mu, s.sigma, s.p_spike = v["mu"], v["sigma"], {int(a): b for a, b in v.get("p_spike", {}).items()}
        s.spikes = list(s.p_spike)
        s._norm = 1 - np.exp(flow._lognormal_logpmf(np.array(s.spikes), s.mu, s.sigma)).sum() if s.spikes else 1.0
        sizes[k] = s
    dirs = {}
    for k, v in model["direction"].items():
        d = Direction()
        d.p_after_buy, d.p_after_sell = v["p_after_buy"], v["p_after_sell"]
        dirs[k] = d
    return arr, sizes, dirs


def score_all(model, raw, orders, label):
    arr, sizes, dirs = build(model)
    rows, resid = {}, {}
    for name, m in arr.items():
        data = raw if "unmerged" in name else orders
        s, z = m.score(data)
        s["half_hour_count_error"] = count_error(data, m.expected_counts())
        s["orders"] = len(data)
        rows[name] = s
        resid[name] = z
    a = pd.DataFrame(rows).T
    s = pd.DataFrame({k: v.score(orders["size"].values) for k, v in sizes.items()}).T
    d = pd.DataFrame({k: v.score(orders.sgn.values, orders.day.values) for k, v in dirs.items()}).T
    for name, t in (("arrivals", a), ("sizes", s), ("direction", d)):
        save(f"{label}_{name}", t)
    by_day = pd.DataFrame({str(day): {f"{k} | fit error": arr[k].score(g)[0]["fit_error"] for k in ("constant", "time of day + self-exciting")}
                           | {f"{k} | count error": count_error(g, arr[k].expected_counts()) for k in ("constant", "time of day")}
                           | {"orders": len(g)} for day, g in orders.groupby("day")}).T
    save(f"{label}_by_day", by_day)
    return a, s, d, resid


def tuning_grid(fills):
    """Fit error on later fit days for each merge window and number of decay speeds, fitted on earlier fit days."""
    days = sorted(fills.day.unique())
    early, late = split_days(days)
    grid, variants = {}, 0
    for ms in (0, 0.5, 1, 2, 5):
        o = flow.to_orders(fills, ms)
        f, v = o[o.day.isin(early)], o[o.day.isin(late)]
        shape = flow.tod_rate(f)
        grid[(ms, 0)] = Arrivals(shape).score(v)[0]["fit_error"]
        for k in (1, 2):
            m = flow.fit_self_exciting(f, shape, k)
            grid[(ms, k)] = m.score(v)[0]["fit_error"]
            grid[(ms, f"triggered share, {k}")] = float(m.alpha.sum())
            variants += 1
        variants += 1
    g = pd.Series(grid).unstack()
    g.index.name = "merge window, ms"
    save("fit_tuning_grid", g)
    err = g[[0, 1, 2]]
    plt.figure(figsize=(6.4, 2.6))
    plt.imshow(err.T.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#104281"]), aspect="auto")
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            v = err.values[i, j]
            plt.text(i, j, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                     color="#ffffff" if v > err.values.mean() else INK)
    plt.xticks(range(len(err)), [f"{m:g}" for m in err.index])
    plt.yticks(range(3), ["none", "one", "two"])
    plt.grid(False)
    plt.xlabel("Merge window, milliseconds")
    plt.ylabel("Decay speeds")
    plt.title("Figure 5. Fit error by merge window and decay speeds")
    fig("fig5_tuning_grid")
    return variants, early, late


def dropped_variants(fills, early, late):
    """Variants tried on the fit days and not kept: minute imbalance and the futures link."""
    o = flow.to_orders(fills, CFG["merge_ms"])
    out = {}
    m = o.assign(m=(o.t // 60).astype(int), v=o.sgn * o["size"]).groupby(["day", "m"]).agg(v=("v", "sum"), tv=("size", "sum"))
    m["x"] = m.v / m.tv
    m["lag"] = m.groupby(level=0).x.shift(1)
    m = m.dropna()
    e, l = m[m.index.get_level_values(0).isin(early)], m[m.index.get_level_values(0).isin(late)]
    slope = np.polyfit(e.lag, e.x, 1)
    out["minute imbalance from previous minute"] = {"slope": slope[0], "r2_later_days": 1 - ((l.x - np.polyval(slope, l.lag)) ** 2).sum() / (l.x ** 2).sum()}

    if CFG.get("aux_tick_files"):
        end = str(max(fills.day))
        q = f"""select ts_event_ns ts from read_parquet({flow._files(paths(CFG['aux_tick_files']))!r})
                where is_trade and symbol = ? and ts_event_ns < {flow._day_end_ns(end)}"""
        g = flow._add_session_time(duckdb.execute(q, [CFG["aux_symbol"]]).df(), "ts")
        g = g[(g.tod >= flow.OPEN_S) & (g.tod < flow.CLOSE_S)]
        idx = pd.MultiIndex.from_product([sorted(o.day.unique()), range(int(T))])
        x = pd.DataFrame({"y": o.groupby(["day", o.t.astype(int)]).size().reindex(idx, fill_value=0),
                          "aux": g.groupby(["day", (g.tod - flow.OPEN_S).astype(int)]).size().reindex(idx, fill_value=0)})
        x["aux_lag"] = x.groupby(level=0).aux.shift(1)
        x["y_lag"] = x.groupby(level=0).y.shift(1)
        x = x.dropna()
        is_early = x.index.get_level_values(0).isin(early)
        r2 = {}
        for cols in (["y_lag"], ["y_lag", "aux_lag"]):
            A = np.c_[np.ones(is_early.sum()), x.loc[is_early, cols]]
            w = np.linalg.lstsq(A, x.y[is_early], rcond=None)[0]
            err = x.y[~is_early] - np.c_[np.ones((~is_early).sum()), x.loc[~is_early, cols]] @ w
            r2["+".join(cols)] = 1 - (err ** 2).sum() / ((x.y[~is_early] - x.y[~is_early].mean()) ** 2).sum()
        out[f"{CFG['aux_symbol']} trades one second earlier"] = {
            "same_second_correlation": x.y.corr(x.aux), "r2_own_lag_only": r2["y_lag"], "r2_with_futures_lag": r2["y_lag+aux_lag"]}
    save("fit_dropped_variants", pd.DataFrame(out).T)
    return len(out) + 1


def stage_fit():
    fills, counts = flow.load_fills(paths(CFG["tick_files"]), CFG["symbol"], end_day=CFG["fit_end"])
    raw, orders = flow.to_orders(fills, 0), flow.to_orders(fills, CFG["merge_ms"])
    background(fills, counts, raw, orders)
    variants, early, late = tuning_grid(fills)
    variants += dropped_variants(fills, early, late)
    model = fit_models(raw, orders)
    variants += len(model["arrivals"]) + len(model["sizes"]) + len(model["direction"])
    # direction on unmerged orders, to show what merging removes
    d_raw = Direction(raw.sgn.values, raw.day.values)
    save("fit_direction_unmerged", pd.DataFrame({"depends on previous order, unmerged": d_raw.score(raw.sgn.values, raw.day.values)
                                                 | {"p_after_buy": d_raw.p_after_buy, "p_after_sell": d_raw.p_after_sell}}).T)
    variants += 1
    _, _, _, resid = score_all(model, raw, orders, "fit")
    save("variants", {"model_variants_this_run": variants})
    gap_figure(resid, "fig4_gaps_fit", "Figure 4. Gaps between orders on the fit days, actual against each model")
    market_maker_view(model, orders, "fit")
    return model


# ---------------------------------------------------------------- validation stage

def gap_figure(resid, name, title):
    plt.figure(figsize=(6.4, 3.0))
    x = np.logspace(-4, 1.2, 300)
    plt.plot(x, 1 - np.exp(-x), color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="A correct model")
    for key, colour, lab in (("constant", ORANGE, "Baseline: constant rate"),
                             ("time of day + self-exciting", BLUE, "Refined: time of day + self-exciting")):
        z = np.sort(resid[key])
        plt.plot(z, np.arange(1, len(z) + 1) / len(z), color=colour, label=lab)
    plt.xscale("log")
    plt.xlim(1e-4, 15)
    plt.xlabel("Gap between orders, rescaled by the model's predicted rate (1 = average)")
    plt.ylabel("Share of gaps shorter")
    plt.legend(frameon=False, fontsize=8, loc="upper left")
    plt.title(title)
    fig(name)


def markouts(orders, quotes):
    """Price move in the order's direction 10 seconds later, against the half-spread a quote earns."""
    rows = []
    for day, o in orders.groupby("day"):
        q = quotes[quotes.day == day]
        mid = pd.Series(q.mid.values, index=q.t.round().astype(int)).reindex(range(1, int(T) + 2)).ffill().values
        k = np.ceil(o.t.values).astype(int) + 10
        ok = k <= int(T)
        later = mid[np.minimum(k, int(T)) - 1]
        o = o.assign(move=o.sgn * (later - o.mid) / o.mid * 1e4, half_spread=(o.ask - o.bid) / 2 / o.mid * 1e4)[ok]
        rows.append(o[["day", "size", "move", "half_spread"]])
    m = pd.concat(rows).dropna()
    m["bucket"] = pd.cut(m["size"], [0, 9, 99, 10**9], labels=["1 to 9", "10 to 99", "100+"])
    day_mean = lambda g: g.groupby("day")[["half_spread", "move"]].mean()
    out = {}
    for name, g in [("all orders", m)] + [(f"{b} shares", m[m.bucket == b]) for b in m.bucket.cat.categories]:
        dm = day_mean(g)
        net = dm.half_spread - dm.move
        out[name] = {"orders": len(g), "half_spread_earned_bps": dm.half_spread.mean(), "price_move_after_10s_bps": dm.move.mean(),
                     "net_to_quote_bps": net.mean(), "net_se_across_days": net.std() / np.sqrt(len(net))}
    return pd.DataFrame(out).T


def market_maker_view(model, orders, label):
    """Burstiness and inventory pressure, actual against simulated from each model, plus what a quote earns when hit."""
    arr, sizes, dirs = build(model)
    real = pd.DataFrame([flow.flow_stats(g.t.values, g.sgn.values, g["size"].values) for _, g in orders.groupby("day")]).mean()
    base = flow.simulate_flow(arr["constant"], sizes["smooth curve"], dirs["coin flip"], CFG["sim_days"], CFG["seed"])
    refined = flow.simulate_flow(arr["time of day + self-exciting"], sizes["smooth curve + round-number spikes"],
                                 dirs["depends on previous order"], CFG["sim_days"], CFG["seed"])
    save(f"{label}_market_maker_view", pd.DataFrame({"actual": real, "baseline model": base, "refined model": refined}))
    days = sorted(orders.day.unique())
    quotes = flow.load_second_quotes(paths(CFG["quote_files"]), start_day=str(days[0]), end_day=str(days[-1]))
    save(f"{label}_markouts", markouts(orders, quotes))


def stage_validate():
    model = json.loads((OUT / "model.json").read_text())
    fills, _ = flow.load_fills(paths(CFG["tick_files"]), CFG["symbol"])
    fills = fills[fills.day > pd.Timestamp(CFG["fit_end"]).date()].reset_index(drop=True)
    raw, orders = flow.to_orders(fills, 0), flow.to_orders(fills, model["merge_ms"])
    print("validation days", sorted(orders.day.unique()))
    a, s, d, resid = score_all(model, raw, orders, "validation")
    gap_figure(resid, "fig4b_gaps_validation", "Gaps between orders on the validation days, actual against each model")
    arr, sizes, dirs = build(model)

    actual = half_hour_counts(orders)
    plt.figure(figsize=(6.4, 3.0))
    plt.bar(range(B), actual.mean(), width=0.72, color=BLUE, label="Actual, average validation day")
    plt.plot(range(B), arr["time of day"].expected_counts(), color=ORANGE, marker="o", markersize=4, label="Refined: time of day")
    plt.plot(range(B), arr["constant"].expected_counts(), color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="Baseline: constant rate")
    plt.xticks(range(B), HALF_HOURS, fontsize=7.5)
    plt.grid(axis="x", visible=False)
    plt.ylabel("Orders per half hour")
    plt.legend(frameon=False, fontsize=8, loc="upper center")
    plt.title("Figure 3. Predicted against actual orders per half hour, validation days")
    fig("fig3_predicted_vs_actual")

    market_maker_view(model, orders, "validation")


# ---------------------------------------------------------------- six-month check

def stage_sixmonth():
    """Share of seconds with a trade, by half hour, predicted from the tick-data model."""
    model = json.loads((OUT / "model.json").read_text())
    arr, _, _ = build(model)
    q = flow.load_second_quotes(paths(CFG["quote_files"]))
    q["b"] = flow.bucket(q.t.values - 1e-9)
    actual = q.groupby(["day", "b"]).traded.mean().unstack()
    rng = np.random.default_rng(CFG["seed"])
    sim = np.zeros(B)
    for _ in range(CFG["sim_days"]):
        t = arr["time of day + self-exciting"].simulate(rng)
        hit = np.bincount(t.astype(int), minlength=int(T)) > 0
        sim += np.array([hit[int(flow.EDGES[b]):int(flow.EDGES[b + 1])].mean() for b in range(B)])
    pred = {"baseline: constant rate": np.full(B, 1 - np.exp(-arr["constant"].shape[0])),
            "refined: time of day + self-exciting": sim / CFG["sim_days"]}
    level = actual.mean(axis=1)
    prev = level.shift(1) / pred["refined: time of day + self-exciting"].mean()
    scaled = pd.DataFrame(np.outer(prev, pred["refined: time of day + self-exciting"]), index=actual.index, columns=actual.columns)
    build_end = pd.Timestamp(CFG["quote_build_end"]).date()
    rows = {}
    for name, days in (("build period", actual.index[actual.index <= build_end]), ("test period", actual.index[actual.index > build_end])):
        a = actual.loc[days]
        r = {"days": len(a), "actual share of seconds with a trade": a.mean().mean(),
             "lowest day": a.mean(axis=1).min(), "highest day": a.mean(axis=1).max()}
        for k, p in pred.items():
            r[f"error, {k}"] = float((np.abs(a - p) / a).mean().mean())
        r["error, refined scaled by the previous day's activity"] = float((np.abs(a - scaled.loc[days]) / a).iloc[1:].mean().mean())
        r["time-of-day shape correlation"] = float(np.corrcoef(a.mean(), pred["refined: time of day + self-exciting"])[0, 1])
        r["day-to-day persistence of activity"] = float(a.mean(axis=1).autocorr())
        rows[name] = r
    save("sixmonth_activity", pd.DataFrame(rows))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("fit", "all"):
        stage_fit()
    if stage in ("validate", "all"):
        stage_validate()
    if stage in ("sixmonth", "all"):
        stage_sixmonth()
