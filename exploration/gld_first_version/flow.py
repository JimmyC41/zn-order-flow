"""Data loading and order flow models: arrivals, sizes and direction.

Inputs are any Parquet files in the course's cleaned top-of-book event layout.
Times are seconds since 09:30:01 New York time within one continuous session.
"""
import glob
import math

import duckdb
import numpy as np
import pandas as pd
from numba import njit
from scipy import optimize, stats

TZ = "America/New_York"
OPEN_S = 9 * 3600 + 30 * 60 + 1   # the first second holds the opening cross
CLOSE_S = 16 * 3600
T = float(CLOSE_S - OPEN_S)       # session length in seconds
B = 13                            # half-hour buckets
EDGES = np.minimum(np.arange(B + 1) * 1800.0, T)
DUR = np.diff(EDGES)
SIZE_SPIKES = [1, 100, 200, 500, 1000]


def bucket(t):
    return np.minimum((np.asarray(t) // 1800).astype(int), B - 1)


# ---------------------------------------------------------------- loading

def _files(patterns):
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No files match {patterns}")
    return files


def _day_end_ns(day):
    return int((pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=1)).value)


def _add_session_time(df, ts_col):
    """Add the New York session date and seconds since midnight, exact to the nanosecond."""
    local = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(TZ)
    df["day"] = local.dt.date
    midnight = local.dt.normalize().dt.tz_convert("UTC").astype("int64")
    df["tod"] = (df[ts_col] - midnight) / 1e9
    return df


def load_fills(patterns, symbol, end_day=None, start_day=None):
    """Trades for one symbol in the continuous session, with a buy or sell sign.

    Sign uses the exchange's aggressor side. Where it is missing, the trade price
    against the midpoint, then the previous price change.
    """
    where = ["is_trade", "symbol = ?", "not is_exact_duplicate_in_source"]
    if end_day:
        where.append(f"ts_event_ns < {_day_end_ns(end_day)}")
    if start_day:
        where.append(f"ts_event_ns >= {int(pd.Timestamp(start_day, tz=TZ).value)}")
    q = f"""select ts_event_ns ts, price_nano/1e9 px, size, side,
                   bid_px_00_nano/1e9 bid, ask_px_00_nano/1e9 ask
            from read_parquet({_files(patterns)!r})
            where {' and '.join(where)}
            order by ts_recv_ns, source_row_number"""
    f = duckdb.execute(q, [symbol]).df()
    f = _add_session_time(f, "ts")
    counts = {"regular_hours": int(((f.tod >= OPEN_S - 1) & (f.tod < CLOSE_S)).sum()),
              "opening_second": int(((f.tod >= OPEN_S - 1) & (f.tod < OPEN_S)).sum())}
    f = f[(f.tod >= OPEN_S) & (f.tod < CLOSE_S)].sort_values("ts", kind="stable").reset_index(drop=True)
    f["t"] = f.tod - OPEN_S
    f["mid"] = (f.bid + f.ask) / 2
    quote_rule = np.sign(f.px - f.mid)
    tick_rule = np.sign(f.groupby("day").px.diff()).replace(0, np.nan).groupby(f.day).ffill().fillna(0)
    f["sgn"] = np.where(f.side == "B", 1, np.where(f.side == "A", -1,
                        np.where(quote_rule != 0, quote_rule, tick_rule))).astype(int)
    counts["no_exchange_side"] = int((f.side == "N").sum())
    counts["signed_by_tick_rule"] = int(((f.side == "N") & (quote_rule == 0)).sum())
    counts["unsigned"] = int((f.sgn == 0).sum())
    f = f[f.sgn != 0].reset_index(drop=True)
    counts["continuous"] = len(f)
    return f, counts


def to_orders(fills, merge_ms):
    """Merge same-side fills no more than merge_ms apart into one order. 0 merges equal timestamps only."""
    gap = fills.groupby("day").ts.diff()
    new = gap.isna() | (gap > merge_ms * 1e6) | (fills.sgn != fills.sgn.shift())
    o = fills.groupby(new.cumsum()).agg(day=("day", "first"), ts=("ts", "first"), t=("t", "first"),
                                        sgn=("sgn", "first"), size=("size", "sum"), fills=("size", "size"),
                                        mid=("mid", "first"), bid=("bid", "first"), ask=("ask", "first"))
    return o.reset_index(drop=True)


def load_second_quotes(patterns, start_day=None, end_day=None):
    """One row per second in regular hours: midpoint and whether a trade printed in that second."""
    where = ["is_nyse_regular_interval", "quote_structurally_valid"]
    if end_day:
        where.append(f"ts_recv_ns < {_day_end_ns(end_day)}")
    if start_day:
        where.append(f"ts_recv_ns >= {int(pd.Timestamp(start_day, tz=TZ).value)}")
    q = f"""select ts_recv_ns tr, ts_event_ns te, (bid_px_00_nano+ask_px_00_nano)/2e9 mid
            from read_parquet({_files(patterns)!r}) where {' and '.join(where)} order by 1"""
    b = duckdb.sql(q).df()
    b = _add_session_time(b, "tr")
    b["traded"] = (b.te > b.tr - 1_000_000_000) & (b.te != b.groupby("day").te.shift(1))
    b = b[(b.tod > OPEN_S) & (b.tod <= CLOSE_S)]
    b["t"] = b.tod - OPEN_S
    return b


# ---------------------------------------------------------------- arrivals

def tod_rate(orders):
    """Orders per second in each half hour, averaged over the days given."""
    n_days = orders.day.nunique()
    return np.bincount(bucket(orders.t.values), minlength=B) / (n_days * DUR)


def _cum_rate(rate):
    cum = np.concatenate([[0.0], np.cumsum(rate * DUR)])
    def at(t):
        b = bucket(t)
        return cum[b] + rate[b] * (t - EDGES[b])
    return at, cum[-1]


@njit(cache=True)
def _excite(t, beta):
    """A[i] = sum over earlier events j of exp(-beta * (t[i] - t[j]))."""
    out = np.zeros(len(t))
    a = 0.0
    for i in range(1, len(t)):
        a = math.exp(-beta * (t[i] - t[i - 1])) * (1.0 + a)
        out[i] = a
    return out


class Arrivals:
    """Arrival model with rate c * shape(t) + sum_k alpha_k * beta_k * exp(-beta_k * (t - t_j)).

    No alphas gives a Poisson process. alpha_k is the average number of orders one order
    triggers at decay speed beta_k (per second). sum(alpha) is the triggered share of orders.
    """

    def __init__(self, shape, c=1.0, alpha=(), beta=()):
        self.shape, self.c = np.asarray(shape, float), c
        self.alpha, self.beta = np.asarray(alpha, float), np.asarray(beta, float)

    def _day(self, t):
        at, total = _cum_rate(self.shape)
        b = bucket(t)
        lam = self.c * self.shape[b]
        comp = self.c * at(t)
        integral = self.c * total
        for a, be in zip(self.alpha, self.beta):
            A = _excite(t, be)
            lam = lam + a * be * A
            comp = comp + a * (np.arange(len(t)) - A)
            integral += a * (1 - np.exp(-be * (T - t))).sum()
        return np.log(lam).sum() - integral, np.diff(comp)

    def score(self, orders):
        """Log-likelihood per order, and fit error: the largest distance between the
        rescaled gaps and the exponential curve a correct model would produce (KS distance)."""
        ll, z = 0.0, []
        for _, d in orders.groupby("day"):
            l, zz = self._day(d.t.values)
            ll += l
            z.append(zz)
        z = np.concatenate(z)
        return {"loglik_per_order": ll / len(orders), "fit_error": stats.kstest(z, "expon").statistic}, z

    def expected_counts(self):
        """Expected orders per half hour on an average day."""
        return self.c * self.shape * DUR / (1 - self.alpha.sum())

    def simulate(self, rng):
        """One day of arrival times, by drawing untriggered orders then their triggered descendants."""
        base = self.c * self.shape * DUR
        gen = np.concatenate([EDGES[b] + rng.random(rng.poisson(base[b])) * DUR[b] for b in range(B)])
        out = [gen]
        while len(gen) and len(self.alpha):
            kids = []
            for a, be in zip(self.alpha, self.beta):
                n = rng.poisson(a, len(gen))
                kids.append(np.repeat(gen, n) + rng.exponential(1 / be, n.sum()))
            gen = np.concatenate(kids)
            gen = gen[gen < T]
            out.append(gen)
        return np.sort(np.concatenate(out))


def fit_self_exciting(orders, shape, k):
    """Maximum likelihood fit of k decay speeds on top of a time-of-day shape."""
    days = [d.t.values for _, d in orders.groupby("day")]

    def nll(theta):
        c, alpha, beta = math.exp(theta[0]), np.exp(theta[1:1 + k]), np.exp(theta[1 + k:])
        if alpha.sum() >= 0.999:
            return 1e12
        m = Arrivals(shape, c, alpha, beta)
        return -sum(m._day(t)[0] for t in days)

    starts = {1: [[math.log(.5), math.log(.5), math.log(10.)], [math.log(.5), math.log(.4), math.log(1000.)]],
              2: [[math.log(.4), math.log(.3), math.log(.3), math.log(200.), math.log(.5)],
                  [math.log(.3), math.log(.4), math.log(.3), math.log(5000.), math.log(.05)]]}[k]
    best = min((optimize.minimize(nll, x0, method="Nelder-Mead",
                                  options={"maxiter": 4000, "xatol": 1e-4, "fatol": 0.1}) for x0 in starts),
               key=lambda o: o.fun)
    th = best.x
    order = np.argsort(-np.exp(th[1 + k:]))
    return Arrivals(shape, math.exp(th[0]), np.exp(th[1:1 + k])[order], np.exp(th[1 + k:])[order])


# ---------------------------------------------------------------- sizes

def _lognormal_logpmf(k, mu, sigma):
    hi = stats.norm.cdf((np.log(k + 0.5) - mu) / sigma)
    lo = stats.norm.cdf((np.log(np.maximum(k - 0.5, 1e-9)) - mu) / sigma)
    return np.log(np.clip(hi - lo, 1e-300, None))


class Sizes:
    """Whole-share lognormal, optionally with extra probability at round-number sizes."""

    def __init__(self, sizes, spikes=()):
        sizes = np.asarray(sizes)
        self.spikes = list(spikes)
        self.p_spike = {m: float((sizes == m).mean()) for m in self.spikes}
        rest = sizes[~np.isin(sizes, self.spikes)]
        self.mu, self.sigma = np.log(rest).mean(), np.log(rest).std()
        self._norm = 1 - np.exp(_lognormal_logpmf(np.array(self.spikes or [0.5]), self.mu, self.sigma)).sum() \
            if self.spikes else 1.0

    def logpmf(self, k):
        k = np.asarray(k)
        out = math.log(1 - sum(self.p_spike.values())) + _lognormal_logpmf(k, self.mu, self.sigma) - math.log(self._norm)
        for m, p in self.p_spike.items():
            out = np.where(k == m, math.log(p), out)
        return out

    def score(self, sizes):
        sizes = np.asarray(sizes)
        draw = self.sample(np.random.default_rng(0), 400_000)
        return {"loglik_per_order": float(self.logpmf(sizes).mean()),
                "share_100_plus_actual": float((sizes >= 100).mean()), "share_100_plus_model": float((draw >= 100).mean()),
                "share_1000_plus_actual": float((sizes >= 1000).mean()), "share_1000_plus_model": float((draw >= 1000).mean()),
                "mean_actual": float(sizes.mean()), "mean_model": float(draw.mean())}

    def sample(self, rng, n):
        body = np.maximum(np.rint(rng.lognormal(self.mu, self.sigma, n)), 1)
        if not self.spikes:
            return body
        while True:   # the body excludes the spike sizes
            bad = np.isin(body, self.spikes)
            if not bad.any():
                break
            body[bad] = np.maximum(np.rint(rng.lognormal(self.mu, self.sigma, bad.sum())), 1)
        pick = rng.choice(len(self.spikes) + 1, n, p=list(self.p_spike.values()) + [1 - sum(self.p_spike.values())])
        return np.where(pick < len(self.spikes), np.array(self.spikes + [0])[pick], body)


# ---------------------------------------------------------------- direction

class Direction:
    """Chance the next order is a buy. Coin flip, or dependent on the previous order."""

    def __init__(self, sgn=None, days=None):
        if sgn is None:
            self.p_after_buy = self.p_after_sell = 0.5
        else:
            prev, nxt = self._pairs(np.asarray(sgn), np.asarray(days))
            self.p_after_buy = float((nxt[prev == 1] == 1).mean())
            self.p_after_sell = float((nxt[prev == -1] == 1).mean())

    @staticmethod
    def _pairs(sgn, days):
        same = days[1:] == days[:-1]
        return sgn[:-1][same], sgn[1:][same]

    def score(self, sgn, days):
        prev, nxt = self._pairs(np.asarray(sgn), np.asarray(days))
        p = np.where(prev == 1, self.p_after_buy, self.p_after_sell)
        buy = nxt == 1
        return {"log_loss": float(-np.mean(np.where(buy, np.log(p), np.log(1 - p)))),
                "accuracy": float(np.mean((p > 0.5) == buy)) if self.p_after_buy != 0.5 else 0.5}

    def sample(self, rng, n):
        u = rng.random(n)
        out = np.empty(n, int)
        last = 1 if u[0] < 0.5 else -1
        pb, ps = self.p_after_buy, self.p_after_sell
        for i in range(n):
            last = 1 if u[i] < (pb if last == 1 else ps) else -1
            out[i] = last
        return out


# ---------------------------------------------------------------- flow a market maker faces

def flow_stats(t, sgn, size):
    """Burstiness and inventory pressure for one day of orders."""
    per_s = np.bincount(t.astype(int), minlength=int(T))
    m = (t // 60).astype(int)
    n_min = int(T // 60) + 1
    per_min = np.bincount(m, minlength=n_min)
    net = np.bincount(m, weights=sgn * size, minlength=n_min)
    return {"dispersion_1s": per_s.var() / per_s.mean(), "dispersion_1min": per_min.var() / per_min.mean(),
            "seconds_with_order": (per_s > 0).mean(), "busiest_second": per_s.max(),
            "net_per_min_sd": net.std(), "net_per_min_p99": np.quantile(np.abs(net), 0.99)}


def simulate_flow(arrivals, sizes, direction, n_days, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_days):
        t = arrivals.simulate(rng)
        rows.append(flow_stats(t, direction.sample(rng, len(t)), sizes.sample(rng, len(t))))
    return pd.DataFrame(rows).mean()
