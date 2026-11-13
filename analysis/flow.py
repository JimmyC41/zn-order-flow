"""Data loading and order flow models: arrivals, sizes and direction.

Inputs are Databento trades-with-quotes (tbbo) Parquet files for one futures contract.
Times are seconds since the session open, New York time.
"""
import glob
import math

import duckdb
import numpy as np
import pandas as pd
from numba import njit
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
from scipy import optimize, stats

TZ = "America/New_York"
OPEN_S = CLOSE_S = T = B = BUCKET_S = EDGES = DUR = None


def configure(open_hm, close_hm, bucket_s):
    global OPEN_S, CLOSE_S, T, B, BUCKET_S, EDGES, DUR
    secs = lambda hm: int(hm[:2]) * 3600 + int(hm[3:]) * 60
    OPEN_S, CLOSE_S, BUCKET_S = secs(open_hm), secs(close_hm), bucket_s
    T = float(CLOSE_S - OPEN_S)
    B = int(T // bucket_s)
    EDGES = np.arange(B + 1) * float(bucket_s)
    DUR = np.diff(EDGES)


def bucket(t):
    return np.minimum((np.asarray(t) // BUCKET_S).astype(int), B - 1)


# ---------------------------------------------------------------- loading

def _files(patterns):
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No files match {patterns}")
    return files


def _ns(day):
    return int(pd.Timestamp(day, tz=TZ).value)


def _add_session_time(df):
    """Add the New York date and seconds since midnight, exact to the nanosecond."""
    local = pd.to_datetime(df.ts, utc=True).dt.tz_convert(TZ)
    df["day"] = local.dt.date
    df["tod"] = (df.ts - local.dt.normalize().dt.tz_convert("UTC").astype("int64")) / 1e9
    return df


def skipped_days(patterns):
    """Days left out of every sample: US holidays, the early closes around Thanksgiving and Christmas,
    and the two trading days before each quarterly contract roll, when trading moves to the next contract."""
    q = f"""select instrument_id iid, min(epoch_ns(ts_event)) first_ts, max(epoch_ns(ts_event)) last_ts
            from read_parquet({_files(patterns)!r}) where action = 'T' group by 1 order by 2"""
    c = duckdb.sql(q).df()
    stamp = lambda ns: pd.Timestamp(ns, tz="UTC").tz_convert(TZ)
    first, last = stamp(c.first_ts.min()).normalize().tz_localize(None), stamp(c.last_ts.max()).normalize().tz_localize(None)
    hol = set(USFederalHolidayCalendar().holidays(first, last)) | set(GoodFriday.dates(first, last))
    hol |= {d + pd.Timedelta(days=1) for d in hol if d.month == 11 and d.day >= 22}      # early close
    hol |= {d - pd.Timedelta(days=1) for d in hol if d.month == 12 and d.day == 25}      # early close
    open_days = [d for d in pd.bdate_range(first, last) if d not in hol]
    skip = set(hol)
    for ns in c.first_ts.iloc[1:]:
        s = stamp(ns)                             # the evening session belongs to the next trade date
        switch = (s + pd.Timedelta(hours=7)).normalize().tz_localize(None)
        skip |= set([d for d in open_days if d < switch][-2:])
    return {d.date() for d in skip}


def load_trades(patterns, start_day, end_day, skip=()):
    """Trades in the session from start_day to end_day inclusive, signed by the exchange's aggressor side."""
    q = f"""select epoch_ns(ts_event) ts, price px, size::double size, side,
                   bid_px_00 bid, ask_px_00 ask, bid_sz_00::double bsz, ask_sz_00::double asz
            from read_parquet({_files(patterns)!r})
            where action = 'T' and epoch_ns(ts_event) >= {_ns(start_day)}
              and epoch_ns(ts_event) < {_ns(pd.Timestamp(end_day) + pd.Timedelta(days=1))}
            order by ts_recv, sequence"""
    f = _add_session_time(duckdb.sql(q).df())
    f = f[(f.tod >= OPEN_S) & (f.tod < CLOSE_S) & ~f.day.isin(skip)]
    counts = {"session_trades": len(f), "no_exchange_side": int((f.side == "N").sum())}
    f = f[f.side != "N"].sort_values("ts", kind="stable").reset_index(drop=True)
    f["t"] = f.tod - OPEN_S
    f["sgn"] = np.where(f.side == "B", 1, -1)
    f["mid"] = (f.bid + f.ask) / 2
    return f.drop(columns=["side", "tod"]), counts


def to_orders(trades, merge_ns):
    """Merge same-side trades no more than merge_ns apart into one order. The quote is the one before the first trade."""
    gap = trades.groupby("day").ts.diff()
    new = gap.isna() | (gap > merge_ns) | (trades.sgn != trades.sgn.shift())
    o = trades.groupby(new.cumsum()).agg(day=("day", "first"), ts=("ts", "first"), t=("t", "first"),
                                         sgn=("sgn", "first"), size=("size", "sum"), trades=("size", "size"),
                                         px=("px", "first"), mid=("mid", "first"), bid=("bid", "first"),
                                         ask=("ask", "first"), bsz=("bsz", "first"), asz=("asz", "first"))
    return o.reset_index(drop=True)


# ---------------------------------------------------------------- arrivals

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
    """Arrival rate c * shape(t) * news(t) + sum_k alpha_k * beta_k * exp(-beta_k * (t - t_j)).

    No alphas gives a Poisson process. alpha_k is the average number of orders one order
    triggers at decay speed beta_k (per second). sum(alpha) is the triggered share of orders.
    news maps a kind of release day to a multiplier per bucket.
    """

    def __init__(self, shape, c=1.0, alpha=(), beta=(), news=None):
        self.shape, self.c = np.asarray(shape, float), c
        self.alpha, self.beta = np.asarray(alpha, float), np.asarray(beta, float)
        self.news = {k: np.asarray(v, float) for k, v in (news or {}).items()}

    def base(self, kind=None):
        return self.c * self.shape * self.news.get(kind, 1.0)

    def day(self, t, kind=None):
        """Log-likelihood, rescaled gaps and the rate at each order for one day."""
        rate = self.base(kind)
        cum = np.concatenate([[0.0], np.cumsum(rate * DUR)])
        b = bucket(t)
        lam = rate[b]
        comp = cum[b] + rate[b] * (t - EDGES[b])
        integral = cum[-1]
        for a, be in zip(self.alpha, self.beta):
            A = _excite(t, be)
            lam = lam + a * be * A
            comp = comp + a * (np.arange(len(t)) - A)
            integral += a * (1 - np.exp(-be * (T - t))).sum()
        return np.log(lam).sum() - integral, np.diff(comp), lam

    def rate_seen(self, t, kind, delay):
        """Rate at each order using only the orders at least `delay` seconds earlier:
        what a trader with that reaction time can know."""
        lam = self.base(kind)[bucket(t)].copy()
        j = np.searchsorted(t, t - delay, side="right") - 1
        seen = j >= 0
        j = np.maximum(j, 0)
        for a, be in zip(self.alpha, self.beta):
            lam += np.where(seen, a * be * (_excite(t, be)[j] + 1) * np.exp(-be * (t - t[j])), 0.0)
        return lam

    def score(self, orders, kinds=None):
        """Log-likelihood per order, and fit error: the largest distance between the rescaled
        gaps and the exponential curve a correct model would produce (KS distance)."""
        kinds = kinds or {}
        ll, z = 0.0, []
        for d, g in orders.groupby("day"):
            l, zz, _ = self.day(g.t.values, kinds.get(d))
            ll += l
            z.append(zz)
        z = np.concatenate(z)
        return {"loglik_per_order": ll / len(orders), "fit_error": stats.kstest(z, "expon").statistic}, z

    def expected_counts(self, kind=None):
        return self.base(kind) * DUR / (1 - self.alpha.sum())

    def simulate(self, rng, kind=None):
        """One day of arrival times: untriggered orders first, then the orders they trigger."""
        n = rng.poisson(self.base(kind) * DUR)
        gen = np.concatenate([EDGES[b] + rng.random(n[b]) * DUR[b] for b in range(B)])
        out = [gen]
        while len(gen) and len(self.alpha):
            kids = []
            for a, be in zip(self.alpha, self.beta):
                k = rng.poisson(a, len(gen))
                kids.append(np.repeat(gen, k) + rng.exponential(1 / be, k.sum()))
            gen = np.concatenate(kids)
            gen = gen[gen < T]
            out.append(gen)
        return np.sort(np.concatenate(out))


def fit_self_exciting(orders, shape, news, kinds, starts):
    """Maximum likelihood fit of the decay speeds on top of a time-of-day shape.
    starts is a list of (c, alphas, betas) starting points. The best fit wins."""
    days = [(g.t.values, kinds.get(d)) for d, g in orders.groupby("day")]
    k = len(starts[0][1])

    def unpack(theta):
        return math.exp(theta[0]), np.exp(theta[1:1 + k]), np.exp(theta[1 + k:])

    def nll(theta):
        c, alpha, beta = unpack(theta)
        if alpha.sum() >= 0.999:
            return 1e15
        m = Arrivals(shape, c, alpha, beta, news)
        return -sum(m.day(t, kind)[0] for t, kind in days)

    fits = [optimize.minimize(nll, np.log(np.concatenate([[c], a, b])), method="Nelder-Mead",
                              options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1.0}) for c, a, b in starts]
    c, alpha, beta = unpack(min(fits, key=lambda o: o.fun).x)
    order = np.argsort(-beta)
    return Arrivals(shape, c, alpha[order], beta[order], news), [float(f.fun) for f in fits]


# ---------------------------------------------------------------- sizes

TOP = 200_000   # largest order size the size curves cover


def _cells(lo, hi, mu, sigma):
    """Lognormal probability of each whole size from lo to hi, a size k covering k - 0.5 to k + 0.5."""
    edges = np.arange(lo - 0.5, hi + 1)
    return np.diff(stats.norm.cdf((np.log(np.maximum(edges, 1e-12)) - mu) / sigma))


class Sizes:
    """Whole-contract order sizes. A lognormal curve, optionally with extra probability at
    chosen sizes (spikes) and a second lognormal curve for the tail above a cutoff."""

    def __init__(self, mu, sigma, p_spike=None, cutoff=None, p_tail=0.0, tail_mu=None, tail_sigma=None):
        self.mu, self.sigma = mu, sigma
        self.p_spike = {int(k): v for k, v in (p_spike or {}).items()}
        self.cutoff, self.p_tail, self.tail_mu, self.tail_sigma = cutoff, p_tail, tail_mu, tail_sigma
        top = int(cutoff) if cutoff else TOP
        body = _cells(1, top, mu, sigma)
        if not self.p_spike and not cutoff:
            body[0] = stats.norm.cdf((math.log(1.5) - mu) / sigma)    # everything below 1.5 is one contract
        body[[s - 1 for s in self.p_spike]] = 0.0
        pmf = body / body.sum() * (1 - sum(self.p_spike.values()) - p_tail)
        for s, p in self.p_spike.items():
            pmf[s - 1] = p
        if cutoff:
            tail = _cells(top + 1, TOP, tail_mu, tail_sigma)
            pmf = np.concatenate([pmf, tail / tail.sum() * p_tail])
        self._pmf = pmf / pmf.sum()

    @classmethod
    def fit(cls, sizes, spikes=(), tail_share=None):
        sizes = np.asarray(sizes)
        if not spikes and not tail_share:
            return cls(float(np.log(sizes).mean()), float(np.log(sizes).std()))
        cutoff = int(np.quantile(sizes, 1 - tail_share)) if tail_share else None
        spikes = [s for s in spikes if not cutoff or s <= cutoff]

        def curve(x, lo, hi, skip=()):
            """Maximum likelihood lognormal for sizes limited to lo..hi, leaving out the spike sizes."""
            vals, n = np.unique(x, return_counts=True)
            keep = ~np.isin(np.arange(lo, hi + 1), skip)

            def nll(th):
                p = _cells(lo, hi, th[0], math.exp(th[1]))
                return -(n * np.log(np.clip(p[vals.astype(int) - lo], 1e-300, None))).sum() + n.sum() * math.log(p[keep].sum())

            th = optimize.minimize(nll, [np.log(x).mean(), math.log(np.log(x).std() + 0.1)], method="Nelder-Mead").x
            return float(th[0]), float(math.exp(th[1]))

        body = sizes[~np.isin(sizes, spikes) & (sizes <= (cutoff or TOP))]
        mu, sigma = curve(body, 1, cutoff or TOP, spikes)
        p_spike = {s: float((sizes == s).mean()) for s in spikes}
        if not cutoff:
            return cls(mu, sigma, p_spike)
        tail = sizes[sizes > cutoff]
        return cls(mu, sigma, p_spike, cutoff, float(len(tail) / len(sizes)), *curve(tail, cutoff + 1, TOP))

    def params(self):
        return {"mu": self.mu, "sigma": self.sigma, "p_spike": self.p_spike, "cutoff": self.cutoff,
                "p_tail": self.p_tail, "tail_mu": self.tail_mu, "tail_sigma": self.tail_sigma}

    def logpmf(self, k):
        return np.log(np.clip(self._pmf[np.minimum(np.asarray(k), TOP).astype(int) - 1], 1e-300, None))

    def sample(self, rng, n):
        return np.searchsorted(np.cumsum(self._pmf), rng.random(n)).clip(0, TOP - 1) + 1.0

    def score(self, sizes, marks=(10, 100, 1000)):
        sizes = np.asarray(sizes)
        k = np.arange(1, TOP + 1)
        out = {"loglik_per_order": float(self.logpmf(sizes).mean()),
               "mean_actual": float(sizes.mean()), "mean_model": float((k * self._pmf).sum()),
               "sd_actual": float(sizes.std()), "sd_model": float(np.sqrt((k ** 2 * self._pmf).sum() - (k * self._pmf).sum() ** 2))}
        for m in marks:
            out[f"share_{m}_plus_actual"] = float((sizes >= m).mean())
            out[f"share_{m}_plus_model"] = float(self._pmf[m - 1:].sum())
        return out


# ---------------------------------------------------------------- direction

@njit(cache=True)
def _signs(u, a, b, log_size):
    """Signs where the chance of a buy is logistic(a + (b0 + b1 * log previous size) * previous sign)."""
    out = np.empty(len(u), np.int64)
    last, last_log = 1, 0.0
    for i in range(len(u)):
        x = a + (b[0] + b[1] * last_log) * last
        last = 1 if u[i] < 1 / (1 + math.exp(-x)) else -1
        last_log = log_size[i]
        out[i] = last
    return out


class Direction:
    """Chance the next order is a buy, from a logistic model on the features given.
    No features is a fixed chance."""

    def __init__(self, w):
        self.w = np.asarray(w, float)

    @classmethod
    def fit(cls, X, buy):
        X = np.c_[np.ones(len(buy)), X]
        w = np.zeros(X.shape[1])
        for _ in range(50):                       # Newton steps
            p = 1 / (1 + np.exp(-X @ w))
            step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (buy - p))
            w += step
            if np.abs(step).max() < 1e-8:
                break
        return cls(w)

    def p_buy(self, X):
        return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ self.w)))

    def score(self, X, buy):
        p = self.p_buy(X)
        return {"log_loss": float(-np.mean(np.where(buy, np.log(p), np.log(1 - p)))),
                "accuracy": float(np.mean((p > 0.5) == buy))}

    def sample(self, rng, sizes):
        """Simulate signs for orders of the given sizes. Uses the weights on the previous sign and,
        if present, on the previous sign times the log of the previous size."""
        b = np.zeros(2)
        b[:len(self.w) - 1] = self.w[1:3]
        return _signs(rng.random(len(sizes)), self.w[0], b, np.log(sizes))


# ---------------------------------------------------------------- flow a market maker faces

def flow_stats(t, sgn, size):
    per_s = np.bincount(t.astype(int), minlength=int(T))
    m = (t // 60).astype(int)
    n_min = int(T // 60)
    net = np.bincount(m, weights=sgn * size, minlength=n_min)
    return {"orders": len(t), "busiest_second": per_s.max(),
            "net_per_min_sd": net.std(), "net_per_min_p99": np.quantile(np.abs(net), 0.99)}


def simulate_flow(arrivals, sizes, direction, n_days, seed, kind=None):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_days):
        t = arrivals.simulate(rng, kind)
        size = sizes.sample(rng, len(t))
        rows.append(flow_stats(t, direction.sample(rng, size), size))
    return pd.DataFrame(rows)
