"""Conditional count forecasts from a frozen exponential Hawkes model.

Times are measured in seconds since session open.
"""
from dataclasses import dataclass
from functools import lru_cache
import bisect
import math


def _eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def _add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def _scale(a, s):
    return [[x * s for x in row] for row in a]


def _mul(a, b):
    return [[math.fsum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def _mv(a, x):
    return [math.fsum(v * w for v, w in zip(row, x)) for row in a]


def _norm(a):
    return max(math.fsum(abs(a[i][j]) for i in range(len(a))) for j in range(len(a)))


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    rate: float


class ConditionalHawkes:
    """q_k = alpha_k * sum_{observed i} exp(-beta_k * (cutoff-t_i)).

    Conditional E[q] obeys q' = Bq + alpha*mu, with
    B = -diag(beta) + alpha beta^T. Coefficients must be nonnegative, subcritical.
    Small dense matrix operations are intended for a few exponential kernels.
    """
    def __init__(self, alpha, beta):
        self.alpha = tuple(float(x) for x in alpha)
        self.beta = tuple(float(x) for x in beta)
        if not self.alpha or len(self.alpha) != len(self.beta):
            raise ValueError('Require equally sized nonempty alpha and beta arrays')
        if any(not math.isfinite(x) or x < 0 for x in self.alpha):
            raise ValueError('alpha must be finite and nonnegative')
        if any(not math.isfinite(x) or x <= 0 for x in self.beta):
            raise ValueError('beta must be finite and positive')
        self.n = math.fsum(self.alpha)
        if self.n >= 1:
            raise ValueError('This module requires a subcritical model: sum(alpha)<1')
        self.B = [[a*b - (self.beta[i] if i == j else 0.0)
                   for j, b in enumerate(self.beta)] for i, a in enumerate(self.alpha)]

    @lru_cache(maxsize=256)
    def operators(self, duration):
        """Return E=exp(Bh), H=int E, J=int (h-s)E(s) ds.

        Taylor series at norm<=0.5, then scaling/squaring with integral identities.
        This avoids subtracting near-equal eigenvalues or subtracting I from E.
        B is Metzler with nonpositive column sums, so E is nonnegative/contractive.
        """
        h = float(duration)
        if not math.isfinite(h) or h < 0:
            raise ValueError('duration must be finite and nonnegative')
        n = len(self.alpha)
        if h == 0:
            z = _scale(_eye(n), 0.0)
            return _eye(n), z, z
        norm = _norm(self.B) * h
        squarings = max(0, math.ceil(math.log2(norm / .5))) if norm > .5 else 0
        step = h / 2**squarings
        a = _scale(self.B, step)
        term = _eye(n)
        e = _eye(n)
        hh = _scale(_eye(n), step)
        jj = _scale(_eye(n), step*step/2)
        for k in range(1, 80):
            term = _scale(_mul(term, a), 1/k)
            e = _add(e, term)
            hh = _add(hh, _scale(term, step/(k+1)))
            jj = _add(jj, _scale(term, step*step/((k+1)*(k+2))))
            if _norm(term) < 2e-17:
                break
        else:
            raise ArithmeticError('Matrix series failed to converge')
        for _ in range(squarings):
            jj = _add(_add(jj, _scale(hh, step)), _mul(e, jj))
            hh = _add(hh, _mul(e, hh))
            e = _mul(e, e)
            step *= 2
        return e, hh, jj

    def propagate(self, state, duration, baseline):
        """Return conditional end-state and expected arrivals over this interval."""
        if len(state) != len(self.alpha) or any(not math.isfinite(x) or x < 0 for x in state):
            raise ValueError('state must contain one finite nonnegative q per kernel')
        if not math.isfinite(baseline) or baseline < 0:
            raise ValueError('baseline must be finite and nonnegative')
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('duration must be finite and nonnegative')
        # Rounding only caches nominal durations to 1 ps, below the input ns resolution.
        h = round(float(duration), 12)
        e, hh, jj = self.operators(h)
        end = [x + baseline*y for x, y in zip(_mv(e, state), _mv(hh, self.alpha))]
        integrated = [x + baseline*y for x, y in zip(_mv(hh, state), _mv(jj, self.alpha))]
        count = baseline*h + math.fsum(b*x for b, x in zip(self.beta, integrated))
        if min(end) < -1e-11 or count < -1e-11:
            raise ArithmeticError('Negative conditional moments from numerical propagation')
        return tuple(max(0.0, x) for x in end), max(0.0, count)

    def state_from_history(self, times, cutoff, session_start=0.0):
        """Observed timestamps <=cutoff only, zero excitation at session_start.

        Times must be sorted and finite. Supply already reconstructed arrival times;
        no completed order size is used. Passing any future timestamp is an error.
        """
        times = list(times)
        if not math.isfinite(session_start) or not math.isfinite(cutoff) or cutoff < session_start:
            raise ValueError('Invalid observation cutoff')
        if any(not math.isfinite(t) or t < session_start or t > cutoff for t in times):
            raise ValueError('History must lie between session_start and cutoff')
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError('History must be sorted')
        return tuple(a*math.fsum(math.exp(-b*(cutoff-t)) for t in times)
                     for a, b in zip(self.alpha, self.beta))

    def history_states(self, times, cutoffs, session_start=0.0):
        """Streaming states at sorted cutoffs. Future events are never incorporated.

        This API accepts the full session array for efficient historical evaluation,
        but yields each state using only its prefix. No lookahead into event sizes.
        """
        if not math.isfinite(session_start) or any(not math.isfinite(t) or t < session_start for t in times):
            raise ValueError('Invalid event timestamp')
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError('Events must be sorted')
        cutoffs = list(cutoffs)
        if any(not math.isfinite(t) or t < session_start for t in cutoffs):
            raise ValueError('Invalid cutoff')
        if any(a > b for a, b in zip(cutoffs, cutoffs[1:])):
            raise ValueError('Cutoffs must be sorted')
        q, last, j = [0.0]*len(self.alpha), session_start, 0
        for cutoff in cutoffs:
            while j < len(times) and times[j] <= cutoff:
                event = times[j]
                q = [x*math.exp(-b*(event-last))+a for x, a, b in zip(q, self.alpha, self.beta)]
                last, j = event, j+1
            q = [x*math.exp(-b*(cutoff-last)) for x, b in zip(q, self.beta)]
            last = cutoff
            yield tuple(q)

    def forecast(self, state, cutoff, start, end, schedule):
        """Predict intensity at start and events in [start,end), given cutoff history.

        Baseline changes between cutoff, start and end are propagated explicitly.
        No conditioning on realized future events or knowledge that an event occurs
        at start. Use fixed-clock anchors when empirically evaluating this forecast.
        """
        if not all(math.isfinite(t) for t in (cutoff, start, end)) or not cutoff <= start <= end:
            raise ValueError('Require finite cutoff <= start <= end')
        # For zero lag the initial state already contains observed events at start;
        # the forecast then counts subsequent arrivals only.
        q, _ = advance(self, state, cutoff, start, schedule)
        intensity = baseline_at(schedule, start) + math.fsum(b*x for b, x in zip(self.beta, q))
        qend, count = advance(self, q, start, end, schedule)
        return {'conditional_intensity_at_start': intensity,
                'expected_events_in_interval': count,
                'state_at_start': q, 'state_at_end': qend}


def validate_schedule(schedule):
    if not schedule:
        raise ValueError('Empty baseline schedule')
    for i, s in enumerate(schedule):
        if not all(math.isfinite(x) for x in (s.start, s.end, s.rate)) or s.end <= s.start or s.rate < 0:
            raise ValueError('Invalid baseline segment')
        if i and abs(s.start-schedule[i-1].end) > 1e-10:
            raise ValueError('Baseline segments must be ordered and contiguous')
    return tuple(schedule)


def baseline_at(schedule, t):
    i = bisect.bisect_right([s.start for s in schedule], t)-1
    if i < 0 or t >= schedule[i].end:
        raise ValueError('Time outside baseline schedule (right endpoints excluded)')
    return schedule[i].rate


def advance(model, state, start, end, schedule):
    if end < start or start < schedule[0].start or end > schedule[-1].end:
        raise ValueError('Forecast interval outside baseline schedule')
    if end == start:
        return model.propagate(state, 0.0, 0.0)
    q, total, cursor = tuple(state), 0.0, start
    i = max(0, bisect.bisect_right([s.start for s in schedule], start)-1)
    while cursor < end:
        segment = schedule[i]
        stop = min(end, segment.end)
        if stop <= cursor:
            raise ValueError('Missing baseline coverage')
        q, count = model.propagate(q, stop-cursor, segment.rate)
        total += count
        cursor, i = stop, i+1
    return q, total


def baseline_integral(schedule, start, end):
    if start < schedule[0].start or end > schedule[-1].end or end < start:
        raise ValueError('Interval outside schedule')
    return math.fsum(s.rate*max(0.0,min(s.end,end)-max(s.start,start)) for s in schedule)


def direct_history_intensity(model, state, cutoff, at, schedule):
    """Existing rate_seen-style feature in intensity units; not a conditional mean."""
    if at < cutoff:
        raise ValueError('at must be >= cutoff')
    return baseline_at(schedule, at)+math.fsum(b*q*math.exp(-b*(at-cutoff)) for b,q in zip(model.beta,state))


def direct_history_count(model, state, cutoff, start, end, schedule):
    """Integrate baseline and decaying excitation from observed history only."""
    if not cutoff <= start <= end:
        raise ValueError('Require cutoff <= start <= end')
    return baseline_integral(schedule,start,end)+math.fsum(
        q*math.exp(-b*(start-cutoff))*(-math.expm1(-b*(end-start)))
        for b,q in zip(model.beta,state))


def from_saved_model(saved, config, kind='none'):
    key = 'time of day + news + self-exciting'
    p = saved['arrivals'][key]
    model = ConditionalHawkes(p['alpha'], p['beta'])
    multipliers = p.get('news',{}).get(kind,[1.0]*len(p['shape']))
    if kind != 'none' and kind not in p.get('news',{}):
        raise ValueError('Unknown announcement category')
    if len(multipliers) != len(p['shape']):
        raise ValueError('Baseline shape and news multiplier lengths differ')
    bucket = config['bucket_seconds']
    to_seconds = lambda hm: int(hm[:2])*3600+int(hm[3:])*60
    duration = to_seconds(config['session_close'])-to_seconds(config['session_open'])
    if duration != bucket*len(p['shape']):
        raise ValueError('Saved shape does not cover the configured session')
    schedule = validate_schedule([Segment(i*bucket,(i+1)*bucket,p['c']*s*n)
                                  for i,(s,n) in enumerate(zip(p['shape'],multipliers))])
    return model,schedule
