# Principles for Designing, Backtesting and Evaluating a Trading Strategy

Strategy-agnostic rules to guide the choice of strategy and the method of the group project, ranked within each section by impact on a one-week project.

## Hypothesis and Edge

- **Write the hypothesis before touching the data**
  - Do: state in one or two sentences why the pattern should exist and why it should persist, before the first backtest. The Student Guide asks for "your Edge" and all key assumptions.
  - Prevents: a story written after the data mining, which "might easily have been the opposite" had the data said the opposite. E.g. a published study "supported" extrasensory perception at odds of 74 billion to 1 and failed to replicate.
  - Source: [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 1

- **Size the edge against the spread first**
  - Do: before coding, estimate the typical move you expect to capture per trade and compare it with the round-trip cost (full bid-ask spread plus fees). Drop ideas where the expected move is not clearly larger.
  - Prevents: an edge that only exists in prices nobody can trade at. E.g. a USO/DNO mean-reversion pair showed a Sharpe of about 1 after 5 bps per side on end-of-day prices, then lost money on nearly every trade on bid-ask quote data.
  - Source: [Chan, Really, Beware of Low Frequency Data](https://epchan.blogspot.com/2016/09/really-beware-of-low-frequency-data.html)

- **Prefer few well-informed trades to many weak ones**
  - Do: if the signal fires constantly, treat that as a warning and raise the entry threshold.
  - Prevents: overtrading. A high opportunity count usually means low information quality, more trades meet more toxic flow, and broad signals capture noise, not edge.
  - Source: Week 9 lecture, Adverse Selection

- **Pick a horizon your data can test**
  - Do: choose a holding period that yields hundreds of independent trades inside the in-sample period, not a handful. See the Sharpe error bar under Evaluation for why.
  - Prevents: a result that cannot be told apart from luck. Small sample size is listed as a core drawback of optimisation.
  - Source: Week 9 lecture, Drawbacks of optimisation. Inference for the "hundreds" figure.

## Data

- **Signal on the mid quote, fill at the bid or ask**
  - Do: compute signals from the bid-ask midpoint. Fill buys at the ask and sells at the bid. Avoid last-trade prices for signals.
  - Prevents: fake mean reversion. Trades bouncing between bid and ask create negative serial correlation in trade-price returns "even if no news hits the market". A reversal rule on trade prices harvests that bounce on paper only (inference).
  - Source: [Ødegaard, The Roll Spread Estimator](https://ba-odegaard.no/teach/notes/liquidity_estimators/roll_spread_estimator/roll_lectures.pdf), [Chan, Really, Beware of Low Frequency Data](https://epchan.blogspot.com/2016/09/really-beware-of-low-frequency-data.html)

- **Test fills on the finest data you have**
  - Do: build signals on bars if you like, but check fills against quote data at the finest resolution available.
  - Prevents: bars hiding quote flip-flops. A gold futures momentum strategy showed a Sharpe of about 1 and 18.7% APR on 1-minute bars, and negative returns on 1-millisecond quote data.
  - Source: [Chan, Beware of Low Frequency Data](https://epchan.blogspot.com/2015/04/beware-of-low-frequency-data.html)

- **Clean the data before any analysis**
  - Do: check for and log:
    - missing ticks and timestamp gaps
    - price jumps and crossed or zero-width quotes
    - symbol mismatches and vendor inconsistencies
    - session times and time zones of every series
  - Prevents: bad data producing wrong signals, bad fills and unintended risk. "A nonlinearity might simply be a bad data point."
  - Source: Week 9 lecture, Types of Risks (2) and Monitoring (2), [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 3b

- **Fix the sample period up front**
  - Do: choose start and end dates and the in-sample and out-of-sample boundary before the first backtest. Never move them or drop awkward days.
  - Prevents: date-shopping. If a model works when the sample starts in 1970 but not in 1960, the model does not work. Deleting crash periods because they hurt the model is the egregious version.
  - Source: [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 3a

- **Decide outlier rules before testing**
  - Do: pick one rule for extreme values (keep, winsorise at a stated level, or drop known bad prints) before seeing results, and check the result does not hinge on it.
  - Prevents: choosing the rule that works. E.g. a strategy that "works" at 5% winsorisation but fails at 1%, with 5% then chosen.
  - Source: [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 3d

## Backtest Design

- **Write every rule before the first backtest**
  - Do: specify entry, exit, position size and execution rules in full, as the Student Guide's trading plan asks.
  - Prevents: rules that drift to fit the backtest. Backtests are for discarding strategies, not improving them. "Never backtest until your model has been fully specified."
  - Source: [Palomar, Portfolio Optimization, 8.3](https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html), the Student Guide, Appendix A.4

- **Use only what was known at decision time**
  - Do: lag every signal so a decision at bar t trades at bar t+1 or later. Use as-of joins when merging feeds. Compute rolling statistics from past data only.
  - Prevents: look-ahead bias. A common form is multiplying weights built with data up to t by returns from t. A moving-average strategy lost its apparent performance once backtested properly.
  - Source: [Palomar, Portfolio Optimization, 8.2](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html)

- **Model the open, the close and the overnight gap**
  - Do: state whether positions are flat at the close. If held overnight, mark them at the next open you could actually trade. Treat the opening auction and pre-market rules as real constraints.
  - Prevents: phantom fills at the open. An opening-gap strategy that "looked wonderful" on midpoint prices was only mildly profitable live, partly because orders had to be sent using pre-market quotes before 9:28am ET.
  - Source: [Chan, Beware of Low Frequency Data](https://epchan.blogspot.com/2015/04/beware-of-low-frequency-data.html), Week 9 lecture, Time Series Observations (1)

- **Respect short-selling limits**
  - Do: if the strategy shorts, state the borrow assumption and cost, or restrict shorts to instruments you could borrow.
  - Prevents: assuming free, unlimited shorting. Borrow costs spiked in 2008 and some stocks were banned from shorting.
  - Source: [Palomar, Portfolio Optimization, 8.2](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html), the Student Guide, Appendix A.7

- **Make every number reproducible from one script**
  - Do: one script runs from raw data to every table and chart, with the same data fields, transformations, fees, slippage and latency throughout.
  - Prevents: in-sample and out-of-sample runs that differ in hidden ways, and results nobody can re-run. The Student Guide requires the code with the report.
  - Source: Week 9 lecture, Pre-Production

## Costs and Execution

- **Charge the full spread and fees on every trade**
  - Do: cross the spread on entry and exit, add commissions and fees, and report results net of costs in both in-sample and out-of-sample periods.
  - Prevents: an edge that is really a cost omission. Costs reduce profitability and significance in every anomaly studied. Few strategies with one-sided monthly turnover above 50% stay significant after costs. Intraday strategies turn over far faster (inference).
  - Source: [Novy-Marx and Velikov](https://www.nber.org/papers/w20721), [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 4c

- **Do not assume limit orders fill when price touches**
  - Do: if using limit orders, count a fill only when price trades through your level, and charge the missed trades at a later market price.
  - Prevents: counting only the good fills. Limit buys fill when price falls and are left behind when it rises, so filled limit orders look like they earn half the spread while the missed ones carry an unseen cost. Limit fills "coincide with adverse price movements".
  - Source: [Hasbrouck, Transaction costs notes](https://pages.stern.nyu.edu/~jhasbrou/Teaching/POST%202015%20Fall/classNotes/STPPTradingCosts.pdf), [DeLise, The Negative Drift of a Limit Order Fill](https://arxiv.org/abs/2407.16527)

- **Add latency between signal and fill**
  - Do: fill at the first quote after a stated delay, not at the quote that produced the signal.
  - Prevents: trading at prices that were gone by the time an order arrived. Latency assumptions must match between backtest and live.
  - Source: Week 9 lecture, Pre-Production

- **Report the break-even cost**
  - Do: rerun at double the assumed cost and report the cost per trade at which net profit hits zero.
  - Prevents: a result that only holds at one optimistic cost guess.
  - Source: inference, building on Week 9 lecture, Other considerations (transaction costs)

- **Check size against available liquidity**
  - Do: compare intended trade size with displayed depth and traded volume. Report capacity, as the Student Guide lists.
  - Prevents: assuming you can trade any size at the quote. The lecture contrasts trading $1 million against $1 billion of exposure and lists liquidity risk as the inability to trade at expected size or price.
  - Source: Week 9 lecture, Strategy Scaling and Types of Risks (1)

## Validation and Overfitting

- **Lock the test set**
  - Do: split by time, tune only on the earlier in-sample period, then run the later out-of-sample period once. Write down the out-of-sample result whatever it is.
  - Prevents: iterated out-of-sample. Revising a strategy after it fails out-of-sample and retesting "is no longer an out-of-sample test. It is overfitting."
  - Source: Week 9 lecture, Implementation of optimisation, [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 4b, [Harvey and Liu](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)

- **Log every configuration you try**
  - Do: keep a trial log of every parameter set, filter, data transformation and variable combination tested, and report the count in the report.
  - Prevents: hidden multiple testing. Of 20 random strategies, one will likely pass t = 2.0 by chance. About 20 iterations finds a false strategy at the 5% level. 20 variables allow 190 pairwise interactions. Not reporting the count "makes it impossible to assess the risk of overfitting".
  - Source: [Bailey, Borwein, López de Prado and Zhu](https://carmamaths.org/jon/backtest.pdf), [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 2, [López de Prado, The 10 Reasons Most ML Funds Fail](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf), Pitfall 2

- **Try few configurations on short data**
  - Do: cap the number of configurations tested. On a few months of data, expect even a small grid to produce an impressive in-sample Sharpe by chance.
  - Prevents: selecting a lucky configuration. Trying 10 configurations with zero true skill yields an expected best in-sample Sharpe of 1.57. With 5 years of data, no more than 45 configurations should be tried. 7 on/off parameters give 128 trials and an expected best Sharpe above 2.6. Scaling the same result to 3 months of data, 10 configurations give an expected best annualised Sharpe of about 3 (inference).
  - Source: [Bailey, Borwein, López de Prado and Zhu](https://carmamaths.org/jon/backtest.pdf)

- **Throw away parameters**
  - Do: remove any factor or parameter that does not clearly matter. Restrict the rest to a coarse grid.
  - Prevents: degrees of freedom that fit noise. A moving-average crossover with lookbacks from 1 to 256 has 65,280 combinations and is "very likely to overfit".
  - Source: Week 9 lecture, Implementation of optimisation, [Carver, The three kinds of overfitting](https://qoppac.blogspot.com/2015/11/the-three-kinds-of-overfitting.html)

- **Pick parameters from a plateau, not a peak**
  - Do: plot performance across a grid of two parameters at a time and choose the middle of a broad region where neighbours also do well.
  - Prevents: a sharp optimum that breaks with any change. The lecture locks in ranges (z1 from 4 to 6, z2 from 4 to 7) where the surface is high and flat, and flags local vs global optima and parameter elasticity as drawbacks.
  - Source: Week 9 lecture, Implementation of optimisation (by example) and Drawbacks of optimisation, [Palomar, Portfolio Optimization, 8.2](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html)

- **Size the split for both fitting and testing**
  - Do: give the out-of-sample period enough days to produce a meaningful trade count. If tuning needs a check, split the in-sample period again or use walk-forward windows that fit only on earlier data.
  - Prevents: an out-of-sample period too short to judge anything. With 1,000 observations, a 90-10 split leaves 100 held out, "difficult to separate the good from the bad". Walk-forward tests one historical path and can still be overfit.
  - Source: [Harvey and Liu](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF), [Carver, The three kinds of overfitting](https://qoppac.blogspot.com/2015/11/the-three-kinds-of-overfitting.html), [López de Prado, The 10 Reasons Most ML Funds Fail](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf), Pitfall 9

- **Haircut the Sharpe for the trials you ran**
  - Do: discount reported Sharpe ratios for the number of trials, or compute the deflated Sharpe ratio from the trial count, sample length, skew and kurtosis.
  - Prevents: reading a marginal Sharpe as real. With 200 tests, a Sharpe of 0.75 falls to 0.32, about 60%. Haircuts exceed 50% when the Sharpe is below 0.4 and are at most 25% above 1.0.
  - Source: [Harvey and Liu](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF), [Bailey and López de Prado, The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

- **Purge overlapping labels if using machine learning**
  - Do: when training a model on labels that span several bars, drop training rows whose label window overlaps the test window, plus a short buffer after it.
  - Prevents: leakage that makes irrelevant features look predictive. A buffer of about 1% of the bars "often suffices".
  - Source: [López de Prado, The 10 Reasons Most ML Funds Fail](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf), Pitfall 8

## Evaluation

- **Report the Sharpe with its error bar**
  - Do: report the standard error of the Sharpe ratio, SE = sqrt((1 + SR²/2) / T) with SR and T in the same period units, and say whether the result is distinguishable from zero.
  - Prevents: mistaking luck for skill. With 60 observations and a true Sharpe of 1.5, the standard error is 0.188 per period. A daily Sharpe of 0.1 over 60 trading days (about 1.6 annualised) has t of about 0.8, and a 1.5 annualised Sharpe needs roughly 450 trading days to reach t = 2 (inference from the formula).
  - Source: [Lo, The Statistics of Sharpe Ratios](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf), Week 9 lecture, Other considerations (skill or luck)

- **Annualise from daily PnL, not per-bar returns**
  - Do: aggregate to daily net PnL before computing an annual Sharpe, and check autocorrelation of the returns used.
  - Prevents: inflated annual Sharpe. Multiplying by the square root of the period count is valid only under special conditions, and serial correlation overstated one hedge fund's annual Sharpe by up to 65%. Intraday bar returns are often autocorrelated (inference).
  - Source: [Lo, The Statistics of Sharpe Ratios](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf)

- **Beat a dumb benchmark**
  - Do: compare against buy-and-hold over the same period and against random entries with the same trade count, holding time and costs. Report alpha, beta and correlation as the Student Guide lists.
  - Prevents: crediting the rule for market drift or chance. A rule trading a random binary sequence looked good until August 2018, then stopped working.
  - Source: [Palomar, Portfolio Optimization, 8.2](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html), the Student Guide, Appendix A.5

- **Show PnL through time, by time of day and against exposure**
  - Do: plot the cumulative net PnL, PnL by hour of the session and by day, and position size beside it.
  - Prevents: a flat or smooth PnL line that only reflects low exposure, and missing when the strategy works or fails (announcements, open vs close, regime shifts).
  - Source: Week 9 lecture, Time Series Observations (1) and (2)

- **Check results do not rest on a few trades**
  - Do: rerun without the best few trades or days and report the change.
  - Prevents: a strategy carried by a few outliers, which are generally unpredictable.
  - Source: [Palomar, Portfolio Optimization, 8.2](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html). Specific test is inference.

- **Test stability across sub-periods**
  - Do: split results by week or volatility regime and report each. A real edge should hold in most of them.
  - Prevents: an edge from one stretch of market conditions. Financial relations change through time and are not physical constants.
  - Source: [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 5a, the Student Guide, Step 6

## Refinement

- **Refine on in-sample, then run baseline and refined out-of-sample once each**
  - Do: make all refinements using in-sample data only. Then run both the baseline and the refined strategy on the locked out-of-sample period and report both, including if the refinement does worse.
  - Prevents: the out-of-sample period becoming a tuning set, which the Student Guide's baseline vs refined comparison invites.
  - Source: the Student Guide, Step 7, Week 9 lecture, Implementation of optimisation. Two-run design is inference.

- **Give each refinement a reason before testing it**
  - Do: write why a change should help (e.g. a cost, adverse selection or time-of-day finding from evaluation) before running it.
  - Prevents: tweaking to fit. Tweaks made after weak results "will generally lead to further overfitting".
  - Source: [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 5c

- **Make realism the first refinement**
  - Do: start with fixes that make the backtest more honest (costs, latency, fill rules, short limits, session handling), not ones that make it more profitable.
  - Prevents: a refined result that improved by fitting rather than by better modelling. The Student Guide lists more realistic modelling first among refinements.
  - Source: the Student Guide, Appendix A.7. Ranking is inference.

- **Prefer removing to adding**
  - Do: try dropping parameters and filters before adding variables. A more complex model must win by an economically significant amount.
  - Prevents: added complexity that widens the gap between backtest and live results.
  - Source: Week 9 lecture, Implementation of optimisation, [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 6b

- **Report the failures**
  - Do: include the trial count, failed variants and a negative out-of-sample result in the critical reflection.
  - Prevents: selection bias in the write-up. "Most experiments will fail to uncover a positive result."
  - Source: [Arnott, Harvey and Markowitz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654), Category 7a

## Risk Management

- **Set risk limits as rules before the backtest**
  - Do: define position limits, notional caps, stop-loss logic and a kill switch as part of the trading plan, and include them in every run.
  - Prevents: large losses the plan never bounded. "A dollar saved is more than a dollar earned."
  - Source: Week 9 lecture, Why do we need risk management? and Pre-Production

- **Size positions by risk, not by conviction**
  - Do: scale position size by recent volatility or a fixed risk budget, using background analysis of volatility.
  - Prevents: PnL that reflects leverage and volatility regime, not skill.
  - Source: Week 9 lecture, Time Series Observations (2), the Student Guide, Appendix A.3

- **Report drawdown and tail risk**
  - Do: report maximum drawdown, value-at-risk and expected shortfall alongside returns.
  - Prevents: judging a strategy on average return when its losses are what end it.
  - Source: the Student Guide, Appendix A.5, Week 9 lecture, Why do we need risk management?

- **Stress test against shocks**
  - Do: rerun on the worst days in the sample and on simulated shocks (wider spreads, a price gap, a stale feed).
  - Prevents: a strategy only tested on the path that happened. Geopolitical instability "tends to kill most algorithmic strategies".
  - Source: Week 9 lecture, Pre-Production and Interesting Risks, [Palomar, Portfolio Optimization, 8.3](https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html)

- **Name the risks you did not model**
  - Do: in the critical reflection, go through each risk type and say how it affects the result:
    - model risk (overfit, mis-specified, unstable)
    - liquidity risk
    - execution and operational risk
    - data risk
    - crowding risk
  - Prevents: a reflection that lists generic limits instead of evidence-based ones.
  - Source: Week 9 lecture, Types of Risks (1) and (2), the Student Guide, Step 8

## Sources

- [Bailey, Borwein, López de Prado and Zhu (2014), Pseudo-Mathematics and Financial Charlatanism](https://carmamaths.org/jon/backtest.pdf): expected best in-sample Sharpe from N skill-less trials, minimum backtest length, reporting the trial count.
- [Bailey and López de Prado (2014), The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf): inputs for correcting the Sharpe for trials, sample length and non-normal returns.
- [Harvey and Liu (2015), Backtesting](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF): Sharpe haircut numbers, limits of out-of-sample tests, split-size trade-off.
- [Arnott, Harvey and Markowitz (2019), A Backtesting Protocol in the Era of Machine Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654) (SSRN working paper): ex ante hypothesis, trial tracking, fixed samples, outliers, iterated out-of-sample, costs, no tweaking, simplicity.
- [López de Prado (2018), The 10 Reasons Most Machine Learning Funds Fail](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf): research through backtesting, label leakage and purging, limits of walk-forward.
- [Lo (2002), The Statistics of Sharpe Ratios](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf): Sharpe standard error and annualisation under serial correlation.
- [Novy-Marx and Velikov, A Taxonomy of Anomalies and Their Trading Costs](https://www.nber.org/papers/w20721): costs reduce every anomaly, 50% monthly turnover threshold.
- [Ødegaard, Trading costs: The Roll Spread Estimator](https://ba-odegaard.no/teach/notes/liquidity_estimators/roll_spread_estimator/roll_lectures.pdf): bid-ask bounce induces serial correlation in trade prices.
- [Hasbrouck, Securities Trading, Chapter 8: Transaction costs](https://pages.stern.nyu.edu/~jhasbrou/Teaching/POST%202015%20Fall/classNotes/STPPTradingCosts.pdf): implementation shortfall, cost components, opportunity cost of unfilled limit orders.
- [DeLise (2024), The Negative Drift of a Limit Order Fill](https://arxiv.org/abs/2407.16527): limit fills coincide with adverse price moves.
- [Carver, The three kinds of overfitting](https://qoppac.blogspot.com/2015/11/the-three-kinds-of-overfitting.html): explicit, implicit and tacit fitting, degrees of freedom, fitting only on past data.
- [Chan (2015), Beware of Low Frequency Data](https://epchan.blogspot.com/2015/04/beware-of-low-frequency-data.html): bar vs quote data results, opening auction constraints.
- [Chan (2016), Really, Beware of Low Frequency Data](https://epchan.blogspot.com/2016/09/really-beware-of-low-frequency-data.html): end-of-day edge that vanished on bid-ask quotes.
- [Palomar, Portfolio Optimization, 8.2 The Seven Sins of Quantitative Investing](https://portfoliooptimizationbook.com/book/8.2-seven-sins.html): look-ahead, survivorship, storytelling, outliers, shorting limits.
- [Palomar, Portfolio Optimization, 8.3 Dangers of Backtesting](https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html): backtests discard rather than improve, full specification before testing, stress tests.
