# Plan

How to execute each section of the FINS3666 group report: a model of the orders arriving in 10-year US Treasury note futures (ZN), fitted, validated and refined (Appendix B track). Check every choice against the rubric and the principles doc.

## Model

- **Objective:** predict three things about the orders hitting ZN's order book on CME, as a market maker quoting there would need, then link them to price
  - When the next order arrives
  - How big it is
  - Whether it is a buy or a sell
- **Baseline, the Week 4 Ho and Stoll picture:**
  - Arrivals at one constant rate (Poisson)
  - Sizes from one lognormal curve
  - Direction as a fixed chance
- **Refined:**
  - Arrival rate that follows the time of day, rises on scheduled news days and rises briefly after each order (self-exciting, or Hawkes)
  - Sizes from the lognormal curve plus extra weight at common sizes and a separate curve for the largest 2%
  - Direction that depends on the previous order, its size and the quote sizes on each side
- **Price link:** the spread a quote earns against what it gives back
- **Result:** all three parts improve on unseen months. Arrivals and direction improve a lot, sizes a little

## Shared Rules

- **Scope:** front-month ZN, 08:20 to 15:00 New York time. Leave out US holidays, early closes and the two days before each contract roll. E.g. 22 days in the year
- **Split:**
  - Fit on November 2025 to April 2026, 116 days
  - Validate on May to August 2026, 80 days. Refinement may use these months
  - Test on September and October 2026, once, after every model is frozen. Done. Don't run them again
  - A statistic added after the test run is reported on the validation months only
- **Trade direction:** the exchange's aggressor side. Drop trades with none. E.g. 1.1% of trades
- **Order merge:** same-side trades under 0.1 milliseconds apart count as one order. E.g. 6.8m trades become 3.6m orders. Below that gap same-side trades outnumber opposite-side ones by 28 times or more, above it by about 5
- **Main score:** joint log-likelihood per order, named before the test. E.g. baseline -4.80, refined -0.89 on validation
- **Other scores, the same for baseline and refined:**
  - Arrivals: fit error of the gaps between orders (0 is perfect), error in orders per 10 minutes
  - Sizes: predicted against actual share of large orders
  - Direction: log-loss and accuracy
- **Simulation check:** before any unseen data, compare simulated days from each model with real days on net buying per minute and the busiest second
- **Variant log:** record every model variant fitted and report the count. It stands at 34
- **Reproducibility:** one script runs from the raw files to every table and chart
- **Intention:** every section, table, score and line of code solves a stated problem. State the reading of each result so the marker does not have to work it out

## Report Sections

### 1. Introduction

- **Why:** the marker wants a clear objective, why order flow matters to market making, and a reason for choosing the topic
- **Source:** Appendix B.2, Week 2 (order flow and imbalance), Week 4 (Ho and Stoll, Kyle, Glosten and Milgrom)
- **Destination:**
  - 250 words
  - Table 1: why ZN, against the first version on GLD
    - Share of the market seen: all of it against one of 16 exchanges
    - Trades with a side from the exchange: 98.9% against 76%
    - Merge window: 0.1 ms set by the data against 1 ms chosen
    - Sample: 12 months against 10 days
- **Method:**
  1. State the objective as three claims to test, one per model part, plus the price link
  2. Explain what a market maker does with each: how often quotes get hit, how large the hits are, whether they pile up on one side
  3. Answer the four B.2 questions in one line each
  4. Use Table 1 to show the dataset was chosen for the model
  5. List assumptions: aggressive orders rebuilt from public trades stand in for client flow, since public data cannot identify clients (Week 8)

### 2. Data and Background Analysis

- **Why:** the marker wants reproducible cleaning and background findings that each set a modelling choice
- **Source:** the fit months only
- **Destination:**
  - 350 words
  - Table 2: dataset summary (months, days, trades, orders, days left out and why, share with no side)
  - Figure 1: orders per 10 minutes on days with and without a release
  - Figure 2: order sizes, share of orders beside share of contracts
  - Figure 3: same-side against opposite-side gaps between trades, the basis of the merge window
  - Table 3: each finding beside the modelling choice it sets
- **Method:**
  1. Clean per Shared Rules and state the timestamp resolution (Week 8)
  2. Produce the six B.3 items
     - Orders per minute. E.g. 76
     - Average and median size. E.g. median 3 contracts, mean 38
     - Size distribution. E.g. the largest 1% of orders carry 31% of contracts
     - Buy against sell volume. E.g. daily buy share 0.46 to 0.54
     - Imbalance through time
     - Time between orders. E.g. median 2 milliseconds, mean 0.79 seconds
  3. Link each finding to a choice
     - Orders cluster, so arrivals need a self-exciting model
     - Activity jumps fivefold in the 10 minutes after an 8:30 release, so the rate needs a news input
     - 38% of orders are 1 contract and sizes pile up at 5, 10, 25, 50 and 100, so the size model needs extra weight there
     - An order repeats the previous side 83% of the time after a 1-contract order and 57% after a 1,000+ order, so direction needs the previous size

### 3. Model Development

- **Why:** the marker wants a formal model of arrivals, size and imbalance, with its logic and assumptions explained
- **Source:** Table 3, Week 4 notation for the baseline
- **Destination:**
  - 300 words
  - Table 4: the three parts, baseline beside refined, each with its formula and parameters
- **Method:**
  1. Write the baseline as Ho and Stoll's two Poisson streams, a buy rate and a sell rate
  2. Write each refined part as a formula and say in one line what it adds
  3. State assumptions: one process within a day, sizes independent of arrival times
  4. Explain how each part relates to trading behaviour. E.g. self-excitation as algorithms slicing orders and reacting to each other (Week 1 order types), queue sizes as traders hitting the thin side (Week 2)
  5. Name the Week 9 model risk: a non-convex fit, so use two starting points and report both

### 4. Testing and Validation

- **Why:** the marker wants predicted against realised on unseen data, in-sample beside out-of-sample
- **Source:** the frozen models, the split
- **Destination:**
  - 250 words
  - Table 5: baseline scores on the fit months. The refinement table carries every model on fit, validation and test months
  - Figure 5: predicted against actual orders by time of day
- **Method:**
  1. Quote the Week 7 pipeline: fit, validate with refinement, then a test used once
  2. Name the main score and the other scores before showing results
  3. State the split dates and that the test ran once

### 5. Evaluation

- **Why:** the marker wants accuracy, stability, robustness and limitations, and whether the result is economically meaningful
- **Source:** baseline results on the fit months
- **Destination:**
  - 500 words
  - Figure 4: actual gaps between orders against what each model predicts
  - Table 6: simulated days against real days. E.g. net buying per minute, baseline 940 contracts against an actual 1,418
  - Table 7: what a quote earns and gives back
- **Method:**
  1. Name the baseline's failures from Table 5, one per model part. E.g. fit error 0.63, counts per 10 minutes off by 49%
  2. Build Table 6 from 200 simulated days per model (Week 7 Monte Carlo)
  3. Build Table 7 with the Week 3 spread measures
     - A quote earns 0.50 of a price step when hit and gives back 0.35 within 10 seconds
     - Per contract it keeps nothing, since large orders take the whole half-spread back
     - The result holds at every horizon from 1 to 60 seconds
  4. State that this is a market-wide figure and not one market maker's result, since Week 7 says market making cannot be backtested
  5. Pass each failure to section 6

### 6. Refinement

- **Why:** the marker wants each change driven by a section 5 finding, re-estimated and compared with the original
- **Source:** section 5 failures, fit and validation months
- **Destination:**
  - 400 words
  - Table 8: one row per change on fit, validation and test months, so the marker sees what each change bought and that the refined model is not overfitted
  - Table 9: direction accuracy by time since the previous order. E.g. 99% under 1 ms, 66% after 1 second
  - Table 10: adverse selection by state. E.g. a quote keeps 0.25 of a price step in the calm half of orders and 0.03 in bursts
  - Table 11: the quoting rule against always quoting
  - Table 12: variants tried and dropped
- **Method:**
  1. Apply one refinement per failure, mapped to B.7
     - Time of day and news: a rate per 10 minutes and per kind of release day
     - Alternative approach: self-exciting arrivals
     - Parameters: common sizes and a separate tail curve
     - Buy and sell classification: the merge window
     - Extra market variables: previous order size and quote sizes
  2. Tune the merge window and the number of decay speeds together on the fit months, the Week 9 two-at-a-time method
  3. Drop what does not help, per the Week 9 rule. E.g. a third decay speed (0.048 against 0.053) and a five-order memory (accuracy unchanged)
  4. Report in one line the two faults the simulation check caught before validation: a tail that was too heavy, and direction that ignored size
  5. Test the Week 9 claim that more trades mean more toxic flow, using the model's own burst measure (Table 10)
  6. State the horizon of each gain. E.g. self-exciting arrivals cut the fit error and leave the 10-minute count error at 38%
  7. Report the quoting rule as losses avoided against always quoting on the same days. E.g. 0.022 of a price step at 1 ms, standard error 0.005
  8. Report the stability of the refined fit error by month and by kind of day (Week 7 stress test on news days)

### 7. Critical Reflection

- **Why:** the marker wants strengths and limits backed by data, and three feasible recommendations with reasoning
- **Source:** every table and figure above
- **Destination:**
  - 450 words
  - Table 13: each strength and limit beside the table or figure that shows it
  - Exactly three recommendations
- **Method:**
  1. Strengths: a locked test used once, a year of data, the simulation check, variant count disclosed
  2. Limits, each with its number
     - Activity over the next 10 minutes is a third off, mostly within the day. E.g. count error 38%, 33% with the day's total known
     - The day's activity level is unexplained. E.g. real days range from 18,000 to 47,000 orders, simulated days from 28,000 to 30,000
     - The busiest second is understated. E.g. 97 orders against an actual 182
     - The best model still fails a strict fit test
     - Aggressive orders only, with no client identity and no resting orders
     - The spread is one price step 99.9% of the time, so the lecture claim that spreads widen with risk shows up in what a quote keeps and not in the spread
     - The quoting rule avoids losses but keeps nothing after fees
  3. Recommendations
     - Update the rate through the day from the last 10 minutes and a trailing daily level
     - Add resting orders and cancellations from full order book data
     - Run the model live as a monitor of bursts and model drift (Week 9)

## Sequence

1. **Sun 15 to Mon 16 Nov:** write the seven sections in prose from the dot-point draft
2. **Tue 17 Nov:** team review of method and results
3. **Wed 18 Nov:** cut to 2,500 words, write the AI disclosure and roles table
4. **Thu 19 Nov:** check against the rubric, convert to PDF
5. **Fri 20 Nov, before 3pm:** submit the report and code. Each member completes the peer evaluation
