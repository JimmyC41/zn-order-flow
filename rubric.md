# Rubric

What the marker wants to see in the FINS3666 group project (Appendix B track: a client order flow model), drawn from the Student Guide, the Peer Evaluation guide and the Week 9 lecture. Check every model, scope, analysis and report decision against it. Lines marked "(inference)" are a reading of the spec, not its text.

## Hard Requirements

- **Due:** End of Week 10, Friday 3:00pm, 20 Nov 2026, via Moodle
- **Weighting:** 30% of the course
- **Task:** Written report
- **Word limit:** 2,500 words maximum, excluding charts, tables, diagrams, etc.
- **Submission items:**
  - Report detailing methodology, analysis and conclusions, as a PDF
  - Code: any computational scripts (e.g. Python, R) or spreadsheets (e.g. Excel) used for background analysis
  - AI Transparency: if AI tools (e.g. ChatGPT, ML models) are used, document their usage and provide justification
- **Data:**
  - "Preferably based on real historical intraday data" (Overview)
  - Topics must allow "meaningful quantitative modelling and empirical testing using historical intraday market data" (Step 2)
  - Split into an initial sample for backtesting and a separate out-of-sample set for testing (Step 3)
  - In-sample must be "an initial but limited data sample" and out-of-sample "a separate unseen historical dataset" (Step 5)
  - All students have FactSet access. Contact the Course Convenor if stuck finding a dataset
- **Peer evaluation:**
  - Done individually in the Moodle Team Evaluation Tool, by the same deadline as the report
  - 10% penalty for not completing it
  - Rate each teammate (not yourself) 0 to 6 on overall contribution: 0 = 0%, 1 = 25%, 2 = 50%, 3 = 65%, 4 = 75%, 5 = 85%, 6 = 100%
  - Five criteria at 20% each: engagement and participation, technical contribution, communication and collaboration, timeliness and reliability, critical thinking and initiative
  - Reflective commentary of 2 to 3 sentences per teammate: most valuable contribution, challenges and how they were handled, whether their grade should differ from the group grade
  - Declare any use of AI tools
- **Roles (suggested, Step 1):**
  - Trading: market and product knowledge
  - Quantitative Analysis: statistics and modelling
  - Technology: coding
  - Sales: writing the report
  - Risk: strategy evaluation and refinement

## Marking Criteria

Four criteria, 25% each. Higher marks go to "logical data driven arguments" and "understanding that extends beyond the standard course materials" (Section 2).

### Topic Selection, Data Processing and Strategy Implementation

- **HD:** "Topic selection is very challenging with clear, logical, well-structured implementation. Uses advanced models and work is explained well."
- **HD vs D:** D is "moderate to advanced" difficulty with "some areas for refinement". HD needs a very challenging topic, advanced models and no loose ends in the build. C notes "further explanation or optimisation is required", so optimisation is expected above C
- **What the marker wants to see (inference):**
  - A stated research objective and a reason for picking this topic over others
  - Every key assumption written down
  - Cleaning steps for the raw intraday data, stated and reproducible from the code
  - A model specified precisely enough that a reader could re-implement it
  - At least one model or technique beyond descriptive statistics (see Beyond Course Material)

### Analysis and Evaluation

- **HD:** "Executes comprehensive analysis & testing of the trading strategy or model. Uses additional metrics effectively and provides insightful evaluation."
- **HD vs D:** D is "solid analysis & evaluation, although some improvements could be made". HD adds metrics beyond the basics and draws insight from them, not just a results table
- **What the marker wants to see (inference):**
  - Results for both in-sample and out-of-sample
  - Predicted against realised for arrivals, sizes and imbalance (B.5)
  - Each result interpreted: does the model perform as expected, and is it economically meaningful, stable and robust
  - Claims backed by numbers, tables or charts

### Critical Reflection, Refinement and Recommendations

- **HD:** "Provides deep, evidence-based reflections on strengths and limitations. Proposes innovative and feasible improvements."
- **HD vs D:** D is "well-supported" but with "some areas for deeper analysis or stronger justification". HD backs every strength and limitation with data and makes recommendations both new and doable
- **What the marker wants to see (inference):**
  - A refined model, re-estimated and compared side by side with the original
  - Each limitation tied to a specific result
  - Exactly three real-world recommendations, each with reasoning

### Group Collaboration and Contribution

- **HD:** "Demonstrates exceptional teamwork. All members contribute equitably, leveraging diverse skills. Evidence of collaboration, role distribution & support."
- **HD vs D:** D allows "some minor imbalances in workload". HD needs equal contribution and visible evidence of it
- **What the marker wants to see (inference):**
  - Roles assigned to each member and matched to their skills
  - Evidence of who did what (e.g. a contribution table, commit history)
  - Peer evaluation scores consistent with an even split

## Report Sections

Seven sections "most projects will typically include" (Section 2). The format "may vary depending on the topic selected". For this track, Step 4 requires "a formal quantitative model rather than relying solely on descriptive statistics".

### 1. Introduction and Research Objective

- **Spec:**
  - Clear statement of the objective of the analysis (B.2)
  - Why client order flow is important to market making strategies (B.2)
  - What order flow helps market makers estimate (Appendix B):
    - Expected trading activity and quote execution frequency
    - Potential inventory accumulation and inventory risk
    - The balance between buying and selling pressure
  - Questions to define the objective (B.2):
    - How frequently do trades arrive in the market?
    - What is the typical size of trades executed?
    - Is order flow balanced between buying and selling?
    - Does client order flow vary during the trading day?
  - Example objectives (B.2):
    - Estimating the arrival rate of trades
    - Modelling the distribution of trade sizes
    - Analysing order flow imbalance between buyers and sellers
- **Marker wants (inference):** an objective stated as testable claims, and a data-driven reason for choosing this topic
- **Week 9 (Interesting Risks):** market makers often get crowded out when liquidity becomes too high

### 2. Data Description and Background Analysis

- **Spec:**
  - Obtain a dataset and clean it with a tool of your choice (Step 3)
  - Split into an initial sample and a separate out-of-sample set (Step 3)
  - Example background analysis (B.3):
    - Number of trades per minute or per hour
    - Average and median trade size
    - Distribution of trade sizes
    - Buy volume versus sell volume
    - Order flow imbalance over time
    - Time between successive trades
- **Marker wants (inference):** each background finding linked to a specific modelling choice in Section 3, and a clear rationale for the dataset chosen (a stated course objective is selecting appropriate datasets)
- **Week 9 (Data Risk, Data Quality and Integrity, garbage in = garbage out):** check for missing ticks, jumps, timestamp gaps, vendor inconsistencies and symbol mismatches

### 3. Model Development

- **Spec:**
  - Develop a model for client order flow and explain its logic and assumptions (B.4)
  - At minimum, a formal model of all three (B.4):
    - Trade arrival rates
    - Trade size
    - Order flow imbalance
  - Possible approaches (B.4):
    - Trade arrival rate models
    - Inter-arrival time models
    - Trade size distribution models
    - Directional order flow models
  - Explain how the model is constructed and how it relates to the behaviour of client order flow (B.4)
- **Marker wants (inference):** each of the three parts written as a formula with its parameters named, and every assumption stated
- **Week 9 (Types of Risks):** model risk, where a model is wrong, mis-specified, overfit or unstable

### 4. Testing and Validation

- **Spec:**
  - Estimate or calibrate on an initial, limited sample (Step 5)
  - Validate on a separate unseen dataset (Step 5)
  - An introductory evaluation of in-sample and out-of-sample results (Step 5)
  - Example tests (B.5):
    - Predicted trade arrival rates against realised trade activity
    - Predicted trade size distributions against realised trade sizes
    - Predicted buy and sell imbalance against realised order flow
    - Whether the model performs consistently across different trading days or market conditions
  - Advanced (B.5): stability across trading days, volatility regimes or market conditions
- **Marker wants (inference):** the same scores for in-sample and out-of-sample so the gap is visible
- **Week 9 (Pre-Production, Backtest to Live Consistency):** same data fields and transformations in both samples, with reproducible results

### 5. Evaluation of Results

- **Spec:**
  - Interpret Step 5 results: does the model perform as expected, and are results economically meaningful, stable and robust (Step 6)
  - Assess (B.6):
    - Model accuracy and stability
    - Robustness across different time periods
    - Limitations of the modelling approach
- **Marker wants (inference):** where the model fails, shown with numbers, and what the model means for a market maker's fills and inventory
- **Week 9 (Other Considerations):** skill or luck
- **Week 9 (Time Series Observations):**
  - Activity through time shows intraday swings, day-to-day variability and regime shifts
  - Shows event sensitivity: vol spikes around announcements, market-open vs close
  - Caveat: short-run patterns are mostly noise, so don't adjust the model on patterns that aren't there
- **Week 9 (Adverse Selection):** more trades means more contact with toxic flow and worse average execution

### 6. Model Refinement

- **Spec:**
  - Refine based on Step 6 findings (Step 7)
  - Explain every change (Step 7)
  - Repeat the estimation and testing with the refined approach and compare with the original (Step 7, B.7)
  - Example refinements (B.7):
    - Adjusting model parameters
    - Incorporating additional market variables
    - Analysing different time periods of the trading day
    - Refining the classification of buy and sell initiated trades
    - Testing alternative modelling approaches
- **Marker wants (inference):** a before and after table on the same scores, and refinements driven by a weakness found in Section 5
- **Week 9 (Implementation of Optimisation):**
  - Throw away useless factors and parameters
  - Start in-sample
  - Optimise two parameters at a time, most impactful first, using a grid of results
  - Lock in the best ranges, then move to the next pair
  - Then run out-of-sample
- **Week 9 (Drawbacks of Optimisation):**
  - Garbage in = garbage out
  - Overfitting
  - Small sample size
  - Local vs global optimum (non-convex problem)
  - Elasticity of parameters
  - Computational costs

### 7. Critical Reflection and Recommendations

- **Spec:**
  - Strengths and limitations, supported with data, tables or charts (Step 8)
  - Top three recommendations to improve the implementation in a real-world setting, with reasoning (Step 8)
- **Marker wants (inference):** recommendations that are specific and feasible, each tied to a limitation found earlier
- **Week 9 (Monitoring):** what to track live
  - Market data health
  - Execution quality: slippage, fill rates
  - Liquidity conditions
  - Signal and model health: prediction drift, abnormal spikes
- **Week 9 (Strategy Scaling):** how would you trade at $1 million exposure vs $1 billion

## Beyond Course Material

The spec signals "advanced" in these places:

- **Section 2:** higher marks for "logical data driven arguments" and understanding "beyond the standard course materials"
- **Marking HD descriptors:**
  - "Uses advanced models"
  - "Uses additional metrics effectively"
  - "Innovative and feasible improvements"
- **Step 4:** "a formal quantitative model rather than relying solely on descriptive statistics"
- **B.5 ("advanced students"):** stability across trading days, volatility regimes or market conditions
- **B.7 refinements:**
  - Additional market variables
  - Time-of-day periods
  - Better buy and sell classification
  - Alternative modelling approaches
- **C.5, C.7 (inventory track):**
  - Simulated stress conditions (e.g. sustained directional flow, high volatility)
  - Adverse selection or price impact
  - Hedging with correlated instruments (e.g. futures)
- **Appendix E:**
  - Probability of Informed Trading
  - Optimal execution: market vs limit orders from order book and trade feed data
  - Combining order book and trade feed data
- **Peer Evaluation:** "extended ideas beyond coursework, e.g. applying AI, ML, advanced risk techniques"

## Scope Implications

Constraints the model must meet to make HD possible. All are inference from the sources.

- **Data:** real trade-by-trade data with a buy or sell side, long enough to fit on some days and validate on others
- **Coverage:** all three parts modelled: arrival rates, trade size and imbalance
- **Challenge:** a model beyond constant rates and averages, since descriptive statistics alone cap Criterion 1
- **Baseline and refined versions:** both fitted on the same data and compared on the same scores
- **Validation:** predicted against realised on unseen days, for each part
- **Stability:** results split by day and time of day
- **Market making link:** what the model implies for quote hits, inventory and adverse selection
- **Report size:** fits in 2,500 words with charts and tables carrying the detail
- **Team:** splits into distinct pieces of work so each member owns a visible part (Criterion 4)

## Open Questions

- **Dataset submission:** the submission list names report, code and AI transparency, not the data. Do we submit the dataset or a sample of it?
- **Word count:** does "etc." exclude references, appendices, the AI transparency statement and code listings?
- **AI transparency format:** a section in the report, an appendix or a separate file?
- **Collaboration evidence:** Criterion 4 is 25% of the mark, but the spec doesn't say how teamwork is shown in the report. Is it peer evaluation alone, or should the report include a contribution or roles table?
- **Peer evaluation timing:** the tool "will become available once your Group Project has been submitted", yet it is due "by the same deadline". Confirm the actual close date
- **10% penalty:** 10% of the project mark or of the course? Do peer scores adjust individual grades?
- **"Client" order flow:** the spec says client order flow. Our trades come from one exchange and carry no client identity. Is exchange trade flow an accepted stand-in?
- **Sample length:** the spec sets no minimum. Is five days to fit and five to validate enough for the marker?
- **Lecture coverage:** only the Week 9 slides are available. Other weeks may hold concepts the marker expects to see applied
