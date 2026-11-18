# FINS3666 Group Project

Context for agents working on the FINS3666 group project: a model of order flow in 10-year Treasury note futures, ZN (Appendix B track), fitted, validated, refined and written up as a report.

- Follow `plan.md`. It sets the model and how each report section is executed
- Check decisions against `rubric.md` (what the marker wants) and `principles.md` (validation practice)
- Every section, table, score and line of code must solve a stated problem. Don't add analysis for observation. State the reading of each result so the marker does not have to work it out
- Splits: fit on November 2025 to April 2026, validate on May to August 2026, test on September and October 2026
- The models are frozen and the test months have been run once. Don't refit and don't run the test months again unless Jimmy says so
- A statistic added after the test run is reported on the validation months only
- Don't change the GitHub repo's visibility. Jimmy does that himself

## Validation residuals

- If asked for arrival residuals, run `python analysis/export_validation_residuals.py` from the repo root. It uses the frozen parameters in `output/model.json` and only the May–August validation files. The README lists the inputs and the validation-only download command.
- Read the local Parquet with `pandas.read_parquet`. Use `rescaled_gap_self_exciting` for the refined model. Group by `session_date_ny` and sort by `order_in_session` before computing lags. Pair gaps only within the same session. The first exported row ends at order 2.
- For Ben's Anderson–Darling check, the reference distribution is exponential with mean 1. Use it to check the gap distribution and use autocorrelation to check serial dependence. Report effect sizes and account for dependence if reporting p-values.
- For day-by-day stability, compare the daily KS distance against the same exponential reference used in the report. Include each day's gap count and use `session_kind` for news-day comparisons. Explain whether errors are widespread or concentrated in a few sessions.
- Keep the raw files, the per-order export, and API keys out of Git. Report any new diagnostics on the validation months only.

## References

- Report draft: `report.md`, in dot points for the team to write up
- Setup and how to run: `README.md`
- Repo: https://github.com/JimmyC41/zn-order-flow
- Analysis code: `analysis/`. `run.py` goes from the raw files to every table and figure in `output/`. `fetch.py` downloads the data. To change the data or the split, edit `config.json`
- Data: `data/databento/ZN_tbbo/`, a year of ZN trades with quotes bought from Databento. `data/news_releases.csv` holds the release calendar. `data/FINS3666_Core_Data/` is the course's GLD and gold futures package
- First version on GLD, kept for the "why ZN" comparison in Table 1: `exploration/gld_first_version/` with outputs in `output/gld/`
- Assignment spec: the Student Guide and Peer Evaluation PDFs
- Lecture slides: `lectures/`
- Python environment: `.venv/`
- Local only, not in the repo: the data folders apart from the release calendar, the PDFs, `lectures/` and `.venv/`
