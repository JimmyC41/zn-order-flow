# FINS3666 Group Project

Context for agents working on the FINS3666 group project: a model of order flow in 10-year Treasury note futures, ZN (Appendix B track), fitted, validated, refined and written up as a report.

- Follow `plan.md`. It sets the model and how each report section is executed
- Check decisions against `rubric.md` (what the marker wants) and `principles.md` (validation practice)
- Every section, table, score and line of code must solve a stated problem. Don't add analysis for observation. State the reading of each result so the marker does not have to work it out
- Splits: fit on November 2025 to April 2026, validate on May to August 2026, test on September and October 2026
- The models are frozen and the test months have been run once. Don't refit and don't run the test months again unless Jimmy says so
- A statistic added after the test run is reported on the validation months only
- Don't change the GitHub repo's visibility. Jimmy does that himself

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
