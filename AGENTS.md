# FINS3666 Group Project

Context for agents working on the FINS3666 group project: a model of order flow in 10-year Treasury note futures, ZN (Appendix B track), fitted, validated, refined and written up as a report.

- Follow `plan.md`. It sets the model and how each report section is executed
- Check decisions against `rubric.md` (what the marker wants) and `principles.md` (validation practice)
- Splits: fit on November 2025 to April 2026, validate on May to August 2026, test on September and October 2026. Refinement may use the validation months. The test months are locked: run them once, after every model is frozen, and only when Jimmy says so

## References

- Assignment spec: the Student Guide and Peer Evaluation PDFs
- Lecture slides: `lectures/`
- Data: `data/databento/ZN_tbbo/`, a year of ZN trades with quotes bought from Databento. `data/news_releases.csv` holds the release calendar. `data/FINS3666_Core_Data/` is the course's GLD and gold futures package
- Analysis code: `analysis/`. `run.py` goes from the raw files to every table and figure in `output/`. `fetch.py` downloads more data. To change the data or the split, edit `config.json`
- First version on GLD, kept for the comparison table: `exploration/gld_first_version/` with outputs in `output/gld/`- Python environment: `.venv/`
