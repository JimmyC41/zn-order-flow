# Supplementary ZN validation

Open the saved tables for [news and order-flow filters](../../output/validation_supplements/news_only_increment_table.csv), [markout sensitivity](../../output/validation_supplements/sensitivity.csv) and [order-count forecasts](../../output/validation_supplements/count_forecast.tsv).

To reproduce the tables, place the team's four monthly ZN Parquet files for May–August 2026 in `data/databento/ZN_tbbo/`. From the repository root, run the following in the project's Python 3.12 environment.

```sh
python analysis/check_validation_supplements.py
```

The script prints the results directory under `data/supplementary_validation_runs/`. It also saves the quote-matching delays and a comparison with the original quoting table. Two tied fills on 2 July can shift the always-quote estimate by 0.00000244 ticks per contract. Historical capture excludes fees, queue position and fill probability.

<details>
<summary>Python packages for a fresh environment</summary>

```sh
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.18.1 pyarrow==26.0.1 numba==0.68.1 duckdb==2.0.0
```
</details>
